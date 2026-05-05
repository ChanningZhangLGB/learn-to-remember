"""
CCME (Contrastive Contextual Memory Encoder) Implementation

This module implements Section 3.3 of the paper:
- Query Encoder Eq(x): Encodes queries into L2-normalized embeddings (lines 241-242)
- Memory Encoder Em(m): Encodes memory bullets into L2-normalized embeddings (lines 243-244)
- CCME Retriever: Hybrid scoring with similarity + reliability (Equation 5)
- CCME Loss: InfoNCE contrastive objective (Equation 6)

Paper References:
- Section 3.3: Contrastive Contextual Memory Encoder (CCME)
- Lines 241-244: Separate Eq (query encoder) and Em (memory encoder)
- Lines 249-251: All embeddings are L2-normalized, sim(q,m) = q^T m
- Equation 5: Hybrid retrieval score = α·sim(Eq(x), Em(m)) + (1-α)·p̂(m)
- Equation 6: L_CCME = -log[exp(Eq·Em+)/Σ exp(Eq·Em_i)]
- Definition 1: Bayesian reliability p̂(m) = (helpful+1)/(helpful+harmful+2)

Architecture:
1. Base embedding: text-embedding-3-small (OpenAI, 1536-dim)
2. Separate trainable projection layers for Eq and Em (→768-dim)
3. L2 normalization for cosine similarity
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
from openai import OpenAI
import os
import faiss
import threading
import copy
from .utils.adapters import Adapter


class BaseEncoder(nn.Module):
    """
    Base encoder class with shared OpenAI API embedding functionality.

    Provides common embedding retrieval from OpenAI API.
    Subclasses (QueryEncoder, MemoryEncoder) add their own projection layers.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        base_dim: int = 1536,
        api_key: Optional[str] = None
    ):
        """
        Initialize base encoder.

        Args:
            embedding_model: OpenAI embedding model name
            base_dim: Dimension of base embedding model
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        super().__init__()
        self.embedding_model = embedding_model
        self.base_dim = base_dim
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _get_clip_model(self):
        """Lazy-load CLIP ViT-L/14 via transformers, on GPU if available."""
        if not hasattr(self, "_clip_model"):
            from transformers import CLIPModel, CLIPProcessor
            import torch as _torch
            _device = "cuda" if _torch.cuda.is_available() else "cpu"
            self._clip_device = _torch.device(_device)
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self._clip_device)
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self._clip_model.eval()
        return self._clip_model, self._clip_processor

    def get_base_embedding(self, text: str) -> np.ndarray:
        """Get base embedding — OpenAI API for standard models, CLIP text encoder for 'clip'."""
        if self.embedding_model == "clip":
            import torch as _torch
            model, processor = self._get_clip_model()
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self._clip_device)
            with _torch.no_grad():
                emb = model.get_text_features(**inputs).float()
                emb = _torch.nn.functional.normalize(emb, dim=-1)
            return emb.squeeze(0).cpu().numpy()
        # text-embedding-3-small has an 8192-token limit; truncate to stay safe
        try:
            import tiktoken as _tiktoken
            _enc = _tiktoken.get_encoding("cl100k_base")
            _tokens = _enc.encode(text)
            if len(_tokens) > 8000:
                text = _enc.decode(_tokens[:8000])
        except Exception:
            # fallback: character-based truncation (~4 chars per token)
            if len(text) > 32000:
                text = text[:32000]
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def get_multimodal_embedding(self, text: str, images: list) -> np.ndarray:
        """
        Fuse text and image embeddings for multimodal queries using dual-CLIP.

        When embedding_model == "clip" (multimodal tasks), both text and images are
        encoded with CLIP ViT-L/14 — the same contrastively-aligned 768d latent space —
        so dot-product similarity against memory embeddings is geometrically meaningful.

        When embedding_model != "clip" (text-only tasks), falls back to text-only
        get_base_embedding() since images should not be present for those tasks.

        Args:
            text: Query text
            images: List of raw image bytes

        Returns:
            L2-normalized fused embedding (base_dim,)
        """
        if not images or self.embedding_model != "clip":
            return self.get_base_embedding(text)

        try:
            import torch as _torch
            import io as _io
            from PIL import Image as _Image

            model, processor = self._get_clip_model()

            # Text embedding via CLIP text encoder (768d)
            text_inputs = processor(text=[text], return_tensors="pt", padding=True,
                                    truncation=True, max_length=77).to(self._clip_device)
            with _torch.no_grad():
                text_feat = model.get_text_features(**text_inputs).float()
                text_feat = _torch.nn.functional.normalize(text_feat, dim=-1)
            text_emb = text_feat.squeeze(0).cpu().numpy()  # (768,)

            # Image embeddings via CLIP visual encoder (768d), mean-pooled
            img_embs = []
            for img_bytes in images:
                try:
                    pil_img = _Image.open(_io.BytesIO(img_bytes)).convert("RGB")
                    img_inputs = processor(images=pil_img, return_tensors="pt").to(self._clip_device)
                    with _torch.no_grad():
                        img_feat = model.get_image_features(**img_inputs).float()
                        img_feat = _torch.nn.functional.normalize(img_feat, dim=-1)
                    img_embs.append(img_feat.squeeze(0).cpu().numpy())
                except Exception:
                    continue

            if not img_embs:
                return text_emb

            img_emb = np.mean(np.stack(img_embs, axis=0), axis=0)  # (768,)
            img_norm = np.linalg.norm(img_emb)
            if img_norm > 1e-8:
                img_emb = img_emb / img_norm

            # Fuse in shared CLIP space: equal-weight average, then re-normalize
            fused = 0.5 * text_emb + 0.5 * img_emb
            fused_norm = np.linalg.norm(fused)
            if fused_norm > 1e-8:
                fused = fused / fused_norm

            return fused.astype(np.float32)

        except Exception as e:
            print(f"  [WARN] Multimodal embedding failed, falling back to text-only: {e}")
            return self.get_base_embedding(text)

    def get_base_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get base embeddings for a batch of texts."""
        if self.embedding_model == "clip":
            import torch as _torch
            model, processor = self._get_clip_model()
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self._clip_device)
            with _torch.no_grad():
                embs = model.get_text_features(**inputs).float()
                embs = _torch.nn.functional.normalize(embs, dim=-1)
            return embs.cpu().numpy()
        # text-embedding-3-small has an 8192-token limit; truncate per-text to stay safe.
        # A single oversized text in the batch would otherwise fail the whole request.
        try:
            import tiktoken as _tiktoken
            _enc = _tiktoken.get_encoding("cl100k_base")
            safe_texts = []
            for t in texts:
                _tokens = _enc.encode(t)
                if len(_tokens) > 8000:
                    safe_texts.append(_enc.decode(_tokens[:8000]))
                else:
                    safe_texts.append(t)
            texts = safe_texts
        except Exception:
            texts = [t[:32000] if len(t) > 32000 else t for t in texts]
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return np.array(
            [item.embedding for item in response.data],
            dtype=np.float32
        )


class QueryEncoder(BaseEncoder):
    """
    Query Encoder Eq(x) for CCME.

    Paper Reference (lines 241-242):
    "Eq: query encoder that maps a task input x into an embedding q"

    Encodes queries into L2-normalized embeddings in the CCME latent spase.
    Has its own trainable projection layer, separate from Em.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        base_dim: int = 1536,
        projection_dim: Optional[int] = None,
        trainable: bool = False,
        api_key: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_hidden: int = 256
    ):
        """
        Initialize Query Encoder Eq.

        Args:
            embedding_model: OpenAI embedding model name
            base_dim: Dimension of base embedding model
            projection_dim: Dimension of projection layer (None = no projection)
            trainable: If True, add trainable projection layer
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            adapter_enabled: If True, use adapter instead of full projection
            adapter_hidden: Hidden dimension for adapter
        """
        super().__init__(embedding_model, base_dim, api_key)

        self.projection_dim = projection_dim if projection_dim else base_dim
        self.trainable = trainable
        self.adapter_enabled = adapter_enabled

        # Eq-specific projection layer (separate from Em)
        if self.adapter_enabled:
            self.adapter = Adapter(
                d_in=self.base_dim,
                d_hidden=adapter_hidden,
                d_out=self.projection_dim
            )
            self.projection = None
        elif self.trainable and self.projection_dim != self.base_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.base_dim, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim)
            )
            self.adapter = None
        else:
            self.projection = None
            self.adapter = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Eq projection layer and L2 normalization.

        Args:
            embeddings: Base embeddings (batch_size, base_dim)

        Returns:
            L2-normalized query embeddings (batch_size, projection_dim)
        """
        if self.adapter is not None:
            embeddings = self.adapter(embeddings)
        elif self.projection is not None and self.trainable:
            embeddings = self.projection(embeddings)

        # L2 normalization (paper lines 249-251)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode_query(self, query: str, with_grad: bool = False, images: list = None):
        """
        Encode a query into L2-normalized embedding.

        For multimodal queries (images provided), fuses CLIP visual features
        with the text embedding before projection. See get_multimodal_embedding().

        Args:
            query: Input query/question string
            with_grad: If True, return tensor with gradients enabled
            images: Optional list of raw image bytes for multimodal queries

        Returns:
            L2-normalized embedding q = Eq(x) (projection_dim,)
        """
        if images:
            base_emb = self.get_multimodal_embedding(query, images)
        else:
            base_emb = self.get_base_embedding(query)

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        base_tensor = torch.from_numpy(base_emb).unsqueeze(0).to(device)

        if with_grad:
            if self.trainable or self.adapter_enabled:
                normalized_emb = self(base_tensor)
            else:
                normalized_emb = F.normalize(base_tensor, p=2, dim=-1)
            return normalized_emb.squeeze(0)

        with torch.no_grad():
            if self.trainable or self.adapter_enabled:
                normalized_emb = self(base_tensor)
            else:
                normalized_emb = F.normalize(base_tensor, p=2, dim=-1)

        return normalized_emb.squeeze(0).cpu().numpy()

    def encode_queries_batch(self, queries: List[str], with_grad: bool = False):
        """
        Encode a batch of queries.

        Args:
            queries: List of query strings
            with_grad: If True, return tensor with gradients enabled

        Returns:
            L2-normalized embeddings (batch_size, projection_dim)
        """
        base_embs = self.get_base_embeddings_batch(queries)

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        base_tensor = torch.from_numpy(base_embs).to(device)

        if with_grad:
            if self.trainable or self.adapter_enabled:
                normalized_embs = self(base_tensor)
            else:
                normalized_embs = F.normalize(base_tensor, p=2, dim=-1)
            return normalized_embs

        with torch.no_grad():
            if self.trainable or self.adapter_enabled:
                normalized_embs = self(base_tensor)
            else:
                normalized_embs = F.normalize(base_tensor, p=2, dim=-1)

        return normalized_embs.cpu().numpy()


class MemoryEncoder(BaseEncoder):
    """
    Memory Encoder Em(m) for CCME.

    Paper Reference (lines 243-244):
    "Em: memory encoder that maps an item m into an embedding m"

    Encodes memory bullets into L2-normalized embeddings.
    Has its own trainable projection layer, separate from Eq.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        base_dim: int = 1536,  # text-embedding-3-small dimension
        projection_dim: Optional[int] = None,
        trainable: bool = False,
        api_key: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_hidden: int = 256
    ):
        """
        Initialize Memory Encoder Em.

        Args:
            embedding_model: OpenAI embedding model name
            base_dim: Dimension of base embedding model
            projection_dim: Dimension of projection layer (None = no projection)
            trainable: If True, add trainable projection layer
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            adapter_enabled: If True, use adapter instead of full projection
            adapter_hidden: Hidden dimension for adapter
        """
        super().__init__(embedding_model, base_dim, api_key)

        self.projection_dim = projection_dim if projection_dim else base_dim
        self.trainable = trainable
        self.adapter_enabled = adapter_enabled

        # Em-specific projection layer (separate from Eq)
        if self.adapter_enabled:
            self.adapter = Adapter(
                d_in=self.base_dim,
                d_hidden=adapter_hidden,
                d_out=self.projection_dim
            )
            self.projection = None
        elif self.trainable and self.projection_dim != self.base_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.base_dim, self.projection_dim),
                nn.ReLU(),
                nn.Linear(self.projection_dim, self.projection_dim)
            )
            self.adapter = None
        else:
            self.projection = None
            self.adapter = None

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Em projection layer and L2 normalization.

        Args:
            embeddings: Base embeddings (batch_size, base_dim)

        Returns:
            L2-normalized memory embeddings (batch_size, projection_dim)
        """
        if self.adapter is not None:
            embeddings = self.adapter(embeddings)
        elif self.projection is not None and self.trainable:
            embeddings = self.projection(embeddings)

        # L2 normalization (paper lines 249-251)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @staticmethod
    def _item_to_text(item: Dict) -> str:
        """Concatenate all memory entry fields into a single text for encoding."""
        parts = []
        if item.get("title"):
            parts.append(f"Title: {item['title']}")
        bullets = item.get("bullets", [])
        if bullets:
            bullets_list = bullets if isinstance(bullets, list) else [str(bullets)]
            parts.append("Insights:\n" + "\n".join(f"- {b}" for b in bullets_list))
        if item.get("example"):
            parts.append(f"Example: {item['example']}")
        tags = item.get("tags", [])
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")
        if item.get("scope"):
            parts.append(f"Scope: {item['scope']}")
        return "\n".join(parts)

    def encode_memory_item(self, memory_item: Dict, with_grad: bool = False):
        """
        Encode a single memory item into L2-normalized embedding.

        Encodes all fields: title, bullets, example, tags, scope.

        Args:
            memory_item: Memory dict with structure {title, bullets, example, tags, scope, meta}
            with_grad: If True, return a torch tensor with gradients enabled

        Returns:
            L2-normalized embedding (projection_dim,)
        """
        text = self._item_to_text(memory_item)

        # Get base embedding
        base_emb = self.get_base_embedding(text)

        # Apply projection and normalization
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        base_tensor = torch.from_numpy(base_emb).unsqueeze(0).to(device)  # (1, base_dim)

        if with_grad:
            if self.trainable or self.adapter_enabled:
                normalized_emb = self(base_tensor)
            else:
                normalized_emb = F.normalize(base_tensor, p=2, dim=-1)
            return normalized_emb.squeeze(0)

        with torch.no_grad():
            if self.trainable or self.adapter_enabled:
                normalized_emb = self(base_tensor)
            else:
                # If not trainable, just normalize
                normalized_emb = F.normalize(base_tensor, p=2, dim=-1)

        return normalized_emb.squeeze(0).cpu().numpy()

    def encode_memory_batch(self, memory_items: List[Dict]) -> np.ndarray:
        """
        Encode a batch of memory items.

        Encodes all fields: title, bullets, example, tags, scope.

        Args:
            memory_items: List of memory dicts

        Returns:
            L2-normalized embeddings (batch_size, projection_dim)
        """
        texts = [self._item_to_text(item) for item in memory_items]

        # Get base embeddings in batch
        base_embs = self.get_base_embeddings_batch(texts)

        # Apply projection and normalization
        with torch.no_grad():
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
            base_tensor = torch.from_numpy(base_embs).to(device)  # (batch_size, base_dim)
            if self.trainable or self.adapter_enabled:
                normalized_embs = self(base_tensor)
            else:
                # If not trainable, just normalize
                normalized_embs = F.normalize(base_tensor, p=2, dim=-1)

        return normalized_embs.cpu().numpy()

    def save_weights(self, path: str):
        """Save trainable weights to file."""
        if self.projection is not None:
            torch.save(self.projection.state_dict(), path)

    def load_weights(self, path: str):
        """Load trainable weights from file."""
        if self.projection is not None:
            self.projection.load_state_dict(torch.load(path))


class CCMERetriever:
    """
    CCME-based retrieval using hybrid score (similarity + reliability).

    Paper Reference (Equation 5):
    score(x, m) = α · sim(Eq(x), Em(m)) + (1 - α) · p̂(m)

    Paper Reference (lines 317-321):
    Uses ANN (Approximate Nearest Neighbor) indexing with FAISS for O(log N)
    sublinear retrieval. Per-domain shard indices for efficient retrieval.

    Uses separate Eq and Em encoders as per paper lines 241-244.
    """

    def __init__(
        self,
        query_encoder: QueryEncoder,
        memory_encoder: MemoryEncoder,
        alpha: float = 0.7,
        top_k: int = 5,
        use_ann: bool = True,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """
        Initialize CCME Retriever with ANN indexing.

        Args:
            query_encoder: QueryEncoder instance Eq (paper lines 241-242)
            memory_encoder: MemoryEncoder instance Em (paper lines 243-244)
            alpha: Balance between similarity and reliability (0-1)
            top_k: Number of items to retrieve
            use_ann: If True, use FAISS ANN indexing; if False, fallback to linear scan
            nlist: Number of clusters for IVF index (paper lines 317-321)
            nprobe: Number of clusters to search at query time
        """
        self.query_encoder = query_encoder  # Eq
        self.memory_encoder = memory_encoder  # Em
        self.alpha = alpha
        self.top_k = top_k
        self.last_retrieval_scores: Dict[str, float] = {}

        # ANN configuration (paper lines 317-321)
        self.use_ann = use_ann
        self.nlist = nlist
        self.nprobe = nprobe

        # Memory embedding cache: {item_id: embedding}
        self.memory_embeddings: Dict[str, np.ndarray] = {}
        # ANN index structures (paper lines 317-321)
        self.faiss_index: Optional[faiss.Index] = None
        self.domain_indices: Dict[str, faiss.Index] = {}
        self.index_to_id: List[str] = []
        self.domain_index_to_id: Dict[str, List[str]] = {}

        # Source query base embeddings for sim_source_query scoring.
        # {query_id: L2-normalized base embedding (1536-dim)}
        # Populated via register_precomputed_query_embedding() before retrieval.
        # {item_id: L2-normalized base embedding of the source query that created it}
        self._query_id_to_base_emb: Dict[str, np.ndarray] = {}
        self.source_query_base_embs: Dict[str, np.ndarray] = {}  # item_id → (N_sq, D)

        # Second ANN index in base space (1536-dim) for source query similarity.
        # Indexes mean source query embedding per item so sim_sq can contribute
        # to candidate generation independently of sim_mem.
        self.sq_faiss_index: Optional[faiss.Index] = None
        self.sq_index_to_id: List[str] = []

        # Embedding dimension (set on first index)
        self.embedding_dim: Optional[int] = None

        # Async indexing (paper line 321)
        self._index_lock = threading.RLock()
        self._indexing_thread: Optional[threading.Thread] = None
        self._pending_index: Optional[faiss.Index] = None
        self._pending_index_to_id: Optional[List[str]] = None
        self._pending_domain_indices: Optional[Dict[str, faiss.Index]] = None
        self._pending_domain_index_to_id: Optional[Dict[str, List[str]]] = None

    def _build_faiss_index(self, embeddings: np.ndarray, nlist: Optional[int] = None) -> faiss.Index:
        """
        Build FAISS index for ANN retrieval.

        Paper lines 317-321: Uses IVF (Inverted File) index for O(log N) retrieval.
        Falls back to flat index for small memory banks.

        Args:
            embeddings: Memory embeddings (N, dim)
            nlist: Number of clusters (None = use self.nlist)

        Returns:
            FAISS index
        """
        n_items, dim = embeddings.shape
        nlist = nlist or self.nlist

        # For small memory banks, use flat index (exact search)
        # IVF requires at least nlist training vectors
        if n_items < nlist * 2:
            # Flat L2 index (exact, but O(N))
            # Since embeddings are L2-normalized, L2 distance ∝ (1 - cosine similarity)
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized vectors
            index.add(embeddings.astype(np.float32))
        else:
            # IVF index for large memory banks (paper lines 317-321)
            # IndexIVFFlat: clusters + flat storage within clusters
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train the index on embeddings
            index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))

            # Set number of clusters to probe at query time
            index.nprobe = self.nprobe

        return index

    def _build_indices_internal(
        self,
        memory_bank: List[Dict],
        embeddings: np.ndarray
    ) -> Tuple[faiss.Index, List[str], Dict[str, faiss.Index], Dict[str, List[str]]]:
        """
        Build FAISS indices (internal, can run in background thread).

        Returns:
            Tuple of (global_index, index_to_id, domain_indices, domain_index_to_id)
        """
        # Build global index
        global_index = self._build_faiss_index(embeddings)

        # Build id mapping
        index_to_id = []
        for item in memory_bank:
            item_id = item.get("id", item.get("title", "unknown"))
            index_to_id.append(item_id)

        # Build per-domain shard indices (paper lines 317-321)
        domain_items: Dict[str, List[Tuple[str, np.ndarray]]] = {}
        for item, emb in zip(memory_bank, embeddings):
            item_id = item.get("id", item.get("title", "unknown"))
            tags = item.get("tags", [])
            domain = tags[0] if tags else "general"
            if domain not in domain_items:
                domain_items[domain] = []
            domain_items[domain].append((item_id, emb))

        domain_indices = {}
        domain_index_to_id = {}
        for domain, items in domain_items.items():
            if len(items) < 2:
                continue
            domain_ids = [item_id for item_id, _ in items]
            domain_embs = np.array([emb for _, emb in items])
            domain_nlist = max(2, min(self.nlist // 4, len(items) // 2))
            domain_indices[domain] = self._build_faiss_index(domain_embs, nlist=domain_nlist)
            domain_index_to_id[domain] = domain_ids

        return global_index, index_to_id, domain_indices, domain_index_to_id

    def _async_index_worker(self, memory_bank: List[Dict], embeddings: np.ndarray):
        """
        Background worker for async index building (paper line 321).
        """
        try:
            global_index, index_to_id, domain_indices, domain_index_to_id = \
                self._build_indices_internal(memory_bank, embeddings)

            # Store in pending slots
            with self._index_lock:
                self._pending_index = global_index
                self._pending_index_to_id = index_to_id
                self._pending_domain_indices = domain_indices
                self._pending_domain_index_to_id = domain_index_to_id

        except Exception as e:
            print(f"  ⚠️ Async indexing failed: {e}")

    def _swap_pending_index(self):
        """Swap pending async index to active (called before retrieval)."""
        with self._index_lock:
            if self._pending_index is not None:
                self.faiss_index = self._pending_index
                self.index_to_id = self._pending_index_to_id
                self.domain_indices = self._pending_domain_indices
                self.domain_index_to_id = self._pending_domain_index_to_id
                self._pending_index = None
                self._pending_index_to_id = None
                self._pending_domain_indices = None
                self._pending_domain_index_to_id = None

    def register_precomputed_query_embedding(self, query_id: str, base_embedding: np.ndarray):
        """
        Register a precomputed base embedding for a query ID.

        Called once per query before retrieval when use_source_query_sim is enabled.
        Embeddings come from the precomputed CSV (text-embedding-3-small, 1536-dim).

        Args:
            query_id: Query ID string (e.g. "Q_001")
            base_embedding: Raw base embedding (1536-dim, will be L2-normalized)
        """
        norm = np.linalg.norm(base_embedding)
        self._query_id_to_base_emb[query_id] = base_embedding / max(norm, 1e-8)

    def _build_source_query_emb_map(self, memory_bank: List[Dict]):
        """
        Build item_id → source query base embedding map from registered query embeddings.
        Uses the first source_query ID stored in each item's meta.
        """
        self.source_query_base_embs = {}
        for item in memory_bank:
            item_id = item.get("id", item.get("title", "unknown"))
            source_queries = item.get("meta", {}).get("source_queries", [])
            # Collect all registered embeddings — item may have accumulated multiple
            # source queries over clustering cycles; store all for max-sim scoring
            sq_embs = [
                self._query_id_to_base_emb[qid]
                for qid in source_queries
                if qid in self._query_id_to_base_emb
            ]
            if sq_embs:
                self.source_query_base_embs[item_id] = np.stack(sq_embs)  # (N_sq, D)

    def index_memory_bank(self, memory_bank: List[Dict], async_mode: bool = False):
        """
        Pre-compute embeddings and build ANN index for all memory items.

        Each item is encoded as: title + bullets + example + tags + scope.

        Paper lines 317-321: Builds FAISS ANN index for O(log N) retrieval.
        Also builds per-domain shard indices for efficient domain-specific retrieval.
        Supports async index updates (paper line 321) to avoid blocking queries.

        Args:
            memory_bank: List of memory items with {id, title, bullets, example, tags, scope, meta}
            async_mode: If True, build index in background thread (paper line 321)
        """
        if len(memory_bank) == 0:
            with self._index_lock:
                self.memory_embeddings = {}
                self.faiss_index = None
                self.index_to_id = []
                self.domain_indices = {}
                self.domain_index_to_id = {}
                self.source_query_base_embs = {}
                self.sq_faiss_index = None
                self.sq_index_to_id = []
            return

        print(f"Indexing {len(memory_bank)} memory items with ANN...")
        embeddings = self.memory_encoder.encode_memory_batch(memory_bank)
        self.embedding_dim = embeddings.shape[1]

        # Store embeddings in cache (always synchronous - needed for fallback)
        with self._index_lock:
            self.index_to_id = []
            for item, emb in zip(memory_bank, embeddings):
                item_id = item.get("id", item.get("title", "unknown"))
                self.memory_embeddings[item_id] = emb
                self.index_to_id.append(item_id)

        # Build source query embedding map (for sim_source_query scoring)
        self._build_source_query_emb_map(memory_bank)

        # Build sq index in base space for dual-index ANN retrieval
        if self.use_ann:
            self._build_sq_index()

        # Build FAISS indices
        if self.use_ann:
            if async_mode:
                # Paper line 321: Async index update
                # Wait for any previous indexing to complete
                if self._indexing_thread is not None and self._indexing_thread.is_alive():
                    self._indexing_thread.join(timeout=0.1)

                # Start background indexing
                self._indexing_thread = threading.Thread(
                    target=self._async_index_worker,
                    args=(copy.deepcopy(memory_bank), embeddings.copy()),
                    daemon=True
                )
                self._indexing_thread.start()
                print(f"  Started async FAISS indexing (paper line 321)")
            else:
                # Synchronous indexing - PRIMARY INDEX
                global_index, index_to_id, domain_indices, domain_index_to_id = \
                    self._build_indices_internal(memory_bank, embeddings)

                with self._index_lock:
                    self.faiss_index = global_index
                    self.index_to_id = index_to_id
                    self.domain_indices = domain_indices
                    self.domain_index_to_id = domain_index_to_id

                print(f"  Built FAISS index: {type(self.faiss_index).__name__}")
                print(f"  Built {len(self.domain_indices)} domain shard indices")

    def _async_index_worker(self, memory_bank: List[Dict], embeddings: np.ndarray):
        """Background worker for async index building."""
        try:
            global_index, index_to_id, domain_indices, domain_index_to_id = \
                self._build_indices_internal(memory_bank, embeddings)
            with self._index_lock:
                self._pending_index = global_index
                self._pending_index_to_id = index_to_id
                self._pending_domain_indices = domain_indices
                self._pending_domain_index_to_id = domain_index_to_id
        except Exception as e:
            print(f"  ⚠️ Async indexing failed: {e}")

    def calculate_reliability(self, meta: Dict) -> float:
        """
        Calculate Bayesian reliability: p̂(m) = (helpful+1)/(helpful+harmful+2)

        Args:
            meta: Metadata dict with {helpful, harmful, used, ...}

        Returns:
            Reliability score [0, 1]
        """
        helpful = meta.get("helpful", 0)
        harmful = meta.get("harmful", 0)

        # Bayesian estimate with Beta(1,1) prior
        # Definition 1: p̂(m) = (helpful+1)/(helpful+harmful+2)
        reliability = (helpful + 1) / (helpful + harmful + 2)

        return reliability

    def _compute_annealed_alpha(self, meta: Dict) -> float:
        """
        Compute exposure-annealed α for an item.

        Paper lines 407-409: α is annealed by exposure count n = helpful + harmful
        so that rare items rely more on similarity (higher α).

        Args:
            meta: Item metadata with helpful/harmful counts

        Returns:
            Annealed α value
        """
        helpful = meta.get("helpful", 0)
        harmful = meta.get("harmful", 0)
        exposure = helpful + harmful

        # Rare items (low exposure) → higher α (rely more on similarity)
        # Well-tested items (high exposure) → use base α (trust reliability more)
        # Formula: α_annealed = base_α + (1 - base_α) * exp(-exposure / scale)
        # This gives α → 1.0 for exposure → 0, and α → base_α for high exposure
        scale = 5.0  # Controls how fast annealing decays
        annealed_alpha = self.alpha + (1 - self.alpha) * np.exp(-exposure / scale)

        return annealed_alpha

    def _ann_search(
        self,
        query_embedding: np.ndarray,
        k_candidates: int,
        domain: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Perform ANN search to get candidate item IDs and similarities.

        Paper lines 317-321: O(log N) retrieval using FAISS.

        Args:
            query_embedding: Query embedding (dim,)
            k_candidates: Number of candidates to retrieve (K' > K)
            domain: Optional domain for shard-specific search

        Returns:
            List of (item_id, similarity) tuples
        """
        query_vec = query_embedding.reshape(1, -1).astype(np.float32)

        if domain and domain in self.domain_indices:
            index = self.domain_indices[domain]
            id_mapping = self.domain_index_to_id[domain]
        else:
            index = self.faiss_index
            id_mapping = self.index_to_id

        if index is None:
            return []

        # Limit k to index size
        actual_k = min(k_candidates, index.ntotal)
        if actual_k == 0:
            return []

        # FAISS search returns (distances, indices)
        # For IndexFlatIP/IndexIVFFlat with METRIC_INNER_PRODUCT, distances = similarities
        similarities, indices = index.search(query_vec, actual_k)

        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= 0 and idx < len(id_mapping):  # Valid index
                item_id = id_mapping[idx]
                results.append((item_id, float(sim)))

        return results

    def _build_sq_index(self):
        """
        Build a FAISS index in base space (1536-dim) over mean source query embeddings.
        Called after _build_source_query_emb_map() so source_query_base_embs is populated.
        Only items that have at least one registered source query are indexed.
        """
        if not self.source_query_base_embs:
            self.sq_faiss_index = None
            self.sq_index_to_id = []
            return

        ids, embs = [], []
        for item_id, sq_stack in self.source_query_base_embs.items():
            # Mean of all source query embeddings for this item
            mean_emb = sq_stack.mean(axis=0)
            norm = np.linalg.norm(mean_emb)
            ids.append(item_id)
            embs.append((mean_emb / max(norm, 1e-8)).astype(np.float32))

        emb_matrix = np.stack(embs)
        self.sq_faiss_index = faiss.IndexFlatIP(emb_matrix.shape[1])
        self.sq_faiss_index.add(emb_matrix)
        self.sq_index_to_id = ids

    def _sq_ann_search(self, query_base_emb: np.ndarray, k_candidates: int) -> List[Tuple[str, float]]:
        """
        ANN search in base space using source query index.
        Returns (item_id, sim_sq) tuples for candidate union.
        """
        if self.sq_faiss_index is None or self.sq_faiss_index.ntotal == 0:
            return []
        actual_k = min(k_candidates, self.sq_faiss_index.ntotal)
        query_vec = query_base_emb.reshape(1, -1).astype(np.float32)
        similarities, indices = self.sq_faiss_index.search(query_vec, actual_k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if 0 <= idx < len(self.sq_index_to_id):
                results.append((self.sq_index_to_id[idx], float(sim)))
        return results

    def retrieve(
        self,
        query_embedding: np.ndarray,
        memory_bank: List[Dict],
        top_k: Optional[int] = None,
        use_exposure_annealing: bool = True,
        domain: Optional[str] = None,
        use_source_query_sim: bool = False,
        source_query_sim_mode: str = "avg",
        current_query_base_emb: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Retrieve top-K memory items using hybrid score with ANN acceleration.

        Base formula (Eq. 5):
            score(x, m) = α · sim_mem + (1-α) · p̂(m)

        With source query similarity enabled:
            sim_combined = avg(sim_mem, sim_sq)   [source_query_sim_mode="avg"]
            sim_combined = max(sim_mem, sim_sq)   [source_query_sim_mode="max"]
            score(x, m)  = α · sim_combined + (1-α) · p̂(m)

        where:
            sim_mem = sim(Eq(x), Em(m))  — projected similarity, full entry content
            sim_sq  = sim_base(x, sq)    — base-space similarity to source query text

        Paper lines 317-321: Uses ANN to get K' candidates (K' > K) in O(log N),
        then applies full hybrid scoring to select final top-K.

        Paper lines 407-409: α is annealed by exposure count so rare items
        rely more on similarity.

        Args:
            query_embedding: Query embedding (base or projected; projected internally if base)
            memory_bank: List of memory items
            top_k: Override default top_k
            use_exposure_annealing: If True, anneal α based on exposure count
            domain: Optional domain hint for shard-specific retrieval
            use_source_query_sim: If True, blend sim_mem with sim_sq
            source_query_sim_mode: "avg" or "max" — how to combine sim_mem and sim_sq
            current_query_base_emb: L2-normalized base embedding of the current query
                                    (required when use_source_query_sim=True)

        Returns:
            Top-K memory items sorted by score
        """
        k = top_k if top_k else self.top_k

        if len(memory_bank) == 0:
            return []

        # Check for pending async index (paper line 321)
        self._swap_pending_index()

        # Project query embedding through Eq if needed
        if query_embedding.shape[-1] == self.query_encoder.base_dim:
            try:
                device = next(self.query_encoder.parameters(), torch.tensor(0)).device
            except StopIteration:
                device = torch.device("cpu")
            query_tensor = torch.from_numpy(query_embedding).unsqueeze(0).float().to(device)
            query_emb_projected = self.query_encoder.forward(query_tensor).squeeze(0).detach().cpu().numpy()
        else:
            query_emb_projected = query_embedding

        # L2-normalize current query base embedding for source query similarity
        sq_query_norm = None
        if use_source_query_sim and current_query_base_emb is not None:
            norm = np.linalg.norm(current_query_base_emb)
            sq_query_norm = current_query_base_emb / max(norm, 1e-8)

        # Build item lookup
        memory_by_id = {
            item.get("id", item.get("title", "unknown")): item
            for item in memory_bank
        }

        # Use ANN if enabled and index exists (paper lines 317-321)
        if self.use_ann and self.faiss_index is not None:
            k_candidates = min(max(k * 4, min(50, len(memory_bank) // 2)), len(memory_bank))
            ann_results = self._ann_search(query_emb_projected, k_candidates, domain)

            # Dual-index union: also retrieve candidates via sq index in base space
            # so high sim_sq items are not pruned before hybrid scoring sees them
            if use_source_query_sim and sq_query_norm is not None and self.sq_faiss_index is not None:
                sq_results = self._sq_ann_search(sq_query_norm, k_candidates)
                # Union by item_id — keep sim_mem from primary index where available
                ann_ids = {item_id for item_id, _ in ann_results}
                for item_id, _ in sq_results:
                    if item_id not in ann_ids:
                        # Item only found via sq index; sim_mem computed from cache
                        if item_id in self.memory_embeddings:
                            sim_mem = float(np.dot(query_emb_projected, self.memory_embeddings[item_id]))
                            ann_results.append((item_id, sim_mem))
                            ann_ids.add(item_id)

            scores = []
            for item_id, ann_similarity in ann_results:
                if item_id not in memory_by_id:
                    continue
                item = memory_by_id[item_id]
                meta = item.get("meta", {})
                sim_mem = float(ann_similarity)
                similarity = self._combine_similarity(
                    sim_mem, item_id, sq_query_norm,
                    use_source_query_sim, source_query_sim_mode
                )
                reliability = self.calculate_reliability(meta)
                alpha = self._compute_annealed_alpha(meta) if use_exposure_annealing else self.alpha
                score = alpha * similarity + (1 - alpha) * reliability
                scores.append((score, item))

        else:
            # Fallback: linear scan
            scores = []
            for item in memory_bank:
                item_id = item.get("id", item.get("title", "unknown"))
                if item_id not in self.memory_embeddings:
                    mem_emb = self.memory_encoder.encode_memory_item(item)
                    self.memory_embeddings[item_id] = mem_emb
                else:
                    mem_emb = self.memory_embeddings[item_id]
                sim_mem = float(np.dot(query_emb_projected, mem_emb))
                similarity = self._combine_similarity(
                    sim_mem, item_id, sq_query_norm,
                    use_source_query_sim, source_query_sim_mode
                )
                meta = item.get("meta", {})
                reliability = self.calculate_reliability(meta)
                alpha = self._compute_annealed_alpha(meta) if use_exposure_annealing else self.alpha
                score = alpha * similarity + (1 - alpha) * reliability
                scores.append((score, item))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:k]
        self.last_retrieval_scores = {
            item.get("id", item.get("title", "unknown")): round(score, 4)
            for score, item in top
        }
        return [item for _, item in top]

    def _combine_similarity(
        self,
        sim_mem: float,
        item_id: str,
        sq_query_norm: Optional[np.ndarray],
        use_source_query_sim: bool,
        mode: str,
    ) -> float:
        """
        Combine sim_mem with sim_sq according to mode.

        Args:
            sim_mem: Projected similarity sim(Eq(x), Em(m)) ∈ [-1, 1]
            item_id: Memory item ID (to look up source query embedding)
            sq_query_norm: L2-normalized base embedding of current query (or None)
            use_source_query_sim: Whether to blend in sim_sq
            mode: "avg" → (sim_mem + sim_sq)/2, "max" → max(sim_mem, sim_sq)

        Returns:
            Combined similarity ∈ [-1, 1]
        """
        if not use_source_query_sim or sq_query_norm is None:
            return sim_mem

        sq_embs = self.source_query_base_embs.get(item_id)
        if sq_embs is None:
            return sim_mem  # No source query registered for this item — fall back

        # sq_embs: (N_sq, D) — score against all source queries, take best match
        sim_sq = float(np.max(sq_embs @ sq_query_norm))

        if mode == "max":
            return max(sim_mem, sim_sq)
        else:  # "avg"
            return (sim_mem + sim_sq) / 2.0


class CCMELoss(nn.Module):
    """
    CCME Loss: InfoNCE for query-memory alignment.

    L_CCME = -Σ log p(m+ | query)

    where p(m+ | query) = exp(sim(Eq(query), Em(m+))/τ) / [exp(sim(Eq(query), Em(m+))/τ) + Σ exp(sim(Eq(query), Em(m-))/τ)]
    """

    def __init__(
        self,
        temperature: float = 0.07,
        adaptive_temperature: bool = True,
        min_temperature: float = 0.05,
        max_temperature: float = 0.15,
        label_smoothing: float = 0.1
    ):
        """
        Initialize CCME Loss.

        Args:
            temperature: InfoNCE temperature
            adaptive_temperature: If True, adjust τ based on success rate
            min_temperature: Minimum temperature
            max_temperature: Maximum temperature
            label_smoothing: Label smoothing factor for negative logits (paper lines 396-397)
        """
        super().__init__()

        self.register_buffer('temperature', torch.tensor(temperature))
        self.adaptive_temperature = adaptive_temperature
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.label_smoothing = label_smoothing

        # Rolling success rate for adaptive temperature
        self.register_buffer('rolling_success_rate', torch.tensor(0.5))
        self.register_buffer('success_momentum', torch.tensor(0.9))

    def info_nce_loss(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for CCME.

        Args:
            query_embeddings: Query embeddings Eq(x) (batch_size, dim)
            positive_embeddings: Positive memory embeddings Em(m+) (batch_size, dim)
            negative_embeddings: Negative memory embeddings Em(m-) (batch_size, num_neg, dim)

        Returns:
            InfoNCE loss scalar
        """
        # Eq. 6: InfoNCE logits use sim(Eq, Em) / τ
        pos_sim = torch.sum(query_embeddings * positive_embeddings, dim=-1) / self.temperature  # (B,)

        # Eq. 6: negative logits share the same temperature scaling
        neg_sim = torch.matmul(
            query_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1) / self.temperature  # (B, num_neg)

        # LogSumExp for numerical stability
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+num_neg)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        # Paper lines 396-397: Apply label smoothing to negative logits
        # This helps avoid false negatives (helpful items mislabeled as negatives)
        loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        return loss

    def update_temperature(self, success_rate: float):
        """
        Adaptively adjust temperature based on rolling success rate.

        Lower success rate → widen τ (smoother, more exploration)
        Higher success rate → tighten τ (sharper, more exploitation)

        Args:
            success_rate: Current success rate [0, 1]
        """
        if not self.adaptive_temperature:
            return

        # Update rolling average
        self.rolling_success_rate = (
            self.success_momentum * self.rolling_success_rate +
            (1 - self.success_momentum) * success_rate
        )

        # Adjust temperature inversely with success rate
        # High success → low temperature (sharper)
        # Low success → high temperature (smoother)
        new_temp = self.max_temperature - (self.max_temperature - self.min_temperature) * self.rolling_success_rate
        self.temperature = torch.clamp(new_temp, self.min_temperature, self.max_temperature)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CCME loss.

        Args:
            query_embeddings: Eq(queries) (batch_size, dim)
            positive_embeddings: Em(m+) (batch_size, dim)
            negative_embeddings: Em(m-) (batch_size, num_neg, dim)

        Returns:
            Tuple of (loss, loss_dict)
        """
        loss = self.info_nce_loss(query_embeddings, positive_embeddings, negative_embeddings)

        loss_dict = {
            'total': loss.item(),
            'temperature': self.temperature.item()
        }

        return loss, loss_dict


# Example usage
if __name__ == "__main__":
    # Initialize separate Eq and Em encoders (paper lines 241-244)
    query_encoder = QueryEncoder(trainable=False)  # Eq
    memory_encoder = MemoryEncoder(trainable=False)  # Em

    # Example memory item
    memory_item = {
        "id": "m_001",
        "title": "Brute-force 24 Game Solver",
        "bullets": [
            "Try all permutations of numbers",
            "Apply each operator (+, -, *, /) to pairs",
            "Recursively solve subproblems"
        ],
        "tags": ["math", "combinatorics"],
        "meta": {
            "helpful": 5,
            "harmful": 1,
            "retrieved_count": 6,
            "last_used_query": 3
        }
    }

    # Encode memory item with Em
    mem_embedding = memory_encoder.encode_memory_item(memory_item)
    print(f"Memory embedding shape (Em): {mem_embedding.shape}")
    print(f"L2 norm: {np.linalg.norm(mem_embedding):.6f}")  # Should be ~1.0

    # Encode query with Eq
    query = "How do I solve the 24 game?"
    query_embedding = query_encoder.encode_query(query)
    print(f"Query embedding shape (Eq): {query_embedding.shape}")
    print(f"L2 norm: {np.linalg.norm(query_embedding):.6f}")  # Should be ~1.0

    # Initialize retriever with both encoders
    retriever = CCMERetriever(query_encoder, memory_encoder, alpha=0.7, top_k=3)

    # Example memory bank
    memory_bank = [memory_item]
    retriever.index_memory_bank(memory_bank)

    print("\nCCME Memory Encoder initialized successfully!")
