"""
CRTE (Contrastive Reflective Trajectory Encoder) Implementation

This module implements Section 3.4 of the paper:
- Trajectory Encoder Et(τ): Encodes sequential reasoning with temporal features
- Reflection Encoder Er(u): Encodes reflective insights hierarchically
- Constrained Temporal Embedding (CTE): Multi-scale temporal position encoding
- CRTE Loss: Bidirectional InfoNCE + clustering + margin (Equation 7-10)

Paper References:
- Section 3.4: Contrastive Reflective Trajectory Encoder (CRTE)
- Equation 7: L_CRTE = L_t→r + L_r→t + λ_clus·L_cluster + λ_mar·L_margin
- Equation 8: CTE γ(k) = [sin(g⊙Ωk), cos(g⊙Ωk)] (Constrained Temporal Embedding)
- Equation 9: L_cluster = Σ ||Er(u_success) - c_success||² (Clustering loss)
- Equation 10: L_margin = max(0, η - sim(success) + sim(failure)) (Margin loss)

Architecture:
1. Trajectory Encoder Et(τ): Base embedding + CTE + attention pooling
2. Reflection Encoder Er(u): Hierarchical structure formatting + base embedding
3. CTE: Multi-scale temporal features prevent phase wrapping in long sequences
4. Combined Loss: Bidirectional alignment + success clustering + failure separation
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
import os
import math
import tiktoken
from .utils.adapters import Adapter

_EMBED_MAX_TOKENS = 8000  # conservative buffer below API's 8192 limit to avoid BPE boundary overruns
_tiktoken_enc = None

def _get_tiktoken():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        _tiktoken_enc = tiktoken.encoding_for_model("gpt-4o")
    return _tiktoken_enc


class ConstrainedTemporalEmbedding(nn.Module):
    """
    CTE: Multi-scale temporal encoding with learnable gating.

    γ(k) = [sin(g⊙Ωk), cos(g⊙Ωk)]

    Prevents phase wrapping in long sequences.
    """

    def __init__(
        self,
        d_temporal: int = 64,
        omega_min: float = 0.01,
        omega_max: float = 10.0,
        trainable: bool = True
    ):
        """
        Initialize CTE.

        Args:
            d_temporal: Dimension of temporal embedding
            omega_min: Minimum frequency
            omega_max: Maximum frequency
            trainable: If True, use learnable gating coefficients
        """
        super().__init__()

        self.d_temporal = d_temporal
        self.d_omega = d_temporal // 2  # Half for sin, half for cos

        # Log-uniform frequency bank
        log_omega = torch.linspace(
            math.log(omega_min),
            math.log(omega_max),
            self.d_omega
        )
        self.register_buffer('omega', torch.exp(log_omega))  # [d_omega]

        # Learnable gating coefficients
        if trainable:
            self.gating = nn.Parameter(torch.ones(self.d_omega))
        else:
            self.register_buffer('gating', torch.ones(self.d_omega))

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal embeddings for positions.

        Args:
            positions: Position indices (batch_size, seq_len) or (seq_len,)

        Returns:
            Temporal embeddings (batch_size, seq_len, d_temporal) or (seq_len, d_temporal)
        """
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)  # (1, seq_len)

        # positions: (batch_size, seq_len)
        # omega: (d_omega,)
        # Compute g⊙Ωk
        gated_omega = self.gating * self.omega  # (d_omega,)
        angles = positions.unsqueeze(-1) * gated_omega.unsqueeze(0).unsqueeze(0)  # (B, L, d_omega)

        # Eq. 8: γ(k) = [sin(g⊙Ωk), cos(g⊙Ωk)]
        sin_emb = torch.sin(angles)
        cos_emb = torch.cos(angles)
        temporal_emb = torch.cat([sin_emb, cos_emb], dim=-1)  # (B, L, d_temporal)

        return temporal_emb.squeeze(0) if positions.size(0) == 1 else temporal_emb


class TrajectoryEncoder(nn.Module):
    """
    Trajectory Encoder Et(τ): Encodes sequential reasoning steps with temporal features.

    Uses base embedding + CTE + attention pooling.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        base_dim: int = 1536,
        projection_dim: int = 768,
        d_temporal: int = 64,
        trainable: bool = False,
        api_key: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_hidden: int = 256
    ):
        """
        Initialize Trajectory Encoder.

        Args:
            embedding_model: OpenAI embedding model
            base_dim: Base embedding dimension
            projection_dim: Output embedding dimension
            d_temporal: CTE dimension
            trainable: If True, add trainable layers
            api_key: OpenAI API key
        """
        super().__init__()

        self.embedding_model = embedding_model
        self.base_dim = base_dim
        self.projection_dim = projection_dim
        self.d_temporal = d_temporal
        self.trainable = trainable
        self.adapter_enabled = adapter_enabled

        # OpenAI client for base embeddings
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        # Constrained Temporal Embedding
        self.cte = ConstrainedTemporalEmbedding(
            d_temporal=d_temporal,
            trainable=trainable
        )

        # Combined dimension after concatenating base + temporal
        combined_dim = base_dim + d_temporal

        if self.adapter_enabled:
            self.adapter = Adapter(
                d_in=combined_dim,
                d_hidden=adapter_hidden,
                d_out=projection_dim
            )
            self.projection = None
        else:
            self.adapter = None
            if trainable:
                # Projection layer
                self.projection = nn.Sequential(
                    nn.Linear(combined_dim, projection_dim),
                    nn.ReLU(),
                    nn.Linear(projection_dim, projection_dim)
                )
            else:
                self.projection = nn.Linear(combined_dim, projection_dim)

        if trainable:
            # Attention pooling weights
            self.attn_query = nn.Linear(projection_dim, projection_dim)
            self.attn_key = nn.Linear(projection_dim, projection_dim)
            self.attn_scale = math.sqrt(projection_dim)
        else:
            self.attn_query = None
            self.attn_key = None

    def _clip_embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts with local CLIP text encoder (used when embedding_model=='clip')."""
        import torch as _torch
        from transformers import CLIPModel as _CLIPModel, CLIPProcessor as _CLIPProcessor
        import torch.nn.functional as _F
        if not hasattr(self, '_clip_model'):
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self._clip_model = _CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            self._clip_model.eval()
            self._clip_processor = _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self._clip_device = device
        inputs = self._clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self._clip_device)
        with _torch.no_grad():
            embs = self._clip_model.get_text_features(**inputs).float()
            embs = _F.normalize(embs, dim=-1)
        return embs.cpu().numpy()

    def _embed_single_safe(self, text: str) -> np.ndarray:
        """
        Embed one text, chunking into 8192-token windows and averaging if it exceeds
        the embedding model's input limit. Only activates when the text is over-limit.
        """
        if self.embedding_model == "clip":
            return self._clip_embed_texts([text])[0]
        enc = _get_tiktoken()
        tokens = enc.encode(text)
        if len(tokens) <= _EMBED_MAX_TOKENS:
            response = self.client.embeddings.create(model=self.embedding_model, input=text)
            return np.array(response.data[0].embedding, dtype=np.float32)
        # Over limit: chunk and average
        chunks = []
        for start in range(0, len(tokens), _EMBED_MAX_TOKENS):
            chunk = enc.decode(tokens[start:start + _EMBED_MAX_TOKENS])
            chunks.append(chunk)
        response = self.client.embeddings.create(model=self.embedding_model, input=chunks)
        chunk_embs = np.array([item.embedding for item in response.data], dtype=np.float32)
        avg = chunk_embs.mean(axis=0)
        avg = (avg / (np.linalg.norm(avg) + 1e-8)).astype(np.float32)
        return avg

    def get_base_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get base embeddings for a list of texts.
        For texts within the 8192-token limit, batch them in one API call.
        For texts exceeding the limit, fall back to chunked embedding + averaging.
        """
        if self.embedding_model == "clip":
            return self._clip_embed_texts(texts)
        enc = _get_tiktoken()
        results = [None] * len(texts)
        normal_indices, normal_texts = [], []

        for i, text in enumerate(texts):
            if len(enc.encode(text)) > _EMBED_MAX_TOKENS:
                results[i] = self._embed_single_safe(text)
            else:
                normal_indices.append(i)
                normal_texts.append(text)

        if normal_texts:
            response = self.client.embeddings.create(
                model=self.embedding_model, input=normal_texts
            )
            for j, idx in enumerate(normal_indices):
                results[idx] = np.array(response.data[j].embedding, dtype=np.float32)

        return np.array(results, dtype=np.float32)

    def format_trajectory_steps(self, steps: List[Dict]) -> List[str]:
        """
        Format trajectory steps for embedding.

        Format: "[T1|analysis] content1"

        Args:
            steps: Parsed trajectory steps with {id, type, content, timestamp, memory_refs}

        Returns:
            List of formatted step strings
        """
        formatted = []
        for step in steps:
            step_id = step.get('id', '')
            step_type = step.get('type', 'unknown')
            timestamp = step.get('timestamp', f'T{step_id}')
            content = step.get('content', '')

            formatted_step = f"[{timestamp}|{step_type}] {content}"
            formatted.append(formatted_step)

        return formatted

    def forward(
        self,
        step_embeddings: torch.Tensor,
        temporal_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: combine step embeddings with temporal features and pool.

        Args:
            step_embeddings: Base embeddings (seq_len, base_dim)
            temporal_embeddings: CTE embeddings (seq_len, d_temporal)

        Returns:
            Trajectory embedding (projection_dim,)
        """
        # Concatenate base + temporal
        combined = torch.cat([step_embeddings, temporal_embeddings], dim=-1)  # (L, base_dim+d_temporal)

        # Project
        if self.adapter is not None:
            projected = self.adapter(combined)
        else:
            projected = self.projection(combined)  # (L, projection_dim)

        # Attention pooling
        if self.trainable and self.attn_query is not None:
            # Gated attention to weight critical steps
            queries = self.attn_query(projected)  # (L, projection_dim)
            keys = self.attn_key(projected)  # (L, projection_dim)

            # Compute attention scores
            attn_scores = torch.matmul(queries, keys.transpose(0, 1)) / self.attn_scale  # (L, L)
            attn_weights = F.softmax(attn_scores.mean(dim=1), dim=0)  # (L,)

            # Weighted sum
            pooled = torch.sum(projected * attn_weights.unsqueeze(-1), dim=0)  # (projection_dim,)
        else:
            # Simple mean pooling
            pooled = projected.mean(dim=0)  # (projection_dim,)

        # L2 normalize
        pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled

    def encode_trajectory(self, steps: List[Dict], with_grad: bool = False):
        """
        Encode a full trajectory.

        Args:
            steps: Parsed trajectory steps

        Returns:
            L2-normalized trajectory embedding (projection_dim,)
        """
        if not steps:
            # Return zero embedding for empty trajectory
            if with_grad:
                device = next(self.parameters()).device
                return torch.zeros(self.projection_dim, device=device)
            return np.zeros(self.projection_dim, dtype=np.float32)

        # Format steps for embedding
        formatted_steps = self.format_trajectory_steps(steps)

        # Get base embeddings
        base_embs = self.get_base_embeddings(formatted_steps)  # (L, base_dim)

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        # Get temporal positions
        positions = torch.arange(len(steps), dtype=torch.float32, device=device)  # (L,)

        # Convert to tensors
        base_tensor = torch.from_numpy(base_embs).to(device)  # (L, base_dim)

        # Get temporal embeddings
        temporal_tensor = self.cte(positions)  # (L, d_temporal)

        if with_grad:
            # Forward pass with grad
            traj_embedding = self(base_tensor, temporal_tensor)  # (projection_dim,)
            return traj_embedding

        with torch.no_grad():
            traj_embedding = self(base_tensor, temporal_tensor)  # (projection_dim,)
        return traj_embedding.cpu().numpy()


class ReflectionEncoder(nn.Module):
    """
    Reflection Encoder Er(u): Encodes reflective insights with hierarchical structure.

    Encodes: memory verdicts + critical observations + new insights
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        base_dim: int = 1536,
        projection_dim: int = 768,
        trainable: bool = False,
        api_key: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_hidden: int = 256
    ):
        """
        Initialize Reflection Encoder.

        Args:
            embedding_model: OpenAI embedding model
            base_dim: Base embedding dimension
            projection_dim: Output embedding dimension
            trainable: If True, add trainable projection layer
            api_key: OpenAI API key
        """
        super().__init__()

        self.embedding_model = embedding_model
        self.base_dim = base_dim
        self.projection_dim = projection_dim
        self.trainable = trainable
        self.adapter_enabled = adapter_enabled

        # OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        if self.adapter_enabled:
            self.adapter = Adapter(
                d_in=base_dim,
                d_hidden=adapter_hidden,
                d_out=projection_dim
            )
            self.projection = None
        else:
            self.adapter = None
            if trainable:
                self.projection = nn.Sequential(
                    nn.Linear(base_dim, projection_dim),
                    nn.ReLU(),
                    nn.Linear(projection_dim, projection_dim)
                )
            else:
                self.projection = nn.Linear(base_dim, projection_dim)

    def format_reflection(self, reflection: Dict) -> str:
        """
        Serialize the full reflector output as text for encoding.

        The reflector JSON is passed as-is so Er captures every field:
          execution_status, trajectory_analysis (critical_steps with
          lesson_and_insights), and memory_evaluation (verdicts + reasons).

        Args:
            reflection: Parsed reflection JSON from reflector

        Returns:
            JSON string of the full reflection
        """
        if not isinstance(reflection, dict):
            return "[INVALID REFLECTION]"
        return json.dumps(reflection, ensure_ascii=False)

    def _extract_reflection_summary(self, reflection: Dict) -> str:
        """
        Extract the semantically dense structured fields from a reflection dict
        when the full JSON exceeds the embedding model's token limit.
        Preserves: execution_status, critical step lessons, and memory verdicts.
        """
        parts = []
        status = reflection.get("execution_status", "")
        if status:
            parts.append(f"status: {status}")
        traj = reflection.get("trajectory_analysis", {})
        for step in traj.get("critical_steps", []):
            lesson = step.get("lesson_and_insights", "")
            if lesson:
                parts.append(f"lesson: {lesson}")
        mem_eval = reflection.get("memory_evaluation", [])
        if isinstance(mem_eval, list):
            for item in mem_eval:
                v = item.get("verdict", "")
                r = item.get("reason", "")
                if v or r:
                    parts.append(f"verdict: {v}. {r}")
        return "\n".join(parts) if parts else ""

    def _clip_embed_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts with local CLIP text encoder (used when embedding_model=='clip')."""
        import torch as _torch
        from transformers import CLIPModel as _CLIPModel, CLIPProcessor as _CLIPProcessor
        import torch.nn.functional as _F
        if not hasattr(self, '_clip_model'):
            device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self._clip_model = _CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            self._clip_model.eval()
            self._clip_processor = _CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self._clip_device = device
        inputs = self._clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self._clip_device)
        with _torch.no_grad():
            embs = self._clip_model.get_text_features(**inputs).float()
            embs = _F.normalize(embs, dim=-1)
        return embs.cpu().numpy()

    def get_base_embedding(self, text: str, reflection_dict: Optional[Dict] = None) -> np.ndarray:
        """
        Get base embedding for a reflection text.
        If the text exceeds the 8192-token embedding limit, extract only the
        structured summary fields (lessons + verdicts) instead of the full JSON.
        Only activates when the text is over-limit.
        """
        if self.embedding_model == "clip":
            return self._clip_embed_texts([text])[0]
        enc = _get_tiktoken()
        tokens = enc.encode(text)
        if len(tokens) > _EMBED_MAX_TOKENS:
            try:
                # Prefer the already-parsed dict to avoid re-parsing JSON
                ref = reflection_dict if reflection_dict is not None else json.loads(text)
                summary = self._extract_reflection_summary(ref)
                if summary:
                    summary_tokens = enc.encode(summary)
                    if len(summary_tokens) > _EMBED_MAX_TOKENS:
                        summary = enc.decode(summary_tokens[:_EMBED_MAX_TOKENS])
                    text = summary
                else:
                    text = enc.decode(tokens[:_EMBED_MAX_TOKENS])
            except Exception:
                text = enc.decode(tokens[:_EMBED_MAX_TOKENS])
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def forward(self, base_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: project and normalize.

        Args:
            base_embedding: Base embedding (base_dim,)

        Returns:
            L2-normalized reflection embedding (projection_dim,)
        """
        if self.adapter is not None:
            projected = self.adapter(base_embedding)
        else:
            projected = self.projection(base_embedding)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized

    def encode_reflection(self, reflection: Dict, with_grad: bool = False):
        """
        Encode a full reflection.

        Args:
            reflection: Parsed reflection JSON

        Returns:
            L2-normalized reflection embedding (projection_dim,)
        """
        # Format reflection
        formatted = self.format_reflection(reflection)

        # Get base embedding — pass the dict so over-limit path can extract summary
        base_emb = self.get_base_embedding(formatted, reflection_dict=reflection)

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        base_tensor = torch.from_numpy(base_emb).to(device)

        if with_grad:
            refl_embedding = self(base_tensor)
            return refl_embedding

        with torch.no_grad():
            refl_embedding = self(base_tensor)
        return refl_embedding.cpu().numpy()


class CRTELoss(nn.Module):
    """
    CRTE Loss: Bidirectional InfoNCE + Clustering + Margin

    L_CRTE = Σ[InfoNCE(t, r) + InfoNCE(r, t)] + λ_clus·L_cluster + λ_mar·L_margin
    """

    def __init__(
        self,
        temperature: float = 0.07,
        lambda_cluster: float = 0.1,
        lambda_margin: float = 0.1,
        margin_eta: float = 0.2
    ):
        """
        Initialize CRTE Loss.

        Args:
            temperature: InfoNCE temperature
            lambda_cluster: Weight for clustering loss
            lambda_margin: Weight for margin loss
            margin_eta: Desired margin width
        """
        super().__init__()

        self.temperature = temperature
        self.lambda_cluster = lambda_cluster
        self.lambda_margin = lambda_margin
        self.margin_eta = margin_eta

        # Running centroid for successful trajectories
        self.register_buffer('success_centroid', None)
        self.register_buffer('centroid_momentum', torch.tensor(0.9))

    def info_nce_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchor: Anchor embeddings (batch_size, dim)
            positive: Positive embeddings (batch_size, dim)
            negatives: Negative embeddings (batch_size, num_neg, dim)

        Returns:
            InfoNCE loss scalar
        """
        # Cosine similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (B,)

        # Negative similarities
        neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / self.temperature  # (B, num_neg)

        # LogSumExp for numerical stability
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+num_neg)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # All first position

        loss = F.cross_entropy(logits, labels)

        return loss

    def clustering_loss(self, success_embeddings: torch.Tensor) -> torch.Tensor:
        """
        L_cluster = Σ ||z - μ_succ||²

        Args:
            success_embeddings: Embeddings from successful trajectories (N, dim)

        Returns:
            Clustering loss scalar
        """
        if success_embeddings.size(0) == 0:
            return torch.tensor(0.0, device=success_embeddings.device)

        # Update centroid with momentum (detach to avoid backprop across steps)
        current_mean = success_embeddings.mean(dim=0)  # (dim,)
        with torch.no_grad():
            if self.success_centroid is None:
                self.success_centroid = current_mean.detach()
            else:
                self.success_centroid = (
                    self.centroid_momentum * self.success_centroid +
                    (1 - self.centroid_momentum) * current_mean.detach()
                )

        # Eq. 9: L_cluster = Σ ||z - μ_succ||²
        centroid = self.success_centroid.detach()
        distances = torch.sum((success_embeddings - centroid.unsqueeze(0)) ** 2, dim=-1)

        return distances.mean()

    def margin_loss(
        self,
        success_embeddings: torch.Tensor,
        failure_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        L_margin = Σ [η - sim(z⁺, μ) + sim(z⁻, μ)]₊

        Args:
            success_embeddings: Success trajectory embeddings (N_succ, dim)
            failure_embeddings: Failure trajectory embeddings (N_fail, dim)

        Returns:
            Margin loss scalar
        """
        if success_embeddings.size(0) == 0 or failure_embeddings.size(0) == 0:
            return torch.tensor(0.0, device=success_embeddings.device)

        if self.success_centroid is None:
            return torch.tensor(0.0, device=success_embeddings.device)

        # Cosine similarity to centroid
        centroid = self.success_centroid.detach()
        success_sim = torch.sum(success_embeddings * centroid.unsqueeze(0), dim=-1)  # (N_succ,)
        failure_sim = torch.sum(failure_embeddings * centroid.unsqueeze(0), dim=-1)  # (N_fail,)

        # Eq. 10: L_margin = [η - sim(z+, μ) + sim(z-, μ)]+
        violations = self.margin_eta - success_sim.unsqueeze(1) + failure_sim.unsqueeze(0)  # (N_succ, N_fail)
        violations = F.relu(violations)  # [·]₊

        return violations.mean()

    def forward(
        self,
        traj_embeddings: torch.Tensor,
        refl_embeddings: torch.Tensor,
        neg_refl_embeddings: torch.Tensor,
        neg_traj_embeddings: torch.Tensor,
        success_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute full CRTE loss.

        Args:
            traj_embeddings: Trajectory embeddings (batch_size, dim) — positives + failures
            refl_embeddings: Reflection embeddings (batch_size, dim) — positives + failures
            neg_refl_embeddings: Per-sample reflection negatives (n_pos, num_neg, dim)
                                 for L_t→r: {uⱼ: j≠i} ∪ {u⁻ from failures}
            neg_traj_embeddings: Per-sample trajectory negatives (n_pos, num_neg, dim)
                                 for L_r→t: {τⱼ: j≠i} ∪ {τ⁻ from failures}
            success_mask: Boolean mask (batch_size,) — True for SUCCESS, False for FAILURE

        Returns:
            Total loss and loss component dict
        """
        n_pos = success_mask.sum().item()
        pos_traj = traj_embeddings[:n_pos]
        pos_refl = refl_embeddings[:n_pos]

        # Bidirectional InfoNCE over positive pairs only
        # L_t→r: for each τᵢ⁺ align to uᵢ⁺ against {uⱼ: j≠i} ∪ {u⁻}
        loss_t2r = self.info_nce_loss(pos_traj, pos_refl, neg_refl_embeddings)
        # L_r→t: for each uᵢ⁺ align to τᵢ⁺ against {τⱼ: j≠i} ∪ {τ⁻}
        loss_r2t = self.info_nce_loss(pos_refl, pos_traj, neg_traj_embeddings)

        loss_info_nce = loss_t2r + loss_r2t

        # Clustering loss (only on successful trajectories)
        success_traj = traj_embeddings[success_mask]
        loss_cluster = self.clustering_loss(success_traj)

        # Margin loss (separate success vs failure trajectories)
        failure_traj = traj_embeddings[~success_mask]
        loss_margin = self.margin_loss(success_traj, failure_traj)

        # Total loss
        total_loss = (
            loss_info_nce +
            self.lambda_cluster * loss_cluster +
            self.lambda_margin * loss_margin
        )

        loss_dict = {
            'total': total_loss.item(),
            'info_nce': loss_info_nce.item(),
            'clustering': loss_cluster.item(),
            'margin': loss_margin.item()
        }

        return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    print("Initializing CRTE encoders...")

    # Initialize encoders (frozen mode)
    traj_encoder = TrajectoryEncoder(trainable=False)
    refl_encoder = ReflectionEncoder(trainable=False)

    # Example trajectory steps (from generator)
    steps = [
        {"id": "1", "type": "analysis", "content": "Analyze the problem structure", "timestamp": "T1", "memory_refs": []},
        {"id": "2", "type": "strategy", "content": "Apply divide and conquer", "timestamp": "T2", "memory_refs": ["m_042"]},
        {"id": "3", "type": "execution", "content": "Implement the solution", "timestamp": "T3", "memory_refs": []}
    ]

    # Example reflection (from reflector)
    reflection = {
        "execution_status": "SUCCESS",
        "trajectory_analysis": {
            "critical_steps": [
                {
                    "step_id": "2",
                    "step_type": "strategy",
                    "description": "Applied divide and conquer",
                    "impact": "POSITIVE",
                    "reasoning": "Unlocked the solution path",
                    "lesson_and_insights": "Use divide and conquer when subproblems are independent",
                    "memory_influence": ["m_042"]
                }
            ]
        },
        "memory_evaluation": [
            {
                "item_id": "m_042",
                "title": "Brute-force 24 Game Solver",
                "usage_context": "Steps 2, 3",
                "verdict": "HELPFUL",
                "reason": "Provided the key algorithmic pattern"
            }
        ]
    }

    # Encode
    traj_emb = traj_encoder.encode_trajectory(steps)
    refl_emb = refl_encoder.encode_reflection(reflection)

    print(f"Trajectory embedding shape: {traj_emb.shape}, L2 norm: {np.linalg.norm(traj_emb):.6f}")
    print(f"Reflection embedding shape: {refl_emb.shape}, L2 norm: {np.linalg.norm(refl_emb):.6f}")
    print(f"Similarity: {np.dot(traj_emb, refl_emb):.6f}")

    print("\nCRTE encoders initialized successfully!")
