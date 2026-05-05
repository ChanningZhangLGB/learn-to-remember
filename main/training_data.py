"""
Training Data Collection for CCME and CRTE

This module implements Algorithm 1 Step 8: Update training buffers B_ccme and B_crte.
Extracts contrastive pairs from each query episode for online encoder training.

Paper References:
- Section 3.5: Training Data Collection
- Algorithm 1 Step 8: Update buffers after each query
- Section 3.3: CCME pair extraction (positive/negative memory items)
- Section 3.4: CRTE pair extraction (trajectory-reflection alignment)

CCME Pairs (for L_CCME training):
- Positive: (Eq(query), Em(memory)) marked HELPFUL by reflector
- Hard negative: (Eq(query), Em(memory)) marked HARMFUL by reflector
- Soft negative: (Eq(query), Em(memory)) marked NEUTRAL by reflector, or retrieved but absent from memory_evaluation

CRTE Pairs (for L_CRTE training):
- Positive: (Et(τ), Er(u)) where execution_status == SUCCESS
- Negative: (Et(τ), Er(u)) where execution_status == FAILURE or mismatched pairs

Buffers maintain temporal diversity with exponential decay weighting.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class TrainingBuffer:
    """
    FIFO buffer for storing contrastive training pairs.

    Implements temporal diversity by maintaining recent examples
    with exponential downweighting for older samples.
    """

    def __init__(self, max_size: int = 1000, temporal_decay: float = 0.9):
        """
        Initialize training buffer.

        Args:
            max_size: Maximum buffer size (FIFO eviction)
            temporal_decay: Weight decay for older samples
        """
        self.max_size = max_size
        self.temporal_decay = temporal_decay
        self.buffer = deque(maxlen=max_size)

    def add(self, item: Dict):
        """Add item to buffer."""
        self.buffer.append(item)

    def sample(self, batch_size: int, apply_decay: bool = True) -> List[Dict]:
        """
        Sample batch from buffer.

        Args:
            batch_size: Number of samples to return
            apply_decay: If True, weight recent samples more heavily

        Returns:
            List of sampled items
        """
        if len(self.buffer) == 0:
            return []

        actual_size = min(batch_size, len(self.buffer))

        if apply_decay:
            # Compute temporal weights (more recent = higher weight)
            weights = np.array([
                self.temporal_decay ** (len(self.buffer) - i - 1)
                for i in range(len(self.buffer))
            ])
            weights = weights / weights.sum()

            # Sample with replacement using temporal weights
            indices = np.random.choice(
                len(self.buffer),
                size=actual_size,
                replace=False,
                p=weights
            )
        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.buffer),
                size=actual_size,
                replace=False
            )

        return [self.buffer[i] for i in indices]

    def get_all(self) -> List[Dict]:
        """Get all items in buffer."""
        return list(self.buffer)

    def clear(self):
        """Clear buffer."""
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class CCMEPairExtractor:
    """
    Extract (query, memory+) and (query, memory-) pairs for CCME training.
    """

    @staticmethod
    def extract_pairs_from_reflection(
        query: str,
        query_embedding: np.ndarray,
        retrieved_memory: List[Dict],
        reflection: Dict,
        query_embedding_is_projected: bool = False,
        embedding_index: Optional[Dict] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract positive and negative pairs from a single query-reflection episode.

        Args:
            query: Original query text
            query_embedding: Base query embedding (1536-dim, from text-embedding-3-small)
            retrieved_memory: Memory items retrieved for this query
            reflection: Parsed reflection JSON
            query_embedding_is_projected: True if query_embedding is already Eq(x)
            embedding_index: Optional dict mapping item_id -> base embedding (1536-dim).
                Used to store memory_base_embedding in each pair for false-negative
                detection in base embedding space (avoids API calls at prune time).

        Returns:
            Tuple of (positive_pairs, negative_pairs)
            Each pair is dict with {query, query_embedding, memory_item, label}
        """
        positive_pairs = []
        negative_pairs = []

        emb_index = embedding_index or {}

        # Index retrieved memory by ID
        memory_by_id = {item["id"]: item for item in retrieved_memory}

        # Get memory evaluation from reflection
        memory_evaluation = reflection.get("memory_evaluation", [])

        for eval_item in memory_evaluation:
            item_id = eval_item.get("item_id")
            verdict = eval_item.get("verdict")

            if item_id not in memory_by_id:
                continue

            memory_item = memory_by_id[item_id]
            memory_base_emb = emb_index.get(item_id)  # 1536-dim base, None if unavailable

            # Positive pairs: HELPFUL
            if verdict == "HELPFUL":
                positive_pairs.append({
                    "query": query,
                    "query_embedding": query_embedding,
                    "query_embedding_is_projected": query_embedding_is_projected,
                    "memory_item": memory_item,
                    "memory_base_embedding": memory_base_emb,
                    "label": "positive",
                    "pair_type": "helpful"
                })

            # Hard negative pairs: HARMFUL
            elif verdict == "HARMFUL":
                negative_pairs.append({
                    "query": query,
                    "query_embedding": query_embedding,
                    "query_embedding_is_projected": query_embedding_is_projected,
                    "memory_item": memory_item,
                    "memory_base_embedding": memory_base_emb,
                    "label": "negative",
                    "pair_type": "harmful"
                })

            # Soft negatives: NEUTRAL verdict — consulted but no clear helpful/harmful signal
            elif verdict == "NEUTRAL":
                negative_pairs.append({
                    "query": query,
                    "query_embedding": query_embedding,
                    "query_embedding_is_projected": query_embedding_is_projected,
                    "memory_item": memory_item,
                    "memory_base_embedding": memory_base_emb,
                    "label": "negative",
                    "pair_type": "soft_neutral"
                })

        # Soft negatives: retrieved but not consulted at all (not in memory_evaluation)
        evaluated_ids = {eval_item.get("item_id") for eval_item in memory_evaluation}
        for item_id, memory_item in memory_by_id.items():
            if item_id not in evaluated_ids:
                memory_base_emb = emb_index.get(item_id)
                negative_pairs.append({
                    "query": query,
                    "query_embedding": query_embedding,
                    "query_embedding_is_projected": query_embedding_is_projected,
                    "memory_item": memory_item,
                    "memory_base_embedding": memory_base_emb,
                    "label": "negative",
                    "pair_type": "soft_unused"
                })

        return positive_pairs, negative_pairs


class CRTEPairExtractor:
    """
    Extract (trajectory, reflection) pairs for CRTE training.
    """

    @staticmethod
    def extract_pairs_from_episode(
        query: str,
        trajectory_steps: List[Dict],
        trajectory_embedding: Optional[np.ndarray],
        reflection: Dict,
        reflection_embedding: Optional[np.ndarray],
        execution_status: str
    ) -> Dict:
        """
        Extract trajectory-reflection pair from a single episode.

        Args:
            query: Original query
            trajectory_steps: Parsed trajectory steps
            trajectory_embedding: Et(τ) embedding (if computed)
            reflection: Parsed reflection JSON
            reflection_embedding: Er(u) embedding (if computed)
            execution_status: SUCCESS/FAILURE/PARTIAL/UNKNOWN

        Returns:
            Pair dict with {query, trajectory, reflection, embeddings, label}
        """
        # Determine label based on execution status
        if execution_status == "SUCCESS":
            label = "positive"
        elif execution_status == "FAILURE":
            label = "negative"
        else:
            label = "neutral"  # PARTIAL or UNKNOWN

        pair = {
            "query": query,
            "trajectory_steps": trajectory_steps,
            "trajectory_embedding": trajectory_embedding,
            "reflection": reflection,
            "reflection_embedding": reflection_embedding,
            "execution_status": execution_status,
            "label": label
        }

        return pair


class TrainingDataCollector:
    """
    High-level collector that manages both CCME and CRTE buffers.
    """

    def __init__(
        self,
        ccme_buffer_size: int = 1000,
        crte_buffer_size: int = 500,
        temporal_decay: float = 0.9
    ):
        """
        Initialize training data collector.

        Args:
            ccme_buffer_size: CCME buffer size
            crte_buffer_size: CRTE buffer size
            temporal_decay: Temporal decay for sampling
        """
        # Separate buffers for positive and negative pairs
        self.ccme_positive_buffer = TrainingBuffer(ccme_buffer_size, temporal_decay)
        self.ccme_negative_buffer = TrainingBuffer(ccme_buffer_size, temporal_decay)

        self.crte_positive_buffer = TrainingBuffer(crte_buffer_size, temporal_decay)
        self.crte_negative_buffer = TrainingBuffer(crte_buffer_size, temporal_decay)

        self.ccme_extractor = CCMEPairExtractor()
        self.crte_extractor = CRTEPairExtractor()

    def collect_from_episode(
        self,
        query: str,
        query_embedding: Optional[np.ndarray],
        retrieved_memory: List[Dict],
        trajectory_steps: List[Dict],
        trajectory_embedding: Optional[np.ndarray],
        reflection: Dict,
        reflection_embedding: Optional[np.ndarray],
        execution_status: str,
        query_embedding_is_projected: bool = False,
        embedding_index: Optional[Dict] = None
    ):
        """
        Collect training pairs from a single agentic episode.

        Args:
            query: Original query
            query_embedding: Base query embedding (1536-dim) — NOT projected
            retrieved_memory: Retrieved memory items
            trajectory_steps: Parsed trajectory
            trajectory_embedding: Et(τ) if available
            reflection: Parsed reflection JSON
            reflection_embedding: Er(u) if available
            execution_status: SUCCESS/FAILURE/etc.
            query_embedding_is_projected: True if query_embedding is already Eq(x)
            embedding_index: Optional dict mapping item_id -> base embedding (1536-dim).
                Stored in each buffer entry for base-space false-negative detection
                during instability pruning — no API calls needed at prune time.
        """
        # Extract CCME pairs
        ccme_pos, ccme_neg = self.ccme_extractor.extract_pairs_from_reflection(
            query=query,
            query_embedding=query_embedding,
            retrieved_memory=retrieved_memory,
            reflection=reflection,
            query_embedding_is_projected=query_embedding_is_projected,
            embedding_index=embedding_index
        )

        # Add to CCME buffers
        for pair in ccme_pos:
            self.ccme_positive_buffer.add(pair)
        for pair in ccme_neg:
            self.ccme_negative_buffer.add(pair)

        # Extract CRTE pair
        crte_pair = self.crte_extractor.extract_pairs_from_episode(
            query=query,
            trajectory_steps=trajectory_steps,
            trajectory_embedding=trajectory_embedding,
            reflection=reflection,
            reflection_embedding=reflection_embedding,
            execution_status=execution_status
        )

        # Add to CRTE buffer
        if crte_pair["label"] == "positive":
            self.crte_positive_buffer.add(crte_pair)
        elif crte_pair["label"] == "negative":
            self.crte_negative_buffer.add(crte_pair)

    def sample_ccme_batch(
        self,
        batch_size: int,
        positive_ratio: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Sample CCME training batch.

        Args:
            batch_size: Total batch size
            positive_ratio: Fraction of positive pairs

        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        n_positive = int(batch_size * positive_ratio)
        n_negative = batch_size - n_positive

        positives = self.ccme_positive_buffer.sample(n_positive)
        negatives = self.ccme_negative_buffer.sample(n_negative)

        return positives, negatives

    def sample_crte_batch(
        self,
        batch_size: int,
        positive_ratio: float = 0.5
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Sample CRTE training batch.

        Args:
            batch_size: Total batch size
            positive_ratio: Fraction of positive pairs

        Returns:
            Tuple of (positive_pairs, negative_pairs)
        """
        n_positive = int(batch_size * positive_ratio)
        n_negative = batch_size - n_positive

        positives = self.crte_positive_buffer.sample(n_positive)
        negatives = self.crte_negative_buffer.sample(n_negative)

        return positives, negatives

    def get_buffer_stats(self) -> Dict:
        """Get statistics about buffer contents."""
        return {
            "ccme_positive": len(self.ccme_positive_buffer),
            "ccme_negative": len(self.ccme_negative_buffer),
            "crte_positive": len(self.crte_positive_buffer),
            "crte_negative": len(self.crte_negative_buffer)
        }

    def clear_all(self):
        """Clear all buffers."""
        self.ccme_positive_buffer.clear()
        self.ccme_negative_buffer.clear()
        self.crte_positive_buffer.clear()
        self.crte_negative_buffer.clear()
