"""
Memory Operations with CCME-based Intelligence

This module implements Section 3.1 of the paper: Memory Merge Operator ⊕
Plus Section 3.5 (lines 548-555): Memory Refinement with Clustering

Paper References:
- Section 3.1: Memory Structure and Operations
- Definition 1: Bayesian reliability p̂(m) = (helpful+1)/(helpful+harmful+2)
- Section 3.1 (Pruning): score(m) = p̂(m) × exp(-λ × staleness)
- Algorithm 1 Step 7: Mi = Mi-1 ⊕ ΔMi (merge operator)
- Algorithm 1 Step 11: Refine memory (cluster → prototype)
- Lines 548-555: Memory refinement using k-medoids clustering

Operations (in order):
1. Reliability updates: Apply helpful/harmful/used counter changes from ΔMi
2. Item updates: Modify existing items based on ΔMi updates
3. Embedding-based deduplication: Find similar items using Em(m) cosine similarity
4. Item merging: Aggregate metadata when combining duplicates
5. Add new items: Insert new entries from ΔMi
6. Temporal decay pruning: Remove low-value items (reliability × exp(-λt))
7. Capacity enforcement: Shard-based constraints to prevent unbounded growth
8. Periodic refinement: Cluster semantically similar items, keep prototypes (k-medoids)

All operations use CCME encoder Em(m) for semantic similarity.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class MemoryOperations:
    """
    CCME-based memory operations for intelligent deduplication, merging, and pruning.
    """

    def __init__(
        self,
        memory_encoder,  # MemoryEncoder instance from ccme_encoder.py
        dedup_threshold: float = 0.95,  # Cosine similarity threshold
        prune_reliability_threshold: float = 0.3,  # Minimum reliability to keep
        temporal_decay_lambda: float = 0.0001,  # Decay rate per time unit (seconds)
        shard_capacity: int = 1000,  # Maximum items per shard
    ):
        """
        Initialize Memory Operations.

        Args:
            memory_encoder: MemoryEncoder instance (Em from CCME)
            dedup_threshold: Cosine similarity threshold for deduplication (0-1)
            prune_reliability_threshold: Min reliability score to keep item
            temporal_decay_lambda: Exponential decay rate for staleness
            shard_capacity: Max items per domain shard
        """
        self.encoder = memory_encoder
        self.dedup_threshold = dedup_threshold
        self.prune_reliability_threshold = prune_reliability_threshold
        self.temporal_decay_lambda = temporal_decay_lambda
        self.shard_capacity = shard_capacity

        # Cache for embeddings: {item_id: embedding}
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def calculate_reliability(self, helpful: int, harmful: int) -> float:
        """
        Bayesian reliability: p̂(m) = (helpful+1)/(helpful+harmful+2)

        Args:
            helpful: Number of helpful uses
            harmful: Number of harmful uses

        Returns:
            Reliability score [0, 1]
        """
        return (helpful + 1) / (helpful + harmful + 2)

    def get_embedding(self, item: Dict, force_recompute: bool = False) -> np.ndarray:
        """
        Get or compute embedding for memory item.

        Args:
            item: Memory item dict
            force_recompute: If True, recompute even if cached

        Returns:
            L2-normalized embedding
        """
        item_id = item.get("id", item.get("title", "unknown"))

        if not force_recompute and item_id in self.embedding_cache:
            return self.embedding_cache[item_id]

        # Compute embedding
        embedding = self.encoder.encode_memory_item(item)
        self.embedding_cache[item_id] = embedding

        return embedding

    def find_similar_items(
        self,
        new_item: Dict,
        existing_items: List[Dict],
        threshold: Optional[float] = None
    ) -> List[Tuple[Dict, float]]:
        """
        Find existing items similar to new item using CCME embeddings.

        Args:
            new_item: New memory item to check
            existing_items: List of existing memory items
            threshold: Similarity threshold (defaults to self.dedup_threshold)

        Returns:
            List of (item, similarity_score) tuples sorted by similarity
        """
        if threshold is None:
            threshold = self.dedup_threshold

        # Get new item embedding
        new_embedding = self.get_embedding(new_item)

        similar_items = []
        for existing_item in existing_items:
            # Get existing item embedding
            existing_embedding = self.get_embedding(existing_item)

            # Cosine similarity (dot product for L2-normalized vectors)
            similarity = float(np.dot(new_embedding, existing_embedding))

            if similarity >= threshold:
                similar_items.append((existing_item, similarity))

        # Sort by similarity descending
        similar_items.sort(key=lambda x: x[1], reverse=True)

        return similar_items

    def merge_items(
        self,
        item1: Dict,
        item2: Dict,
        prefer_higher_reliability: bool = True
    ) -> Dict:
        """
        Merge two similar items, aggregating metadata.

        Strategy:
        - Keep structure (title, bullets, tags) from higher reliability item
        - Aggregate metadata counters (helpful, harmful, used)
        - Merge tags and source_queries
        - Update timestamps to most recent

        Args:
            item1: First memory item
            item2: Second memory item
            prefer_higher_reliability: If True, keep structure from higher reliability item

        Returns:
            Merged memory item
        """
        meta1 = item1.get("meta", {})
        meta2 = item2.get("meta", {})

        # Calculate reliabilities
        rel1 = self.calculate_reliability(
            meta1.get("helpful", 0),
            meta1.get("harmful", 0)
        )
        rel2 = self.calculate_reliability(
            meta2.get("helpful", 0),
            meta2.get("harmful", 0)
        )

        # Determine which item to keep structure from
        if prefer_higher_reliability and rel1 >= rel2:
            base_item = item1.copy()
            other_item = item2
        elif prefer_higher_reliability and rel2 > rel1:
            base_item = item2.copy()
            other_item = item1
        else:
            # Default to first item
            base_item = item1.copy()
            other_item = item2

        # Aggregate metadata counters
        merged_meta = base_item.get("meta", {}).copy()
        merged_meta["helpful"] = meta1.get("helpful", 0) + meta2.get("helpful", 0)
        merged_meta["harmful"] = meta1.get("harmful", 0) + meta2.get("harmful", 0)
        merged_meta["used"] = meta1.get("used", 0) + meta2.get("used", 0)

        # Use most recent timestamp
        q1 = meta1.get("last_used_query")
        q2 = meta2.get("last_used_query")
        merged_meta["last_used_query"] = max(q1, q2) if q1 is not None and q2 is not None else (q1 if q1 is not None else q2)

        # Merge tags (union)
        tags1 = set(item1.get("tags", []))
        tags2 = set(item2.get("tags", []))
        merged_tags = list(tags1 | tags2)

        # Merge source_queries
        sources1 = set(meta1.get("source_queries", []))
        sources2 = set(meta2.get("source_queries", []))
        merged_meta["source_queries"] = list(sources1 | sources2)

        # Update base item
        base_item["tags"] = merged_tags
        base_item["meta"] = merged_meta

        return base_item

    def deduplicate_new_items(
        self,
        new_items: List[Dict],
        existing_memory: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Deduplicate new items against existing memory using CCME embeddings.

        Strategy:
        - For each new item, find similar existing items
        - If similar items found, merge and update existing
        - If no similar items, add as new

        Args:
            new_items: List of new memory items to add
            existing_memory: Current memory bank

        Returns:
            Tuple of (items_to_add, items_to_merge)
            - items_to_add: New items with no similar existing items
            - items_to_merge: Dicts with {existing_id, new_item, similarity}
        """
        items_to_add = []
        items_to_merge = []

        for new_item in new_items:
            # Find similar existing items
            similar = self.find_similar_items(new_item, existing_memory)

            if similar:
                # Merge with most similar existing item
                most_similar_item, similarity = similar[0]
                items_to_merge.append({
                    "existing_id": most_similar_item.get("id"),
                    "existing_item": most_similar_item,
                    "new_item": new_item,
                    "similarity": similarity
                })
            else:
                # No similar items, add as new
                items_to_add.append(new_item)

        return items_to_add, items_to_merge

    def apply_merges(
        self,
        memory_bank: List[Dict],
        merge_actions: List[Dict]
    ) -> List[Dict]:
        """
        Apply merge actions to memory bank.

        Args:
            memory_bank: Current memory bank
            merge_actions: List of merge dicts from deduplicate_new_items()

        Returns:
            Updated memory bank with merged items
        """
        # Create a copy and index by ID
        updated_bank = [item.copy() for item in memory_bank]
        id_to_idx = {item["id"]: idx for idx, item in enumerate(updated_bank)}

        for action in merge_actions:
            existing_id = action["existing_id"]
            existing_item = action["existing_item"]
            new_item = action["new_item"]

            if existing_id in id_to_idx:
                idx = id_to_idx[existing_id]
                # Merge items
                merged_item = self.merge_items(existing_item, new_item)
                updated_bank[idx] = merged_item

                # Update embedding cache
                self.get_embedding(merged_item, force_recompute=True)

        return updated_bank

    def calculate_prune_score(
        self,
        item: Dict,
        current_query: Optional[int] = None,
        temporal_decay_lambda_query: float = 0.05
    ) -> float:
        """
        Calculate pruning score for an item.

        Score = reliability * exp(-λ_q * query_gap)

        Staleness is measured in queries (learning-progress scale).
        If last_used_query or current_query is unavailable, decay = 1.0 (no penalty).

        Higher score = keep, lower score = prune.

        Args:
            item: Memory item
            current_query: Current query index
            temporal_decay_lambda_query: Decay rate per query gap (default 0.05;
                exp(-0.05*30) ≈ 0.22 after 30 unused queries)

        Returns:
            Prune score (higher is better)
        """
        meta = item.get("meta", {})

        helpful = meta.get("helpful", 0)
        harmful = meta.get("harmful", 0)
        reliability = self.calculate_reliability(helpful, harmful)

        last_used_query = meta.get("last_used_query")
        if current_query is not None and last_used_query is not None:
            query_gap = max(0, current_query - last_used_query)
            decay_factor = np.exp(-temporal_decay_lambda_query * query_gap)
        else:
            decay_factor = 1.0  # No data yet — no penalty

        return float(reliability * decay_factor)

    def prune_low_quality_items(
        self,
        memory_bank: List[Dict],
        current_query: Optional[int] = None,
        temporal_decay_lambda_query: float = 0.02,
        pruning_mode: str = "AND"
    ) -> Tuple[List[Dict], List[str]]:
        """
        Prune items based on reliability × query-based temporal decay score.

        Pruning modes:
        - "AND": prune only if BOTH prune_score < threshold AND reliability < threshold
                 Conservative. Recommended: threshold=0.2, lambda_q=0.05
        - "OR":  prune if EITHER prune_score < threshold OR reliability < threshold
                 Aggressive. Recommended: threshold=0.15, lambda_q=0.02

        Args:
            memory_bank: Current memory bank
            current_query: Current query index for staleness computation
            temporal_decay_lambda_query: Decay rate per query gap
            pruning_mode: "AND" (conservative) or "OR" (aggressive)

        Returns:
            Tuple of (pruned_memory_bank, pruned_item_ids)
        """
        items_to_keep = []
        pruned_ids = []

        for item in memory_bank:
            meta = item.get("meta", {})
            retrieved_count = meta.get("retrieved_count", 0)
            score = self.calculate_prune_score(
                item,
                current_query=current_query,
                temporal_decay_lambda_query=temporal_decay_lambda_query
            )

            helpful = meta.get("helpful", 0)
            harmful = meta.get("harmful", 0)
            reliability = self.calculate_reliability(helpful, harmful)

            if pruning_mode == "OR":
                should_prune = (score < self.prune_reliability_threshold or
                                reliability < self.prune_reliability_threshold)
            else:  # AND
                should_prune = (score < self.prune_reliability_threshold and
                                reliability < self.prune_reliability_threshold)

            if should_prune:
                item_id = item.get("id", "unknown")
                pruned_ids.append(item_id)
                if item_id in self.embedding_cache:
                    del self.embedding_cache[item_id]
            else:
                items_to_keep.append(item)

        return items_to_keep, pruned_ids

    def enforce_shard_capacity(
        self,
        memory_bank: List[Dict],
        shard_key: str = "domain"
    ) -> Tuple[List[Dict], List[str]]:
        """
        Enforce capacity constraints per domain shard.

        If shard exceeds capacity, remove lowest prune-score items.

        Args:
            memory_bank: Current memory bank
            shard_key: How to shard (currently only "domain" supported)

        Returns:
            Tuple of (capacity_enforced_bank, removed_item_ids)
        """
        # Group items by domain (first tag before '.')
        shards = {}
        for item in memory_bank:
            tags = item.get("tags", [])
            if tags:
                # Use first tag's domain
                domain = tags[0].split(".")[0] if "." in tags[0] else tags[0]
            else:
                domain = "general"

            shards.setdefault(domain, []).append(item)

        # Enforce capacity per shard
        kept_items = []
        removed_ids = []

        for domain, items in shards.items():
            if len(items) <= self.shard_capacity:
                # Under capacity, keep all
                kept_items.extend(items)
            else:
                # Over capacity, prune lowest scoring
                # Calculate scores
                scored_items = [
                    (item, self.calculate_prune_score(item))
                    for item in items
                ]
                # Sort by score descending
                scored_items.sort(key=lambda x: x[1], reverse=True)

                # Keep top K, remove rest
                kept_items.extend([item for item, score in scored_items[:self.shard_capacity]])

                # Track removed
                for item, score in scored_items[self.shard_capacity:]:
                    item_id = item.get("id", "unknown")
                    removed_ids.append(item_id)
                    # Remove from cache
                    if item_id in self.embedding_cache:
                        del self.embedding_cache[item_id]

        return kept_items, removed_ids

    def apply_full_merge_operator(
        self,
        memory_bank: List[Dict],
        delta_m: Dict,
        current_query: Optional[int] = None,
        temporal_decay_lambda_query: float = 0.02,
        pruning_mode: str = "AND"
    ) -> Tuple[List[Dict], Dict[str, any]]:
        """
        Full ⊕ operator: Mi = Mi-1 ⊕ ΔMi

        Performs:
        1. Update metadata counters (reliability_updates)
        2. Update existing items (updates_to_existing)
        3. Deduplicate and add new items (new_items)
        4. Prune low-quality items (automatic)
        5. Enforce shard capacity (automatic)

        Args:
            memory_bank: Current memory bank Mi-1
            delta_m: Curation output ΔMi
            current_query: Current query index for staleness computation
            temporal_decay_lambda_query: Decay rate per query gap
            pruning_mode: "AND" (conservative) or "OR" (aggressive)

        Returns:
            Tuple of (updated_memory_bank Mi, operation_stats)
        """
        from .memory_formatter import (
            apply_reliability_updates,
            apply_item_updates,
            add_new_items_to_memory
        )

        updated_bank = memory_bank.copy()
        stats = {
            "reliability_updates": 0,
            "items_updated": 0,
            "items_added": 0,
            "items_merged": 0,
            "items_pruned": 0,
            "capacity_removed": 0
        }

        # 1. Apply reliability updates
        reliability_updates = delta_m.get("reliability_updates", [])
        if reliability_updates:
            updated_bank = apply_reliability_updates(updated_bank, reliability_updates)
            stats["reliability_updates"] = len(reliability_updates)

        # 2. Apply updates to existing items
        existing_updates = delta_m.get("updates_to_existing", [])
        if existing_updates:
            updated_bank = apply_item_updates(updated_bank, existing_updates)
            stats["items_updated"] = len(existing_updates)

            # Recompute embeddings for updated items
            updated_ids = {upd.get("item_id") for upd in existing_updates}
            for item in updated_bank:
                if item.get("id") in updated_ids:
                    self.get_embedding(item, force_recompute=True)

        # 3. Deduplicate and add new items
        new_items = delta_m.get("new_items", [])
        if new_items:
            items_to_add, items_to_merge = self.deduplicate_new_items(new_items, updated_bank)

            # Apply merges first
            if items_to_merge:
                updated_bank = self.apply_merges(updated_bank, items_to_merge)
                stats["items_merged"] = len(items_to_merge)

            # Then add truly new items
            if items_to_add:
                updated_bank = add_new_items_to_memory(updated_bank, items_to_add)
                stats["items_added"] = len(items_to_add)

                # Compute embeddings for new items
                for item in items_to_add:
                    self.get_embedding(item)

        # 4. Prune low-quality items
        updated_bank, pruned_ids = self.prune_low_quality_items(
            updated_bank,
            current_query=current_query,
            temporal_decay_lambda_query=temporal_decay_lambda_query,
            pruning_mode=pruning_mode
        )
        stats["items_pruned"] = len(pruned_ids)

        # 5. Enforce shard capacity
        updated_bank, capacity_removed_ids = self.enforce_shard_capacity(updated_bank)
        stats["capacity_removed"] = len(capacity_removed_ids)

        return updated_bank, stats


    def refine_memory_with_clustering(
        self,
        memory_bank: List[Dict],
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 2,
        use_crte_embeddings: bool = True,
        crte_encoder = None
    ) -> Tuple[List[Dict], Dict[str, any]]:
        """
        Memory refinement via clustering (Paper lines 548-555, Algorithm 1 Step 11).

        Consolidates redundant or semantically similar memory entries using k-medoids
        clustering. Selects one prototype (medoid) per cluster, removing dominated
        or unreliable items.

        Paper Reference (lines 548-555):
        "Periodic refinement consolidates redundant or outdated strategies. CRTE
        embeddings are used to cluster semantically equivalent or functionally
        similar memory entries, and a k-medoids or facility-location algorithm
        selects one prototype per cluster while removing dominated or unreliable items."

        IMPORTANT: Per the paper, CRTE embeddings (specifically the Reflection Encoder Er)
        should be used for clustering. Each memory item's bullets/insights are encoded
        by Er(u) to create the clustering spase. The Trajectory Encoder is only used
        during CRTE training as "labels" - at refinement time, only insight embeddings
        matter.

        Args:
            memory_bank: Current memory bank
            n_clusters: Number of clusters (if None, auto-determined)
            min_cluster_size: Minimum items to form a cluster worth consolidating
            use_crte_embeddings: If True (default), use CRTE Reflection Encoder Er(u).
                                 This is the paper-aligned behavior.
            crte_encoder: ReflectionEncoder instance for CRTE embeddings.
                         Required when use_crte_embeddings=True.

        Returns:
            Tuple of (refined_memory_bank, refinement_stats)
        """
        if len(memory_bank) < min_cluster_size:
            return memory_bank, {"clusters_found": 0, "items_removed": 0, "prototypes_kept": len(memory_bank)}

        def _memory_item_to_reflection(item: Dict) -> Dict:
            """Map a memory item to a dict for Er encoding via json.dumps.
            Passes all semantic fields directly — title, bullets, example,
            tags, scope — matching the entry structure.
            """
            return {
                "title": item.get("title", ""),
                "bullets": item.get("bullets", []),
                "example": item.get("example", ""),
                "tags": item.get("tags", []),
                "scope": item.get("scope", ""),
            }

        # Validate CRTE encoder when using CRTE embeddings (paper-aligned default)
        if use_crte_embeddings and crte_encoder is None:
            # Fallback to CCME if no CRTE encoder provided, but warn
            import warnings
            warnings.warn(
                "use_crte_embeddings=True but no crte_encoder provided. "
                "Falling back to CCME embeddings. For paper-aligned behavior, "
                "pass the ReflectionEncoder as crte_encoder."
            )
            use_crte_embeddings = False

        # Compute embeddings for all items
        # Paper: Use Er(u) - Reflection Encoder on insights/bullets
        embeddings = []
        valid_items = []

        for item in memory_bank:
            try:
                if use_crte_embeddings and crte_encoder is not None:
                    # Paper-aligned: Use CRTE Reflection Encoder Er(u)
                    # Each memory item's bullets = insights, encoded by Er
                    if hasattr(crte_encoder, 'encode_reflection'):
                        emb = crte_encoder.encode_reflection(_memory_item_to_reflection(item))
                    else:
                        # Fallback if wrong encoder type passed
                        emb = self.get_embedding(item)
                else:
                    # Fallback: use CCME memory embeddings Em(m)
                    emb = self.get_embedding(item)
                embeddings.append(emb)
                valid_items.append(item)
            except Exception:
                # If embedding fails, keep item without clustering
                valid_items.append(item)
                embeddings.append(None)

        # Filter out items with failed embeddings
        items_with_embeddings = [(item, emb) for item, emb in zip(valid_items, embeddings) if emb is not None]
        items_without_embeddings = [item for item, emb in zip(valid_items, embeddings) if emb is None]

        if len(items_with_embeddings) < min_cluster_size:
            return memory_bank, {"clusters_found": 0, "items_removed": 0, "prototypes_kept": len(memory_bank)}

        items_for_clustering = [item for item, _ in items_with_embeddings]
        embedding_matrix = np.array([emb for _, emb in items_with_embeddings])

        # Auto-determine number of clusters if not specified
        if n_clusters is None:
            # Heuristic: sqrt(N) clusters, minimum 2, maximum N/2
            n_clusters = max(2, min(int(np.sqrt(len(items_for_clustering))), len(items_for_clustering) // 2))

        n_clusters = min(n_clusters, len(items_for_clustering))

        # Run k-medoids clustering
        medoid_indices, cluster_assignments = self._k_medoids(
            embedding_matrix,
            n_clusters=n_clusters,
            max_iterations=100
        )

        # Select prototypes and consolidate clusters
        refined_items = []
        items_removed = 0

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_item_indices = np.where(cluster_mask)[0]

            if len(cluster_item_indices) == 0:
                continue

            if len(cluster_item_indices) == 1:
                # Single item cluster, keep as-is
                refined_items.append(items_for_clustering[cluster_item_indices[0]])
            else:
                # Multi-item cluster: select prototype as most reliable member,
                # fall back to medoid if no reliability signal exists yet
                medoid_idx = medoid_indices[cluster_id]
                best_idx = medoid_idx
                best_reliability = -1.0
                for idx in cluster_item_indices:
                    meta = items_for_clustering[idx].get("meta", {})
                    helpful = meta.get("helpful", 0)
                    harmful = meta.get("harmful", 0)
                    rel = (helpful + 1) / (helpful + harmful + 2)
                    if rel > best_reliability:
                        best_reliability = rel
                        best_idx = idx
                prototype = items_for_clustering[best_idx].copy()

                # Aggregate from all cluster members
                total_helpful = 0
                total_harmful = 0
                total_retrieved = 0
                last_used_query = prototype.get("meta", {}).get("last_used_query")
                all_tags = set(prototype.get("tags", []))
                all_sources = set(prototype.get("meta", {}).get("source_queries", []))

                for idx in cluster_item_indices:
                    item = items_for_clustering[idx]
                    meta = item.get("meta", {})
                    total_helpful += meta.get("helpful", 0)
                    total_harmful += meta.get("harmful", 0)
                    total_retrieved += meta.get("retrieved_count", 0)
                    # Keep the most recent usage across all members
                    member_last = meta.get("last_used_query")
                    if member_last is not None:
                        last_used_query = member_last if last_used_query is None else max(last_used_query, member_last)
                    all_tags.update(item.get("tags", []))
                    all_sources.update(meta.get("source_queries", []))

                # Update prototype with aggregated fields
                prototype_meta = prototype.get("meta", {}).copy()
                prototype_meta["helpful"] = total_helpful
                prototype_meta["harmful"] = total_harmful
                prototype_meta["retrieved_count"] = total_retrieved
                prototype_meta["last_used_query"] = last_used_query
                prototype_meta["source_queries"] = list(all_sources)
                prototype_meta["cluster_size"] = len(cluster_item_indices)
                prototype["meta"] = prototype_meta
                prototype["tags"] = list(all_tags)

                refined_items.append(prototype)
                items_removed += len(cluster_item_indices) - 1

                # Clean up embedding cache for removed items
                for idx in cluster_item_indices:
                    if idx != medoid_idx:
                        removed_id = items_for_clustering[idx].get("id", "")
                        if removed_id in self.embedding_cache:
                            del self.embedding_cache[removed_id]

        # Add back items without embeddings
        refined_items.extend(items_without_embeddings)

        stats = {
            "clusters_found": n_clusters,
            "items_removed": items_removed,
            "prototypes_kept": len(refined_items),
            "original_count": len(memory_bank)
        }

        return refined_items, stats

    def _k_medoids(
        self,
        X: np.ndarray,
        n_clusters: int,
        max_iterations: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        K-medoids clustering algorithm (PAM - Partitioning Around Medoids).

        Unlike k-means which uses centroids (mean of cluster), k-medoids uses
        actual data points (medoids) as cluster centers. This is more robust
        to outliers and works directly with cosine similarity.

        Args:
            X: Embedding matrix (N x D)
            n_clusters: Number of clusters K
            max_iterations: Maximum iterations

        Returns:
            Tuple of (medoid_indices, cluster_assignments)
        """
        n_samples = X.shape[0]

        if n_clusters >= n_samples:
            # Each item is its own cluster
            return np.arange(n_samples), np.arange(n_samples)

        # Compute pairwise cosine similarity matrix
        # For L2-normalized vectors: sim = X @ X.T
        similarity_matrix = X @ X.T
        # Convert to distance (1 - similarity for clustering)
        distance_matrix = 1 - similarity_matrix

        # Initialize medoids randomly (or use k-means++ style initialization)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        medoid_indices = rng.choice(n_samples, size=n_clusters, replace=False)

        for iteration in range(max_iterations):
            # Assign each point to nearest medoid
            distances_to_medoids = distance_matrix[:, medoid_indices]
            cluster_assignments = np.argmin(distances_to_medoids, axis=1)

            # Update medoids
            new_medoid_indices = np.zeros(n_clusters, dtype=int)
            for k in range(n_clusters):
                cluster_mask = cluster_assignments == k
                cluster_indices = np.where(cluster_mask)[0]

                if len(cluster_indices) == 0:
                    # Empty cluster, keep old medoid
                    new_medoid_indices[k] = medoid_indices[k]
                else:
                    # Find point that minimizes total distance to other cluster members
                    cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                    total_distances = cluster_distances.sum(axis=1)
                    best_idx = cluster_indices[np.argmin(total_distances)]
                    new_medoid_indices[k] = best_idx

            # Check convergence
            if np.array_equal(medoid_indices, new_medoid_indices):
                break

            medoid_indices = new_medoid_indices

        # Final assignment
        distances_to_medoids = distance_matrix[:, medoid_indices]
        cluster_assignments = np.argmin(distances_to_medoids, axis=1)

        return medoid_indices, cluster_assignments


# Convenience functions for backward compatibility
def apply_memory_updates_with_ccme(
    memory_bank: List[Dict],
    delta_m: Dict,
    memory_encoder,
    **kwargs
) -> Tuple[List[Dict], Dict]:
    """
    Apply full memory update with CCME-based intelligence.

    This is the main entry point for the ⊕ operator.

    Args:
        memory_bank: Current memory bank Mi-1
        delta_m: Curation output ΔMi
        memory_encoder: MemoryEncoder instance from ccme_encoder
        **kwargs: Additional config (dedup_threshold, prune_threshold, etc.)

    Returns:
        Tuple of (updated_memory_bank Mi, operation_stats)
    """
    mem_ops = MemoryOperations(memory_encoder, **kwargs)
    return mem_ops.apply_full_merge_operator(memory_bank, delta_m)


def refine_memory_with_clustering(
    memory_bank: List[Dict],
    memory_encoder,
    n_clusters: Optional[int] = None,
    crte_encoder = None,
    use_crte_embeddings: bool = True,
    **kwargs
) -> Tuple[List[Dict], Dict]:
    """
    Standalone function for memory refinement with clustering.

    Paper Reference (lines 548-555, Algorithm 1 Step 11):
    "Refine memory (cluster → prototype); rollback if stability metrics degrade"

    This should be called periodically after online training (every K_upd steps).

    IMPORTANT: Per the paper, this function defaults to using CRTE embeddings
    (Reflection Encoder Er) for clustering memory items. Each memory item's
    bullets/insights are encoded as e_u = Er(u). Pass the ReflectionEncoder
    as crte_encoder for paper-aligned behavior.

    Args:
        memory_bank: Current memory bank
        memory_encoder: MemoryEncoder instance from CCME (used as fallback)
        n_clusters: Number of clusters (auto if None)
        crte_encoder: ReflectionEncoder instance for CRTE embeddings.
                     Required for paper-aligned behavior.
        use_crte_embeddings: If True (default), use CRTE Reflection Encoder.
        **kwargs: Additional MemoryOperations config

    Returns:
        Tuple of (refined_memory_bank, refinement_stats)
    """
    mem_ops = MemoryOperations(memory_encoder, **kwargs)
    return mem_ops.refine_memory_with_clustering(
        memory_bank,
        n_clusters=n_clusters,
        crte_encoder=crte_encoder,
        use_crte_embeddings=use_crte_embeddings
    )
