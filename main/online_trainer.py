"""
Online Contrastive Training for CCME and CRTE

This module implements Section 3.5 and Algorithm 1 (Steps 8-11):
Test-time adaptation via online encoder training with stability monitoring.

Paper References:
- Section 3.5: Online Training Protocol
- Algorithm 1 Steps 8-11: Online training every K_upd queries
- Equation 6: L_CCME (InfoNCE loss for query-memory alignment)
- Equation 7: L_CRTE (Combined trajectory-reflection loss)

Training Protocol (every K_upd steps):
1. Sample recent contrastive pairs from buffers (B_ccme, B_crte)
2. Train Eq, Em via L_CCME on positive/negative memory pairs
3. Train Et, Er via L_CRTE on success/failure trajectory-reflection pairs
4. Monitor stability: Check rolling loss variance
5. Rollback if unstable: Revert to last checkpoint if loss increases significantly
6. Save checkpoint: Store current encoder states for potential rollback
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from collections import deque
import copy


def _to_tensor(x: Union[np.ndarray, torch.Tensor], device: str = "cpu") -> torch.Tensor:
    """Convert numpy array or tensor to tensor on specified device."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device).float()
    elif isinstance(x, torch.Tensor):
        return x.to(device).float()
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(x)}")


class OnlineTrainer:
    """
    Online trainer for CCME and CRTE encoders.

    Implements periodic updates every K_upd steps with stability monitoring.
    """

    def __init__(
        self,
        query_encoder,  # QueryEncoder (Eq) with trainable=True
        memory_encoder,  # MemoryEncoder (Em) with trainable=True
        trajectory_encoder,  # TrajectoryEncoder (Et) with trainable=True
        reflection_encoder,  # ReflectionEncoder (Er) with trainable=True
        ccme_loss,  # CCMELoss
        crte_loss,  # CRTELoss
        data_collector,  # TrainingDataCollector
        k_upd: int = 50,  # Update every K_upd queries
        num_train_steps: int = 10,  # Gradient steps per update
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        learning_rate_ccme: Optional[float] = None,
        learning_rate_crte: Optional[float] = None,
        lambda_crte: float = 1.0,  # Weight for CRTE loss
        stability_window: int = 20,            # Shared default window for stability monitoring
        stability_threshold: float = 0.1,      # Shared default max allowed loss degradation
        ccme_stability_window: Optional[int] = None,
        ccme_stability_threshold: Optional[float] = None,
        crte_stability_window: Optional[int] = None,
        crte_stability_threshold: Optional[float] = None,
        device: Optional[str] = None,
        refine_every_updates: int = 0,
        refinement_mode: str = "fixed",        # "fixed" or "adaptive"
        redundancy_threshold: float = 0.65,    # Mean pairwise sim threshold for adaptive mode
        min_items_for_refinement: int = 5,     # Min memory bank size before refinement fires
        refine_use_crte_embeddings: bool = True,
        min_ccme_positive: int = 5,
        min_crte_positive: int = 2,
        n_clusters_mode: str = "heuristic",
        n_clusters_fixed: int = 10,
        post_refinement_grace: int = 0,
        ccme_prune_on_instability: bool = False,
        ccme_false_negative_threshold: float = 0.55,
        ccme_reliability_prune_threshold: float = 0.35,
        crte_prune_on_instability: bool = False,
        crte_stale_positive_threshold: float = 0.15,
        crte_trivial_negative_threshold: float = 0.05,
        memory_ops_cfg: Optional[Dict] = None
    ):
        """
        Initialize online trainer.

        Paper Reference (Algorithm 1, lines 9-10):
        - Train Eq, Em on B_ccme via L_CCME
        - Train Et, Er on B_crte via L_CRTE

        Args:
            query_encoder: QueryEncoder (Eq) with trainable layers - paper lines 241-242
            memory_encoder: MemoryEncoder (Em) with trainable layers - paper lines 243-244
            trajectory_encoder: TrajectoryEncoder (Et) with trainable layers - paper lines 245-246
            reflection_encoder: ReflectionEncoder (Er) with trainable layers - paper lines 247-248
            ccme_loss: CCMELoss instance
            crte_loss: CRTELoss instance
            data_collector: TrainingDataCollector instance
            k_upd: Update frequency (every K queries)
            num_train_steps: Number of gradient steps per update
            batch_size: Training batch size
            learning_rate: Default learning rate for Adam (fallback)
            learning_rate_ccme: Optional CCME optimizer learning rate override
            learning_rate_crte: Optional CRTE optimizer learning rate override
            lambda_crte: Weight for CRTE loss
            stability_window: Rolling window for stability monitoring
            stability_threshold: Maximum allowed loss increase before rollback
            device: torch device
        """
        # CCME encoders (Eq, Em)
        self.query_encoder = query_encoder  # Eq
        self.memory_encoder = memory_encoder  # Em

        # CRTE encoders (Et, Er)
        self.trajectory_encoder = trajectory_encoder  # Et
        self.reflection_encoder = reflection_encoder  # Er

        self.ccme_loss = ccme_loss
        self.crte_loss = crte_loss
        self.data_collector = data_collector

        self.k_upd = k_upd
        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_ccme = learning_rate_ccme if learning_rate_ccme is not None else learning_rate
        self.learning_rate_crte = learning_rate_crte if learning_rate_crte is not None else learning_rate
        self.lambda_crte = lambda_crte
        self.ccme_stability_window = ccme_stability_window if ccme_stability_window is not None else stability_window
        self.ccme_stability_threshold = ccme_stability_threshold if ccme_stability_threshold is not None else stability_threshold
        self.crte_stability_window = crte_stability_window if crte_stability_window is not None else stability_window
        self.crte_stability_threshold = crte_stability_threshold if crte_stability_threshold is not None else stability_threshold
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.refine_every_updates = refine_every_updates
        self.refinement_mode = refinement_mode
        self.redundancy_threshold = redundancy_threshold
        self.min_items_for_refinement = min_items_for_refinement
        self.refine_use_crte_embeddings = refine_use_crte_embeddings
        self.min_ccme_positive = min_ccme_positive
        self.min_crte_positive = min_crte_positive
        self.n_clusters_mode = n_clusters_mode
        self.n_clusters_fixed = n_clusters_fixed
        self.post_refinement_grace = post_refinement_grace
        self.post_refinement_grace_remaining = 0
        self.ccme_prune_on_instability = ccme_prune_on_instability
        self.ccme_false_negative_threshold = ccme_false_negative_threshold
        self.ccme_reliability_prune_threshold = ccme_reliability_prune_threshold
        self.crte_prune_on_instability = crte_prune_on_instability
        self.crte_stale_positive_threshold = crte_stale_positive_threshold
        self.crte_trivial_negative_threshold = crte_trivial_negative_threshold
        allowed_mem_ops_keys = {
            "dedup_threshold",
            "prune_reliability_threshold",
            "temporal_decay_lambda",
            "shard_capacity",
        }
        self.memory_ops_cfg = {
            k: v for k, v in (memory_ops_cfg or {}).items() if k in allowed_mem_ops_keys
        }

        # Move modules to device
        self.query_encoder.to(self.device)  # Eq
        self.memory_encoder.to(self.device)  # Em
        self.trajectory_encoder.to(self.device)  # Et
        self.reflection_encoder.to(self.device)  # Er
        self.ccme_loss.to(self.device)
        self.crte_loss.to(self.device)

        # Optimizers: Train Eq and Em together for CCME (paper Algorithm 1 line 9)
        # Respect each encoder's .trainable flag so ablations (CCME-off / CRTE-off) skip
        # optimizer creation cleanly even when adapter params exist.
        ccme_enabled = getattr(query_encoder, "trainable", True) and getattr(memory_encoder, "trainable", True)
        crte_enabled = getattr(trajectory_encoder, "trainable", True) and getattr(reflection_encoder, "trainable", True)
        ccme_params = list(query_encoder.parameters()) + list(memory_encoder.parameters()) if ccme_enabled else []
        crte_params = list(trajectory_encoder.parameters()) + list(reflection_encoder.parameters()) if crte_enabled else []

        if len(ccme_params) > 0:
            self.ccme_optimizer = optim.Adam(ccme_params, lr=self.learning_rate_ccme)
            self.ccme_trainable = True
        else:
            self.ccme_optimizer = None
            self.ccme_trainable = False

        if len(crte_params) > 0:
            self.crte_optimizer = optim.Adam(crte_params, lr=self.learning_rate_crte)
            self.crte_trainable = True
        else:
            self.crte_optimizer = None
            self.crte_trainable = False

        # Freeze parameters when component training is disabled (ablation).
        if not self.ccme_trainable:
            for p in list(query_encoder.parameters()) + list(memory_encoder.parameters()):
                p.requires_grad = False
        if not self.crte_trainable:
            for p in list(trajectory_encoder.parameters()) + list(reflection_encoder.parameters()):
                p.requires_grad = False

        # Tracking
        self.step_count = 0
        self.update_count = 0
        self.loss_history = {
            'ccme': deque(maxlen=self.ccme_stability_window),
            'crte': deque(maxlen=self.crte_stability_window)
        }

        # Per-component checkpoints for independent CCME / CRTE rollback
        self.ccme_checkpoint = None
        self.crte_checkpoint = None
        self.last_stable_state = None  # combined reference kept for backward compat
        self.save_checkpoint()

    def save_ccme_checkpoint(self):
        """Save Eq and Em states for independent CCME rollback."""
        self.ccme_checkpoint = {
            'query_encoder': copy.deepcopy(self.query_encoder.state_dict()),
            'memory_encoder': copy.deepcopy(self.memory_encoder.state_dict()),
        }

    def save_crte_checkpoint(self):
        """Save Et and Er states for independent CRTE rollback."""
        self.crte_checkpoint = {
            'trajectory_encoder': copy.deepcopy(self.trajectory_encoder.state_dict()),
            'reflection_encoder': copy.deepcopy(self.reflection_encoder.state_dict()),
        }

    def rollback_ccme(self):
        """Rollback Eq and Em to last stable CCME checkpoint."""
        if self.ccme_checkpoint is not None:
            self.query_encoder.load_state_dict(self.ccme_checkpoint['query_encoder'])
            self.memory_encoder.load_state_dict(self.ccme_checkpoint['memory_encoder'])
            print("⚠️  Rolled back CCME (Eq, Em) to last stable checkpoint")

    def rollback_crte(self):
        """Rollback Et and Er to last stable CRTE checkpoint."""
        if self.crte_checkpoint is not None:
            self.trajectory_encoder.load_state_dict(self.crte_checkpoint['trajectory_encoder'])
            self.reflection_encoder.load_state_dict(self.crte_checkpoint['reflection_encoder'])
            print("⚠️  Rolled back CRTE (Et, Er) to last stable checkpoint")

    def save_checkpoint(self):
        """Save all encoder states (combined + per-component)."""
        self.save_ccme_checkpoint()
        self.save_crte_checkpoint()
        # keep last_stable_state in sync for external code that reads it
        self.last_stable_state = {
            'query_encoder': self.ccme_checkpoint['query_encoder'],
            'memory_encoder': self.ccme_checkpoint['memory_encoder'],
            'trajectory_encoder': self.crte_checkpoint['trajectory_encoder'],
            'reflection_encoder': self.crte_checkpoint['reflection_encoder'],
        }

    def rollback(self):
        """Rollback all encoders to last stable checkpoint."""
        self.rollback_ccme()
        self.rollback_crte()

    def check_stability_components(self) -> Tuple[bool, bool]:
        """
        Check per-component stability independently.
        Returns (ccme_stable, crte_stable).
        Enables selective rollback: an unstable CCME does not force CRTE rollback.
        """
        def _component_stable(history, window, threshold) -> bool:
            if len(history) < window // 2:
                return True
            hist_list = list(history)
            # Split the available history into equal halves — older vs recent.
            # Using hardcoded [-5:] / [:-5] ignores the actual window size and
            # produces near-empty "older" when history is short.
            mid = len(hist_list) // 2
            older = np.mean(hist_list[:mid])
            recent = np.mean(hist_list[mid:])
            increase = (recent - older) / (older + 1e-8)
            return increase < threshold

        ccme_stable = _component_stable(
            self.loss_history['ccme'],
            self.ccme_stability_window,
            self.ccme_stability_threshold,
        )
        crte_stable = _component_stable(
            self.loss_history['crte'],
            self.crte_stability_window,
            self.crte_stability_threshold,
        )
        return ccme_stable, crte_stable

    def check_stability(self) -> bool:
        """Check if training is stable (both CCME and CRTE).
        Returns True if stable, False if either component is degraded."""
        ccme_stable, crte_stable = self.check_stability_components()
        return ccme_stable and crte_stable

    def _calculate_reliability(self, meta: Dict) -> float:
        """Calculate Bayesian reliability: p̂(m) = (helpful+1)/(helpful+harmful+2)"""
        helpful = meta.get("helpful", 0)
        harmful = meta.get("harmful", 0)
        return (helpful + 1) / (helpful + harmful + 2)

    def train_ccme_step(self, false_negative_threshold: float = 0.8) -> Dict[str, float]:
        """
        Perform one CCME training step.

        Paper-aligned implementation:
        - Lines 394-396: Exclude items with p̂(m) > ρ from negative pool to avoid false negatives

        Args:
            false_negative_threshold: Reliability threshold ρ; items with p̂(m) > ρ are excluded
                                      from negatives to avoid false negatives (default 0.8)

        Returns:
            Loss dict
        """
        # Sample batch
        positives, negatives = self.data_collector.sample_ccme_batch(
            batch_size=self.batch_size,
            positive_ratio=0.5
        )

        if len(positives) == 0 or len(negatives) == 0:
            return {'ccme_loss': 0.0}

        # Paper lines 394-396: Filter out high-reliability items from negatives
        # to avoid false negatives (helpful items mislabeled as negatives)
        filtered_negatives = []
        for neg_pair in negatives:
            neg_mem = neg_pair.get("memory_item", {})
            meta = neg_mem.get("meta", {})
            reliability = self._calculate_reliability(meta)
            if reliability <= false_negative_threshold:
                filtered_negatives.append(neg_pair)

        # If all negatives filtered out, use original (with warning)
        if len(filtered_negatives) == 0:
            filtered_negatives = negatives

        # Prepare tensors
        query_embs = []
        pos_mem_embs = []
        neg_mem_embs_list = []

        expected_projected = None
        mixed_projected = False
        for pos_pair in positives:
            # Get query embedding (assume already embedded)
            if pos_pair.get("query_embedding") is not None:
                query_emb = pos_pair["query_embedding"]
            else:
                # If not provided, would need query encoder Eq (not implemented here)
                continue
            is_projected = bool(pos_pair.get("query_embedding_is_projected", False))
            if expected_projected is None:
                expected_projected = is_projected
            if is_projected != expected_projected:
                mixed_projected = True
                continue

            # Get positive memory embedding (projected, with grad if trainable)
            pos_mem = pos_pair["memory_item"]
            pos_mem_emb = self.memory_encoder.encode_memory_item(pos_mem, with_grad=True)

            # Sample negatives for this query (use filtered negatives)
            neg_embs = []
            for neg_pair in filtered_negatives[:min(8, len(filtered_negatives))]:  # Limit to 8 negatives
                neg_mem = neg_pair["memory_item"]
                neg_mem_emb = self.memory_encoder.encode_memory_item(neg_mem, with_grad=True)
                neg_embs.append(neg_mem_emb)

            if len(neg_embs) > 0:
                query_embs.append(query_emb)
                pos_mem_embs.append(pos_mem_emb)
                neg_mem_embs_list.append(torch.stack(neg_embs))

        if len(query_embs) == 0:
            if mixed_projected:
                print("⚠️  CCME batch has mixed projected/unprojected query embeddings; "
                      "skipping mismatched pairs. Standardize buffers to avoid drops.")
            return {'ccme_loss': 0.0}

        # Convert queries to tensors and project through Eq (query encoder)
        # Paper Algorithm 1 line 9: Train Eq, Em via L_CCME
        # Handle mixed numpy/tensor inputs from buffer
        query_tensor = torch.stack([_to_tensor(q, self.device) for q in query_embs])
        if expected_projected:
            # Embeddings are already Eq(x); just normalize for safety.
            query_tensor = F.normalize(query_tensor, p=2, dim=-1)
        else:
            if self.query_encoder.trainable or self.query_encoder.adapter_enabled:
                query_tensor = self.query_encoder(query_tensor)  # Use Eq for queries
            else:
                query_tensor = F.normalize(query_tensor, p=2, dim=-1)

        pos_tensor = torch.stack([_to_tensor(e, self.device) for e in pos_mem_embs]).to(self.device)

        # Pad negatives to same length
        max_negs = max(arr.shape[0] for arr in neg_mem_embs_list)
        padded_negs = []
        for neg_arr in neg_mem_embs_list:
            if neg_arr.shape[0] < max_negs:
                pad_size = max_negs - neg_arr.shape[0]
                padding = torch.zeros((pad_size, neg_arr.shape[1]), device=neg_arr.device)
                neg_arr = torch.cat([neg_arr, padding], dim=0)
            padded_negs.append(neg_arr)

        neg_tensor = torch.stack([_to_tensor(e, self.device) for e in padded_negs]).to(self.device)

        # Forward pass: L_CCME with Eq(query) and Em(memory)
        loss, loss_dict = self.ccme_loss(query_tensor, pos_tensor, neg_tensor)

        # Backward pass: Update both Eq and Em (paper Algorithm 1 line 9)
        if self.ccme_trainable and self.ccme_optimizer is not None and loss.requires_grad:
            self.ccme_optimizer.zero_grad()
            loss.backward()
            # Clip gradients for both Eq and Em
            torch.nn.utils.clip_grad_norm_(self.query_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.memory_encoder.parameters(), max_norm=1.0)
            self.ccme_optimizer.step()
        elif self.ccme_trainable and self.ccme_optimizer is not None and not loss.requires_grad:
            print("⚠️  CCME loss has no grad; skipping optimizer step")

        return {'ccme_loss': loss_dict['total']}

    def train_crte_step(self) -> Dict[str, float]:
        """
        Perform one CRTE training step.

        Paper-aligned implementation:
        - Positive pairs: (τᵢ⁺, uᵢ⁺) from SUCCESS episodes
        - Negative reflections for L_t→r: in-batch mismatches {uⱼ: j≠i} + failure reflections u⁻
        - Negative trajectories for L_r→t: in-batch mismatches {τⱼ: j≠i} + failure trajectories τ⁻
        - Failure trajectories also passed with success_mask=False for L_cluster and L_margin

        Returns:
            Loss dict
        """
        # Sample batch
        positives, negatives = self.data_collector.sample_crte_batch(
            batch_size=self.batch_size,
            positive_ratio=0.7
        )

        if len(positives) == 0:
            return {'crte_loss': 0.0}

        # Encode positive (SUCCESS) trajectories and reflections
        pos_traj_embs = []
        pos_refl_embs = []
        for pair in positives:
            pos_traj_embs.append(self.trajectory_encoder.encode_trajectory(
                pair["trajectory_steps"], with_grad=True))
            pos_refl_embs.append(self.reflection_encoder.encode_reflection(
                pair["reflection"], with_grad=True))

        if len(pos_traj_embs) == 0:
            return {'crte_loss': 0.0}

        # Encode negative (FAILURE) trajectories AND reflections
        neg_traj_embs = []
        neg_refl_embs = []
        for pair in negatives:
            neg_traj_embs.append(self.trajectory_encoder.encode_trajectory(
                pair["trajectory_steps"], with_grad=True))
            neg_refl_embs.append(self.reflection_encoder.encode_reflection(
                pair["reflection"], with_grad=True))

        # Build full batch: positives first, then failures (for clustering/margin via success_mask)
        all_traj_embs = pos_traj_embs + neg_traj_embs
        all_refl_embs = pos_refl_embs + neg_refl_embs

        traj_tensor = torch.stack([_to_tensor(e, self.device) for e in all_traj_embs]).to(self.device).float()
        refl_tensor  = torch.stack([_to_tensor(e, self.device) for e in all_refl_embs]).to(self.device).float()

        n_pos = len(pos_traj_embs)
        n_neg = len(neg_traj_embs)
        total  = n_pos + n_neg

        # success_mask: True for positives, False for failures
        # Used by L_cluster (only success trajectories) and L_margin (separate success vs failure)
        success_tensor = torch.zeros(total, dtype=torch.bool, device=self.device)
        success_tensor[:n_pos] = True

        # Build per-sample negative sets for InfoNCE (both directions):
        # L_t→r negatives for sample i: {uⱼ : j≠i} ∪ {u⁻ from failures}
        # L_r→t negatives for sample i: {τⱼ : j≠i} ∪ {τ⁻ from failures}
        # Only compute InfoNCE over the positive-pair portion of the batch (first n_pos)
        pos_traj_tensor = traj_tensor[:n_pos]
        pos_refl_tensor = refl_tensor[:n_pos]

        neg_refl_list = []
        neg_traj_list = []
        for i in range(n_pos):
            # In-batch reflection mismatches (other positive reflections)
            r_negs = [pos_refl_tensor[j] for j in range(n_pos) if j != i]
            # Add failure reflections
            if n_neg > 0:
                r_negs += [refl_tensor[n_pos + k] for k in range(n_neg)]
            if len(r_negs) == 0:
                r_negs = [torch.zeros_like(pos_refl_tensor[0])]
            neg_refl_list.append(torch.stack(r_negs))

            # In-batch trajectory mismatches (other positive trajectories)
            t_negs = [pos_traj_tensor[j] for j in range(n_pos) if j != i]
            # Add failure trajectories
            if n_neg > 0:
                t_negs += [traj_tensor[n_pos + k] for k in range(n_neg)]
            if len(t_negs) == 0:
                t_negs = [torch.zeros_like(pos_traj_tensor[0])]
            neg_traj_list.append(torch.stack(t_negs))

        def pad_neg_list(lst):
            max_n = max(t.size(0) for t in lst)
            padded = []
            for t in lst:
                if t.size(0) < max_n:
                    t = torch.cat([t, torch.zeros(max_n - t.size(0), t.size(1), device=t.device)], dim=0)
                padded.append(t)
            return torch.stack(padded)  # (n_pos, max_n, dim)

        neg_refl_tensor = pad_neg_list(neg_refl_list)
        neg_traj_tensor = pad_neg_list(neg_traj_list)

        # Forward pass — pass full traj/refl batch so clustering/margin see all episodes
        loss, loss_dict = self.crte_loss(
            traj_tensor, refl_tensor, neg_refl_tensor, neg_traj_tensor, success_tensor
        )

        # Backward pass
        if self.crte_trainable and self.crte_optimizer is not None and loss.requires_grad:
            self.crte_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trajectory_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.reflection_encoder.parameters(), max_norm=1.0)
            self.crte_optimizer.step()
        elif self.crte_trainable and self.crte_optimizer is not None and not loss.requires_grad:
            print("⚠️  CRTE loss has no grad; skipping optimizer step")

        return loss_dict

    def should_update(self) -> bool:
        """Check if should perform update based on step count."""
        return self.step_count > 0 and self.step_count % self.k_upd == 0

    def update(
        self,
        success_rate: Optional[float] = None,
        memory_bank: Optional[List[Dict]] = None,
        retriever=None
    ) -> Dict[str, any]:
        """
        Perform online update of encoders.

        Args:
            success_rate: Current rolling success rate for adaptive temperature
            memory_bank: Current memory bank for refinement trigger check
            retriever: CCMERetriever instance — provides memory_embeddings cache
                       used by adaptive refinement trigger (no extra API calls)

        Returns:
            Update statistics dict
        """
        print(f"\n{'='*60}")
        print(f"Online Update #{self.update_count + 1} (Step {self.step_count})")
        print(f"{'='*60}")

        # Check buffer sizes
        buffer_stats = self.data_collector.get_buffer_stats()
        print(f"Buffer sizes: {buffer_stats}")

        ccme_ready = buffer_stats['ccme_positive'] >= self.min_ccme_positive
        crte_ready = buffer_stats['crte_positive'] >= self.min_crte_positive

        if not ccme_ready and not crte_ready:
            print("⚠️  Insufficient training data, skipping update")
            return {
                'status': 'skipped',
                'reason': 'insufficient_data',
                'update_count': self.update_count,
                'avg_ccme_loss': 0.0,
                'avg_crte_loss': 0.0,
                'is_stable': True,
                'buffer_stats': buffer_stats,
                'memory_refinement': None
            }

        # Update adaptive temperature
        if success_rate is not None:
            self.ccme_loss.update_temperature(success_rate)

        # Training loop — wrapped so a training failure never blocks memory refinement
        ccme_losses = []
        crte_losses = []
        avg_ccme_loss = 0.0
        avg_crte_loss = 0.0
        is_stable = True
        status = 'success'

        # Set all four encoders to train mode (Eq, Em, Et, Er)
        self.query_encoder.train()  # Eq
        self.memory_encoder.train()  # Em
        self.trajectory_encoder.train()  # Et
        self.reflection_encoder.train()  # Er

        try:
            for step in range(self.num_train_steps):
                # Train CCME (Eq, Em via L_CCME)
                if ccme_ready:
                    ccme_loss_dict = self.train_ccme_step()
                    if ccme_loss_dict['ccme_loss'] > 0:
                        ccme_losses.append(ccme_loss_dict['ccme_loss'])

                # Train CRTE (Et, Er via L_CRTE)
                if crte_ready:
                    crte_loss_dict = self.train_crte_step()
                    if crte_loss_dict.get('total', 0) > 0:
                        crte_losses.append(crte_loss_dict['total'])

                if (step + 1) % 5 == 0:
                    print(f"  Step {step+1}/{self.num_train_steps}: "
                          f"CCME={np.mean(ccme_losses[-5:]):.4f}, "
                          f"CRTE={np.mean(crte_losses[-5:]):.4f}")

            # Record losses
            avg_ccme_loss = np.mean(ccme_losses) if ccme_losses else 0.0
            avg_crte_loss = np.mean(crte_losses) if crte_losses else 0.0

            self.loss_history['ccme'].append(avg_ccme_loss)
            self.loss_history['crte'].append(avg_crte_loss)

            # Per-component stability check (skip during post-refinement grace period)
            if self.post_refinement_grace_remaining > 0:
                print(f"[Stability] Post-refinement grace: "
                      f"{self.post_refinement_grace_remaining} updates remaining, skipping check")
                self.post_refinement_grace_remaining -= 1
                ccme_stable = crte_stable = True
            else:
                ccme_stable, crte_stable = self.check_stability_components()

            is_stable = ccme_stable and crte_stable

            if ccme_stable:
                print("✓ CCME stable, saving CCME checkpoint")
                self.save_ccme_checkpoint()
            else:
                print("⚠️  CCME instability detected! Rolling back Eq, Em.")
                self.rollback_ccme()
                if self.ccme_prune_on_instability:
                    self._prune_ccme_buffer_on_instability()
                status = 'ccme_rolled_back'

            if crte_stable:
                print("✓ CRTE stable, saving CRTE checkpoint")
                self.save_crte_checkpoint()
            else:
                print("⚠️  CRTE instability detected! Rolling back Et, Er.")
                self.rollback_crte()
                if self.crte_prune_on_instability:
                    self._prune_crte_buffer_on_instability()
                status = 'both_rolled_back' if status == 'ccme_rolled_back' else 'crte_rolled_back'

            # Keep combined last_stable_state in sync for external access
            self.last_stable_state = {
                'query_encoder': self.ccme_checkpoint['query_encoder'],
                'memory_encoder': self.ccme_checkpoint['memory_encoder'],
                'trajectory_encoder': self.crte_checkpoint['trajectory_encoder'],
                'reflection_encoder': self.crte_checkpoint['reflection_encoder'],
            }

            if ccme_ready and not crte_ready:
                status = 'ccme_only'
            elif crte_ready and not ccme_ready:
                status = 'crte_only'

        except Exception as _train_exc:
            import traceback as _tb
            import sys as _sys
            print(
                f"⚠️  Training loop failed ({type(_train_exc).__name__}: {_train_exc}); "
                f"skipping encoder update but continuing to memory refinement",
                file=_sys.stderr, flush=True,
            )
            _tb.print_exc(file=_sys.stderr)
            _sys.stderr.flush()
            status = 'train_failed'
        finally:
            # Always restore eval mode regardless of training outcome
            self.query_encoder.eval()
            self.memory_encoder.eval()
            self.trajectory_encoder.eval()
            self.reflection_encoder.eval()

        memory_refinement = None
        refined_memory_bank = None
        refinement_mean_sim = None
        refinement_triggered = False
        if memory_bank is None:
            should_refine = False
        elif self.refinement_mode == "adaptive":
            should_refine, refinement_mean_sim = self._should_refine_adaptive(memory_bank, retriever=retriever)
            refinement_triggered = should_refine
        else:  # "fixed"
            should_refine = (
                bool(self.refine_every_updates) and
                (self.update_count + 1) % self.refine_every_updates == 0
            )

        if should_refine:
            n_clusters = self.n_clusters_fixed if self.n_clusters_mode == "fixed" else None
            refined_memory_bank, memory_refinement = self.refine_memory(
                memory_bank=memory_bank,
                n_clusters=n_clusters,
                use_crte_embeddings=self.refine_use_crte_embeddings
            )
            # Post-refinement grace: skip stability checks for N updates so encoders
            # can re-adapt to the restructured memory bank without false rollbacks.
            if self.post_refinement_grace > 0:
                self.post_refinement_grace_remaining = self.post_refinement_grace
                print(f"[Memory Refinement] Post-refinement grace activated: "
                      f"{self.post_refinement_grace} updates")

        self.update_count += 1

        stats = {
            'status': status,
            'update_count': self.update_count,
            'avg_ccme_loss': avg_ccme_loss,
            'avg_crte_loss': avg_crte_loss,
            'is_stable': is_stable,
            'buffer_stats': buffer_stats,
            'memory_refinement': memory_refinement,
            'refinement_mean_sim': refinement_mean_sim,
            'refinement_triggered': refinement_triggered,
            'refinement_threshold': self.redundancy_threshold,
        }
        if refined_memory_bank is not None:
            stats['refined_memory_bank'] = refined_memory_bank

        print(f"{'='*60}\n")

        return stats

    def _should_refine_adaptive(self, memory_bank: List[Dict], retriever=None) -> Tuple[bool, Optional[float]]:
        """
        Trigger refinement when memory bank exhibits semantic redundancy.

        Computes mean pairwise cosine similarity across all memory embeddings.
        When this exceeds redundancy_threshold, the bank has accumulated enough
        overlapping entries to make clustering worthwhile — regardless of how
        many training updates have elapsed.

        Guards:
        - Requires at least min_items_for_refinement items (clustering on 2-3 items is wasteful)
        - Uses memory_embeddings cache from CCMERetriever — no extra API calls
        """
        if len(memory_bank) < self.min_items_for_refinement:
            return False, None

        # Collect embeddings from retriever's cache (CCMERetriever.memory_embeddings)
        emb_cache = getattr(retriever, "memory_embeddings", {}) if retriever is not None else {}
        embs = []
        for item in memory_bank:
            item_id = item.get("id", item.get("title", "unknown"))
            emb = emb_cache.get(item_id)
            if emb is not None:
                embs.append(emb)

        if len(embs) < self.min_items_for_refinement:
            return False, None

        emb_matrix = np.stack(embs)  # (N, D) — already L2-normalized
        sim_matrix = emb_matrix @ emb_matrix.T  # (N, N)

        # Mean of upper triangle — exclude diagonal (self-similarity = 1.0)
        n = len(embs)
        upper_idx = np.triu_indices(n, k=1)
        mean_sim = float(np.mean(sim_matrix[upper_idx]))

        should_trigger = mean_sim >= self.redundancy_threshold
        if should_trigger:
            print(f"[Memory Refinement] *** TRIGGERED *** mean_sim={mean_sim:.4f} >= threshold={self.redundancy_threshold} "
                  f"(n={n} embeddings, {len(memory_bank)} bank items)")
        else:
            print(f"[Memory Refinement] check: mean_sim={mean_sim:.4f} < threshold={self.redundancy_threshold} "
                  f"(n={n} embeddings, {len(memory_bank)} bank items) — not yet")
        return should_trigger, mean_sim

    def _prune_ccme_buffer_on_instability(self):
        """
        Quality-based pruning of CCME buffers on detected instability.

        Positive buffer: remove low-reliability positives.
        Reliability = (helpful+1)/(helpful+harmful+2) — Bayesian-smoothed signal
        accumulated across episodes. This is the label-grounded measure of whether
        a memory item is genuinely useful. Low reliability = the memory was often
        unhelpful/harmful, so training to align it with the query is a noisy signal.
        NOTE: low Eq(q)·Em(m) projected similarity is NOT the right signal — it
        identifies hard positives, which are the most informative training examples.

        Negative buffer: remove false negatives using BASE embedding similarity.
        Both q_emb (1536-dim) and memory_base_embedding (1536-dim) live in the fixed
        text-embedding-3-small space — stable across training updates, same dimension,
        no projection needed. If base_sim(q, m) > ccme_false_negative_threshold, the
        memory is semantically related despite the negative label, producing adversarial
        gradients that conflict with legitimate positives.

        Safety guard: positive pruning is skipped if it would reduce positive count
        below min_ccme_positive.
        """
        pos_buf = self.data_collector.ccme_positive_buffer
        neg_buf = self.data_collector.ccme_negative_buffer

        # --- Prune positive buffer: remove low-reliability positives ---
        pos_items = pos_buf.get_all()
        if pos_items:
            keep_pos = []
            n_unreliable = 0
            for entry in pos_items:
                meta = entry.get("memory_item", {}).get("meta", {})
                helpful = meta.get("helpful", 0)
                harmful = meta.get("harmful", 0)
                total_obs = helpful + harmful
                # Require at least 2 observations before pruning: a single failure
                # on an otherwise-unseen entry (helpful=0, harmful=1) gives
                # reliability=0.333, which would incorrectly fire at threshold=0.35.
                # With min 2 observations the worst credible case is (0,2) → 0.2,
                # safely below any reasonable threshold.
                if total_obs < 2:
                    keep_pos.append(entry)
                    continue
                reliability = (helpful + 1) / (total_obs + 2)  # Bayesian smoothing
                if reliability < self.ccme_reliability_prune_threshold:
                    n_unreliable += 1
                else:
                    keep_pos.append(entry)

            if n_unreliable > 0:
                if len(keep_pos) >= self.min_ccme_positive:
                    pos_buf.buffer = type(pos_buf.buffer)(keep_pos, maxlen=pos_buf.buffer.maxlen)
                    print(f"[CCME Prune] Removed {n_unreliable} low-reliability positives "
                          f"(reliability < {self.ccme_reliability_prune_threshold}); "
                          f"{len(keep_pos)} remain")
                else:
                    print(f"[CCME Prune] Positive prune skipped: would drop below "
                          f"min_ccme_positive={self.min_ccme_positive}")

        # --- Prune negative buffer: remove false negatives via base embedding similarity ---
        # Both q_emb and memory_base_embedding are 1536-dim base vectors — same space,
        # fixed and stable regardless of encoder training state.
        neg_items = neg_buf.get_all()
        keep_neg = []
        n_false_neg = 0
        for entry in neg_items:
            q_emb = entry.get("query_embedding")
            mem_base_emb = entry.get("memory_base_embedding")
            if q_emb is None or mem_base_emb is None:
                keep_neg.append(entry)
                continue
            q_np = np.array(q_emb, dtype=np.float32)
            m_np = np.array(mem_base_emb, dtype=np.float32)
            if q_np.shape != m_np.shape:
                keep_neg.append(entry)
                continue
            q_norm = q_np / (np.linalg.norm(q_np) + 1e-8)
            m_norm = m_np / (np.linalg.norm(m_np) + 1e-8)
            sim = float(np.dot(q_norm, m_norm))
            if sim > self.ccme_false_negative_threshold:
                n_false_neg += 1
                continue  # false negative — semantically related despite negative label
            keep_neg.append(entry)
        if n_false_neg:
            neg_buf.buffer = type(neg_buf.buffer)(keep_neg, maxlen=neg_buf.buffer.maxlen)
            print(f"[CCME Prune] Removed {n_false_neg} false negatives "
                  f"(base_sim > {self.ccme_false_negative_threshold}); {len(keep_neg)} remain")

    def _prune_crte_buffer_on_instability(self):
        """
        Quality-based pruning of CRTE buffers on detected instability.

        Positive buffer: remove stale or mismatched temporal-reflection pairs
        where cosine similarity in base space is below crte_stale_positive_threshold.
        These yield near-zero gradients and dilute the positive signal.

        Negative buffer: remove trivially easy negatives (sim < crte_trivial_negative_threshold).
        These are already maximally separated in embedding space; the encoder extracts
        no gradient signal from them and they waste batch capacity.

        Note: CRTE is structurally negative-heavy on math tasks (most episodes across
        different problems should not align). Do NOT guard on imbalance ratio here.

        Safety guard: pruning is skipped if it would reduce the positive count below
        min_crte_positive.
        """
        pos_buf = self.data_collector.crte_positive_buffer
        neg_buf = self.data_collector.crte_negative_buffer

        def _base_sim(entry: Dict) -> Optional[float]:
            """Cosine similarity using cached base embeddings."""
            t_emb = entry.get("trajectory_embedding")
            r_emb = entry.get("reflection_embedding")
            if t_emb is None or r_emb is None:
                return None
            t_np = np.array(t_emb, dtype=np.float32)
            r_np = np.array(r_emb, dtype=np.float32)
            if t_np.shape != r_np.shape:
                return None
            t_norm = t_np / (np.linalg.norm(t_np) + 1e-8)
            r_norm = r_np / (np.linalg.norm(r_np) + 1e-8)
            return float(np.dot(t_norm, r_norm))

        # --- Prune positive buffer: remove stale pairs ---
        pos_items = pos_buf.get_all()
        keep_pos = []
        n_stale = 0
        for entry in pos_items:
            sim = _base_sim(entry)
            if sim is not None and sim < self.crte_stale_positive_threshold:
                n_stale += 1
                continue  # stale/mismatched trajectory-reflection pair
            keep_pos.append(entry)
        if n_stale > 0:
            if len(keep_pos) >= self.min_crte_positive:
                pos_buf.buffer = type(pos_buf.buffer)(keep_pos, maxlen=pos_buf.buffer.maxlen)
                print(f"[CRTE Prune] Removed {n_stale} stale positives "
                      f"(sim < {self.crte_stale_positive_threshold}); {len(keep_pos)} remain")
            else:
                print(f"[CRTE Prune] Positive prune skipped: would drop below "
                      f"min_crte_positive={self.min_crte_positive}")

        # --- Prune negative buffer: remove trivially easy negatives ---
        neg_items = neg_buf.get_all()
        keep_neg = []
        n_trivial = 0
        for entry in neg_items:
            sim = _base_sim(entry)
            if sim is not None and sim < self.crte_trivial_negative_threshold:
                n_trivial += 1
                continue  # trivially separated — no gradient signal
            keep_neg.append(entry)
        if n_trivial > 0:
            neg_buf.buffer = type(neg_buf.buffer)(keep_neg, maxlen=neg_buf.buffer.maxlen)
            print(f"[CRTE Prune] Removed {n_trivial} trivial negatives "
                  f"(sim < {self.crte_trivial_negative_threshold}); {len(keep_neg)} remain")

    def refine_memory(
        self,
        memory_bank: List[Dict],
        n_clusters: Optional[int] = None,
        use_crte_embeddings: bool = True
    ) -> Tuple[List[Dict], Dict]:
        """
        Refine memory using clustering after training.

        Paper Reference (Algorithm 1 Step 11, lines 548-555):
        "Refine memory (cluster → prototype); rollback if stability metrics degrade"

        This consolidates semantically similar memory entries using k-medoids
        clustering. Should be called after online training completes successfully.

        IMPORTANT (Paper-Aligned Behavior):
        Per the paper, CRTE embeddings (specifically the Reflection Encoder Er) are
        used for clustering. Each memory item's bullets/insights are encoded as
        e_u = Er(u). This creates a semantic spase where functionally similar
        strategies cluster together, regardless of surfase-level text differences.

        The Trajectory Encoder (Et) is only used during CRTE training as "labels"
        for learning - at refinement time, only insight embeddings (Er) are used
        for the clustering spase.

        Args:
            memory_bank: Current memory bank
            n_clusters: Number of clusters (auto if None)
            use_crte_embeddings: If True (default), use CRTE Reflection Encoder Er.
                                This is the paper-aligned behavior.

        Returns:
            Tuple of (refined_memory_bank, refinement_stats)
        """
        from .memory_operations import MemoryOperations

        print(f"[Memory Refinement] Starting clustering on {len(memory_bank)} items...")
        if use_crte_embeddings:
            print(f"[Memory Refinement] Using CRTE Reflection Encoder Er (paper-aligned)")
        else:
            print(f"[Memory Refinement] Using CCME Memory Encoder Em (fallback)")

        # Create memory operations instance
        mem_ops = MemoryOperations(self.memory_encoder, **self.memory_ops_cfg)

        # Per paper: Use Reflection Encoder (Er) for clustering insights
        crte_encoder = self.reflection_encoder if use_crte_embeddings else None

        # Perform refinement
        refined_memory, refinement_stats = mem_ops.refine_memory_with_clustering(
            memory_bank,
            n_clusters=n_clusters,
            crte_encoder=crte_encoder,
            use_crte_embeddings=use_crte_embeddings
        )

        print(f"[Memory Refinement] Complete: {refinement_stats['original_count']} → "
              f"{refinement_stats['prototypes_kept']} items "
              f"({refinement_stats['items_removed']} removed via clustering)")

        return refined_memory, refinement_stats

    def step(
        self,
        success_rate: Optional[float] = None,
        memory_bank: Optional[List[Dict]] = None,
        retriever=None
    ) -> Optional[Dict]:
        """
        Increment step counter and trigger update if needed.

        Args:
            success_rate: Current success rate
            memory_bank: Current memory bank for refinement trigger check
            retriever: CCMERetriever instance for embedding cache access

        Returns:
            Update stats if update was performed, None otherwise
        """
        self.step_count += 1

        if self.should_update():
            return self.update(success_rate, memory_bank=memory_bank, retriever=retriever)

        return None
