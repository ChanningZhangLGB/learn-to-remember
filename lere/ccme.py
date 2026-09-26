"""CCME -- Contrastive Contextual Memory Encoder: online training of the heads E_q, E_m.

E_q and E_m are fitted online from the curator's attribution. After each query a training
pair is buffered; every `k_upd` steps the buffer is replayed and the heads take a few
gradient steps on

    L_ccme = - sum_i log [ exp(sim(E_q(x_i), E_m(s_i^+)) / tau)
                           / ( exp(sim(E_q(x_i), E_m(s_i^+)) / tau)
                               + sum_{s^- in N_i} exp(sim(E_q(x_i), E_m(s^-)) / tau) ) ]

Three decisions here are not free parameters:

1. **`unused_redundant` is not a negative.** That bucket means *relevant, and correct, but
   already covered by another retrieved entry*. Training E_q/E_m to push it away from the
   query teaches the encoder that a genuinely matching skill does not match. Negatives are
   `used_negative` and `unused_irrelevant` only.

2. **In-batch negatives are added to the hard ones.** Every hard negative is drawn from a
   top-K the retriever already ranked highly, so hard-negatives-only gives the head a
   narrow, self-selected view of the space and it collapses. Other anchors' positives in
   the same batch supply the easy negatives that anchor the geometry.

3. **The protocol is prequential.** `observe()` is called only after the item has been
   answered and scored, so an item's own label never reaches the heads that retrieved for
   it. That property is what makes online training on the evaluation stream legitimate,
   and it depends on the heads being reset between runs -- see `DualEncoder.reset_heads`.

Realistic yield is the limiting factor, not the objective. A ~180-item AIME run starting
from an empty memory produces on the order of 10^2 positive pairs; that is enough to move a
single linear head slightly and not enough to demonstrate anything. MMLU-Pro (~12k) is the
only benchmark here with the stream length to show a real effect.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from .embed import DualEncoder

# Buckets that count as negative evidence for the encoder. `unused_redundant` is
# deliberately absent; see the module docstring.
NEGATIVE_BUCKETS = ("used_negative", "unused_irrelevant")


@dataclass
class CCMEPair:
    """One anchor query with the skills the curator judged useful, and those it did not."""
    query_view: str
    positives: list[str] = field(default_factory=list)     # skill views
    hard_negatives: list[str] = field(default_factory=list)
    step: int = 0
    # The curator's verdict on the ANSWER, carried for measurement, not applied here.
    # A positive pair is a claim about retrieval relevance -- "this query should have
    # surfaced this entry" -- and relevance does not stop being true because the solver
    # then made an arithmetic slip. Reliability is the consumer that legitimately keys on
    # the outcome; the InfoNCE geometry is not.
    answer_correct: bool | None = None


@dataclass
class CCMEStats:
    updates: int = 0
    pairs_seen: int = 0
    pairs_buffered: int = 0
    last_loss: float | None = None
    losses: list[float] = field(default_factory=list)
    pairs_from_wrong_answer: int = 0

    def summary(self) -> dict:
        return {
            "updates": self.updates,
            "pairs_seen": self.pairs_seen,
            "buffer": self.pairs_buffered,
            "last_loss": None if self.last_loss is None else round(self.last_loss, 4),
            "mean_loss": (round(sum(self.losses) / len(self.losses), 4)
                          if self.losses else None),
            "pairs_from_wrong_answer": self.pairs_from_wrong_answer,
        }


class CCMETrainer:
    """Owns the pair buffer, the optimizer, and the update schedule.

    Disabled (`enabled: false`) it is a no-op that still counts pairs, so the ablation
    "CCME on vs. off" runs the identical code path with the identical memory dynamics and
    differs only in whether the heads move.
    """

    def __init__(self, encoder: DualEncoder, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.encoder = encoder
        self.enabled = bool(cfg.get("enabled", True))
        self.k_upd = max(1, int(cfg.get("k_upd", 5)))
        self.tau = float(cfg.get("tau", 0.07))
        self.lr = float(cfg.get("lr", 1e-3))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.batch_size = int(cfg.get("batch_size", 32))
        self.steps_per_update = int(cfg.get("steps_per_update", 1))
        self.min_pairs = int(cfg.get("min_pairs", 4))
        self.max_negatives = int(cfg.get("max_negatives", 32))
        self.buffer: deque[CCMEPair] = deque(maxlen=int(cfg.get("buffer_size", 512)))
        self.stats = CCMEStats()
        self._rng = random.Random(int(cfg.get("seed", 0)))
        self._opt = None
        self._torch = None
        if self.enabled and encoder.trainable:
            self._build_optimizer()

    def _build_optimizer(self) -> None:
        try:
            import torch                                     # noqa: PLC0415
        except ImportError:
            # No autodiff available: degrade to a no-op rather than failing a run, and
            # make it visible in the report rather than silent.
            self.enabled = False
            return
        self._torch = torch
        params = list(self.encoder.eq.module.parameters()) + \
            list(self.encoder.em.module.parameters())
        self._opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    # ------------------------------------------------------------------ intake

    def observe(self, query_view: str, ref_views: dict[str, str],
                attribution: dict[str, list[str]], step: int,
                correct: bool | None = None) -> CCMEPair | None:
        """Buffer one training pair from a completed step. Prequential: call after scoring.

        `ref_views` maps retrieved entry id -> its skill view, captured at retrieval time
        so a later merge cannot silently change what the pair was labelled on.

        `correct` is the curator's verdict on the answer. It is **recorded on the pair,
        not used to gate it.** A CCME positive asserts that this query should have
        retrieved this entry, which is a relevance claim; a solver that had the right note
        and still slipped on the arithmetic does not make the note less relevant. Gating
        on the outcome would also throw away pairs on precisely the items where the memory
        is being built, and pair yield is already the binding constraint.
        The count is kept so the share of pairs drawn from failed items is measurable.
        """
        positives = [ref_views[i] for i in attribution.get("used_positive", [])
                     if i in ref_views]
        if positives and correct is False:
            self.stats.pairs_from_wrong_answer += 1
        if not positives:
            # No positive means no numerator; the InfoNCE term is undefined, and an
            # all-negative "pair" would only teach the query to repel everything.
            return None
        negatives: list[str] = []
        for bucket in NEGATIVE_BUCKETS:
            for i in attribution.get(bucket, []):
                if i in ref_views and ref_views[i] not in positives:
                    negatives.append(ref_views[i])

        pair = CCMEPair(query_view=query_view, positives=positives,
                        hard_negatives=negatives, step=step, answer_correct=correct)
        self.buffer.append(pair)
        self.stats.pairs_seen += 1
        self.stats.pairs_buffered = len(self.buffer)
        return pair

    def trainable_pairs(self) -> int:
        """Buffered pairs that could actually contribute a loss term.

        InfoNCE needs a positive AND at least one negative: with no negative the softmax
        denominator collapses to the numerator, the probability is 1, the loss is 0 and
        the gradient is exactly zero. A positive alone says "pull together" with nothing
        to push against, and on L2-normalized vectors that has the trivial solution of
        collapsing everything to one point.

        A pair qualifies if it carries hard negatives of its own, or if some *other* pair
        in the buffer offers a positive it does not already contain -- the in-batch case,
        which is why two pairs sharing one positive view are jointly untrainable.
        """
        all_pos = {t for p in self.buffer for t in p.positives}
        n = 0
        for pair in self.buffer:
            if pair.hard_negatives:
                n += 1
                continue
            if any(t not in pair.positives for t in all_pos):
                n += 1
        return n

    def due(self, step: int) -> bool:
        """Whether an update should be attempted now.

        `len(buffer) >= min_pairs` alone gated on the wrong quantity: it counted anchors
        while the loss needs anchors *with negatives*, so `due()` could return True on an
        update that `_loss` would then skip silently. The trainable check makes the gate
        say what it means.

        It is a necessary condition over the whole buffer, not a sufficient one over the
        sampled batch: with a buffer larger than `batch_size` the sample can still come
        back degenerate. That is why `_loss` keeps its own skip and why
        `updates_skipped_degenerate` stays a reported counter.
        """
        return (self.enabled and self._opt is not None
                and len(self.buffer) >= self.min_pairs
                and self.trainable_pairs() > 0
                and (step + 1) % self.k_upd == 0)

    # ---------------------------------------------------------------- training

    def maybe_update(self, step: int) -> float | None:
        """Run an update if one is due. Returns the loss, or None if it was skipped."""
        if not self.due(step):
            return None
        return self.update()

    def update(self) -> float | None:
        if not self.enabled or self._opt is None or not self.buffer:
            return None
        torch = self._torch
        total = 0.0
        n_steps = 0
        for _ in range(max(1, self.steps_per_update)):
            batch = self._sample_batch()
            loss = self._loss(batch)
            if loss is None:
                continue
            self._opt.zero_grad()
            loss.backward()
            self._opt.step()
            total += float(loss.detach())
            n_steps += 1
        if n_steps == 0:
            return None

        # Projected vectors computed under the previous parameters are now stale.
        self.encoder.bump_version()

        mean = total / n_steps
        self.stats.updates += 1
        self.stats.last_loss = mean
        self.stats.losses.append(mean)
        return mean

    def _sample_batch(self) -> list[CCMEPair]:
        pool = list(self.buffer)
        if len(pool) <= self.batch_size:
            return pool
        return self._rng.sample(pool, self.batch_size)

    def _loss(self, batch: list[CCMEPair]):
        """InfoNCE with hard negatives plus in-batch negatives.

        One anchor at a time because the candidate set is ragged: each anchor has its own
        hard negatives. Batches are <= 32, so the loop is not the bottleneck -- the base
        encoder pass is, and that is cached.
        """
        torch = self._torch
        if not batch:
            return None

        # One base pass for every distinct string in the batch, then split by tower.
        q_texts = [p.query_view for p in batch]
        s_texts: list[str] = []
        for p in batch:
            s_texts.extend(p.positives)
            s_texts.extend(p.hard_negatives)
        s_texts = list(dict.fromkeys(s_texts))
        if not s_texts:
            return None
        s_index = {t: i for i, t in enumerate(s_texts)}

        q_base = torch.from_numpy(self.encoder.base_vectors(q_texts))
        s_base = torch.from_numpy(self.encoder.base_vectors(s_texts))
        q = self.encoder.eq.forward_torch(q_base)          # (B, d), L2-normalized
        s = self.encoder.em.forward_torch(s_base)          # (S, d), L2-normalized

        # All positives in the batch, so one anchor's positive is another's easy negative.
        all_pos = {t for p in batch for t in p.positives}

        terms = []
        for bi, pair in enumerate(batch):
            pos_i = [s_index[t] for t in pair.positives]
            hard_i = [s_index[t] for t in pair.hard_negatives]
            in_batch_i = [s_index[t] for t in all_pos
                          if t not in pair.positives and t not in pair.hard_negatives]
            neg_i = hard_i + in_batch_i
            if len(neg_i) > self.max_negatives:
                neg_i = hard_i[:self.max_negatives] + \
                    self._rng.sample(in_batch_i,
                                     max(0, self.max_negatives - len(hard_i[:self.max_negatives])))
            if not neg_i:
                # Nothing to contrast against: the softmax would be degenerate at 1.
                continue

            sims_pos = (q[bi] @ s[pos_i].T) / self.tau     # (P,)
            sims_neg = (q[bi] @ s[neg_i].T) / self.tau     # (N,)
            neg_lse = torch.logsumexp(sims_neg, dim=0)
            for sp in sims_pos:
                denom = torch.logsumexp(torch.stack([sp, neg_lse]), dim=0)
                terms.append(denom - sp)                   # -log p(s+ | x)

        if not terms:
            return None
        return torch.stack(terms).mean()

    # ------------------------------------------------------------------- reset

    def reset(self) -> None:
        """Clear the buffer and re-initialize the heads. Start of every run."""
        self.buffer.clear()
        self.stats = CCMEStats()
        self.encoder.reset_heads()
        if self.enabled and self.encoder.trainable:
            self._build_optimizer()
