"""Top-K retrieval: similarity floor, domain filter, alpha-mix with reliability, MMR.

The scoring rule is the convex combination from the design document:

    score(x, s) = alpha * sim(E_q(x), E_m(s)) + (1 - alpha) * r_hat(s)

with both terms in [0, 1] so the weights mean what they say. `r_hat` is a Beta posterior
mean and is bounded by construction; `sim` is a cosine and is **not**, so it is clamped at
zero before the mix. Two details make it behave:

* **A negative cosine is clamped to zero before the domain penalty.** The penalty is
  multiplicative, so on a negative `sim` it moved the score *up*: at penalty 0.25 an entry
  at -0.5 in a mismatched domain scored -0.375, i.e. a domain mismatch improved it. The
  floor hides this at any non-negative threshold, which is every configuration shipped, but
  it made `sim_threshold: -1.0` -- the natural way to disable the floor entirely -- silently
  invert the domain filter. Clamping also makes the [0, 1] claim above true unconditionally
  rather than true-by-accident-of-the-floor. Ordering among negative candidates is lost,
  which costs nothing: a negative cosine means actively dissimilar, and ranking within that
  is noise.

* **The relevance floor is applied to raw `sim`, before the mix.** Under an additive rule a
  well-established entry clears any floor on reliability alone -- at alpha=0.7 an entry
  with sim 0.1 and r_hat 0.95 scores 0.355, which would pass a floor set on the mixed
  score. Gating on raw similarity keeps "nothing here is relevant" reachable, which is the
  cold-start and no-match path: an empty result is a normal outcome, and Solver handles k=0.
  Returning the least-bad entry instead spends context and invites force-fitting.

* **alpha is weaker early than it looks.** With no evidence r_hat is exactly 0.5 for every
  entry, so (1 - alpha) * r_hat is a constant offset and ranking is pure similarity. The
  reliability term only starts separating entries once evidence accumulates, so alpha's
  effective weight grows over a run rather than being constant. That is the Laplace prior
  doing its job, but it needs stating: an alpha ablation measures something different at
  step 20 than at step 2000.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import PlannerOutput, MemoryEntry, domain_affinity
from .memory import MemoryBank


def parse_threshold(value) -> float | None:
    """`sim_threshold` as a number, or None meaning no floor at all.

    `false` and `null` both disable it. A sentinel is clearer than the numeric workaround:
    `0.0` still drops genuinely negative cosines, and `-1.0` reads as a magic number. With
    no floor every active entry is a candidate and `top_k` alone decides, which is the
    "just give me the k most similar" configuration.
    """
    if value is None or value is False:
        return None
    return float(value)


@dataclass
class RetrievedRef:
    entry: MemoryEntry
    raw_sim: float
    score: float
    domain_affinity: float

    @property
    def id(self) -> str:
        return self.entry.id

    @property
    def identity_view(self) -> str:
        """The exact string E_m encoded, captured for the CCME training pair."""
        return self.entry.identity_view()


def _render_for_prompt(refs: list[RetrievedRef]) -> str:
    """The full entry goes to the solver -- everything except meta.

    Retrieval indexes `identity_view` (title, domain, tags) but the solver needs the bullets
    and the example, which are the actionable part. meta is withheld: it is our usage
    bookkeeping, and showing helpful/harmful counts to Solver would let the solver defer to a
    popularity statistic instead of judging the note on its merits.
    """
    if not refs:
        return "(no relevant memory entries were retrieved)"
    blocks = []
    for r in refs:
        e = r.entry
        bullets = "\n".join(f"  - {b}" for b in e.bullets)
        blocks.append(
            f"[{e.id}] {e.title}\n"
            f"  domain: {e.domain}   tags: {', '.join(e.tags)}\n"
            f"{bullets}\n"
            f"  example: {e.example}"
        )
    return "\n\n".join(blocks)


class Retriever:
    def __init__(self, memory: MemoryBank, cfg: dict) -> None:
        self.memory = memory
        self.cfg = cfg or {}

    def retrieve(self, plan: PlannerOutput, step: int) -> list[RetrievedRef]:
        cfg = self.cfg
        candidates = self.memory.active()
        if not candidates:
            return []                                    # cold start: a normal outcome

        qvec = self.memory.encoder.encode_query(plan.query_view())
        mat = np.stack([self.memory.vector(e) for e in candidates])
        sims = mat @ qvec                                # both sides L2-normalized

        mode = cfg.get("domain_filter", "soft")
        penalty = float(cfg.get("domain_penalty", 0.25))
        partial = float(cfg.get("domain_partial_credit", 0.6))
        alpha = float(cfg.get("alpha", 0.7))
        threshold = parse_threshold(cfg.get("sim_threshold", 0.6))

        scored: list[RetrievedRef] = []
        for entry, raw in zip(candidates, sims):
            raw = float(raw)
            if threshold is not None and raw < threshold:  # floor on raw sim, pre-mix
                continue
            aff = domain_affinity(plan.domain, entry.domain, partial)
            if mode == "hard" and aff == 0.0:
                continue
            # Clamp before the penalty: it is multiplicative, and on a negative cosine
            # it would raise the score rather than lower it. See the module docstring.
            sim = max(0.0, raw)
            if mode == "soft":
                sim *= 1.0 - penalty * (1.0 - aff)
            score = alpha * sim + (1.0 - alpha) * entry.meta.reliability
            scored.append(RetrievedRef(entry, raw, score, aff))

        if not scored:
            return []

        scored.sort(key=lambda r: r.score, reverse=True)
        selected = self._mmr(scored, int(cfg.get("top_k", 3)),
                             float(cfg.get("mmr_lambda", 0.7)))
        # `note_retrieved` is deliberately NOT called here. It mutates retrieved_count and
        # last_used_step, and SPEC section 9 promises the read phase sees a frozen
        # snapshot; a write here is a read-modify-write race the moment items run in
        # parallel, which MMLU-Pro requires. The pipeline records it at the batch
        # boundary instead.
        return selected

    def _mmr(self, ranked: list[RetrievedRef], k: int, lam: float) -> list[RetrievedRef]:
        """Maximal marginal relevance, so Top-K is not K paraphrases of one entry.

        Matters here specifically because consolidation runs at a high cosine; entries
        between the link and merge thresholds are legitimately similar and would otherwise
        crowd out the slate.
        """
        if k <= 0:
            return []
        if len(ranked) <= 1 or lam >= 1.0:
            return ranked[:k]

        vecs = {r.id: self.memory.vector(r.entry) for r in ranked}
        selected = [ranked[0]]
        pool = ranked[1:]
        while pool and len(selected) < k:
            best, best_val = None, -np.inf
            for cand in pool:
                redundancy = max(
                    float(vecs[cand.id] @ vecs[s.id]) for s in selected
                )
                val = lam * cand.score - (1.0 - lam) * redundancy
                if val > best_val:
                    best, best_val = cand, val
            selected.append(best)
            pool.remove(best)
        return selected

    @staticmethod
    def render(refs: list[RetrievedRef]) -> str:
        """Format the slate for the Solver prompt's {{references}} slot."""
        return _render_for_prompt(refs)
