"""Credit assignment and the write path: attribution -> evidence, proposal -> guard ->
consolidate -> write.

Both halves of the loop that v0 left open live here. v0's C3 emitted a positive/negative
memory verdict and proposed new entries, and nothing consumed either: no arrow from the
error analyzer back to the memory bank, and no rule for how helpful/harmful/reliability
would ever change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .guard import GuardResult, check_entry
from .schema import CuratorOutput, ProposedEntry
from .store import SkillBook
from .verify import VerificationSignal


@dataclass
class CurationResult:
    created: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)   # (title, reasons)
    evidence_applied: dict[str, tuple[float, float]] = field(default_factory=dict)
    quarantined: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)

    @property
    def guard_rejection_rate(self) -> float:
        total = len(self.created) + len(self.merged) + len(self.rejected)
        return len(self.rejected) / total if total else 0.0


# ----------------------------------------------------------- credit assignment

def apply_attribution(book: SkillBook, curation: CuratorOutput,
                      signal: VerificationSignal, cfg: dict,
                      query_id: str, step: int) -> dict[str, tuple[float, float]]:
    """Turn C3's attribution plus the verification outcome into evidence counters.

    Eq. (12) of the paper. An earlier version scaled evidence twice: by a per-source `gain`
    (exec 0.5/1.0, judge 0.3/0.6) and, for a used note on a wrong answer, by a
    `root_cause`-keyed blame factor (0.25-1.0). Both were hand-set hypotheses, and together
    they made positive evidence arrive at a fraction of the weight of negative evidence,
    which showed up directly in the CCQS geometry: the heads learned to push negatives away
    and barely moved positives. Both are removed. Evidence is `confidence` alone, symmetric:

        used_positive & correct  ->  helpful += confidence
        used_positive & wrong    ->  harmful += confidence
        used_negative & wrong    ->  harmful += confidence
        used_negative & correct  ->  nothing (noise, not evidence)

    `gains` and `blame_by_root_cause` / `used_but_wrong_factor` are ignored if present in
    a config. `root_cause` is still recorded on the step for analysis.
    """
    conf = float(signal.confidence)
    correct = bool(signal.correct)

    applied: dict[str, tuple[float, float]] = {}

    for mid in curation.attribution.get("used_positive", []):
        if correct:
            h, x = conf, 0.0
        else:
            h, x = 0.0, conf
        book.apply_evidence(mid, helpful=h, harmful=x, query_id=query_id, step=step)
        applied[mid] = (h, x)

    for mid in curation.attribution.get("used_negative", []):
        # Blame only lands when the answer was actually wrong. A note that "misled" on a
        # correct answer is noise, not evidence.
        h, x = (0.0, conf) if not correct else (0.0, 0.0)
        if x:
            book.apply_evidence(mid, helpful=0.0, harmful=x, query_id=query_id, step=step)
        applied[mid] = (h, x)

    # unused_* buckets carry no evidence; they feed the retrieval-precision metric only.
    return applied


# ------------------------------------------------------------- consolidation

def nearest_entry(book: SkillBook, proposal: ProposedEntry,
                  same_domain_only: bool = True) -> tuple[str | None, float]:
    """Nearest existing entry to a proposal, over skill views.

    Skill views exclude bullets. Including them would make two proposals of the same skill
    that each contribute a different step score far below the merge threshold, so both get
    written -- the exact near-duplicate growth consolidation is meant to prevent. Since
    retrieval now indexes the same rendering, consolidation and retrieval agree on what
    "the same skill" means, which they did not in an earlier version.
    """
    pool = [e for e in book.entries.values()
            if not same_domain_only or e.domain == proposal.domain]
    if not pool:
        return None, 0.0
    pvec = book.encoder.encode_skill([proposal.skill_view()])[0]
    mat = np.stack([book.vector(e) for e in pool])
    sims = mat @ pvec
    idx = int(np.argmax(sims))
    return pool[idx].id, float(sims[idx])


def consolidate_and_write(book: SkillBook, proposals: list[ProposedEntry],
                          question_text: str, gold_answer: str | None,
                          answer_type: str, cfg: dict, query_id: str,
                          step: int) -> CurationResult:
    """Guard, then merge-or-create, for each proposed entry."""
    result = CurationResult()
    cur_cfg = cfg.get("curation", {})
    guard_cfg = cfg.get("guard", {})

    merge_t = float(cur_cfg.get("merge_threshold", 0.90))
    link_t = float(cur_cfg.get("link_threshold", 0.80))
    max_bullets = int(cur_cfg.get("max_bullets_after_merge", 8))
    max_props = int(cur_cfg.get("max_proposals_per_query", 3))

    for proposal in proposals[:max_props]:
        verdict: GuardResult = check_entry(
            proposal, question_text, gold_answer, answer_type, guard_cfg
        )
        if not verdict.accepted:
            result.rejected.append((proposal.title, verdict.reason_str))
            book.log(step, query_id, "reject", "-",
                     f"{proposal.title} :: {verdict.reason_str}")
            continue

        target_id, sim = nearest_entry(book, proposal)
        if target_id is not None and sim >= merge_t:
            book.merge(target_id, proposal, query_id, step, max_bullets=max_bullets)
            result.merged.append(target_id)
        else:
            related = [target_id] if (target_id and sim >= link_t) else []
            entry = book.create(proposal, query_id, step, related_ids=related)
            result.created.append(entry.id)

    return result


def maintenance_pass(book: SkillBook, cfg: dict, query_id: str,
                     step: int) -> tuple[list[str], list[str]]:
    """Quarantine proven-harmful entries, then enforce capacity."""
    quarantined = book.quarantine_pass(cfg.get("pruning", {}), step, query_id)
    pruned = book.prune_to_capacity(cfg.get("pruning", {}), step, query_id)
    return quarantined, pruned
