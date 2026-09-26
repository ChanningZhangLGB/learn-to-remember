"""Verification signal: one contract, four sources.

This is what makes label-free and GT-available the same system rather than two. Every
source produces the same `VerificationSignal`, and everything downstream -- meta updates,
quarantine, curation -- reads only that. Swapping supervision becomes a config change, so
the label-free result is an ablation on supervision strength rather than a different
pipeline.

The asymmetric gains encode a real property of self-consistency: disagreement among
samples is decent evidence of an error, while agreement is weak evidence of correctness,
because models are confidently and consistently wrong. So negative evidence is trusted
more than positive evidence in label-free mode.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .answers import canonical_answer, is_correct
from .tools import ToolResult


@dataclass
class VerificationSignal:
    correct: bool | None
    confidence: float
    source: str
    detail: str = ""
    # Which row of the `gains` table prices this verdict. Defaults to `source` and differs
    # only when the curator overrode: the item really was supervised by `source` (that is
    # what the ablation asks) but the verdict came from a model, so it is priced as
    # `judge`. Splitting the two keeps `source` honest for reporting without letting an
    # unbacked verdict draw a measured source's weight.
    gains_source: str | None = None

    def gains(self, cfg: dict) -> tuple[float, float]:
        """(gain_positive, gain_negative) for whichever source prices this verdict."""
        table = (cfg.get("gains") or {}).get(self.gains_source or self.source) or {}
        # v2: evidence is confidence alone; this table is no longer consulted by the
        # write path. Kept so callers still resolve, defaulting to 1.0/1.0.
        return float(table.get("positive", 1.0)), float(table.get("negative", 1.0))


# --------------------------------------------------------------------- sources

def signal_from_gt(pred: str, gold: str, answer_type: str,
                   n_options: int = 10, question: str | None = None) -> VerificationSignal:
    ok = is_correct(pred, gold, answer_type, n_options=n_options,
                    question=question)
    # Curator reads `detail`, so in GT mode this is where the curator learns the gold answer
    # (the same information ACE's Reflector receives in its GT arm). Say what it is and
    # how it may be used; the leak guard still rejects any entry that restates it.
    return VerificationSignal(
        correct=ok, confidence=1.0, source="gt",
        detail=f"pred={canonical_answer(pred, answer_type, n_options)!r} "
               f"gold={canonical_answer(gold, answer_type, n_options)!r} "
               f"(gold is the ground-truth answer: use it to locate where the reasoning "
               f"went right or wrong; never state it in a proposed entry)",
    )


def signal_from_consistency(samples: list[str], answer_type: str,
                            n_options: int = 10) -> VerificationSignal:
    """Majority vote over n independent solver samples. No labels required.

    `correct` here means 'agrees with the consensus', not 'is right' -- that is the whole
    limitation of label-free operation, and the asymmetric gains exist because of it.
    """
    canon = [canonical_answer(s, answer_type, n_options) for s in samples]
    canon = [c for c in canon if c]
    if not canon:
        return VerificationSignal(None, 0.0, "consistency", "no parseable samples")
    counts = Counter(canon)
    top, n_top = counts.most_common(1)[0]
    agreement = n_top / len(canon)
    return VerificationSignal(
        correct=(canon[0] == top),
        confidence=float(agreement),
        source="consistency",
        detail=f"majority={top!r} agreement={agreement:.2f} n={len(canon)}",
    )


def _distinct_outputs(runs) -> list:
    """Distinct stdout values across successful runs, whitespace-normalized."""
    seen = []
    for r in runs or []:
        out = " ".join((r.stdout or "").split())
        if out and out not in seen:
            seen.append(out)
    return seen


def signal_from_exec(code: str, expected: str, answer_type: str,
                     timeout: int = 20,
                     recorded: "ToolResult | None" = None,
                     recorded_runs=None) -> VerificationSignal:
    """Compare a program's real output against the answer the solver committed to.

    A real signal with no labels -- the strongest option available in label-free mode on
    AIME and the numeric part of MathVista.

    **Agreement here is weaker than it looks, and the confidences say so.** When Solver reads
    its answer off its own program's output -- the normal pattern on AIME -- agreement is
    guaranteed by construction: a wrong program yields a wrong answer and the two still
    match. That checks transcription, not correctness. Disagreement is the informative
    direction, because a solver that committed to something its own program contradicts is
    reliably in trouble. So confidence is 0.5 on agreement and 0.9 on disagreement, the
    same asymmetry `consistency` uses and for the same reason.

    `recorded` is the execution Solver already performed during its own reasoning
    (`lere/tools.py`). Prefer it: re-running the same program costs a second subprocess
    and, worse, can disagree with the first if the code is not deterministic -- which
    would make the verification signal depend on which of two runs it happened to read.
    Falling back to running the code covers `tools.enabled: false`, where Solver wrote code
    but nothing executed it.

    `recorded_runs` is every successful execution. `recorded` is the LAST of them, which
    is the answer-producing program only by coincidence: a solver that computes its answer
    and then runs one more exploratory program leaves the exploratory output as the thing
    verified. When the successful runs printed conflicting values this source cannot say
    which one the answer came from, so agreement is discounted further and the conflict is
    named in `detail`.

    This is the *post-hoc check*, not the solver's tool. `lere/tools.py` runs during Solver's
    reasoning and feeds the result back to it; this runs afterwards and feeds Curator.
    """
    if recorded is not None:
        if not recorded.ok:
            return VerificationSignal(
                False, 0.5, "exec", f"execution failed: {recorded.error}"
            )
        out = (recorded.stdout or "").strip()
        if not out:
            return VerificationSignal(None, 0.0, "exec", "recorded run printed nothing")
        agrees = is_correct(out, expected, answer_type)
        outputs = _distinct_outputs(recorded_runs)
        conflict = len(outputs) > 1
        if agrees:
            # Transcription-consistency, not correctness. See the docstring.
            conf = 0.3 if conflict else 0.5
            detail = ("answer matches the program it was read from (transcription check, "
                      f"not an independent one); stdout={out[:80]!r}")
            if conflict:
                detail += f"; CONFLICTING successful runs printed {outputs!r}"
        else:
            conf = 0.9
            detail = (f"answer {expected[:40]!r} CONTRADICTS its own program's output "
                      f"{out[:80]!r}")
            if conflict:
                detail += f"; successful runs printed {outputs!r}"
        return VerificationSignal(correct=agrees, confidence=conf, source="exec",
                                  detail=detail)

    if not code or code.strip() in ("", "N/A"):
        return VerificationSignal(None, 0.0, "exec", "no code supplied")

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "check.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=timeout, cwd=td,
            )
        except subprocess.TimeoutExpired:
            return VerificationSignal(None, 0.0, "exec", "timeout")

    if proc.returncode != 0:
        return VerificationSignal(
            False, 0.5, "exec", f"nonzero exit: {proc.stderr.strip()[:200]}"
        )
    out = proc.stdout.strip()
    if not out:
        return VerificationSignal(None, 0.0, "exec", "no stdout")
    agrees = is_correct(out, expected, answer_type)
    return VerificationSignal(
        correct=agrees, confidence=0.5 if agrees else 0.9, source="exec",
        detail=(f"answer matches the program it was read from (transcription check); "
                f"stdout={out[:80]!r}" if agrees else
                f"answer {expected[:40]!r} CONTRADICTS program output {out[:80]!r}"),
    )


def signal_from_judge(judge_correct: bool | None, judge_confidence: float,
                      cfg: dict) -> VerificationSignal:
    """Weakest source: the solver's own self-report, capped.

    `judge_correct` must be supplied by the caller. It used to default to None here and
    the pipeline never passed it, so this source returned `correct=None` for every item --
    a signal that moved no counter and taught nothing, while the config still claimed a
    verification source was active. The pipeline now derives it from Solver's own confidence
    (`>= 0.5`), which is honest about what it is: the weakest source in the table, capped
    at `judge_confidence_cap` precisely because self-judging is optimistic.
    """
    cap = float(cfg.get("judge_confidence_cap", 0.6))
    return VerificationSignal(
        correct=judge_correct,
        confidence=min(float(judge_confidence or 0.0), cap),
        source="judge",
        detail=f"self-report capped at {cap}",
    )


# Sources whose verdict Curator may not overrule. `gt` is ground truth: letting the curator
# second-guess a gold label would corrupt the supervised arm, which is the control the
# `gt` vs `consistency` ablation depends on. Every other source is a model or a mechanical
# check that can be, and on `exec` routinely is, wrong in a way Curator can see.
AUTHORITATIVE_SOURCES = ("gt",)


def resolve_verdict(signal: VerificationSignal, curator_correct: bool | None,
                    cfg: dict | None = None) -> tuple:
    """Decide which verdict is the label, and say whether the curator overrode.

    Returns `(signal_to_use, overridden)`.

    Returns `(signal_to_use, overridden)`. `source` is preserved -- the item genuinely was
    supervised by that source, which is what the ablation asks -- but the verdict is
    **priced as `judge`** via `gains_source`, because a model's judgement is what produced
    it.

    That split fixes a real inversion. Keeping `exec`'s gains gave a verdict with no
    mechanical backing (`exec` returned null: 0.6 x 1.0 = 0.60) *more* evidence than one
    with a real measurement behind it (0.5 x 1.0 = 0.50). Priced as `judge` the ordering
    is restored: 0.6 x 0.6 = 0.36, below the measured 0.50 and above a transcription
    check's 0.25. The override itself stays visible in `verdict_overridden` on the step
    record and in the `curator_overrode_signal` counter, so nothing is hidden by keeping
    the source label truthful.

    Confidence is never raised above `judge_confidence_cap`, because an overriding verdict
    is Curator's judgement rather than a stronger measurement. But it is not clamped *down* to a
    signal that had nothing to say: when the source returned `correct=None` (no code ran,
    no samples drawn) its confidence is 0.0, and carrying that through made the curator's
    verdict weightless. Two of the three `used_negative` events in the first 30-item runs
    applied no evidence at all for exactly this reason. Supplying a verdict where none
    existed is the judge case, so it carries the judge cap.
    """
    cfg = cfg or {}
    if curator_correct is None:
        return signal, False
    if signal.source in AUTHORITATIVE_SOURCES:
        return signal, False
    if signal.correct is not None and bool(curator_correct) == bool(signal.correct):
        return signal, False

    cap = float(cfg.get("judge_confidence_cap", 0.6))
    was = "null" if signal.correct is None else str(signal.correct).lower()
    conf = cap if signal.correct is None else min(float(signal.confidence), cap)
    return (VerificationSignal(
        correct=bool(curator_correct),
        confidence=conf,
        source=signal.source,
        gains_source="judge",
        detail=f"curator verdict {str(curator_correct).lower()} OVERRODE "
               f"{signal.source}={was}; original detail: {signal.detail}",
    ), True)


def build_signal(source: str, *, pred: str = "", gold: str | None = None,
                 answer_type: str = "free", n_options: int = 10,
                 samples: list[str] | None = None, code: str = "",
                 recorded: "ToolResult | None" = None, recorded_runs=None,
                 question: str | None = None,
                 judge_correct: bool | None = None, judge_confidence: float = 0.0,
                 cfg: dict | None = None) -> VerificationSignal:
    """Dispatch on config, with a documented degradation path.

    If a source is unavailable for an item -- no gold label, no code -- it falls back to
    the next weakest rather than failing, and the resulting signal records which source
    actually produced it so the run report can break results down by supervision.

    Every fallback is *visible*: `signal.source` names the source that actually fired, and
    a degraded item is counted in the run report. An earlier version fell back from
    `consistency` to `judge` with `judge_correct=None`, which produced a signal carrying
    no information at all while the config still said "consistency" -- the ablation looked
    like it ran, and had not.
    """
    cfg = cfg or {}
    if source == "gt":
        if gold is None:
            return build_signal("consistency", pred=pred, answer_type=answer_type,
                                n_options=n_options, samples=samples, code=code,
                                recorded=recorded, recorded_runs=recorded_runs,
                                question=question, judge_correct=judge_correct,
                                judge_confidence=judge_confidence, cfg=cfg)
        return signal_from_gt(pred, gold, answer_type, n_options, question=question)

    if source == "exec":
        sig = signal_from_exec(code, pred, answer_type, recorded=recorded,
                               recorded_runs=recorded_runs)
        if sig.correct is None and samples:
            return signal_from_consistency(samples, answer_type, n_options)
        return sig

    if source == "consistency":
        if not samples:
            return signal_from_judge(judge_correct, judge_confidence, cfg)
        return signal_from_consistency(samples, answer_type, n_options)

    if source == "judge":
        return signal_from_judge(judge_correct, judge_confidence, cfg)

    raise ValueError(f"unknown verification source: {source!r}")
