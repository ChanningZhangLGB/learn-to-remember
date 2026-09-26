"""The streaming loop with batched writes.

Batching is not an optimization. Within a batch, retrieval reads a frozen snapshot and all
writes are buffered to the boundary. That does two things: it makes ordering deterministic
at batch granularity when items are processed in parallel (MMLU-Pro is ~12k items and will
be), and it removes the concurrent read-modify-write race that otherwise produces duplicate
entries and lost counter updates. `batch_size = 1` is pure online.

C2 runs as a bounded loop rather than a single call: it may write a program, receive what
it actually printed, and reason on from there. The loop is driven entirely prompt-side --
the transcript of previous calls is rendered into the next prompt -- so the `LLM` protocol
stays a one-shot `complete_json` and a provider client remains a thin wrapper. See
`lere/tools.py` for the executor and the reason its output overwrites what C2 claims.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .answers import is_correct, normalize_mcq
from .ccqs import CCQSTrainer
from .curate import apply_attribution, consolidate_and_write, maintenance_pass
from .llm import LLM, extract_json, load_prompt, render
from .retrieve import Retriever
from .schema import (CuratorOutput, PlannerOutput, SolverOutput, VocabViolations,
                     audit_solver_verdict, vocabulary_block)
from .store import SkillBook
from .tools import ToolConfig, ToolTranscript, probe_modules, run_python
from .verify import VerificationSignal, build_signal, resolve_verdict


@dataclass
class Item:
    """One benchmark question."""
    id: str
    question: str
    answer_type: str = "free"
    gold: str | None = None
    n_options: int = 10
    image: bytes | None = None
    dataset: str = "unknown"
    meta: dict = field(default_factory=dict)


@dataclass
class StepRecord:
    step: int
    item_id: str
    dataset: str
    domain: str
    retrieved: list[str]
    answer: str
    gold: str | None
    correct: bool
    signal_source: str
    signal_confidence: float
    used_positive: list[str] = field(default_factory=list)
    used_negative: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    book_size: int = 0
    latency_s: float = 0.0
    llm_calls: int = 0
    ccqs_pair: bool = False
    # The curator's own verdict, and whether it displaced the mechanical signal. Recorded
    # per item because the disagreement rate between C3 and each source is the number that
    # says how much the label-free supervision is actually worth.
    curator_correct: bool | None = None
    verdict_overridden: bool = False
    # C3's failure diagnosis. Recorded per step because it now modulates blame, and its
    # distribution has never been measured against anything.
    root_cause: str = "none"
    ccqs_pair_from_wrong_answer: bool = False
    tool_expected: bool = False
    tool_calls: int = 0
    tool_failures: int = 0
    error: str | None = None


@dataclass
class _Pending:
    """One completed read-phase item, waiting for the batch boundary.

    Views are captured at retrieval time so a later merge cannot change what a CCQS
    training pair was labelled against.
    """
    item: Item
    refs: list
    solver: SolverOutput
    signal: VerificationSignal
    curation: CuratorOutput
    record: StepRecord
    query_view: str
    ref_views: dict[str, str]


@dataclass
class RunReport:
    steps: list[StepRecord] = field(default_factory=list)
    violations: VocabViolations = field(default_factory=VocabViolations)
    ccqs: dict = field(default_factory=dict)
    permutation_seed: int | None = None

    @property
    def accuracy(self) -> float:
        graded = [s for s in self.steps if s.gold is not None]
        return sum(s.correct for s in graded) / len(graded) if graded else 0.0

    @property
    def retrieval_precision(self) -> float:
        """Retrieved entries the curator judged genuinely helpful."""
        total = sum(len(s.retrieved) for s in self.steps)
        pos = sum(len(s.used_positive) for s in self.steps)
        return pos / total if total else 0.0

    @property
    def harm_rate(self) -> float:
        """Retrieved entries that misled the solver. The headline number for GPQA-D."""
        total = sum(len(s.retrieved) for s in self.steps)
        neg = sum(len(s.used_negative) for s in self.steps)
        return neg / total if total else 0.0

    @property
    def guard_rejection_rate(self) -> float:
        proposed = sum(len(s.created) + len(s.merged) + len(s.rejected)
                       for s in self.steps)
        rejected = sum(len(s.rejected) for s in self.steps)
        return rejected / proposed if proposed else 0.0

    @property
    def total_latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    @property
    def mean_latency_s(self) -> float:
        """Wall-clock per item. LeRe spends >= 3 LLM calls where a CoT baseline spends 1,
        so accuracy has to be read against this and against `llm_calls`, not on its own."""
        return self.total_latency_s / len(self.steps) if self.steps else 0.0

    @property
    def total_llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def tool_call_rate(self) -> float:
        """Mean executions per item. Read beside `llm_calls_per_item`: a tool round-trip
        costs a call, so a rise here is a rise in the cost the baseline must be matched
        against."""
        return (sum(s.tool_calls for s in self.steps) / len(self.steps)
                if self.steps else 0.0)

    @property
    def tool_success_rate(self) -> float:
        """Fraction of executions that ran and exited zero.

        A low value is a solver problem, not a harness problem -- it means C2 is writing
        code that does not run, and every failure burned a call. Watch it during the pilot
        and cap `tools.max_calls_per_item` accordingly."""
        calls = sum(s.tool_calls for s in self.steps)
        fails = sum(s.tool_failures for s in self.steps)
        return (calls - fails) / calls if calls else 0.0

    @property
    def tool_agreement(self) -> float:
        """Accuracy on items that used a tool, minus accuracy on items that did not.

        The number that says whether tool use is earning its cost. Returns 0.0 when either
        side is empty."""
        with_t = [s for s in self.steps if s.tool_calls and s.gold is not None]
        without = [s for s in self.steps if not s.tool_calls and s.gold is not None]
        if not with_t or not without:
            return 0.0
        a = sum(s.correct for s in with_t) / len(with_t)
        b = sum(s.correct for s in without) / len(without)
        return a - b

    def accumulation_curve(self, bucket: int = 25) -> list[tuple[int, float]]:
        """Accuracy per bucket of `bucket` items -- the learning curve over the stream."""
        out: list[tuple[int, float]] = []
        graded = [s for s in self.steps if s.gold is not None]
        for i in range(0, len(graded), bucket):
            window = graded[i:i + bucket]
            if window:
                out.append((i // bucket, sum(s.correct for s in window) / len(window)))
        return out

    def summary(self) -> dict:
        return {
            "n": len(self.steps),
            "accuracy": round(self.accuracy, 4),
            "retrieval_precision": round(self.retrieval_precision, 4),
            "harm_rate": round(self.harm_rate, 4),
            "guard_rejection_rate": round(self.guard_rejection_rate, 4),
            "errors": sum(1 for s in self.steps if s.error),
            "mean_latency_s": round(self.mean_latency_s, 4),
            "total_latency_s": round(self.total_latency_s, 2),
            "llm_calls": self.total_llm_calls,
            "llm_calls_per_item": (round(self.total_llm_calls / len(self.steps), 2)
                                   if self.steps else 0.0),
            "tool_call_rate": round(self.tool_call_rate, 3),
            "tool_success_rate": round(self.tool_success_rate, 3),
            "tool_agreement": round(self.tool_agreement, 4),
            "permutation_seed": self.permutation_seed,
            "ccqs": self.ccqs,
            "violations": dict(self.violations),
        }


class Pipeline:
    def __init__(self, book: SkillBook, llm: LLM, cfg: dict) -> None:
        self.book = book
        self.llm = llm
        self.cfg = cfg
        self.retriever = Retriever(book, cfg.get("retrieval", {}))
        self.ccqs = CCQSTrainer(book.encoder, cfg.get("ccqs", {}))
        self.tools = ToolConfig.from_cfg(cfg.get("tools", {}))
        # Probed once per run, through the executor, and cached. See tools.probe_modules.
        self._modules = (probe_modules(self.tools) if self.tools.enabled else [])
        # `prompts.dir` selects an alternative prompt set (default `prompts/`). The
        # resolved path is recorded so a run's report says which set produced it.
        self.prompt_dir = (cfg.get("prompts") or {}).get("dir")
        self.p_c1 = load_prompt("c1_planner.md", self.prompt_dir)
        self.p_c2 = load_prompt("c2_solver.md", self.prompt_dir)
        self.p_c3 = load_prompt("c3_curator.md", self.prompt_dir)
        # Optional observer, fired once per completed step at the end of the write phase.
        # None by default and never consulted otherwise, so it cannot change a run; it
        # exists because a demo needs the book, the heads and the loss AFTER the update,
        # and reconstructing that from outside would mean reimplementing `_flush`.
        self.on_step = None

    # ------------------------------------------------------------- components

    def plan(self, item: Item, violations: VocabViolations) -> PlannerOutput:
        prompt = render(
            self.p_c1, query=item.question, image_present=str(item.image is not None),
            answer_type=item.answer_type, dataset=item.dataset,
            tools_available=str(self.tools.enabled).lower(),
            vocabulary=vocabulary_block(),
        )
        raw = self.llm.complete_json(prompt, component="c1", image=item.image)
        return PlannerOutput.parse(raw, violations)

    def solve(self, item: Item, refs, plan: PlannerOutput,
              violations: VocabViolations) -> tuple[SolverOutput, ToolTranscript, int]:
        """Run C2 to a final answer, executing any code it asks for along the way.

        Returns the parsed output, the execution transcript, and the number of LLM calls
        spent -- which is variable now, and is the denominator of the matched-cost
        comparison the paper turns on.

        A turn that asks for a tool is not an answer, so it does not count against the
        item; a turn that answers ends the loop. The budget is bounded twice over: by
        `max_calls_per_item`, and by a final forced turn that tells C2 its budget is gone
        and it must commit. Without the forced turn a model that keeps requesting tools
        would never produce an answer and the item would fail for a reason that has
        nothing to do with the question.
        """
        transcript = ToolTranscript()
        calls = 0
        budget = self.tools.max_calls_per_item if self.tools.enabled else 0

        while True:
            exhausted = transcript.calls >= budget
            prompt = render(
                self.p_c2, query=item.question, answer_type=item.answer_type,
                k=len(refs), references=Retriever.render(refs),
                tools_available=str(self.tools.enabled).lower(),
                tool_expected=str(plan.tool_expected).lower(),
                tool_budget=str(max(0, budget - transcript.calls)),
                tool_transcript=(transcript.render() or "(no code has been run yet)"),
                tool_guidance=self._tool_guidance(budget, transcript, plan),
                available_modules=(", ".join("`%s`" % m for m in self._modules)
                                   or "(the standard library only)"),
            )
            raw = self.llm.complete_json(prompt, component="c2", image=item.image)
            calls += 1

            wants_tool = (
                isinstance(raw, dict)
                and str(raw.get("action", "")).strip().lower() == "tool"
                and not exhausted
            )
            if not wants_tool:
                solver = SolverOutput.parse(raw, violations)
                self._reconcile_tool_output(solver, transcript, violations)
                return solver, transcript, calls

            code = str(raw.get("code", ""))
            prior = transcript.find_repeat(code)
            if prior is not None:
                # Identical program, identical sandbox, identical output. Echo the first
                # run's result and say so, rather than burning a subprocess to reproduce
                # it. The turn is still charged, so the budget still forces a commit.
                violations.bump("solver_repeated_identical_code")
                transcript.add_repeat(prior)
            else:
                transcript.add(run_python(code, self.tools))

    def _tool_guidance(self, budget: int, transcript: ToolTranscript,
                       plan: PlannerOutput) -> str:
        if not self.tools.enabled:
            return ("Code execution is DISABLED for this run. Do not emit a tool action; "
                    "solve by reasoning alone and set \"coding\" to \"N/A\".")
        left = budget - transcript.calls
        if left <= 0:
            return ("Your code-execution budget is spent. You must emit the final answer "
                    "object now, reasoning from the tool results above.")
        if transcript.calls == 0:
            # The planner's guess is worth stating once, before any evidence exists. After
            # a program has run, repeating it competes with the model's own results: it
            # kept saying "code would help" on every turn, and the solver kept agreeing
            # with it by re-running the same program.
            hint = ("The planner judged that running code would make this answer more "
                    "reliable. " if plan.tool_expected else "")
            return (f"{hint}You may run code up to {left} more time(s). Run code when it "
                    f"would settle the answer more reliably than reasoning alone; answer "
                    f"directly when it would not.")
        ran = transcript.calls
        return (f"You have already run code {ran} time(s) this item and the results are "
                f"above. You may run code up to {left} more time(s), but only if a "
                f"DIFFERENT program would change your answer -- to test a case the last "
                f"one missed, to fix a bug you can see in it, or to check a value it did "
                f"not print. Re-running the same program returns the same output and "
                f"wastes a turn. If the results above already settle the answer, or if "
                f"you cannot say what a new program would do differently, emit the final "
                f"answer object now.")

    def _reconcile_tool_output(self, solver: SolverOutput, transcript: ToolTranscript,
                               violations: VocabViolations) -> None:
        """Replace the solver's claim about its own program with what the program did.

        The failure this closes: C2's prompt asks for `coding_result`, C3's prompt is
        handed that string as evidence, and before `tools.py` existed nothing had ever run
        the code -- so the field was the model's guess at its own output, presented
        downstream as a measurement. Overwriting it means a model that misreports cannot
        mislead the curator, and the disagreement is counted rather than hidden.
        """
        solver.tool_calls = transcript.calls
        solver.tool_failures = transcript.failures
        if not transcript.results:
            return

        last = transcript.last_success or transcript.results[-1]
        claimed = " ".join((solver.coding_result or "").split())
        actual = " ".join((last.stdout or "").split())
        if claimed and claimed not in ("N/A", "n/a") and claimed != actual:
            violations.bump("solver_fabricated_coding_result")

        solver.coding = last.code
        solver.coding_result = (last.stdout if last.ok
                                else f"[{last.error}] {last.stderr}"[:2000])
        solver.tool_executed = True

    def curate(self, item: Item, refs, solver: SolverOutput, transcript: ToolTranscript,
               signal: VerificationSignal, violations: VocabViolations) -> CuratorOutput:
        prompt = render(
            self.p_c3, query=item.question, references=Retriever.render(refs),
            solver_output=_solver_digest(solver, transcript),
            signal_source=signal.source,
            signal_correct="null" if signal.correct is None else str(signal.correct).lower(),
            signal_confidence=f"{signal.confidence:.2f}", signal_detail=signal.detail,
            vocabulary=vocabulary_block(),
        )
        raw = self.llm.complete_json(prompt, component="c3", image=item.image)
        return CuratorOutput.parse(raw, [r.id for r in refs], violations)

    # ------------------------------------------------------------------- loop

    def run(self, items: Sequence[Item], report: RunReport | None = None) -> RunReport:
        """One streaming pass. The book starts empty and Ep/Es start at identity.

        Both resets are required for the prequential protocol to hold across repeated
        runs. `pass@1` over 5-10 runs means 5-10 *independent* passes; if the heads or the
        book carried over, run n would answer items whose labels had already shaped the
        retriever in runs 1..n-1, and an item would no longer be predicted before its own
        label was used. Vary `run.seed` across those runs so they differ in question
        order -- repeating one order only measures decoding noise.

        `reset_per_run` empties the book as well as the heads. It used to reset only the
        heads, so a caller looping this method to collect pass@1 carried the book silently
        from pass to pass and accuracy drifted upward for a reason that had nothing to do
        with the method.
        """
        report = report or RunReport()
        run_cfg = self.cfg.get("run", {})
        batch_size = max(1, int(run_cfg.get("batch_size", 8)))
        write_enabled = bool(run_cfg.get("write_enabled", True))

        if bool(run_cfg.get("reset_per_run", True)):
            # The book is emptied only on a writing pass. A frozen pass (SPEC section 1)
            # reads a book built elsewhere on a disjoint corpus -- that book IS the
            # experiment, and clearing it would leave the frozen arm retrieving from
            # nothing while still reporting a number. The heads reset either way: they are
            # cheap to rebuild, CCQS does not train on a frozen pass, and identity is the
            # defined starting state.
            if write_enabled:
                self.book.reset()
            self.ccqs.reset()

        ordered = list(items)
        seed = run_cfg.get("seed")
        if seed is not None:
            random.Random(int(seed)).shuffle(ordered)
            report.permutation_seed = int(seed)

        pending: list[_Pending] = []

        for step, item in enumerate(ordered):
            try:
                entry = self._process(item, step, report)
            except Exception as exc:                     # one bad item must not kill a run
                report.steps.append(StepRecord(
                    step=step, item_id=item.id, dataset=item.dataset, domain="other",
                    retrieved=[], answer="", gold=item.gold, correct=False,
                    signal_source="none", signal_confidence=0.0,
                    book_size=len(self.book), error=f"{type(exc).__name__}: {exc}",
                ))
                continue

            pending.append(entry)
            if write_enabled and len(pending) >= batch_size:
                self._flush(pending, report)
                pending = []

        if write_enabled and pending:
            self._flush(pending, report)
        elif pending:
            # Frozen pass: no evidence, no entries, and no CCQS intake -- the book and the
            # heads are both read-only. Retrieval bookkeeping is still recorded so the
            # frozen arm reports which entries it actually used.
            for p in pending:
                self.book.note_retrieved(p.record.retrieved, p.record.step)
                report.steps.append(p.record)

        report.ccqs = self.ccqs.stats.summary()
        return report

    def _process(self, item: Item, step: int, report: RunReport) -> _Pending:
        """Read-only phase: plan, retrieve against the frozen book, solve, verify, curate."""
        v = report.violations
        t0 = time.perf_counter()

        plan = self.plan(item, v)
        refs = self.retriever.retrieve(plan, step)
        solver, transcript, solver_calls = self.solve(item, refs, plan, v)
        audit_solver_verdict(solver, [r.id for r in refs], v)

        vcfg = self.cfg.get("verification", {})
        samples, sample_calls = self._consistency_samples(item, refs, plan, v, vcfg,
                                                          solver.answer)
        signal = build_signal(
            vcfg.get("source", "gt"), pred=solver.answer, gold=item.gold,
            answer_type=item.answer_type, n_options=item.n_options,
            samples=samples, code=solver.coding,
            recorded=transcript.last_success, recorded_runs=transcript.successes,
            question=item.question,
            # The judge source is C2's own confidence, which is the only self-report that
            # exists before C3 runs. Capped downstream; see verify.signal_from_judge.
            judge_correct=(solver.confidence >= 0.5),
            judge_confidence=solver.confidence, cfg=vcfg,
        )
        curation = self.curate(item, refs, solver, transcript, signal, v)

        # C3 has now read the reasoning. On every source but `gt` its verdict is the label
        # that assigns credit and gates CCQS intake; `gt` is ground truth and stands.
        signal, overridden = resolve_verdict(signal, curation.correct, vcfg)
        if overridden:
            v.bump("curator_overrode_signal")

        # Accuracy is always scored against gold when gold exists, independently of the
        # verification source. Otherwise a label-free run could not be compared to a
        # supervised one at all.
        if item.gold is not None:
            correct = is_correct(solver.answer, item.gold, item.answer_type,
                                 n_options=item.n_options, question=item.question)
            # How much of the score rests on the lenient option-text path rather than on
            # a letter. Counted because that path is substring-based (DC's rule, adopted
            # for comparability) and can credit a short numeric option on a wrong answer.
            if correct and item.answer_type == "mcq_letter" and \
                    normalize_mcq(str(solver.answer), item.n_options) is None:
                v.bump("mcq_scored_by_option_text")
        else:
            correct = bool(signal.correct)

        record = StepRecord(
            step=step, item_id=item.id, dataset=item.dataset, domain=plan.domain,
            retrieved=[r.id for r in refs], answer=solver.answer, gold=item.gold,
            correct=correct, signal_source=signal.source,
            signal_confidence=signal.confidence,
            curator_correct=curation.correct,
            verdict_overridden=overridden,
            root_cause=curation.root_cause,
            used_positive=list(curation.attribution.get("used_positive", [])),
            used_negative=list(curation.attribution.get("used_negative", [])),
            book_size=len(self.book),
            latency_s=time.perf_counter() - t0,
            llm_calls=1 + solver_calls + sample_calls + 1,   # C1 + C2 turns + votes + C3
            tool_expected=plan.tool_expected,
            tool_calls=transcript.calls,
            tool_failures=transcript.failures,
        )
        return _Pending(
            item=item, refs=refs, solver=solver, signal=signal, curation=curation,
            record=record, query_view=plan.query_view(),
            ref_views={r.id: r.skill_view for r in refs},
        )

    def _consistency_samples(self, item: Item, refs, plan: PlannerOutput,
                             violations: VocabViolations, vcfg: dict,
                             first_answer: str) -> tuple[list[str] | None, int]:
        """Draw extra solver samples for the label-free majority vote.

        Only drawn when the configured source actually needs them -- `consistency`, or
        `gt` on an item with no gold label, which degrades to it. Everything else returns
        None and spends nothing.

        The committed answer is sample 0, and the extra samples run through the identical
        path (same references, same tool budget). Drawing them any other way would make
        the vote a comparison between two different systems. The cost is real and belongs
        in the results table: `consistency_samples: 5` multiplies the solver's share of
        the bill by five, which is exactly why the honest baseline is self-consistency at
        a *matched* call budget rather than a single CoT pass.
        """
        source = vcfg.get("source", "gt")
        needs = source == "consistency" or (source == "gt" and item.gold is None)
        if not needs:
            return None, 0

        n = int(vcfg.get("consistency_samples", 5))
        samples = [first_answer]
        calls = 0
        for _ in range(max(0, n - 1)):
            extra, _, extra_calls = self.solve(item, refs, plan, violations)
            samples.append(extra.answer)
            calls += extra_calls
        return samples, calls

    def _flush(self, pending: list[_Pending], report: RunReport) -> None:
        """Write phase: apply the batch's buffered evidence and entries in order.

        CCQS intake happens here, after the item has been answered and scored, which is
        what makes the protocol prequential rather than transductive.
        """
        for p in pending:
            # Retrieval bookkeeping is a write, so it lands here rather than inside
            # `Retriever.retrieve` -- the read phase must see a frozen snapshot.
            self.book.note_retrieved(p.record.retrieved, p.record.step)

            apply_attribution(self.book, p.curation, p.signal,
                              self.cfg.get("verification", {}), p.item.id, p.record.step)

            pair = self.ccqs.observe(p.query_view, p.ref_views, p.curation.attribution,
                                     p.record.step, correct=p.signal.correct)
            p.record.ccqs_pair = pair is not None
            p.record.ccqs_pair_from_wrong_answer = (
                pair is not None and pair.answer_correct is False)

            if p.curation.proposed_entries:
                res = consolidate_and_write(
                    self.book, p.curation.proposed_entries, p.item.question, p.item.gold,
                    p.item.answer_type, self.cfg, p.item.id, p.record.step,
                )
                p.record.created = res.created
                p.record.merged = res.merged
                p.record.rejected = res.rejected

            maintenance_pass(self.book, self.cfg, p.item.id, p.record.step)
            p.record.book_size = len(self.book)
            report.steps.append(p.record)

            loss = self.ccqs.maybe_update(p.record.step)
            if self.on_step is not None:
                self.on_step(p, report, loss)


def _solver_digest(solver: SolverOutput, transcript: ToolTranscript | None = None) -> str:
    """Reconstruct C2's output for the C3 prompt, verbatim in content.

    When code ran, the real transcript is included and labelled as executed. C3's job is
    to catch errors the solver missed, and the difference between "the solver says its
    program printed 204" and "the program printed 204" is most of what makes that
    possible.
    """
    used = "\n".join(f"    used {u.get('id')}: {u.get('how_it_helped', '')}"
                     for u in solver.used) or "    (none)"
    unused = "\n".join(f"    unused {u.get('id')}: {u.get('why_not', '')} "
                       f"[{u.get('reason_code', '')}]"
                       for u in solver.unused) or "    (none)"

    if transcript is not None and transcript.results:
        code_block = (
            f"code execution ({transcript.calls} call(s), "
            f"{transcript.failures} failed) -- ACTUALLY EXECUTED, output is real:\n"
            f"{transcript.render()}"
        )
    else:
        code_block = (f"coding (NOT executed, the solver's own text):\n{solver.coding}\n\n"
                      f"coding_result (the solver's claim, unverified): "
                      f"{solver.coding_result}")

    return (
        f"reasoning_trajectory:\n{solver.reasoning_trajectory}\n\n"
        f"{code_block}\n\n"
        f"reference_verdict (the solver's own claim, not fact):\n{used}\n{unused}\n\n"
        f"answer: {solver.answer}\n"
        f"confidence: {solver.confidence}"
    )
