"""Run LeRe on one stream with one backbone and record everything the paper reports.

    python scripts/run_lere.py --model gpt-4.1-mini --stream AIME_2025
    python scripts/run_lere.py --model gemini-3.1-flash-lite --stream MathVista --top-k 5
    python scripts/run_lere.py --model gpt-4o-mini --stream GPQA_Diamond --no-ccme
    python scripts/run_lere.py --model gpt-4o-mini --stream AIME_2024 --dry-run   # no API

The configuration is configs/lere.yaml (Table 4) with configs/models/<model>.yaml merged
over its `llm:` block; `--set section.key=value` overrides anything else. The merged
config is saved in the run directory as config.snapshot.yaml.

Ablations (Table 2):
  --no-planner   retrieval keyed on the raw question; the Planner is never called
  --no-ccme      CCME heads stay at the identity (frozen sentence encoder)
  --no-exec      the Solver cannot execute code
Sensitivity (Table 8):  --top-k {1,3,5,10}

Output, in --out (default runs/<model>/<stream>[_<variant>]/):
  steps.jsonl       per item: plan, retrieval, solver, verification signal, curation,
                    gold correctness, latency and the LLM calls it made
  llm_calls.jsonl   every Planner / Solver / Curator call: prompt, raw text, tokens, cost
  retrieval.jsonl   every candidate entry: raw similarity, domain weight, score, MMR pick
  ccme.jsonl        per step: CCME pair yield, buffer, updates, loss, head statistics
  heads/            E_q / E_m weights at the start, after every update, and at the end
  memory/, memory_final.json   the memory bank before and after each item
  report.json       accuracy, cost, latency, CCME statistics and consistency checks

Nothing here changes the method: the recorders hang off `Pipeline.on_step` and
`ProviderLLM.on_call`. On Linux, sentence-transformers may need a newer libstdc++ than the
system one; `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib` fixes it (scripts/run_stream.sh).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lere.datasets import load_jsonl  # noqa: E402
from lere.embed import build_encoder  # noqa: E402
from lere.llm import EchoLLM  # noqa: E402
from lere.pipeline import Pipeline, RunReport  # noqa: E402
from lere.providers import ProviderLLM  # noqa: E402
from lere.memory import MemoryBank  # noqa: E402
from lere.trace import (CallRecorder, JsonlWriter, TracedLLM, TracingRetriever,  # noqa: E402
                        head_stats, head_weights)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "lere.yaml"
STREAMS = yaml.safe_load((REPO_ROOT / "configs" / "streams.yaml").read_text(encoding="utf-8"))


def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def apply_set(cfg: dict, assignment: str) -> None:
    """`section.key=value`, value parsed as YAML (so 5, false, null, 0.7 work)."""
    path, _, raw = assignment.partition("=")
    if not _:
        raise SystemExit("--set expects section.key=value, got %r" % assignment)
    node = cfg
    keys = path.split(".")
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(raw)


def resolve_items(stream: str) -> Path:
    if stream in STREAMS:
        path = REPO_ROOT / "data" / "subsets" / (STREAMS[stream] + ".jsonl")
        if not path.is_file() and path.with_suffix(".ids.json").is_file():
            raise SystemExit("%s is not redistributed; run scripts/data/fetch_streams.py first"
                             % path.name)
        return path
    p = Path(stream)
    if p.is_file():
        return p
    raise SystemExit("unknown stream %r; use a path or one of: %s" % (stream, ", ".join(STREAMS)))


def disable_planner() -> None:
    """The "w/o Planner" ablation: k_i = x_i instead of phi(pi_i).

    Planner is never called. The raw question becomes the semantic context (and hence the
    retrieval key), with no image description, no domain (normalized to `other`, so every
    entry gets the same soft domain weight), no tags and no code hint for the Solver.
    """
    from lere.schema import PlannerOutput

    def _plan_without_c1(self, item, violations):
        return PlannerOutput(semantic_context=item.question, domain="", tags=[],
                             retrieval_query="", visual_context=None,
                             reason="planner ablated: raw-question retrieval key",
                             tool_expected=False)

    Pipeline.plan = _plan_without_c1


class _DryRunLLM:
    """Offline stand-in that exercises every recorder: it retrieves, attributes, and
    proposes, which `EchoLLM` deliberately does not. Not a model, and not a baseline --
    its only job is to prove the instrumentation writes what it claims before the run
    costs money.
    """

    _REF_RE = re.compile(r"^\[(m_\d+)\]", re.MULTILINE)

    # Varied on purpose. A stub that proposes one topic makes every proposal merge into
    # one entry, every CCME pair share a single positive, and every update get skipped as
    # degenerate -- which is correct behaviour but leaves the trainer
    # path unexercised. These four give the buffer distinct positives and hard negatives.
    _TOPICS = [
        ("math.number_theory", ["strategy.casework"], "digit and divisibility casework"),
        ("math.geometry", ["strategy.invariant"], "similar triangles and area ratios"),
        ("math.combinatorics", ["strategy.backward_induction"], "counting arrangements"),
        ("math.algebra", ["tool.sympy"], "systems of nonlinear equations"),
    ]

    def __init__(self) -> None:
        self._turns = 0
        self._n = 0

    @property
    def _topic(self):
        return self._TOPICS[(self._n - 1) % len(self._TOPICS)]

    def complete_json(self, prompt: str, *, component: str, image=None) -> dict:
        if component == "planner":
            self._turns = 0
            self._n += 1
            domain, tags, blurb = self._topic
            return {"semantic_context": blurb, "visual_context": None,
                    "domain": domain, "tags": tags,
                    "retrieval_query": "%s | domain: %s" % (blurb, domain),
                    "tool_expected": True, "reason": "dry run"}
        if component == "solver":
            self._turns += 1
            if self._turns == 1:
                return {"action": "tool", "tool": "python",
                        "code": "print(%d)" % (70 + self._n), "why": "dry run"}
            refs = self._REF_RE.findall(prompt)
            return {"action": "answer", "reasoning_trajectory": "dry-run trajectory",
                    "coding": "print(%d)" % (70 + self._n),
                    "coding_result": str(70 + self._n),
                    "reference_verdict": {
                        "used": [{"id": r, "how_it_helped": "framed the casework"}
                                 for r in refs[:1]],
                        "unused": [{"id": r, "why_not": "different subproblem",
                                    "reason_code": "irrelevant"} for r in refs[1:]]},
                    "answer": str(70 + self._n), "confidence": 0.6}
        if component == "curator":
            refs = self._REF_RE.findall(prompt)
            return {"verification": {"correct": True, "reasoning_sound": True,
                                     "verdict_source": "signal", "reason": "dry run",
                                     "root_cause": "none"},
                    "attribution": {"used_positive": refs[:1], "used_negative": [],
                                    "unused_irrelevant": refs[1:],
                                    "unused_redundant": []},
                    "lesson": "split the count by digit place",
                    "sufficient": 0,
                    "proposed_entries": [{
                        "title": "%s, variant %d" % (self._topic[2], self._n),
                        "bullets": ["Name the constrained quantity first.",
                                    "Combine the independent parts afterwards."],
                        "example": "worked instance %d" % self._n,
                        "domain": self._topic[0],
                        "tags": self._topic[1]}]}
        raise ValueError(component)



def main() -> int:
    ap = argparse.ArgumentParser(description="Run LeRe on one stream.")
    ap.add_argument("--model", required=True,
                    help="backbone: a file stem in configs/models/ "
                         "(gemini-3.1-flash-lite, gpt-4.1-mini, gpt-4o-mini)")
    ap.add_argument("--stream", required=True,
                    help="a stream name from configs/streams.yaml, or a subset .jsonl path")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--top-k", type=int, default=None, help="retrieved entries K")
    ap.add_argument("--no-planner", action="store_true", help="ablation: raw-question key")
    ap.add_argument("--no-ccme", action="store_true", help="ablation: frozen encoder")
    ap.add_argument("--no-exec", action="store_true", help="ablation: no code execution")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="override any config value (repeatable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N items of the stream (a quick probe)")
    ap.add_argument("--out", default=None, help="run directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="scripted stand-in instead of the provider: exercises every "
                         "component and recorder with no API spend")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_file = REPO_ROOT / "configs" / "models" / (args.model + ".yaml")
    if not model_file.is_file():
        raise SystemExit("no backbone config %s" % model_file)
    cfg = deep_merge(cfg, yaml.safe_load(model_file.read_text(encoding="utf-8")))
    variant = []
    if args.top_k is not None:
        cfg["retrieval"]["top_k"] = args.top_k
        variant.append("k%d" % args.top_k)
    if args.no_ccme:
        cfg["ccme"]["enabled"] = False
        variant.append("no_ccme")
    if args.no_exec:
        cfg["tools"]["enabled"] = False
        variant.append("no_exec")
    if args.no_planner:
        disable_planner()
        variant.append("no_planner")
    cfg.setdefault("run", {})["no_planner"] = bool(args.no_planner)   # recorded only
    for a in args.set:
        apply_set(cfg, a)

    items_path = resolve_items(args.stream)
    args.items = str(items_path)
    items = load_jsonl(items_path)
    if args.limit:
        items = items[:args.limit]

    name = items_path.stem if args.stream not in STREAMS else args.stream
    out = Path(args.out) if args.out else REPO_ROOT / "runs" / args.model / (
        "_".join([name] + variant) + ("_dry" if args.dry_run else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory").mkdir(exist_ok=True)
    (out / "heads").mkdir(exist_ok=True)

    (out / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                              encoding="utf-8")
    with (out / "items.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it.id, "gold": it.gold,
                                 "question": it.question}, ensure_ascii=False) + "\n")

    w_calls = JsonlWriter(out / "llm_calls.jsonl")
    w_retr = JsonlWriter(out / "retrieval.jsonl")
    w_ccme = JsonlWriter(out / "ccme.jsonl")
    w_steps = JsonlWriter(out / "steps.jsonl")

    recorder = CallRecorder(w_calls)
    if args.dry_run:
        llm = TracedLLM(_DryRunLLM(), recorder)
    else:
        llm = ProviderLLM.from_config(cfg, on_call=recorder)
        print("client:", llm)

    encoder = build_encoder(cfg.get("embedding", {}))
    memory = MemoryBank(encoder=encoder)
    pipe = Pipeline(memory, llm, cfg)

    class _SnapshottingRetriever(TracingRetriever):
        """Saves the memory as it stood when the query arrived.

        `on_step` fires after the write phase, so `memory/step_NN.json` is the memory AFTER
        item NN. Retrieval is the first thing that touches the memory on an item, so this is
        the only place to capture the state the query was actually scored against.
        """

        def retrieve(self, plan, step):
            memory.save(out / "memory" / ("step_%02d_before.json" % step))
            return super().retrieve(plan, step)

    pipe.retriever = _SnapshottingRetriever(memory, cfg.get("retrieval", {}), w_retr)

    np.savez(out / "heads" / "start.npz", **head_weights(encoder))
    # `DualEncoder.reset_heads()` bumps the version itself, so `version > 0` is NOT evidence
    # that CCME moved anything -- the post-reset baseline is already 1. The honest
    # trigger is `stats.updates` increasing.
    state = {"updates": 0, "written": 0}

    def on_step(p, report, loss) -> None:
        step = p.record.step
        enc_stats = head_stats(encoder)
        cstats = pipe.ccme.stats

        # Full weights only when the heads actually moved: 384x384 x2 per step does not
        # survive to a 12k-item run, and an unchanged matrix carries no information.
        if cstats.updates != state["updates"]:
            np.savez(out / "heads" / ("update_step%02d.npz" % step),
                     **head_weights(encoder))
            state["updates"] = cstats.updates

        memory.save(out / "memory" / ("step_%02d.json" % step))

        # New write-log rows since the previous step: create / merge / reject / quarantine
        # / prune, with the detail string each carries.
        writes = [vars(r) for r in memory.write_log[state["written"]:]]
        state["written"] = len(memory.write_log)

        # Every proposal scored against every entry now in the memory. This is the raw
        # material for calibrating the merge / link thresholds (real should-merge / should-separate
        # pairs), and it is measured AFTER the write, so a proposal that merged appears
        # here scored against the entry it merged into.
        proposal_sims = []
        for prop in p.curation.proposed_entries:
            pv = encoder.encode_entries([prop.identity_view()])[0]
            scored = sorted(
                ({"entry_id": e.id, "entry_identity_view": e.identity_view(),
                  "cosine": round(float(pv @ memory.vector(e)), 6)}
                 for e in memory.active()),
                key=lambda r: r["cosine"], reverse=True)[:5]
            proposal_sims.append({"proposal_title": prop.title,
                                  "proposal_identity_view": prop.identity_view(),
                                  "measured_after_write": True,
                                  "nearest": scored})

        # An update that was scheduled but produced no loss is the documented degenerate
        # case: every buffered pair had the same positive and no
        # negatives, so the softmax has nothing to contrast and the step is correctly
        # skipped. Distinguishing it from "not due yet" is the difference between a
        # trainer bug and a data-yield fact.
        was_due = pipe.ccme.due(step)
        w_ccme.write({
            "step": step, "ccme_loss": loss,
            "pair_buffered_this_step": bool(p.record.ccme_pair),
            "update_was_due": was_due,
            "update_skipped_degenerate": bool(was_due and loss is None),
            "distinct_positives_in_buffer": len(
                {v for pair in pipe.ccme.buffer for v in pair.positives}),
            "pairs_from_wrong_answer": pipe.ccme.stats.pairs_from_wrong_answer,
            "trainable_pairs": pipe.ccme.trainable_pairs(),
            "pairs_with_hard_negatives": sum(
                1 for pair in pipe.ccme.buffer if pair.hard_negatives),
            "stats": pipe.ccme.stats.summary(),
            "losses": list(pipe.ccme.stats.losses),
            "buffer_len": len(pipe.ccme.buffer),
            "enabled": pipe.ccme.enabled, "k_upd": pipe.ccme.k_upd,
            "min_pairs": pipe.ccme.min_pairs, "heads": enc_stats,
        })

        retr = getattr(pipe.retriever, "last", {}) or {}
        sig = p.signal
        w_steps.write({
            "step": step,
            "item_id": p.item.id,
            "question": p.item.question,
            "gold": p.item.gold,
            "planner": {"domain": p.record.domain, "query_view": p.query_view,
                        "tool_expected": p.record.tool_expected},
            "retrieval": {
                "n_candidates": retr.get("n_candidates"),
                "n_passed_floor": retr.get("n_passed_floor"),
                "best_raw_sim": retr.get("best_raw_sim"),
                "sim_threshold": retr.get("sim_threshold"),
                "selected": retr.get("selected_scores"),
                "encoder_version": retr.get("encoder_version"),
            },
            "solver": {
                "answer": p.solver.answer, "confidence": p.solver.confidence,
                "tool_calls": p.record.tool_calls,
                "tool_failures": p.record.tool_failures,
                "tool_executed": getattr(p.solver, "tool_executed", False),
                "coding": p.solver.coding, "coding_result": p.solver.coding_result,
                "reasoning_trajectory": p.solver.reasoning_trajectory,
            },
            # The whole point of the label-free arm: `signal_correct` is what the run
            # learned from, `gold_correct` is what it is scored on, and they are allowed
            # to disagree. The disagreement rate is the number to read.
            "signal": {"source": sig.source, "correct": sig.correct,
                       "confidence": sig.confidence, "detail": sig.detail},
            # Curator is the verdict authority on every source but `gt`; these say whether it
            # exercised that and whether the exercise was right.
            "verdict": {
                "curator_correct": p.record.curator_correct,
                "overrode_signal": p.record.verdict_overridden,
                "root_cause": p.record.root_cause,
                "ccme_pair_from_wrong_answer": p.record.ccme_pair_from_wrong_answer,
            },
            "gold_correct": p.record.correct,
            "signal_correct": sig.correct,
            "signal_agrees_with_gold": (None if sig.correct is None
                                        else bool(sig.correct) == bool(p.record.correct)),
            "curation": {
                "attribution": p.curation.attribution,
                "lesson": getattr(p.curation, "lesson", None),
                "n_proposed": len(p.curation.proposed_entries),
                "created": p.record.created, "merged": p.record.merged,
                "rejected": p.record.rejected,
                "write_log": writes,
                "proposal_similarities": proposal_sims,
            },
            "memory": memory.snapshot_stats(),
            "llm_calls": recorder.slice_for_step(step),
            "llm_calls_this_step": p.record.llm_calls,
            "latency_s": p.record.latency_s,
        })
        done = step + 1
        graded = [s for s in report.steps if s.gold is not None]
        hits = sum(1 for s in graded if s.correct)
        print("  [%2d/%2d] %-16s ans=%-6s gold=%-4s %-5s | acc %2d/%-2d = %5.1f%% | "
              "refs=%d sig=%-6s memory=%2d pairs=%d loss=%s"
              % (done, len(items), p.item.id, (p.solver.answer or "-")[:6], p.item.gold,
                 "OK" if p.record.correct else "WRONG",
                 hits, len(graded), 100.0 * hits / max(1, len(graded)),
                 len(p.refs),
                 "%s%s" % (sig.correct, "*" if p.record.verdict_overridden else ""),
                 len(memory), pipe.ccme.stats.pairs_seen,
                 "-" if loss is None else round(loss, 4)), flush=True)

    pipe.on_step = on_step

    print("running %d items from %s" % (len(items), args.items), flush=True)
    print("  config=%s  sim_threshold=%s  k_upd=%s  source=%s"
          % (args.config, cfg["retrieval"]["sim_threshold"], cfg["ccme"]["k_upd"],
             cfg["verification"]["source"]), flush=True)
    report = RunReport()
    t0 = time.perf_counter()
    pipe.run(items, report=report)
    wall = time.perf_counter() - t0

    np.savez(out / "heads" / "end.npz", **head_weights(encoder))
    memory.save(out / "memory_final.json")

    traced = {json.loads(l)["step"] for l in
              (out / "steps.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    errored = [{"step": s.step, "item_id": s.item_id, "error": s.error}
               for s in report.steps if s.error]

    payload = {
        "config": args.config,
        "model": args.model,
        "stream": args.stream,
        "variant": variant or ["default"],
        "items_file": args.items,
        "n_items": len(items),
        "wall_s": round(wall, 1),
        "summary": report.summary(),
        "usage": (llm.usage.summary() if hasattr(llm, "usage") else None),
        "heads_final": head_stats(encoder),
        "checks": {
            # A step that raised never reaches the write phase, so it has a StepRecord and
            # no trace row. Reconciled rather than left as a silent gap in the files.
            "steps_recorded": len(report.steps),
            "steps_traced": len(traced),
            "errored_steps": errored,
            "order_preserved": [s.item_id for s in report.steps] == [i.id for i in items],
            "permutation_seed": report.permutation_seed,
            "signal_vs_gold_disagreements": sum(
                1 for l in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["signal_agrees_with_gold"] is False),
            "signal_unavailable": sum(
                1 for l in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["signal_correct"] is None),
            "ccme_updates": pipe.ccme.stats.updates,
            "ccme_pairs_seen": pipe.ccme.stats.pairs_seen,
            "ccme_updates_skipped_degenerate": sum(
                1 for l in (out / "ccme.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["update_skipped_degenerate"]),
            "heads_moved": head_stats(encoder)["eq"]["delta_from_identity"] > 0.0,
            "curator_overrode_signal": report.violations.get("curator_overrode_signal", 0),
            "ccme_pairs_from_wrong_answer": pipe.ccme.stats.pairs_from_wrong_answer,
            "root_cause_distribution": {
                rc: sum(1 for s in report.steps if s.root_cause == rc)
                for rc in sorted({s.root_cause for s in report.steps})},
            "curator_verdict_vs_gold": {
                "agree": sum(1 for s in report.steps
                             if s.curator_correct is not None
                             and bool(s.curator_correct) == s.correct),
                "disagree": sum(1 for s in report.steps
                                if s.curator_correct is not None
                                and bool(s.curator_correct) != s.correct),
                "no_verdict": sum(1 for s in report.steps if s.curator_correct is None),
            },
        },
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2, default=str),
                                     encoding="utf-8")

    for w in (w_calls, w_retr, w_ccme, w_steps):
        w.close()

    print("\n--- summary ---")
    print(json.dumps(payload["summary"], indent=2, default=str))
    print("--- usage ---")
    print(json.dumps(payload["usage"], indent=2))
    print("--- checks ---")
    print(json.dumps(payload["checks"], indent=2, default=str))
    print("--- heads ---")
    print(json.dumps(payload["heads_final"], indent=2))
    print("\nrun dir:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
