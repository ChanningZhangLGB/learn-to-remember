#!/usr/bin/env python
"""Score LeRe run directories with the paper's shared scorer (scripts/eval/scoring.py).

    python scripts/eval/score_runs.py runs/gpt-4.1-mini/AIME_2025
    python scripts/eval/score_runs.py runs/*/* --csv results.csv
    python scripts/eval/score_runs.py runs/*/* --table          # model x stream accuracy

Each run is re-scored from its per-item records (steps.jsonl), never from a summary. The
item metadata (answer type, options) comes from the subset file the run used. Items that
raised during the run have no record and count as incorrect: the denominator is always the
full stream. Reported accuracy is the audited column (see scoring.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import score_item  # noqa: E402

STREAMS = {v: k for k, v in __import__("yaml").safe_load(
    (REPO / "configs" / "streams.yaml").read_text()).items()}


def subset_for(run: Path) -> Path:
    """The subset file whose ids match the run's items, in order."""
    ids = [json.loads(l)["id"] for l in (run / "items.jsonl").open(encoding="utf-8")]
    rep = json.loads((run / "report.json").read_text()) if (run / "report.json").is_file() else {}
    cand = [Path(rep["items_file"])] if rep.get("items_file") else []
    cand += sorted((REPO / "data" / "subsets").glob("*.jsonl"))
    for c in cand:
        if c.is_file():
            sub = [json.loads(l)["id"] for l in c.open(encoding="utf-8")]
            if sub[:len(ids)] == ids:
                return c
    raise SystemExit(f"{run}: no subset in data/subsets/ matches its items.jsonl")


def score_run(run: Path) -> dict:
    sub = subset_for(run)
    items = [json.loads(l) for l in sub.open(encoding="utf-8")]
    n_run = sum(1 for _ in (run / "items.jsonl").open(encoding="utf-8"))
    by_id = {it["id"]: it for it in items}
    steps = [json.loads(l) for l in (run / "steps.jsonl").open(encoding="utf-8") if l.strip()]
    base = aud = 0
    for r in steps:
        it = by_id[r["item_id"]]
        pred = (r.get("solver") or {}).get("answer")
        b, a = score_item(pred, it["gold"], it["question"], it["answer_type"],
                          int(it.get("n_options") or 10))
        base += b
        aud += a
    rep = json.loads((run / "report.json").read_text()) if (run / "report.json").is_file() else {}
    usage = rep.get("usage") or {}
    summ = rep.get("summary") or {}
    lat = [s.get("latency_s") or 0.0 for s in steps]
    snap = run / "config.snapshot.yaml"
    model = rep.get("model") or (__import__("yaml").safe_load(snap.read_text())["llm"]["model"]
                                 if snap.is_file() else run.parent.name)
    return {
        "run": str(run), "model": model,
        "stream": STREAMS.get(sub.stem, sub.stem), "variant": "+".join(rep.get("variant") or []),
        "n": n_run, "scored": len(steps), "errored": n_run - len(steps),
        "correct": aud, "accuracy": 100.0 * aud / n_run if n_run else float("nan"),
        "accuracy_base": 100.0 * base / n_run if n_run else float("nan"),
        "cost_usd": usage.get("cost_usd"), "runtime_min": sum(lat) / 60.0,
        "llm_calls_per_item": summ.get("llm_calls_per_item"),
        "ccme_updates": (summ.get("ccme") or summ.get("ccqs") or {}).get("updates"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, help="write one row per run")
    ap.add_argument("--table", action="store_true", help="print a model x stream table")
    ap.add_argument("--brief", action="store_true", help="one line per run")
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        if not (run / "steps.jsonl").is_file():
            continue
        r = score_run(run)
        rows.append(r)
        cost = "" if r["cost_usd"] is None else f"  ${r['cost_usd']:.2f}"
        print(f"{r['model']:<22} {r['stream']:<21} {r['variant'] or 'default':<12} "
              f"acc {r['accuracy']:5.1f}  ({r['correct']}/{r['n']}, {r['errored']} errored)"
              f"  {r['runtime_min']:.0f} min{cost}")
    if args.csv and rows:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if args.table and rows:
        order = list(STREAMS.values())
        models = sorted({r["model"] for r in rows})
        cell = {(r["model"], r["stream"]): r["accuracy"] for r in rows if not r["variant"]
                or r["variant"] == "default"}
        short = {"AIME_2024": "AIME24", "AIME_2025": "AIME25", "AIME_2020_2025": "AIME20-25",
                 "MATH": "MATH", "GPQA_Diamond": "GPQA", "MMLU_Pro_Engineering": "MMLU-Eng",
                 "MMLU_Pro_Physics": "MMLU-Phy", "HLE_Exact": "HLE", "MathVista": "MathVista",
                 "MMMU_Pro_Standard_4": "MMMU-S4", "MMMU_Pro_Standard_10": "MMMU-S10",
                 "MMMU_Pro_Vision": "MMMU-V"}
        print("\n" + "model".ljust(22) + "".join(short.get(s, s[:9]).rjust(10) for s in order)
              + "    avg")
        for m in models:
            vals = [cell.get((m, s)) for s in order]
            got = [v for v in vals if v is not None]
            print(m.ljust(22) + "".join(("%.1f" % v if v is not None else "-").rjust(10)
                                        for v in vals)
                  + ("%7.1f" % (sum(got) / len(got)) if len(got) == len(order) else "      -"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
