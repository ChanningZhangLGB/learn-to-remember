#!/usr/bin/env python
"""Sensitivity to the number of retrieved entries K (Appendix C.3.1, Table 8).

    python scripts/analysis/k_sweep_significance.py runs/*/*

Groups runs by backbone and K (read from each run's config.snapshot.yaml), scores every
item with the paper's scorer, and compares settings on paired per-query correctness,
since every run answers the same queries in the same order:

  mean accuracy     mean over the streams of a backbone
  Cochran's Q       any difference across K
  McNemar (exact)   K = 3 against each alternative, Holm-corrected within a backbone

Only streams present for every K of a backbone enter the test. If a (backbone, K, stream)
has several runs, the first one given is used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from score_runs import subset_for  # noqa: E402
from scoring import score_item  # noqa: E402


def correctness(run: Path) -> tuple:
    sub = subset_for(run)
    items = [json.loads(l) for l in sub.open(encoding="utf-8")]
    got = {}
    for line in (run / "steps.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        got[r["item_id"]] = (r.get("solver") or {}).get("answer")
    vec = np.array([int(score_item(got[it["id"]], it["gold"], it["question"], it["answer_type"],
                                   int(it.get("n_options") or 10))[1])
                    if it["id"] in got else 0 for it in items])   # errored item = incorrect
    return sub.stem, vec


def holm(pvals: dict) -> dict:
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        prev = min(1.0, max(prev, (len(pvals) - i) * p))
        out[k] = prev
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    data = {}
    for run in args.runs:
        snap = run / "config.snapshot.yaml"
        if not (run / "steps.jsonl").is_file() or not snap.is_file():
            continue
        cfg = yaml.safe_load(snap.read_text())
        if not cfg["ccqs"]["enabled"] or not cfg["tools"]["enabled"] or \
                (cfg.get("run") or {}).get("no_planner"):
            continue                                   # ablation arms are not part of the sweep
        stream, vec = correctness(run)
        data.setdefault(cfg["llm"]["model"], {}).setdefault(
            int(cfg["retrieval"]["top_k"]), {}).setdefault(stream, vec)
    for model, byk in sorted(data.items()):
        ks = sorted(byk)
        if 3 not in ks or len(ks) < 2:
            print(f"{model}: need K=3 and at least one other K")
            continue
        streams = sorted(set.intersection(*(set(byk[k]) for k in ks)))
        means = {k: 100 * np.mean([byk[k][s].mean() for s in streams]) for k in ks}
        X = np.stack([np.concatenate([byk[k][s] for s in streams]) for k in ks], 1)
        c, r, n = X.sum(0), X.sum(1), X.sum()
        q = (len(ks) - 1) * (len(ks) * np.sum(c ** 2) - n ** 2) / (len(ks) * n - np.sum(r ** 2))
        p_q = stats.chi2.sf(q, len(ks) - 1)
        i3 = ks.index(3)
        raw = {}
        for j, k in enumerate(ks):
            if k == 3:
                continue
            only3 = int(((X[:, i3] == 1) & (X[:, j] == 0)).sum())
            onlyk = int(((X[:, i3] == 0) & (X[:, j] == 1)).sum())
            raw[k] = stats.binomtest(min(only3, onlyk), only3 + onlyk, 0.5).pvalue \
                if only3 + onlyk else 1.0
        adj = holm(raw)
        print(f"{model}  ({len(streams)} streams, {X.shape[0]} paired queries)")
        print("  mean accuracy: " + "  ".join(f"K={k}: {means[k]:.1f}" for k in ks))
        print(f"  Cochran's Q = {q:.2f}, p = {p_q:.3g}")
        print("  K=3 vs (Holm p): " + "  ".join(f"K={k}: {adj[k]:.2g}" for k in sorted(adj)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
