#!/usr/bin/env python
"""Does CCME reshape the retrieval geometry? (Appendix C.2.1, Table 6, Figure 8)

    python scripts/analysis/ccme_geometry.py runs/*/*            # Table 6
    python scripts/analysis/ccme_geometry.py runs/*/* --curves curves.json   # + Figure 8 data

For every run whose CCME heads were updated at least once, the training triples the run
produced (anchor = Planner key; positives = HELPFUL entries; negatives = HARMFUL or
IRRELEVANT entries; REDUNDANT excluded, as in training) are embedded once with the frozen
encoder and scored under the identity heads and under the run's final heads. Per anchor
we take the mean cosine to its helpful and to its harmful/irrelevant entries; the table
reports the mean change of each and of their margin, with a two-sided Wilcoxon
signed-rank test on the per-anchor margin change. Pooled rows combine the anchors of all
runs of a backbone.

--curves additionally writes, per backbone, the pooled means (95% bootstrap intervals) at
25 / 50 / 75 / 100% of each run's head versions, and the held-out margin: triples from
later in the stream scored under the current heads vs. under the identity heads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pairs import (SentenceTransformerEncoder, base_vectors, collect_pairs,  # noqa: E402
                    head_files, per_anchor)


def run_label(run):
    rep = os.path.join(run, "report.json")
    snap = os.path.join(run, "config.snapshot.yaml")
    model = None
    if os.path.isfile(rep):
        model = json.load(open(rep)).get("model")
    if not model and os.path.isfile(snap):
        import yaml
        model = yaml.safe_load(open(snap))["llm"]["model"]
    return model or os.path.basename(os.path.dirname(run)), os.path.basename(run.rstrip("/"))


def wilcoxon_p(x):
    try:
        return float(stats.wilcoxon(x).pvalue) if len(x) >= 3 and np.any(x != 0) else float("nan")
    except ValueError:
        return float("nan")


def boot(x, rng, n=1000):
    if len(x) < 2:
        return (float(x.mean()), float(x.mean()))
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--csv", help="write the table as CSV")
    ap.add_argument("--curves", help="write Figure 8 data as JSON")
    args = ap.parse_args()
    enc = SentenceTransformerEncoder("all-MiniLM-L6-v2")
    rng = np.random.default_rng(0)
    rows, pooled, curves = [], {}, {}
    for run in args.runs:
        if not os.path.isdir(os.path.join(run, "heads")):
            continue
        heads = head_files(run)
        pairs = collect_pairs(run)
        if len(heads) < 2 or not pairs:        # CCME never updated: nothing to measure
            continue
        model, name = run_label(run)
        base = base_vectors(pairs, enc)
        W = [dict(np.load(f)) for _, _, f in heads]
        s0, s1 = per_anchor(pairs, base, W[0]), per_anchor(pairs, base, W[-1])
        dpos, dneg = s1[:, 0] - s0[:, 0], s1[:, 1] - s0[:, 1]
        dm = dpos - dneg
        pooled.setdefault(model, []).append((dpos, dneg, dm))
        rows.append(dict(model=model, run=name, anchors=len(pairs), d_helpful=dpos.mean(),
                         d_harmful=dneg.mean(), d_margin=dm.mean(), p_wilcoxon=wilcoxon_p(dm),
                         anchors_gaining=float((dm > 0).mean())))
        if args.curves:
            c = curves.setdefault(model, {"train": {}, "heldout": {}})
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                h = int(round(frac * (len(heads) - 1)))
                r = per_anchor(pairs, base, W[h])
                c["train"].setdefault(str(frac), []).append(r.tolist())
                fut = [p for p in pairs if p["step"] > heads[h][1]] if h > 0 else []
                if len(fut) >= 3:
                    rh, r0 = per_anchor(fut, base, W[h]), per_anchor(fut, base, W[0])
                    c["heldout"].setdefault(str(frac), []).append(
                        ((rh[:, 0] - rh[:, 1]).tolist(), (r0[:, 0] - r0[:, 1]).tolist()))
    for model, parts in pooled.items():
        dpos, dneg, dm = (np.concatenate([p[i] for p in parts]) for i in range(3))
        rows.append(dict(model=model, run="POOLED", anchors=len(dm), d_helpful=dpos.mean(),
                         d_harmful=dneg.mean(), d_margin=dm.mean(), p_wilcoxon=wilcoxon_p(dm),
                         anchors_gaining=float((dm > 0).mean())))
    rows.sort(key=lambda r: (r["model"], r["run"] == "POOLED", r["run"]))
    print("%-22s %-34s %7s %9s %9s %9s %9s %8s" % ("model", "run", "anchors", "d_help",
                                                    "d_harm", "d_margin", "p", "gain%"))
    for r in rows:
        print("%-22s %-34s %7d %+9.2f %+9.2f %+9.2f %9.2g %7.0f%%" % (
            r["model"], r["run"], r["anchors"], r["d_helpful"], r["d_harmful"], r["d_margin"],
            r["p_wilcoxon"], 100 * r["anchors_gaining"]))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if args.curves:
        out = {}
        for model, c in curves.items():
            tr = {}
            for frac, runs in c["train"].items():
                a = np.concatenate([np.array(r) for r in runs])
                tr[frac] = dict(helpful=a[:, 0].mean(), helpful_ci=boot(a[:, 0], rng),
                                harmful=a[:, 1].mean(), harmful_ci=boot(a[:, 1], rng))
            ho = {}
            for frac, runs in c["heldout"].items():
                cur = np.concatenate([np.array(r[0]) for r in runs])
                ini = np.concatenate([np.array(r[1]) for r in runs])
                ho[frac] = dict(margin_current=cur.mean(), margin_identity=ini.mean(), n=len(cur))
            out[model] = dict(train=tr, heldout=ho)
        json.dump(out, open(args.curves, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
