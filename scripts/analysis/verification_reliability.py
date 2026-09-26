#!/usr/bin/env python
"""How reliable is LeRe's label-free verdict? (Appendix C.2.2, Table 7, Proposition A.9)

    python scripts/analysis/verification_reliability.py runs/*/*

Compares the verification signal each step learned from (steps.jsonl `signal_correct`,
formed without reference answers) with gold correctness (`gold_correct`, used here only
to score it). Steps with no verdict are excluded and counted. Per backbone, pooled over
its runs:

  agreement   P(verdict == gold)
  precision   P(correct | verdict "correct")
  gold acc.   base rate of correct answers (over all recorded steps)
  recall      zeta_1 = P(verdict "correct" | correct)
  FPR         zeta_0 = P(verdict "correct" | wrong)

zeta_1 > zeta_0 is the condition under which reliability stays ordered by true
helpfulness (Proposition A.9). Also reported: the chi-square test of association per run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from scipy.stats import chi2_contingency


def counts(run):
    tp = fp = tn = fn = none = n = gold_all = 0
    for line in open(os.path.join(run, "steps.jsonl")):
        s = json.loads(line)
        n += 1
        sig, gold = s["signal_correct"], bool(s["gold_correct"])
        gold_all += gold
        if sig is None:
            none += 1
        elif sig and gold:
            tp += 1
        elif sig:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, none=none, steps=n, gold_all=gold_all)


def metrics(c):
    n = c["tp"] + c["fp"] + c["tn"] + c["fn"]
    div = lambda a, b: a / b if b else float("nan")
    return dict(N=n, agreement=div(c["tp"] + c["tn"], n), precision=div(c["tp"], c["tp"] + c["fp"]),
                gold_acc=div(c["gold_all"], c["steps"]), recall=div(c["tp"], c["tp"] + c["fn"]),
                fpr=div(c["fp"], c["fp"] + c["tn"]))


def model_of(run):
    rep = os.path.join(run, "report.json")
    if os.path.isfile(rep) and json.load(open(rep)).get("model"):
        return json.load(open(rep))["model"]
    snap = os.path.join(run, "config.snapshot.yaml")
    if os.path.isfile(snap):
        import yaml
        return yaml.safe_load(open(snap))["llm"]["model"]
    return os.path.basename(os.path.dirname(run))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    by_model, significant = {}, {}
    for run in args.runs:
        if not os.path.isfile(os.path.join(run, "steps.jsonl")):
            continue
        c = counts(run)
        m = model_of(run)
        agg = by_model.setdefault(m, dict(tp=0, fp=0, tn=0, fn=0, none=0, steps=0, gold_all=0))
        for k in agg:
            agg[k] += c[k]
        table = [[c["tp"], c["fp"]], [c["fn"], c["tn"]]]
        if min(sum(r) for r in table) > 0 and min(sum(col) for col in zip(*table)) > 0:
            p = chi2_contingency(table, correction=False)[1]
            s = significant.setdefault(m, [0, 0])
            s[0] += p < 0.05
            s[1] += 1
    print("%-22s %6s %9s %9s %9s %9s %9s %8s %s" % ("backbone", "N", "agree", "precision",
          "gold_acc", "zeta1", "zeta0", "z1-z0", "runs with chi2 p<0.05"))
    for m, c in sorted(by_model.items()):
        r = metrics(c)
        s = significant.get(m, [0, 0])
        print("%-22s %6d %9.3f %9.3f %9.3f %9.3f %9.3f %+8.3f %d/%d" % (
            m, r["N"], r["agreement"], r["precision"], r["gold_acc"], r["recall"], r["fpr"],
            r["recall"] - r["fpr"], s[0], s[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
