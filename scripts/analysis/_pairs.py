"""Rebuild the CCME training triples of a run and the saved head versions.

Positives are entries the Curator labelled HELPFUL (`used_positive`); negatives are HARMFUL
or IRRELEVANT (`used_negative`, `unused_irrelevant`); REDUNDANT entries are excluded, as in
training (Eq. 10). The anchor is the Planner's key as the run embedded it.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lere.embed import SentenceTransformerEncoder, l2_normalize  # noqa: E402,F401

NEG = ("used_negative", "unused_irrelevant")
def head_files(run):
    d = os.path.join(run, "heads")
    ups = sorted((f for f in os.listdir(d) if f.startswith("update_step")), key=lambda f: int(f[len("update_step"):-4]))
    out = [("start", -1, os.path.join(d, "start.npz"))]
    out += [("upd%02d" % (i + 1), int(f[len("update_step"):-4]), os.path.join(d, f)) for i, f in enumerate(ups)]
    return out

def collect_pairs(run):
    ret = {r["step"]: r for r in (json.loads(l) for l in open(os.path.join(run, "retrieval.jsonl")))}
    pairs = []
    for l in open(os.path.join(run, "steps.jsonl")):
        s = json.loads(l); r = ret.get(s["step"])
        if not r or not r.get("candidates"): continue
        view = {c["entry_id"]: c["skill_view"] for c in r["candidates"]}
        att = s["curation"]["attribution"]
        pos = [view[i] for i in att.get("used_positive", []) if i in view]
        neg = [view[i] for b in NEG for i in att.get(b, []) if i in view]
        if pos and neg:
            pairs.append(dict(step=s["step"], q=r["query_view"], pos=pos, neg=neg))
    return pairs


def per_anchor(pairs, base, W):
    """(anchors, 2): mean cosine of each anchor to its helpful and to its harmful entries."""
    ep, es = W["ep"], W["es"]
    out = []
    for p in pairs:
        q = l2_normalize(base[p["q"]][None] @ ep.T)[0]
        sp = l2_normalize(np.array([base[t] for t in p["pos"]]) @ es.T) @ q
        sn = l2_normalize(np.array([base[t] for t in p["neg"]]) @ es.T) @ q
        out.append((float(sp.mean()), float(sn.mean())))
    return np.array(out)


def base_vectors(pairs, enc):
    texts = sorted({p["q"] for p in pairs} | {t for p in pairs for t in p["pos"] + p["neg"]})
    return dict(zip(texts, enc.encode(texts)))
