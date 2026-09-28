#!/usr/bin/env python
"""Rebuild the GPQA-Diamond and HLE-Exact streams in data/subsets/ from their sources.

Both benchmarks ask that their questions not be republished in plain text (GPQA ships
password-protected with a canary string; HLE is gated and asks not to re-upload it), so the
repository keeps only <stream>.ids.json: item ids, order, metadata and a SHA-256 of each
question and gold answer. This script fetches the text, writes <stream>.jsonl exactly as
used in the paper and checks it byte for byte against the recorded file hash.

    python scripts/data/fetch_streams.py              # both streams
    python scripts/data/fetch_streams.py --only hle   # one of them

HLE is a gated dataset: accept its terms on Hugging Face and run `huggingface-cli login`
first. GPQA-Diamond is read from the Dynamic Cheatsheet release the paper used, pinned to
one commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBSETS = REPO / "data" / "subsets"

DC_COMMIT = "1cf258c936edc28ef045202621bf9463f70a1988"
DC_GPQA_URL = ("https://raw.githubusercontent.com/suzgunmirac/dynamic-cheatsheet/%s/"
               "data/GPQA_Diamond/data-00000-of-00001.arrow" % DC_COMMIT)


def gpqa_rows() -> dict:
    """meta.source_index -> (question, gold), from the Dynamic Cheatsheet arrow file."""
    import pyarrow as pa

    blob = urllib.request.urlopen(DC_GPQA_URL).read()
    rows = pa.ipc.open_stream(pa.BufferReader(blob)).read_all().to_pylist()
    return {i: (r["input"].strip(), r["target"].strip()) for i, r in enumerate(rows)}


def hle_rows() -> dict:
    """meta.hle_id -> (question, gold), from the gated cais/hle test split."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("cais/hle", "data/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path, columns=["id", "question", "answer"]).to_pylist()
    return {r["id"]: (r["question"], r["answer"].strip()) for r in table}


# stream key -> (subset stem, row loader, meta field that indexes the loader's rows)
SOURCES = {
    "gpqa": ("gpqa_diamond_dcorder198", gpqa_rows, "source_index"),
    "hle": ("hle_dcorder250", hle_rows, "hle_id"),
}


def rebuild(stem: str, load, key: str) -> bool:
    spec = json.loads((SUBSETS / f"{stem}.ids.json").read_text(encoding="utf-8"))
    rows = load()
    lines, bad = [], []
    for item in spec["items"]:
        question, gold = rows[item["meta"][key]]
        digest = hashlib.sha256((question + "\x00" + gold).encode()).hexdigest()
        if digest != item["sha256"]:
            bad.append(item["id"])
        rest = {k: v for k, v in item.items() if k not in ("id", "answer_type", "sha256")}
        record = {"id": item["id"], "question": question,
                  "answer_type": item["answer_type"], "gold": gold}
        record.update(rest)
        lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    data = "".join(lines).encode("utf-8")
    ok = not bad and hashlib.sha256(data).hexdigest() == spec["file_sha256"]
    if ok:
        (SUBSETS / f"{stem}.jsonl").write_bytes(data)
    print(f"  {stem}: {len(lines) - len(bad)}/{len(lines)} items match"
          + ("" if ok else " -- NOT written (source differs from the paper's copy)"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    args = ap.parse_args()
    ok = True
    for name in args.only:
        try:
            ok &= rebuild(*SOURCES[name])
        except Exception as exc:  # gated repo, no network, ...
            ok = False
            print(f"  ERROR {name}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            if "Gated" in type(exc).__name__:
                print("  -> accept the dataset terms on huggingface.co and run "
                      "`huggingface-cli login`, then re-run with --only " + name)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
