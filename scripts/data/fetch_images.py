#!/usr/bin/env python
"""Download the images referenced by the frozen multimodal subsets in data/subsets/.

The subset files (*.jsonl) fix the items, their order and their gold answers; images are
not redistributed here and are written next to each subset as <subset>_images/<image_file>,
which is where `lere.datasets.load_jsonl` looks for them. Bytes are written exactly as
stored in the Hugging Face release, so the files are identical to the ones used in the
paper (checked with --verify against a SHA-256 manifest).

    python scripts/data/fetch_images.py                       # all four image subsets
    python scripts/data/fetch_images.py --only mathvista hle  # a subset of them
    python scripts/data/fetch_images.py --verify              # check hashes only

HLE is a gated dataset: accept its terms on Hugging Face and run `huggingface-cli login`
first. Downloads total about 3.8 GB of parquet shards (cached by huggingface_hub).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBSETS = REPO / "data" / "subsets"
HASHES = SUBSETS / "image_sha256.json"

# subset stem -> (HF repo, parquet directory prefix, id column, image columns)
SOURCES = {
    "mathvista": [("mathvista_testmini_dcorder250", "AI4Math/MathVista", "data/testmini-",
                   "pid", ["decoded_image"])],
    "mmmu": [
        ("mmmu_pro_standard_4_dcorder250", "MMMU/MMMU_Pro", "standard (4 options)/", "id",
         ["image_%d" % i for i in range(1, 8)]),
        ("mmmu_pro_standard_10_dcorder250", "MMMU/MMMU_Pro", "standard (10 options)/", "id",
         ["image_%d" % i for i in range(1, 8)]),
        ("mmmu_pro_vision_dcorder250", "MMMU/MMMU_Pro", "vision/", "id", ["image"]),
    ],
    "hle": [("hle_dcorder250", "cais/hle", "data/test-", "id", ["image"])],
}


def source_key(stem: str, meta: dict) -> str:
    """The id the item carries in the upstream release."""
    if stem.startswith("mathvista"):
        return str(meta["pid"])
    if stem.startswith("mmmu"):
        return str(meta["mmmu_id"])
    return str(meta["hle_id"])


def to_bytes(value):
    """Image cell -> raw bytes (parquet struct, raw bytes, or an HLE data URI)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("bytes")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, str) and value.startswith("data:image/"):
        return base64.b64decode(value.split(",", 1)[1])
    return None


def fetch(stem: str, repo: str, prefix: str, id_col: str, img_cols: list) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    rows = [json.loads(l) for l in (SUBSETS / f"{stem}.jsonl").open(encoding="utf-8")]
    wanted = {source_key(stem, r["meta"]): r["image_file"] for r in rows if r.get("image_file")}
    out_dir = SUBSETS / f"{stem}_images"
    out_dir.mkdir(exist_ok=True)
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                    if f.startswith(prefix) and f.endswith(".parquet"))
    found = {}
    for shard in shards:
        path = hf_hub_download(repo, shard, repo_type="dataset")
        table = pq.read_table(path, columns=[id_col] + [c for c in img_cols
                                                       if c in pq.read_schema(path).names])
        ids = [str(x) for x in table.column(id_col).to_pylist()]
        cols = {c: table.column(c).to_pylist() for c in table.column_names if c != id_col}
        for i, key in enumerate(ids):
            if key in wanted and key not in found:
                blob = next((b for b in (to_bytes(cols[c][i]) for c in img_cols if c in cols)
                             if b), None)
                if blob:
                    (out_dir / wanted[key]).write_bytes(blob)
                    found[key] = wanted[key]
        print(f"  {stem}: {shard} -> {len(found)}/{len(wanted)}", flush=True)
        if len(found) == len(wanted):
            break
    missing = sorted(set(wanted) - set(found))
    if missing:
        print(f"  WARNING {stem}: {len(missing)} images not found, e.g. {missing[:3]}")
    return found


def verify(stems) -> bool:
    ref = json.loads(HASHES.read_text())
    ok = True
    for stem in stems:
        bad = [name for name, h in ref[stem].items()
               if not (SUBSETS / f"{stem}_images" / name).is_file()
               or hashlib.sha256((SUBSETS / f"{stem}_images" / name).read_bytes()).hexdigest() != h]
        print(f"  {stem}: {len(ref[stem]) - len(bad)}/{len(ref[stem])} images match")
        ok &= not bad
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    ap.add_argument("--verify", action="store_true", help="only check hashes")
    args = ap.parse_args()
    jobs = [j for k in args.only for j in SOURCES[k]]
    if not args.verify:
        for job in jobs:
            try:
                fetch(*job)
            except Exception as exc:  # gated repo, no network, ...
                print(f"  ERROR {job[0]} from {job[1]}: {type(exc).__name__}: "
                      f"{str(exc).splitlines()[0]}")
                if "Gated" in type(exc).__name__:
                    print("  -> accept the dataset terms on huggingface.co and run "
                          "`huggingface-cli login`, then re-run with --only "
                          + next(k for k, v in SOURCES.items() if job in v))
    return 0 if verify([j[0] for j in jobs]) else 1


if __name__ == "__main__":
    sys.exit(main())
