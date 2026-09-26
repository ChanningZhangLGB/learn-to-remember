"""Dataset loaders. One function per dataset, each returning `list[Item]`.

`pipeline.Item` is the whole interface. `lere/answers.py` owns normalization -- a loader
that "helpfully" strips a `\\boxed{}` or pads an AIME answer to three digits would put two
different normalizers in the run and the disagreement would surface as an accuracy delta
with no cause.

Why pyarrow rather than `datasets.load_from_disk`: the sets on disk are HuggingFace
`save_to_disk` directories, but `import datasets` fails in this environment (pandas cannot
find `GLIBCXX_3.4.29`). The arrow files themselves are the artifact and pyarrow reads them
with no pandas in the path, so the loader depends on less, not more.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .answers import normalize_mcq
from .pipeline import Item

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "dataset"

# AIME answers are integers 0-999 by construction. Anything else in the gold column is a
# defect in the source, not something to coerce.
_AIME_GOLD_RE = re.compile(r"^\d{1,3}$")


class DatasetError(RuntimeError):
    """A dataset could not be loaded as specified. Raised rather than worked around: a
    silently dropped item changes the denominator of every reported number."""


# --------------------------------------------------------------------------- arrow io

def read_hf_disk(path: str | Path) -> list:
    """Read a HuggingFace `save_to_disk` directory into a list of row dicts.

    Shards are read in filename order, which is the order `save_to_disk` wrote them, so
    row order is the dataset's own order. `cache-*.arrow` files are ignored: they are
    `map`/`filter` leftovers, not the dataset.
    """
    import pyarrow as pa  # noqa: PLC0415

    d = Path(path)
    if not d.is_dir():
        raise DatasetError("not a dataset directory: %s" % d)
    shards = sorted(d.glob("data-*.arrow"))
    if not shards:
        raise DatasetError("no data-*.arrow shards in %s" % d)

    rows: list = []
    for shard in shards:
        with pa.memory_map(str(shard), "rb") as src:
            rows.extend(pa.ipc.open_stream(src).read_all().to_pylist())
    return rows


# ------------------------------------------------------------------------------- AIME

def load_aime(path: str | Path, *, dataset: str | None = None, limit: int | None = None,
              offset: int = 0, on_bad_gold: str = "raise") -> list:
    """AIME 2020-2025 in any of the on-disk layouts, in the dataset's own order.

    Columns are `input` (question) and `target` (answer); `metadata` is present in some
    sets and absent in others, which is why the id is synthesized when it is missing
    rather than assumed.

    The worked `Solution` that AIME_2024's metadata carries is dropped here and never
    reaches `Item`. It is the answer with its derivation attached: putting it in `meta`
    would leave a leak one careless `str(item.meta)` away, and `lere/guard.py` protects
    the memory from the *question*, not from us.

    `offset`/`limit` slice before anything else, so "the first 10" means the first 10 rows
    of the file and stays reproducible. With `on_bad_gold="skip"` that ordering means the
    result can be SHORTER than `limit` -- deliberately: a stable window with a reported
    n beats a shifting window that always returns the number you asked for.
    """
    d = Path(path)
    name = dataset or d.name
    rows = read_hf_disk(d)

    if offset or limit is not None:
        end = None if limit is None else offset + int(limit)
        rows = rows[offset:end]
        if limit is not None and len(rows) < int(limit):
            raise DatasetError(
                "%s: asked for %d items from offset %d, only %d available"
                % (name, int(limit), offset, len(rows)))

    items: list = []
    for i, row in enumerate(rows):
        index = offset + i
        meta_col = row.get("metadata") or {}
        question = (row.get("input") or "").strip()
        gold = (row.get("target") or "").strip()

        if not question:
            raise DatasetError("%s row %d has an empty question" % (name, index))

        if not _AIME_GOLD_RE.match(gold) or not 0 <= int(gold) <= 999:
            # AIME_2020_2025 has one of these: `080 or 081 (both were accepted)`.
            if on_bad_gold == "skip":
                continue
            raise DatasetError(
                "%s row %d (%s): gold %r is not an AIME integer 0-999. Pass "
                "on_bad_gold='skip' to drop such rows, and report the changed n."
                % (name, index, meta_col.get("ID", "no-id"), gold))

        # AIME_2025 has no metadata column at all, so the row index is the only stable
        # identity it has. Zero-padded so ids sort in dataset order.
        item_id = str(meta_col.get("ID") or "%s-%04d" % (name, index))

        items.append(Item(
            id=item_id,
            question=question,
            answer_type="integer",
            gold=gold,
            image=None,
            dataset=name,
            meta={
                "source_index": index,
                "year": meta_col.get("Year"),
                "part": meta_col.get("Part"),
                # Recorded, never carried: see the docstring.
                "solution_withheld": bool((meta_col.get("Solution") or "").strip()),
            },
        ))
    return items


def load_aime_2025(limit: int | None = None, offset: int = 0) -> list:
    return load_aime(DATASET_DIR / "AIME_2025", dataset="AIME_2025",
                     limit=limit, offset=offset)


def load_aime_2024(limit: int | None = None, offset: int = 0) -> list:
    return load_aime(DATASET_DIR / "AIME_2024", dataset="AIME_2024",
                     limit=limit, offset=offset)


# ------------------------------------------------------------------------- GPQA

# Four, not the ten `Item` defaults to. `prompts/taxonomy.md` calls this out because it is
# the one dataset where the default is wrong, and a wrong `n_options` makes `normalize_mcq`
# accept letters that were never offered.
GPQA_N_OPTIONS = 4


def load_gpqa(path: str | Path, *, dataset: str | None = None,
              limit: int | None = None, offset: int = 0) -> list:
    """GPQA-Diamond. Columns are `input` (question with its options inline) and `target`
    (`"(C)"`).

    The options stay embedded in the question text rather than being parsed out: that is
    how the source stores them, Solver's prompt asks for the letter alone, and `answers.py`
    already accepts `(C)` and `C` interchangeably. Splitting them here would add a second
    parser with nothing to gain.

    No metadata column, so the row index is the identity, as with AIME_2025.
    """
    d = Path(path)
    name = dataset or d.name
    rows = read_hf_disk(d)

    if offset or limit is not None:
        end = None if limit is None else offset + int(limit)
        rows = rows[offset:end]
        if limit is not None and len(rows) < int(limit):
            raise DatasetError(
                "%s: asked for %d items from offset %d, only %d available"
                % (name, int(limit), offset, len(rows)))

    items: list = []
    for i, row in enumerate(rows):
        index = offset + i
        question = (row.get("input") or "").strip()
        gold = (row.get("target") or "").strip()
        if not question:
            raise DatasetError("%s row %d has an empty question" % (name, index))
        if normalize_mcq(gold, GPQA_N_OPTIONS) is None:
            raise DatasetError(
                "%s row %d: gold %r is not one of the %d option letters"
                % (name, index, gold, GPQA_N_OPTIONS))
        items.append(Item(
            id="%s-%04d" % (name, index),
            question=question,
            answer_type="mcq_letter",
            gold=gold,
            n_options=GPQA_N_OPTIONS,
            image=None,
            dataset=name,
            meta={"source_index": index},
        ))
    return items


def load_gpqa_diamond(limit: int | None = None, offset: int = 0) -> list:
    return load_gpqa(DATASET_DIR / "GPQA_Diamond", dataset="GPQA_Diamond",
                     limit=limit, offset=offset)


# -------------------------------------------------------------------- MMLU-Pro

# Markers are line-initial inside the Options block. Restricting to that block matters:
# `(A)` also appears in ordinary question prose, and counting those would inflate the
# option count.
_MMLU_OPTION_RE = re.compile(r"^\(([A-Z])\)", re.MULTILINE)


def mmlu_n_options(question: str, default: int = 10) -> int:
    """How many choices this item actually offers.

    `prompts/taxonomy.md` says MMLU-Pro is 10 options, and most items are, but 89 of 969
    Engineering items and 57 of 1299 Physics items offer fewer -- four, in the common case
    ("Stack is also known as ... (A) FIFO (B) Flash (C) LIFO (D) LILO"). A flat 10 would
    let `normalize_mcq` accept a letter that was never on offer, scoring a hallucinated
    option as a legitimate answer, so the count is taken per item.
    """
    head, sep, tail = question.partition("Options:")
    body = tail if sep else question
    letters = _MMLU_OPTION_RE.findall(body)
    return (ord(max(letters)) - ord("A") + 1) if letters else default


def load_mmlu_pro(path: str | Path, *, dataset: str | None = None,
                  limit: int | None = None, offset: int = 0) -> list:
    """One MMLU-Pro category. Columns are `input` and `target` (`"(C)"`), as for GPQA.

    Options stay inline in the question, which is how the source stores them and what
    `answers.py` already parses.
    """
    d = Path(path)
    name = dataset or d.name
    rows = read_hf_disk(d)

    if offset or limit is not None:
        end = None if limit is None else offset + int(limit)
        rows = rows[offset:end]
        if limit is not None and len(rows) < int(limit):
            raise DatasetError(
                "%s: asked for %d items from offset %d, only %d available"
                % (name, int(limit), offset, len(rows)))

    items: list = []
    for i, row in enumerate(rows):
        index = offset + i
        question = (row.get("input") or "").strip()
        gold = (row.get("target") or "").strip()
        if not question:
            raise DatasetError("%s row %d has an empty question" % (name, index))
        n_opt = mmlu_n_options(question)
        letter = normalize_mcq(gold, n_opt)
        if letter is None:
            raise DatasetError(
                "%s row %d: gold %r is not one of the %d options this item offers"
                % (name, index, gold, n_opt))
        items.append(Item(
            id="%s-%04d" % (name, index),
            question=question,
            answer_type="mcq_letter",
            gold=gold,
            n_options=n_opt,
            image=None,
            dataset=name,
            meta={"source_index": index, "n_options": n_opt},
        ))
    return items


def load_mmlu_pro_engineering(limit: int | None = None, offset: int = 0) -> list:
    return load_mmlu_pro(DATASET_DIR / "MMLU_Pro_Engineering",
                         dataset="MMLU_Pro_Engineering", limit=limit, offset=offset)


def load_mmlu_pro_physics(limit: int | None = None, offset: int = 0) -> list:
    return load_mmlu_pro(DATASET_DIR / "MMLU_Pro_Physics",
                         dataset="MMLU_Pro_Physics", limit=limit, offset=offset)


# -------------------------------------------------------------------- MathVista

def _letter(i: int) -> str:
    return chr(ord("A") + i)


def load_mathvista(path: str | Path, *, dataset: str | None = None,
                   limit: int | None = None, offset: int = 0) -> list:
    """MathVista testmini subset. Multimodal: every item carries an image.

    Three things this subset does that the format table in `prompts/taxonomy.md` does not
    lead you to expect, all verified against the 250 rows rather than assumed:

    * Every row is `multi_choice` with `answer_type: "text"`. There are no integer or
      float items here, so `answer_type` is `mcq_letter` for all of them and the float
      tolerance path never runs.
    * `answer` is the choice **text** ("97", "Yes", "Common water flea"), not a letter.
      Gold is the letter at that choice's index; the mapping was checked to line up with
      the lettered options already rendered in `query` for all 250 rows.
    * `n_options` varies from 2 to 7, so letters run A..G. It is per item, never a
      constant.

    `query` is used as the question because it is the dataset's own canonical rendering,
    with the lettered choices inline -- the same reasoning as the GPQA loader.
    """
    d = Path(path)
    name = dataset or d.name
    rows = read_hf_disk(d)

    if offset or limit is not None:
        end = None if limit is None else offset + int(limit)
        rows = rows[offset:end]
        if limit is not None and len(rows) < int(limit):
            raise DatasetError(
                "%s: asked for %d items from offset %d, only %d available"
                % (name, int(limit), offset, len(rows)))

    items: list = []
    for i, row in enumerate(rows):
        index = offset + i
        question = (row.get("query") or "").strip()
        choices = list(row.get("choices") or [])
        answer = row.get("answer")
        if not question:
            raise DatasetError("%s row %d has an empty query" % (name, index))
        if not choices:
            raise DatasetError("%s row %d has no choices" % (name, index))
        if answer not in choices:
            raise DatasetError(
                "%s row %d: answer %r is not among its choices" % (name, index, answer))

        img = row.get("decoded_image") or {}
        blob = img.get("bytes") if isinstance(img, dict) else None
        if not blob:
            raise DatasetError("%s row %d has no image bytes" % (name, index))

        meta = row.get("metadata") or {}
        items.append(Item(
            id="%s-%s" % (name, row.get("pid") or index),
            question=question,
            answer_type="mcq_letter",
            gold=_letter(choices.index(answer)),
            n_options=len(choices),
            image=blob,
            dataset=name,
            meta={"source_index": index, "pid": row.get("pid"),
                  "answer_text": answer, "n_choices": len(choices),
                  "category": meta.get("category"), "task": meta.get("task"),
                  "source": meta.get("source")},
        ))
    return items


def load_mathvista_testmini_250(limit: int | None = None, offset: int = 0) -> list:
    return load_mathvista(DATASET_DIR / "MathVista_testmini_250",
                          dataset="MathVista_testmini_250", limit=limit, offset=offset)


# ------------------------------------------------------------------- frozen subsets

def dump_jsonl(items, path: str | Path) -> Path:
    """Freeze a subset to disk so the exact items a run used are recoverable.

    A run identified only as "the first 10 of AIME_2025" stops being reproducible the
    moment the directory is re-exported. The file is the record.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    img_dir = p.with_suffix("")
    img_dir = img_dir.with_name(img_dir.name + "_images")
    with p.open("w", encoding="utf-8") as fh:
        for it in items:
            row = {
                "id": it.id, "question": it.question, "answer_type": it.answer_type,
                "gold": it.gold, "n_options": it.n_options, "dataset": it.dataset,
                "meta": it.meta,
            }
            if it.image is not None:
                # Images go to a sidecar directory rather than being base64'd inline: a
                # 1.5 MB PNG per row would make the subset file unreadable and unable to
                # be diffed, and the point of freezing a subset is that a human can see
                # exactly which items a run used.
                img_dir.mkdir(parents=True, exist_ok=True)
                fname = "%s%s" % (_safe_name(it.id), _image_ext(it.image))
                (img_dir / fname).write_bytes(it.image)
                row["image_file"] = fname
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def load_jsonl(path: str | Path) -> list:
    p = Path(path)
    img_dir = p.with_suffix("")
    img_dir = img_dir.with_name(img_dir.name + "_images")
    out: list = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        blob = None
        if d.get("image_file"):
            f = img_dir / d["image_file"]
            if not f.is_file():
                raise DatasetError(
                    "%s references image %s but %s is missing; the subset is incomplete "
                    "and the item would be scored on text the model never saw"
                    % (p, d["image_file"], f))
            blob = f.read_bytes()
        out.append(Item(
            id=d["id"], question=d["question"], answer_type=d.get("answer_type", "free"),
            gold=d.get("gold"), n_options=int(d.get("n_options", 10)), image=blob,
            dataset=d.get("dataset", "unknown"), meta=d.get("meta") or {},
        ))
    return out


def _safe_name(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", item_id)


_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", ".png"), (b"\xff\xd8\xff", ".jpg"),
                (b"GIF87a", ".gif"), (b"GIF89a", ".gif"))


def _image_ext(blob: bytes) -> str:
    for magic, ext in _IMAGE_MAGIC:
        if blob.startswith(magic):
            return ext
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    return ".bin"
