# Evaluation streams

`subsets/` holds the twelve streams of the paper (Section 4.1, Appendix B.1) as frozen JSONL
files: one item per line, in the exact order every method processed them. Each line has
`id`, `question`, `answer_type` (`integer`, `math`, `mcq_letter`, `free`), `gold`,
`n_options`, `dataset`, `meta` (source ids and indices) and, for image items, `image_file`.
`configs/streams.yaml` maps the stream names used by the scripts to these files.

GPQA-Diamond and HLE-Exact are the exception: both benchmarks ask that their questions not be
republished in plain text, so only `<stream>.ids.json` is committed (ids, order, metadata and
a SHA-256 of each question and gold answer). Rebuild the two JSONL files before running them:

```bash
python scripts/data/fetch_streams.py   # byte-identical to the paper's copies, hash-checked
```

| Stream | File | Items | Answer type |
| --- | --- | ---: | --- |
| AIME 2024 | `aime_2024_dcorder30.jsonl` | 30 | integer |
| AIME 2025 | `aime_2025_dcorder30.jsonl` | 30 | integer |
| AIME 2020-2025 | `aime_2020_2025_dcorder162.jsonl` | 162 | integer |
| MATH | `math500_dcorder250.jsonl` | 250 | LaTeX |
| GPQA-Diamond | `gpqa_diamond_dcorder198.jsonl` | 198 | 4 options |
| MMLU-Pro Engineering | `mmlu_pro_engineering_dcorder250.jsonl` | 250 | up to 10 options |
| MMLU-Pro Physics | `mmlu_pro_physics_dcorder250.jsonl` | 250 | up to 10 options |
| HLE-Exact | `hle_dcorder250.jsonl` | 250 | letter or number (34 with an image) |
| MathVista | `mathvista_testmini_dcorder250.jsonl` | 250 | options, image |
| MMMU-Pro Standard-4 | `mmmu_pro_standard_4_dcorder250.jsonl` | 250 | mostly 4 options, image |
| MMMU-Pro Standard-10 | `mmmu_pro_standard_10_dcorder250.jsonl` | 250 | up to 10 options, image |
| MMMU-Pro Vision | `mmmu_pro_vision_dcorder250.jsonl` | 250 | question and options in the image |

## How the streams were drawn

Every stream is a fixed random permutation of its source pool with seed 10
(`np.random.default_rng(10).permutation`, equivalent to `datasets.shuffle(seed=10)`),
following Dynamic Cheatsheet; streams larger than 250 items keep the first 250.

* **AIME 2024 / 2025**: all 30 problems of each year. **AIME 2020-2025**: 162 problems pooled
  from these years; one problem whose official answer accepts two values was dropped
  (`aime_2020_2025_dcorder162.manifest.json`).
* **MATH**: 250 items of MATH-500, free-form LaTeX answers.
* **GPQA-Diamond**: all 198 questions.
* **MMLU-Pro Engineering / Physics**: 250 items of each discipline.
* **HLE-Exact**: the Humanity's Last Exam questions whose reference answer is an option letter
  or a bare number (so every item is scored by exact match, with no LLM judge), balanced over
  the eight HLE categories (31 or 32 each); `hle_dcorder250.manifest.json` has the details.
* **MathVista**: 250 multiple-choice problems from the testmini split.
* **MMMU-Pro**: one draw of 250 single-image questions, evaluated under all three official
  settings. The three files contain the same questions in the same order, so their differences
  isolate the answer format.

## Images

Images are not redistributed. Fetch them from the Hugging Face releases (about 3.8 GB of
parquet shards, cached by `huggingface_hub`):

```bash
python scripts/data/fetch_images.py            # MathVista, MMMU-Pro, HLE
python scripts/data/fetch_images.py --verify   # SHA-256 check against image_sha256.json
```

The bytes are written unchanged to `subsets/<stream>_images/`, and the check confirms they
are identical to the images used in the paper. HLE (`cais/hle`) is gated: accept its terms on
Hugging Face and run `huggingface-cli login` first. Run `fetch_streams.py` before
`fetch_images.py`, which reads the HLE stream file. The other text-only streams need no download.

Please follow the license of each source dataset: AIME (via Hugging Face `HuggingFaceH4/aime_2024`,
`MathArena/aime_2025`), MATH, GPQA, MMLU-Pro, Humanity's Last Exam, MathVista and MMMU-Pro.
