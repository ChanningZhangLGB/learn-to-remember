# Datasets

Text-only datasets are bundled in this repository. The three multimodal
datasets (MMMU-Pro, MathVista) are **not** included due to file-size
limits — they are publicly available and can be downloaded with a few
lines of code.

## Bundled (in this repo)

| Directory                       | # queries | Modality |
|---------------------------------|-----------|----------|
| `AIME_2024/`                    |        30 | text     |
| `AIME_2025/`                    |        30 | text     |
| `GPQA_Diamond/`                 |       198 | text     |
| `MMLU_Pro_Engineering_250/`     |       250 | text     |
| `MMLU_Pro_Physics_250/`         |       250 | text     |

## Download separately

| Directory (after download)                           | # queries | Source |
|------------------------------------------------------|-----------|--------|
| `MathVista_testmini_250/`                            |       250 | [`AI4Math/MathVista`](https://huggingface.co/datasets/AI4Math/MathVista) (testmini split, first 250) |
| `MMMU_Pro/MMMU_Pro_standard_4_250/`                  |       250 | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) (`standard (4 options)` split, first 250) |
| `MMMU_Pro/MMMU_Pro_standard_10_250/`                 |       250 | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) (`standard (10 options)` split, first 250) |
| `MMMU_Pro/MMMU_Pro_vision_250/`                      |       250 | [`MMMU/MMMU_Pro`](https://huggingface.co/datasets/MMMU/MMMU_Pro) (`vision` split, first 250) |

### Download snippet

```python
from datasets import load_dataset

# MathVista (testmini → first 250)
ds = load_dataset("AI4Math/MathVista", split="testmini")
ds.select(range(250)).save_to_disk("data/MathVista_testmini_250")

# MMMU-Pro (standard 4-option / 10-option / vision)
for cfg, out in [
    ("standard (4 options)",  "data/MMMU_Pro/MMMU_Pro_standard_4_250"),
    ("standard (10 options)", "data/MMMU_Pro/MMMU_Pro_standard_10_250"),
    ("vision",                "data/MMMU_Pro/MMMU_Pro_vision_250"),
]:
    ds = load_dataset("MMMU/MMMU_Pro", cfg, split="test")
    ds.select(range(250)).save_to_disk(out)
```

The dataset directories follow HuggingFace's `save_to_disk` layout
(`data-00000-of-00001.arrow`, `dataset_info.json`, `state.json`) — the
same layout the bundled text datasets use, and the layout
`run_lere_experiment.py` expects.

### CLIP embeddings

The `embeddings/*.csv` files for MathVista and MMMU-Pro are bundled in
this repo, so once you have the raw datasets you can run experiments
without recomputing them. To recompute from scratch:

```bash
python scripts/precompute_mathvista_embeddings.py
python scripts/precompute_mmmu_pro_embeddings.py
python scripts/recompute_mmmu_pro_vision_embeddings.py
```
