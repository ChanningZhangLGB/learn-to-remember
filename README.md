# Learn to Remember (LeRe)

Test-time adaptation through learned, contrastive memory.

LeRe is an agentic context engineering framework that learns to **remember**
which past insights help on which queries. Each query is solved by a
Generator → Reflector → Curator pipeline that maintains a structured memory
bank. Two trainable encoders — **CCME** (Contrastive Contextual Memory
Encoder) and **CRTE** (Contrastive Reflection-Trajectory Encoder) — are
updated online so retrieval and curation improve as the agent sees more
queries.

This repo contains the algorithm, the datasets used in our experiments, the
precomputed text/CLIP embeddings, all configs needed to reproduce the paper
results, and a small `usage_example.npy` showing the memory-bank format.

---

## Repository layout

```
Learn_to_remember/
├── main/                           # Algorithm (LeRe core)
│   ├── ccme_encoder.py             # CCME: query/memory encoders + contrastive loss
│   ├── crte_encoder.py             # CRTE: trajectory/reflection encoders + temporal embedding
│   ├── memory_operations.py        # Add / dedupe / prune / refine memory bank
│   ├── training_data.py            # Online buffers, contrastive pair extraction
│   ├── online_trainer.py           # Test-time training loop
│   ├── language_model_lere.py      # LLM call + Python-execution hook
│   ├── model_call.py               # Generic LLM client (litellm wrapper)
│   ├── run_lere_experiment.py      # End-to-end experiment runner
│   └── utils/
│       ├── lere_pipeline.py        # Generator → Reflector → Curator orchestrator
│       ├── lere_extractor.py       # Parse structured outputs from each stage
│       ├── memory_formatter.py     # Render memory items into prompts
│       ├── adapters.py             # Trainable projection heads
│       ├── dataset_ordering.py     # Query ordering / shuffling
│       └── execute_code.py         # Sandboxed Python execution
│
├── prompts/                        # Generator / Reflector / Curator / Synthesizer prompts
├── configs/                        # Per-(dataset, model, mode) JSON configs
├── data/                           # 9 paper datasets (HuggingFace .arrow format)
├── embeddings/                     # Precomputed text/CLIP embeddings (CSV)
├── figures/                        # Paper figures (cumulative accuracy + memory size)
├── docs/LeRe_Pipeline.md           # Algorithm + multimodal + Synthesizer spec
├── scripts/                        # Embedding precomputation utilities
├── test_framework/run_experiment.sh  # Entry-point launcher
├── example_usage.py                # Minimal end-to-end example
├── usage_example.npy               # Sample memory-bank state (see below)
├── config.env.example              # API-key template (copy → config.env)
└── requirements.txt
```

---

## Setup

```bash
# 1. Create environment
conda create -n lere python=3.11 -y
conda activate lere
pip install -r requirements.txt

# 2. Configure API keys
cp config.env.example config.env
# Edit config.env — fill in keys for the providers you intend to use
```

You only need keys for the providers you actually call. `OPENAI_API_KEY_EMBED`
is used for `text-embedding-3-small` (memory-item embeddings) regardless of
which LLM you query, so it is required for all runs.

---

## Reproducing paper results

Each entry in `configs/` corresponds to one (dataset, model, retrieval mode)
cell of the result tables. To reproduce a single run:

```bash
bash test_framework/run_experiment.sh \
    configs/AIME_2024/gemini-2.5-flash-lite/ccme_topk/aime2024_gemini_ccme_topk_run1.json
```

Outputs land under `results/<model>/<dataset>/<mode>/run_<id>/<timestamp>/`:

| File                              | Contents                                              |
|-----------------------------------|-------------------------------------------------------|
| `*_results.jsonl`                 | One record per query: input, trajectory, final answer |
| `final_memory_bank.json`          | Memory-bank state at end of run                       |
| `embedding_index.pkl`             | `{memory_id: 1536-dim embedding}`                     |
| `experiment_summary.json`         | Aggregate metrics + config snapshot                   |
| `input_output_log/query_NNN/`     | Per-query Generator/Reflector/Curator inputs+outputs  |
| `checkpoints/`                    | CCME / CRTE encoder weights                           |

### Datasets included

| Dataset                       | Domain         | # queries |
|-------------------------------|----------------|-----------|
| `AIME_2024`                   | Math (text)    |        30 |
| `AIME_2025`                   | Math (text)    |        30 |
| `GPQA_Diamond`                | Science (text) |       198 |
| `MMLU_Pro_Engineering_250`    | MMLU-Pro       |       250 |
| `MMLU_Pro_Physics_250`        | MMLU-Pro       |       250 |
| `MathVista_testmini_250`      | Vision+Math    |       250 |
| `MMMU_Pro_standard_4_250`    | Multimodal     |       250 |
| `MMMU_Pro_standard_10_250`   | Multimodal     |       250 |
| `MMMU_Pro_vision_250`        | Vision-only    |       250 |

Multimodal datasets ship with their CLIP embeddings; text datasets use
`text-embedding-3-small`.

### Retrieval modes (per-dataset config sub-folders)

| Folder                  | Behaviour                                                 |
|-------------------------|-----------------------------------------------------------|
| `ccme_topk/`            | CCME retrieval, top-k memory items                        |
| `ccme_topk_synth/`      | + forward-looking Synthesizer synthesis stage              |
| `ccme_topk_synth_v1/`   | Synthesizer v1 (refined contract; see `docs/LeRe_Pipeline.md`) |
| `past_sol_plus_ccme/`   | Inject k past solutions alongside CCME-retrieved items    |
| `ablation/`             | `no_ccme`, `no_crte`, `no_ccme_crte` ablations            |

---

## Memory-bank format (`usage_example.npy`)

```python
import numpy as np

data = np.load("usage_example.npy", allow_pickle=True).item()

memory_bank = data["memory_bank"]   # list[dict] — one entry per memory item
embeddings  = data["embeddings"]    # np.ndarray, shape (N, 1536), float32

print(memory_bank[0]["title"])
# → "Analyzing Nested Absolute Value and Trigonometric Functions"

print(memory_bank[0].keys())
# → dict_keys(['title', 'bullets', 'example', 'tags', 'scope', 'meta', 'id'])
```

Each memory item has:

| Field     | Type        | Description                                            |
|-----------|-------------|--------------------------------------------------------|
| `id`      | `str`       | Unique identifier (`m_NNN`)                            |
| `title`   | `str`       | Short name for the strategy / insight                  |
| `bullets` | `list[str]` | Actionable reasoning steps                             |
| `example` | `str`       | Concrete worked example                                |
| `tags`    | `list[str]` | Semantic tags (`reasoning.*`, `strategy.*`, etc.)      |
| `scope`   | `str`       | When this memory applies                               |
| `meta`    | `dict`      | `helpful` / `harmful` counts, `source_queries`, etc.   |

Cosine-similarity retrieval over the embeddings is the baseline retrieval
operator; CCME re-ranks the candidates using its trained projection.

---

## End-to-end example

`example_usage.py` instantiates a minimal LeRe agent (CCME + CRTE +
MemoryOperations + OnlineTrainer + Pipeline) and walks through a single
query. It is the shortest path from "what does the API look like" to
"running it on your own task".

```bash
python example_usage.py
```

---

## Citation

Paper: *Sparse and Uncertainty-Aware Agentic Context Engineering for
Robust Test-Time Adaptation.* (See `docs/LeRe_Pipeline.md` for the
algorithmic specification.)

---

## License

Code is released under the MIT License. Dataset files retain their original
licenses (AIME, GPQA, MMLU-Pro, MMMU-Pro, MathVista) — see each dataset's
upstream source for terms.
