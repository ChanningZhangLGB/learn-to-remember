# Learn to Remember (LeRe)

Code for *Learn to Remember: Geometric Memory for Inference-Time Self-Improvement in Language
Models* (under review at ICLR 2027).

LeRe adapts a frozen LLM at inference time without ground-truth labels or weight updates. A
**Planner** turns each query into a structured retrieval key; **CCME**, two linear heads on a
frozen sentence encoder, retrieves the top-K memory entries and is trained online from the
Curator's attributions; a **Solver** answers, optionally running code; a **Curator** verifies
the answer, credits or blames each retrieved entry and proposes new ones, which the **GCM**
gate (Guard, Consolidate, Maintain) filters into the memory bank.

## Repository layout

```
learn-to-remember-AD37/
├── lere/                            # Algorithm (LeRe core)
│   ├── pipeline.py                  # Plan → retrieve → solve → curate loop (Algorithm 1)
│   ├── retrieve.py                  # Retrieval score (Eq. 3) + MMR top-K selection
│   ├── embed.py                     # Frozen encoder + CCME projection heads E_q / E_m
│   ├── ccqs.py                      # CCME online contrastive training (Eq. 10-11)
│   ├── curate.py                    # Credit assignment (Eq. 12) + Consolidate (merge / link)
│   ├── guard.py                     # Guard: answer-leakage / restatement rejection
│   ├── store.py                     # Memory bank, reliability (Eq. 8), quarantine, pruning
│   ├── verify.py                    # Label-free verification signal
│   ├── tools.py                     # Restricted Python execution for the Solver
│   ├── providers.py                 # OpenAI / Gemini clients + cost ledger
│   ├── llm.py                       # LLM interface + offline stand-in
│   ├── schema.py                    # Planner / Solver / Curator output schemas
│   ├── answers.py                   # Answer normalization
│   ├── datasets.py                  # Stream loading (JSONL + images)
│   └── trace.py                     # Run recorders (calls, retrieval, CCME)
│
├── prompts/                         # Planner / Solver / Curator prompts + vocabularies
├── configs/
│   ├── lere.yaml                    # All hyperparameters (Table 4)
│   ├── models/                      # Backbone + list prices (Table 3)
│   └── streams.yaml                 # The 12 evaluation streams
├── data/subsets/                    # 12 streams, fixed evaluation order (JSONL)
├── scripts/
│   ├── run_lere.py                  # Run one stream (ablation / K flags)
│   ├── run_stream.sh                # Entry-point launcher
│   ├── run_main.sh                  # Table 1: 12 streams × 3 backbones
│   ├── run_ablations.sh             # Table 2: w/o Planner / CCME / Execution
│   ├── run_k_sweep.sh               # Table 8: K ∈ {1, 5, 10}
│   ├── eval/                        # Shared answer scorer (App. B.3) + run scoring
│   ├── analysis/                    # CCME geometry, verification reliability, K-sweep tests
│   └── data/fetch_images.py         # Image download + SHA-256 check
├── tests/                           # Unit tests (no API access)
├── API_key.txt.example              # API-key template (copy → API_key.txt)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt                 # tested on Python 3.8
python -m pytest tests/ -q                      # no API access needed
python scripts/data/fetch_images.py             # images for MathVista, MMMU-Pro, HLE
```

Set `OPENAI_API_KEY` and/or `GEMINI_API_KEY` (or copy `API_key.txt.example` to `API_key.txt`).
HLE is gated on Hugging Face: run `huggingface-cli login` before fetching its images. The
twelve evaluation streams are in `data/subsets/`, in the order all methods processed them
(see `data/README.md`).

## Usage

```bash
scripts/run_stream.sh gpt-4.1-mini AIME_2025               # one stream, one backbone
scripts/run_stream.sh gpt-4o-mini AIME_2025 --dry-run      # offline, no API calls
python scripts/eval/score_runs.py runs/gpt-4.1-mini/AIME_2025
```

Backbones: `gemini-3.1-flash-lite`, `gpt-4.1-mini`, `gpt-4o-mini`. Streams are listed in
`configs/streams.yaml`. Options: `--top-k K`, `--no-planner`, `--no-ccme`, `--no-exec`,
`--set section.key=value`, `--limit N`, `--out DIR`.

All hyperparameters (Table 4) are in `configs/lere.yaml`; backbones and prices are in
`configs/models/`. Each run directory records every LLM call, retrieval candidate, CCME update
and memory write, plus `report.json` with accuracy, cost and latency.

## Reproducing the paper

```bash
scripts/run_main.sh                    # Table 1: 12 streams x 3 backbones
scripts/run_ablations.sh               # Table 2: w/o Planner, CCME, Execution
scripts/run_k_sweep.sh <backbone>      # Table 8: K in {1, 5, 10}

python scripts/eval/score_runs.py runs/*/* --table                 # accuracy (Appendix B.3 scorer)
python scripts/analysis/ccme_geometry.py runs/*/*                  # Table 6, Figure 8
python scripts/analysis/verification_reliability.py runs/*/*      # Table 7
python scripts/analysis/k_sweep_significance.py runs/*/*          # Table 8 tests
```

Runs use temperature 0 and at most 2,048 output tokens per call, but hosted models can change
over time, so results may differ slightly. Baselines were run with the official Dynamic
Cheatsheet (https://github.com/suzgunmirac/dynamic-cheatsheet) and ACE implementations on the
same streams and scorer.

## Code vs. paper names

| Paper | Code |
| --- | --- |
| Planner / Solver / Curator | `c1` / `c2` / `c3` |
| memory bank, entry | `SkillBook`, `SkillEntry` |
| CCME, heads E_q / E_m | `ccqs`, `Ep` / `Es` |
| HELPFUL / HARMFUL / IRRELEVANT / REDUNDANT | `used_positive` / `used_negative` / `unused_irrelevant` / `unused_redundant` |

## Note

The Solver executes model-written Python in a restricted subprocess (no network, no API keys,
time and memory limits). This is not a full sandbox; run experiments in an isolated
environment.
