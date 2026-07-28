# Configuration reference

Complete hyperparameter settings for the reported experiments, on two backbones (Gemini-2.5-flash-lite, GPT-4o-mini) and nine benchmarks.

Benchmarks are grouped into two **stream lengths**:

* **Short stream** -- AIME-2024, AIME-2025 (30 queries each)
* **Long stream** -- GPQA-Diamond (198), MathVista-testmini (250), MMLU-Pro Engineering (250), MMLU-Pro Physics (250), MMMU-Pro Standard-4 (250), MMMU-Pro Standard-10 (250), MMMU-Pro Vision (250)

## 1. Settings identical across all configurations

These hold for every benchmark, backbone and method variant.

**`adapters`**

| Setting | Value |
|---|---|
| `enabled` | `true` |
| `hidden_dim` | `384` |

**`ccme`**

| Setting | Value |
|---|---|
| `adaptive_temperature` | `true` |
| `temperature` | `0.07` |
| `use_exposure_annealing` | `true` |

**`ccme_training`**

| Setting | Value |
|---|---|
| `buffer_size` | `512` |
| `false_negative_threshold` | `0.75` |
| `learning_rate` | `2e-05` * |
| `min_positive` | `1` |
| `prune_on_instability` | `true` |
| `reliability_prune_threshold` | `0.35` |
| `stability_threshold` | `0.1` |

**`crte`**

| Setting | Value |
|---|---|
| `d_temporal` | `64` |
| `lambda_cluster` | `0.1` |
| `lambda_margin` | `0.1` |
| `margin_eta` | `0.2` |

**`crte_training`**

| Setting | Value |
|---|---|
| `buffer_size` | `256` |
| `lambda_crte` | `1.0` |
| `learning_rate` | `2e-05` * |
| `min_positive` | `1` |
| `prune_on_instability` | `true` |
| `stability_threshold` | `0.1` |
| `stale_positive_threshold` | `0.15` |
| `trivial_negative_threshold` | `0.05` |

**`experiment_control`**

| Setting | Value |
|---|---|
| `save_detailed_outputs` | `true` |
| `shuffle_seed` | `null` |
| `verbose` | `true` |

**`llm`**

| Setting | Value |
|---|---|
| `execute_python_code` | `true` |
| `temperature` | `0.0` |

**`online_training`**

| Setting | Value |
|---|---|
| `enable_training` | `true` |
| `num_train_steps` | `4` |
| `temporal_decay` | `0.9` |

**`per_query_memory_operations`**

| Setting | Value |
|---|---|
| `dedup_threshold` | `0.75` |
| `prune_reliability_threshold` | `0.35` |
| `pruning_mode` | `OR` |
| `staleness_decay_lambda` | `0.01` |

**`periodic_memory_refinement`**

| Setting | Value |
|---|---|
| `min_items_for_refinement` | `5` |
| `n_clusters_mode` | `fixed` |
| `post_refinement_grace` | `2` |
| `refinement_mode` | `fixed` |
| `shard_capacity` | `10` |

**`retrieval`**

| Setting | Value |
|---|---|
| `ccme_alpha` | `0.9` |
| `use_full_bank` | `false` |

\* Learning rate is `2e-05` in all but 3 configurations, which use `4e-05`, `5e-05`. CCME and CRTE always share the same value.

One setting depends on the backbone:

| Block | Setting | Gemini-2.5-flash-lite | GPT-4o-mini |
|---|---|---|---|
| `llm` | `max_tokens` | `16384` | `2048` |

## 2. Short stream vs long stream

Update cadence and memory-maintenance schedule scale with the length of the query stream. All other settings are shared.

| Block | Setting | Short stream (30 queries) | Long stream (198-250 queries) |
|---|---|---|---|
| `ccme_training` | `stability_window` | `10` | `20` |
| `crte_training` | `stability_window` | `10` | `20` |
| `experiment_control` | `save_interval` | `5` | `10` |
| `online_training` | `batch_size` | `4` | `8` |
| `online_training` | `k_upd` | `3` | `8` |
| `periodic_memory_refinement` | `redundancy_threshold` | `0.65` | `0.75` |
| `periodic_memory_refinement` | `refine_every_updates` | `3` | `4` |

Two settings scale with the exact benchmark size rather than the stream class:

| Block | Setting | 30 queries | 198 queries | 250 queries |
|---|---|---|---|---|
| `per_query_memory_operations` | `min_queries_before_operations` | `5` | `5` | `10` |
| `periodic_memory_refinement` | `n_clusters_fixed` | `10` | `10` | `16` |

## 3. Encoder settings by modality

The frozen base embedder differs between text-only and multimodal benchmarks; the trainable projection is the same architecture in both cases.

| Block | Setting | Text benchmarks | Multimodal benchmarks |
|---|---|---|---|
| `adapters` | `base_dim` | `1536` | `768` |
| `adapters` | `embedding_model` | `text-embedding-3-small` | `clip` |
| `adapters` | `projection_dim` | `768` | `512` |

Multimodal benchmarks: `MMMU_Pro_standard_10_250`, `MMMU_Pro_standard_4_250`, `MMMU_Pro_vision_250`, `MathVista_testmini_250`. All others are text-only.

The projection in both cases is `Linear(base_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, projection_dim)` followed by L2 normalisation.

## 4. What distinguishes LeRe, LeRe-R and LeRe-S

The three variants differ only in the `retrieval` block; every setting in Sections 1 to 3 is shared between them.

| Variant | Memory retrieval | Past solutions injected | Synthesizer |
|---|---|---|---|
| **LeRe** | CCME top-k | no | no |
| **LeRe-R** | CCME top-k | yes | no |
| **LeRe-S** | CCME top-k | no | yes |

### Short stream

| `retrieval` setting | LeRe | LeRe-R | LeRe-S |
|---|---|---|---|
| `ccme_alpha` | `0.9` | `0.9` | `0.9` |
| `inject_past_solutions` | `false` | `true` | `false` |
| `past_solutions_top_k` | `3` | `3` (x3), `2` (x3), `1` (x1) | `3` |
| `retrieve_top_k` | `3` | `3` (x6), `2` (x1) | `3` (x7), `2` (x1) |
| `use_full_bank` | `false` | `false` | `false` |
| `synthesizer` | `disabled` | `disabled` | `enabled` |
| `synthesizer implementation` | `-` | `-` | `v1` (x4), `v2` (x4) |

### Long stream

| `retrieval` setting | LeRe | LeRe-R | LeRe-S |
|---|---|---|---|
| `ccme_alpha` | `0.9` | `0.9` | `0.9` |
| `inject_past_solutions` | `false` | `true` | `false` (x16), `true` (x1) |
| `past_solutions_top_k` | `3` | `3` (x18), `2` (x1), `1` (x1) | `3` (x16), `2` (x1) |
| `retrieve_top_k` | `3` | `3` (x14), `2` (x3), `1` (x3) | `3` (x16), `2` (x1) |
| `use_full_bank` | `false` | `false` | `false` |
| `synthesizer` | `disabled` | `disabled` | `enabled` |
| `synthesizer implementation` | `-` | `-` | `v1` (x14), `v2` (x3) |

`(xN)` marks settings that were not uniform across the runs behind a column. The LeRe-S column pools two synthesizer implementations, `v1` and `v2`.

## 5. A complete configuration

Short stream, LeRe-S, Gemini-2.5-flash-lite. Absolute paths are replaced with `<PROJECT_ROOT>`, internal directory and codenames are normalised to the names used in the paper, and the dataset description and inline mode guide are elided as free text.

```json
{
  "experiment": {
    "name": "ccme_topk_synth_v1_run1",
    "task": "AIME_2024",
    "approach_name": "LeRe",
    "run_id": 1
  },
  "llm": {
    "model_name": "gemini/gemini-2.5-flash-lite",
    "temperature": 0.0,
    "max_tokens": 16384,
    "execute_python_code": true
  },
  "adapters": {
    "embedding_model": "text-embedding-3-small",
    "base_dim": 1536,
    "projection_dim": 768,
    "hidden_dim": 384,
    "enabled": true
  },
  "ccme": {
    "temperature": 0.07,
    "adaptive_temperature": true,
    "use_exposure_annealing": true
  },
  "crte": {
    "d_temporal": 64,
    "lambda_cluster": 0.1,
    "lambda_margin": 0.1,
    "margin_eta": 0.2
  },
  "online_training": {
    "enable_training": true,
    "k_upd": 3,
    "num_train_steps": 4,
    "batch_size": 4,
    "temporal_decay": 0.9,
    "device": "cuda"
  },
  "ccme_training": {
    "buffer_size": 512,
    "learning_rate": 2e-05,
    "min_positive": 1,
    "stability_window": 10,
    "stability_threshold": 0.1,
    "prune_on_instability": true,
    "false_negative_threshold": 0.75,
    "reliability_prune_threshold": 0.35
  },
  "crte_training": {
    "buffer_size": 256,
    "learning_rate": 2e-05,
    "lambda_crte": 1.0,
    "min_positive": 1,
    "stability_window": 10,
    "stability_threshold": 0.1,
    "prune_on_instability": true,
    "stale_positive_threshold": 0.15,
    "trivial_negative_threshold": 0.05
  },
  "per_query_memory_operations": {
    "dedup_threshold": 0.75,
    "prune_reliability_threshold": 0.35,
    "pruning_mode": "OR",
    "staleness_decay_lambda": 0.01,
    "min_queries_before_operations": 5
  },
  "periodic_memory_refinement": {
    "shard_capacity": 10,
    "n_clusters_mode": "fixed",
    "n_clusters_fixed": 10,
    "redundancy_threshold": 0.65,
    "min_items_for_refinement": 5,
    "refinement_mode": "fixed",
    "refine_every_updates": 3,
    "post_refinement_grace": 2
  },
  "retrieval": {
    "use_full_bank": false,
    "retrieve_top_k": 3,
    "ccme_alpha": 0.9,
    "inject_past_solutions": false,
    "past_solutions_top_k": 3,
    "synthesizer_v1": true
  },
  "experiment_control": {
    "max_n_samples": 30,
    "verbose": true,
    "save_interval": 5,
    "shuffle_seed": null,
    "save_detailed_outputs": true
  },
  "paths": {
    "data_dir": "<PROJECT_ROOT>/data/AIME_2024",
    "prompts_dir": "<PROJECT_ROOT>/prompts",
    "results_root": "<PROJECT_ROOT>/results/new_frame",
    "precomputed_embeddings_csv": "<PROJECT_ROOT>/embeddings/AIME_2024.csv"
  }
}
```
