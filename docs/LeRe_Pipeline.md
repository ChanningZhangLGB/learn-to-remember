# LeRe Pipeline

This document describes the Learn-to-Remember (LeRe) framework: how a query
flows through the agent, how the memory bank is structured, how the two
trainable encoders (CCME, CRTE) are updated online, and how the optional
Prospector mode anticipates knowledge for upcoming queries. Read together
with [`README.md`](../README.md) (setup + reproduction) and the per-query
prompts in [`prompts/`](../prompts/).

---

## 1. Architecture overview

```
                 ┌──────────────────┐
   query x  ───► │   CCME retrieval │ ── retrieved memory items M_x
                 │   (Eq, Em, p̂)   │
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Generator     │ ── trajectory τ + answer ŷ
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Reflector     │ ── reflection r:
                 │                  │     • verdict per memory item
                 │                  │     • new memory proposals
                 └──────────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │    Curator       │ ── memory bank B updated
                 │  (per-query ops) │
                 └──────────────────┘
                          │
                          ▼ (every k_upd queries)
                 ┌──────────────────┐
                 │  Online trainer  │ ── CCME loss → (Eq, Em)
                 │                  │     CRTE loss → (Et, Er)
                 └──────────────────┘
```

Four trainable encoders share a single adapter contract (see `adapters`
config block: `embedding_model`, `base_dim`, `projection_dim`, `hidden_dim`).

| Encoder | Symbol | Trained via | Role |
|---|---|---|---|
| Query encoder | E_q | L_CCME | Project query into contrastive retrieval space |
| Memory encoder | E_m | L_CCME | Project memory entry into contrastive retrieval space |
| Trajectory encoder | E_t | L_CRTE | Encode reasoning trajectory steps |
| Reflection encoder | E_r | L_CRTE | Encode structured reflection / insight |

---

## 2. Memory entry structure

```json
{
  "id": "m_014",
  "title": "Short descriptive title",
  "scope": "specific | general",
  "bullets": ["Key insight 1", "Key insight 2", "..."],
  "example": "Concrete worked example",
  "tags": ["algebra", "modular-arithmetic"],
  "keywords": ["key", "terms"],
  "source_queries": ["Q_006", "Q_018"],
  "helpful": 3,
  "harmful": 0,
  "retrieved_count": 5,
  "last_used_query": "Q_018"
}
```

When the memory encoder E_m embeds an entry, all semantic fields are
concatenated into a single text:

```
text = "Title: " + title
     + "\nInsights:\n- " + bullets.join("\n- ")
     + "\nExample: " + example
     + "\nTags: "  + tags.join(", ")
     + "\nScope: " + scope
```

This same surface is used for clustering during periodic refinement, so
training, retrieval, and refinement all see one representation.

---

## 3. Retrieval (CCME)

```
score(x, m) = α · sim(E_q(x), E_m(m)) + (1 − α) · p̂(m)
```

- `sim(·, ·)` — cosine similarity in the projected (contrastive) space.
- `p̂(m) = (helpful + 1) / (helpful + harmful + 2)` — Bayesian reliability prior.
- `α` (`retrieval.ccme_alpha`) — semantic-vs-reliability blend.

### Source-query similarity (`sim_sq`)

Periodic refinement merges memory items into prototypes that retain the
union of all `source_queries`. At retrieval time we additionally compute,
in **raw base-API space** (un-projected — provenance is invariant to
training state):

```
sim_sq(x, m) = max over sq in m.source_queries of cos(emb(x), emb(sq))
```

The combined score with `source_query_sim_mode: max` is then
`max(score, sim_sq)`. This rescues memories whose contrastive
representation has drifted but whose generating query is highly similar to
the current one.

---

## 4. Generator → Reflector → Curator pipeline

### Generator
Produces a solution attempt with optional Python execution.
- Receives the query, the retrieved memory items M_x, and the generator
  prompt template.
- If the generated code raises, the model is re-prompted with the error
  trace and retries (bounded).
- Outputs: trajectory τ (chain of reasoning steps + tool calls) and the
  final answer ŷ.

### Reflector
Produces a structured JSON reflection r:
- `answer` — the model's final answer (re-extracted, may differ from ŷ)
- `memory_evaluation` — for **each** retrieved item: `helpful` or
  `harmful` plus a justification
- `new_memory` — proposed new entries (title, scope, bullets, example,
  tags, keywords) the reflector thinks the bank is missing for queries
  like x

### Curator
Decides admission for each new-memory proposal, then runs the
**per-query memory operations**:

| Op | Param | Behaviour |
|---|---|---|
| Deduplication | `dedup_threshold` | Drop near-duplicates (cosine sim > τ) |
| Reliability pruning | `prune_reliability_threshold` | Drop entries with very low p̂ |
| Staleness decay | `staleness_decay_lambda` | Decay reliability of entries not recently retrieved |
| Pruning combinator | `pruning_mode` | `OR` / `AND` over the previous two |
| Warm-up guard | `min_queries_before_operations` | Skip dedup + pruning until N queries seen |

The warm-up guard prevents the curator from collapsing early entries on
sparse signal — important on small datasets (e.g. AIME, 30 queries; we set
the guard to 5).

---

## 5. Online training

### 5.1 Buffers

| Buffer | Content |
|---|---|
| `ccme_positive_buffer` | (query, **helpful** memory) pairs |
| `ccme_negative_buffer` | (query, **harmful** memory) pairs |
| `crte_positive_buffer` | (trajectory, reflection) pairs from successful episodes |
| `crte_negative_buffer` | (trajectory, reflection) pairs from failed episodes |

Sampling uses temporal-decay weighting (`temporal_decay`): more recent
pairs are sampled with higher probability.

### 5.2 Schedule

Every `k_upd` queries, run `num_train_steps` gradient steps:
- `(E_q, E_m)` updated jointly with L_CCME (InfoNCE).
- `(E_t, E_r)` updated jointly with L_CRTE (InfoNCE + clustering + margin terms).

### 5.3 Per-component stability monitoring

CCME and CRTE are watched independently using rolling windows:

```
increase = (mean of last 5 losses) − (mean of older losses)
           ─────────────────────────────────────────────────
                       (mean of older losses + ε)

unstable_if  increase ≥ stability_threshold
```

If CCME is unstable, only `(E_q, E_m)` rolls back to the previous
checkpoint. If CRTE is unstable, only `(E_t, E_r)` rolls back. The stable
component keeps adapting.

### 5.4 Quality-based buffer pruning on instability

Pruning targets the *source* of conflicting gradients, not buffer-size
imbalance.

**CCME**
- *Positive buffer*: drop outliers — entries with
  `sim < μ − outlier_sigma_factor · σ` under the current encoder. They
  oppose the consensus gradient.
- *Negative buffer*: drop **false negatives** — entries with
  `sim > false_negative_threshold`. They are semantically close but
  labelled negative; they generate adversarial gradients.
- Safety: never reduce positives below `min_positive`.

**CRTE** (uses cached base embeddings — no extra API calls)
- *Positive buffer*: drop **stale pairs** with
  `cos(t_emb, r_emb) < stale_positive_threshold` (no alignment, no
  gradient signal).
- *Negative buffer*: drop **trivial easy negatives** with
  `cos(t_emb, r_emb) < trivial_negative_threshold` (already maximally
  separated, no learning signal).
- Note: CRTE is structurally negative-heavy on math reasoning. *Do not*
  guard on positive/negative ratio.

---

## 6. Periodic memory refinement

### 6.1 Triggers

| Mode | Condition |
|---|---|
| `fixed` | Every `refine_every_updates` training cycles |
| `adaptive` | When mean pairwise cosine similarity over the bank exceeds `redundancy_threshold` |

Adaptive mode reuses the cached E_m embeddings — no API cost.

### 6.2 K-medoids in E_r space

1. Encode every memory item with E_r.
2. Partition into `n_clusters` clusters (`n_clusters_mode: heuristic |
   fixed`).
3. **Prototype = highest-reliability member**, `(helpful+1)/(helpful+harmful+2)`,
   *not* the geometric medoid. This rewards items with empirical evidence.
4. **Prototype metadata merge**:
   - `helpful`, `harmful`, `retrieved_count`: summed across cluster members
   - `tags`, `keywords`, `source_queries`: union (full provenance preserved)
   - `last_used_query`: most recent across members
   - `title`, `scope`, `bullets`, `example`: taken from the prototype

### 6.3 Post-refinement grace period

After refinement the bank distribution shifts; encoders trained on the old
distribution may show a temporary loss spike. To prevent false rollbacks,
stability checks are skipped for `post_refinement_grace` updates. After
that, normal monitoring resumes.

---

## 7. Multimodal handling

The encoder family is selected at startup from the dataset, **not** from
the config:

| Task | Datasets | E_q / E_m | dim |
|---|---|---|---|
| Text-only | AIME, GPQA, MMLU-Pro | OpenAI `text-embedding-3-small` | 1536 |
| Multimodal | MMMU-Pro (standard / vision), MathVista | CLIP ViT-L/14 (text + image, HF) | 768 |

CLIP's text and image encoders were contrastively trained to share a
latent space, so a fused (text + image) query embedding can be compared
against a text-only memory embedding by dot product. Bridging CLIP (768d)
and `text-embedding-3-small` (1536d) through a projection layer was tried
and rejected: it couples incompatible latent spaces and produces
unreliable similarity scores. `projection_dim` therefore defaults to
`base_dim` and the projection is the identity unless `adapters.enabled`
is explicitly set.

Detection happens in `_init_components()` of `run_lere_experiment.py` via
`_is_mmmu_pro_task()`, `_is_mmmu_pro_vision_task()`, and
`_is_mathvista_task()`.

---

## 8. Optional: Prospector (`ccme_topk_prosp_v1`)

Standard mode adapts memory *after* each query is solved. The Prospector
adds a forward-looking step: a curator-style LLM call that anticipates
what knowledge query `Q_{i+1}` is likely to need, runs **before** that
query is processed.

Prospector v1 is **admission-gated** to keep the bank clean:

1. After Q_i finishes, the Prospector proposes `k` candidate memory
   entries for Q_{i+1}.
2. These candidates are kept in an **ephemeral buffer**, not added to the
   bank.
3. When Q_{i+1} runs, the generator + reflector see the union of CCME
   top-k items (from the bank) and the ephemeral buffer — they cannot
   tell which is which.
4. The reflector's `memory_evaluation` decides each ephemeral candidate's
   fate:
   - Verdict `HELPFUL` → admitted into the bank (before the curator runs)
   - Verdict `HARMFUL` or unused → discarded with no trace
5. CCME training operates on the **effective** retrieved set: admitted
   ephemerals act like normal bank entries; rejected ephemerals
   contribute zero gradient.

This preserves bank quality (only empirically validated entries persist)
while still capturing the upside of anticipatory curation.

Earlier modes are kept for ablation: `ccme_topk_prosp` (v0; entries enter
the bank immediately, no admission gate), `ccme_topk` (no prospector at
all), `past_sol_plus_ccme` (inject the previous queries' solutions
alongside CCME-retrieved items).

---

## 9. Config block layout

Each block has a single responsibility; no parameter appears in two
blocks.

```json
{
  "experiment":          { ... },
  "llm":                 { ... },
  "adapters":            { "embedding_model", "base_dim", "projection_dim", "hidden_dim", "enabled" },
  "ccme":                { "temperature", "adaptive_temperature", "use_exposure_annealing" },
  "crte":                { "d_temporal", "lambda_cluster", "lambda_margin", "margin_eta" },
  "online_training": {
    "enable_training", "k_upd", "num_train_steps", "batch_size",
    "temporal_decay", "refinement_mode", "refine_every_updates",
    "post_refinement_grace", "device"
  },
  "ccme_training": {
    "buffer_size", "learning_rate", "min_positive",
    "stability_window", "stability_threshold",
    "prune_on_instability", "false_negative_threshold", "outlier_sigma_factor"
  },
  "crte_training": {
    "buffer_size", "learning_rate", "lambda_crte", "min_positive",
    "stability_window", "stability_threshold",
    "prune_on_instability", "stale_positive_threshold", "trivial_negative_threshold"
  },
  "per_query_memory_operations": {
    "dedup_threshold", "prune_reliability_threshold",
    "pruning_mode", "staleness_decay_lambda", "min_queries_before_operations"
  },
  "periodic_memory_refinement": {
    "shard_capacity", "n_clusters_mode", "n_clusters_fixed",
    "redundancy_threshold", "min_items_for_refinement"
  },
  "retrieval":           { "retrieve_top_k", "ccme_alpha", "use_source_query_sim", "source_query_sim_mode", "prospector_v1" },
  "experiment_control":  { ... },
  "paths":               { ... }
}
```
