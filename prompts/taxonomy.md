# Closed vocabularies

Both `domain` and `tags` are **closed sets**. Free-text values break the domain filter
(§4 of SPEC) and fragment the tag space, which silently degrades retrieval. Any value
outside these lists is coerced to `other` at ingest by `lere/schema.py` and counted in
`vocab_violations`.

## `domain` — 14 MMLU-Pro categories, with `math` refined

Base categories (from MMLU-Pro), covering GPQA-Diamond, MMMU-Pro and MathVista as well:

```
biology            business           chemistry          computer_science
economics          engineering        health             history
law                philosophy         physics            psychology
other
```

`math` is split, because AIME and MathVista are the datasets where a domain filter has to
do real work and undivided `math` would match everything:

```
math.algebra       math.number_theory math.combinatorics math.geometry
math.probability   math.precalculus   math.calculus      math.other
```

Domain matching is prefix-aware: `math.geometry` matches `math.*` at reduced weight
(`domain_partial_credit`, default 0.6) and a non-`math` domain at zero.

## `tags` — five prefixes, always `prefix.specific`

| prefix | meaning | examples |
| --- | --- | --- |
| `strategy.` | a reasoning move | `strategy.equation_manipulation`, `strategy.casework`, `strategy.invariant`, `strategy.backward_induction` |
| `tool.` | a computational capability the solver can actually invoke | `tool.sympy`, `tool.numeric_search`, `tool.brute_force_check`, `tool.exact_arithmetic` |
| `knowledge.` | a domain fact or law that must be recalled | `knowledge.thermodynamics`, `knowledge.stereochemistry` |
| `pitfall.` | a recurring failure to avoid | `pitfall.unit_conversion`, `pitfall.off_by_one`, `pitfall.degenerate_case` |
| `format.` | answer-shape handling | `format.multiple_choice`, `format.integer_0_999`, `format.boxed` |

Rules:

- 1 to 4 tags per entry. More than 4 makes every entry match everything.
- Always `prefix.snake_case`. A tag with no listed prefix is dropped.
- `knowledge.*` entries are the weakest kind of skill entry — they tend to be one-shot
  facts rather than transferable procedures. Prefer `strategy.*` and `pitfall.*`.
- `tool.*` is the channel by which "this problem type wants code" reaches retrieval. C1's
  `tool_expected` flag controls *execution* and is deliberately kept out of the retrieval
  key; a `tool.*` tag is how the same judgement becomes a durable, retrievable skill.

## Answer formats by dataset

| dataset | `answer_type` | normalization |
| --- | --- | --- |
| AIME 2024 / 2025 | `integer` | strip `\boxed{}`, strip leading zeros, range 0–999 |
| GPQA-Diamond | `mcq_letter` | single letter A–D |
| MMLU-Pro | `mcq_letter` | single letter A–J (10 options) |
| MMMU-Pro | `mcq_letter` | single letter; also a vision-only split where the question is inside the image |
| MathVista | `integer` / `float` / `mcq_letter` | per-item; float compared with relative tolerance |
| HLE-Exact (Humanity's Last Exam, letter+number pool, category-balanced) | `mcq_letter` (153, up to 22 options) / `free` (97 single numbers) | letter A–Z bounded by `n_options`; numbers by canonical exact match; 100% string-scoreable, no judge |
| AA-Omniscience-Public | `free` | canonical exact match for accuracy; the Index (+1/−1/0) needs the post-hoc grader for abstentions |
