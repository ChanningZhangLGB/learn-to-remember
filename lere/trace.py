"""Run instrumentation: record what each component saw, what it returned, and what the
retriever and the heads were doing at the time.

None of this changes a run. Everything here either reads state or hangs off the two
optional observer hooks (`Pipeline.on_step`, `ProviderLLM.on_call`), so the traced and
untraced paths are the same code.

What it captures, and why each piece is here rather than derived later:

* **Every LLM call**, prompt and raw text included. `extract_json` discards the raw text,
  and that is the only place a fence, a preamble or a near-miss schema violation shows up.
* **Every retrieval candidate, not just the selected ones.** `Retriever` drops anything
  below `sim_threshold` and returns a list; on a cold-start run the informative fact is
  usually "the best candidate scored 0.43 against a 0.60 floor", and a returned empty list
  cannot tell you that. It is also the raw material for calibrating thresholds.
* **The encoder version on every vector.** A CCME update invalidates every projected
  vector (`DualEncoder.bump_version`). An unversioned embedding dump stops being
  interpretable the moment the heads move.
* **Head geometry per step, full weights per update.** `||W - I||` is the number that
  answers "did the heads actually move"; it is cheap enough to keep at 12k items, which
  full 384x384 dumps are not.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from .retrieve import Retriever, parse_threshold
from .schema import domain_affinity


def _jsonable(obj):
    """Make anything the pipeline holds writable as JSON, without losing numbers."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [round(float(x), 6) for x in obj.ravel().tolist()]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj):
        return _jsonable(asdict(obj))
    if hasattr(obj, "to_dict"):
        return _jsonable(obj.to_dict())
    return str(obj)


class JsonlWriter:
    """Append-only, flushed per line: a run that dies at item 7 still has items 1-6."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.n = 0

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")
        self._fh.flush()
        self.n += 1

    def close(self) -> None:
        self._fh.close()


# ------------------------------------------------------------------- encoder geometry

def head_weights(encoder) -> dict:
    """E_q and E_m as plain arrays, whichever backend is live."""
    out = {}
    for name, head in (("eq", encoder.eq), ("em", encoder.em)):
        if head.module is not None:
            out[name] = head.module.weight.detach().numpy().copy()
        else:
            out[name] = np.array(head._w, copy=True)
    return out


def head_stats(encoder) -> dict:
    """Cheap per-step geometry. `delta_from_identity` is the one that matters: the heads
    start at identity by construction, so any nonzero value is CCME having moved them."""
    stats = {"version": int(encoder.version), "trainable": bool(encoder.trainable)}
    for name, w in head_weights(encoder).items():
        eye = np.eye(w.shape[0], w.shape[1], dtype=w.dtype)
        stats[name] = {
            "shape": list(w.shape),
            "frobenius_norm": float(np.linalg.norm(w)),
            "delta_from_identity": float(np.linalg.norm(w - eye)),
            "max_abs_delta_from_identity": float(np.max(np.abs(w - eye))),
        }
    return stats


# ---------------------------------------------------------------------- LLM recording

class CallRecorder:
    """Collects every provider call. Wire it in as `ProviderLLM(on_call=recorder)`."""

    def __init__(self, writer: JsonlWriter, advance_on: str = "planner") -> None:
        self.writer = writer
        self.calls: list = []
        self.step = -1
        # Planner runs exactly once per item and always first (`Pipeline._process`), so it is
        # the item boundary. The driver cannot supply the step itself: `Pipeline.run`
        # owns the loop, and calls happen inside it.
        self.advance_on = advance_on
        self._t0 = time.perf_counter()

    def __call__(self, *, component, messages, raw_text, reply, parsed, attempts,
                 spec) -> None:
        if component == self.advance_on:
            self.step += 1
        # The prompt is the last user turn; earlier turns exist only on a parse retry.
        prompt = messages[0]["content"]
        if isinstance(prompt, list):         # multimodal: text part plus image parts
            prompt = "\n".join(p.get("text", "[image]") for p in prompt)
        record = {
            "step": self.step,
            "call_index": len(self.calls),
            "component": component,
            "model": spec.model,
            "temperature": spec.temperature,
            "parse_attempts": attempts,
            "turns_in_request": len(messages),
            "prompt": prompt,
            "raw_response": raw_text,
            "parsed": parsed,
            "prompt_tokens": reply.prompt_tokens,
            "cached_prompt_tokens": reply.cached_prompt_tokens,
            "completion_tokens": reply.completion_tokens,
            "elapsed_s": round(time.perf_counter() - self._t0, 3),
        }
        self.calls.append(record)
        self.writer.write(record)

    def slice_for_step(self, step: int) -> list:
        return [c for c in self.calls if c["step"] == step]


class TracedLLM:
    """Wrap any `LLM` so its calls reach a `CallRecorder`.

    `ProviderLLM` has the `on_call` hook and needs no wrapper; this exists so an offline
    stub exercises the identical recording path, which is the only way to test the
    recorders without spending API credit.
    """

    class _Reply:
        prompt_tokens = 0
        cached_prompt_tokens = 0
        completion_tokens = 0

    class _Spec:
        model = "offline-stub"
        temperature = 0.0

    def __init__(self, inner, recorder: CallRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def complete_json(self, prompt: str, *, component: str, image=None) -> dict:
        parsed = self.inner.complete_json(prompt, component=component, image=image)
        self.recorder(component=component, messages=[{"role": "user", "content": prompt}],
                      raw_text=json.dumps(parsed), reply=self._Reply(), parsed=parsed,
                      attempts=1, spec=self._Spec())
        return parsed


# ---------------------------------------------------------------- retrieval recording

class TracingRetriever(Retriever):
    """`Retriever` plus a record of every candidate it considered.

    The scoring loop below duplicates `Retriever.retrieve`, which is a liability, so it
    self-checks: every id the real retriever returned must appear in this table having
    passed the floor. A divergence means the duplication has drifted and is flagged in the
    record rather than silently producing a plausible, wrong table.
    """

    def __init__(self, memory, cfg, writer: JsonlWriter) -> None:
        super().__init__(memory, cfg)
        self.writer = writer
        self.last: dict = {}

    def retrieve(self, plan, step: int) -> list:
        cfg = self.cfg
        candidates = self.memory.active()
        qview = plan.query_view()
        qvec = self.memory.encoder.encode_query(qview)

        threshold = parse_threshold(cfg.get("sim_threshold", 0.6))
        mode = cfg.get("domain_filter", "soft")
        penalty = float(cfg.get("domain_penalty", 0.25))
        partial = float(cfg.get("domain_partial_credit", 0.6))
        alpha = float(cfg.get("alpha", 0.7))

        rows = []
        for entry in candidates:
            evec = self.memory.vector(entry)
            raw = float(evec @ qvec)
            aff = domain_affinity(plan.domain, entry.domain, partial)
            adjusted = raw * (1.0 - penalty * (1.0 - aff)) if mode == "soft" else raw
            rows.append({
                "entry_id": entry.id,
                "title": entry.title,
                "identity_view": entry.identity_view(),
                "entry_domain": entry.domain,
                "raw_sim": round(raw, 6),
                "passed_floor": threshold is None or raw >= threshold,
                "domain_affinity": round(float(aff), 6),
                "domain_adjusted_sim": round(float(adjusted), 6),
                "reliability": round(float(entry.meta.reliability), 6),
                "mixed_score": round(alpha * float(adjusted)
                                     + (1.0 - alpha) * float(entry.meta.reliability), 6),
                "es_vector": evec,
            })
        rows.sort(key=lambda r: r["mixed_score"], reverse=True)

        selected = super().retrieve(plan, step)
        chosen = {r.id for r in selected}
        for r in rows:
            r["selected"] = r["entry_id"] in chosen
        # The duplication self-check described in the class docstring.
        drift = sorted(chosen - {r["entry_id"] for r in rows if r["passed_floor"]})

        record = {
            "step": step,
            "encoder_version": int(self.memory.encoder.version),
            "query_view": qview,
            "planner_domain": plan.domain,
            "ep_query_vector": qvec,
            "sim_threshold": threshold,   # null == no floor
            "alpha": alpha,
            "top_k": int(cfg.get("top_k", 3)),
            "mmr_lambda": float(cfg.get("mmr_lambda", 0.7)),
            "n_candidates": len(rows),
            "n_passed_floor": sum(1 for r in rows if r["passed_floor"]),
            "n_selected": len(selected),
            "best_raw_sim": rows[0]["raw_sim"] if rows else None,
            "selected_ids": [r.id for r in selected],
            "selected_scores": [{"entry_id": r.id, "raw_sim": round(r.raw_sim, 6),
                                 "score": round(r.score, 6),
                                 "domain_affinity": round(r.domain_affinity, 6)}
                                for r in selected],
            "candidates": rows,
            "scoring_drift": drift,
        }
        self.last = record
        self.writer.write(record)
        return selected
