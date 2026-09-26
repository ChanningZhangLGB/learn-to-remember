"""LLM interface and prompt rendering.

Deliberately thin: one `complete_json` method. Swapping providers or models must not touch
pipeline logic, and the components must stay separable so Planner/Solver/Curator can run on different
models (a common and useful ablation -- a small planner with a large solver).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class LLM(Protocol):
    def complete_json(self, prompt: str, *, component: str,
                      image: bytes | None = None) -> dict:
        """Return the parsed JSON object the component's prompt asked for."""
        ...


def load_prompt(name: str, prompt_dir: str | Path | None = None) -> str:
    """Read a prompt template.

    `prompt_dir` selects an alternative prompt set, so a revised set can be A/B'd against
    the shipped one without editing files in place. A relative path resolves against the
    repo root. Defaults to `prompts/`.
    """
    base = PROMPT_DIR if prompt_dir is None else Path(prompt_dir)
    if not base.is_absolute():
        base = PROMPT_DIR.parent / base
    path = base / name
    if not path.is_file():
        raise FileNotFoundError(
            "prompt %s not found in %s; a run with a missing prompt would silently render "
            "an empty template" % (name, base))
    return path.read_text(encoding="utf-8")


def render(template: str, **slots: object) -> str:
    """Fill {{slot}} placeholders. Missing slots become empty rather than raising --
    an unfilled optional slot should not abort a 12k-item run."""
    out = template
    for key, value in slots.items():
        out = out.replace("{{" + key + "}}", "" if value is None else str(value))
    return re.sub(r"\{\{[a-z_]+\}\}", "", out)


# A backslash that JSON does not allow: not one of the seven legal escapes, or a \u that
# is not followed by four hex digits. Models writing LaTeX inside a JSON string emit these
# constantly ("\dfrac", "\(", "\usepackage"), and json.loads rejects the whole object.
_BAD_ESCAPE_RE = re.compile(r'\\(?:u(?![0-9a-fA-F]{4})|[^"\\/bfnrtu])')


def _loads(text: str) -> dict:
    """Strict JSON first; on failure retry once with invalid backslash escapes doubled.

    The repair only ever runs on text that already failed `json.loads`, so it can widen
    what parses but can never change the result of a parse that previously succeeded.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _BAD_ESCAPE_RE.sub(lambda m: "\\" + m.group(0), text)
        if repaired == text:
            raise
        return json.loads(repaired)


def extract_json(text: str) -> dict:
    """Recover a JSON object from model output.

    Models emit fences and preambles despite instructions not to; failing the item over
    formatting would confound the actual measurement.
    """
    text = (text or "").strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return _loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return _loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in model output")


class EchoLLM:
    """Offline stub for smoke-testing the pipeline with no API access.

    Returns schema-valid, deliberately unimpressive output. It exercises every code path
    -- parsing, guard, consolidation, evidence -- without pretending to solve anything.

    `tool_code` makes it request one execution before answering, which is the only way to
    exercise the Solver tool loop without a provider. The stub emits the tool action on its
    first Solver turn and the answer on every turn after, so the loop terminates whatever the
    budget is.
    """

    def __init__(self, answer: str = "A", tool_code: str | None = None,
                 claim_result: str | None = None) -> None:
        self.answer = answer
        self.tool_code = tool_code
        self.claim_result = claim_result
        self.calls: list[str] = []
        self._solver_turns = 0

    def complete_json(self, prompt: str, *, component: str,
                      image: bytes | None = None) -> dict:
        self.calls.append(component)
        if component == "planner":
            # Planner marks the start of a new item, which is what resets the per-item tool
            # budget. Without this the stub emits a tool action on the first item of a run
            # and never again, and a multi-item smoke test silently exercises the tool
            # loop once instead of every step.
            self._solver_turns = 0
            return {
                "semantic_context": "placeholder problem type from the echo stub",
                "visual_context": None,
                "domain": "other",
                "tags": ["strategy.casework"],
                "retrieval_query": "placeholder problem type | domain: other",
                "tool_expected": self.tool_code is not None,
                "reason": "stub",
            }
        if component == "solver":
            self._solver_turns += 1
            if self.tool_code is not None and self._solver_turns == 1:
                return {"action": "tool", "tool": "python", "code": self.tool_code,
                        "why": "stub execution"}
            return {
                "action": "answer",
                "reasoning_trajectory": "stub trajectory",
                "coding": "N/A" if self.tool_code is None else self.tool_code,
                "coding_result": ("N/A" if self.claim_result is None
                                  else self.claim_result),
                "reference_verdict": {"used": [], "unused": []},
                "answer": self.answer,
                "confidence": 0.1,
            }
        if component == "curator":
            return {
                "verification": {
                    "correct": False, "reasoning_sound": False,
                    "verdict_source": "signal", "reason": "stub",
                    "root_cause": "conceptual_gap",
                },
                "attribution": {
                    "used_positive": [], "used_negative": [],
                    "unused_irrelevant": [], "unused_redundant": [],
                },
                "lesson": "none",
                "sufficient": 1,
                "proposed_entries": [],
            }
        raise ValueError(f"unknown component: {component}")
