"""Answer normalization and matching for the six target datasets.

Answer matching is where benchmark harnesses quietly lose points: a correct `204` scored
wrong because the model wrote `\\boxed{204}`, or `C` scored wrong because it wrote
`(C) 3.14`. Every rule here is deliberately conservative -- it strips presentation, never
content.
"""

from __future__ import annotations

import re
from typing import Any

BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
# A-Z, not A-J: HLE multiple choice runs to 22 options. `normalize_mcq` still rejects any
# letter past `n_options`, so datasets with <= 10 options score exactly as before.
LETTER_ONLY_RE = re.compile(r"^\s*\(?([A-Z])\)?\s*[.:)]?\s*$", re.IGNORECASE)
LEADING_LETTER_RE = re.compile(r"^\s*\(?([A-Z])\)?\s*[.:)]\s+", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:[\d,]*\d)?(?:\.\d+)?")
ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*", re.IGNORECASE
)


def strip_presentation(text: str) -> str:
    """Remove LaTeX/markdown wrapping and 'the answer is' framing."""
    s = str(text).strip()
    m = BOXED_RE.search(s)
    if m:
        s = m.group(1)
    s = s.replace("$", "").replace("\\!", "").replace("\\,", "")
    s = s.strip().strip("`").strip()
    s = ANSWER_PREFIX_RE.sub("", s)
    s = s.strip().rstrip(".").strip()
    return s


def normalize_mcq(text: str, n_options: int = 10) -> str | None:
    """Extract a single option letter. Returns None when no unambiguous letter is found."""
    s = strip_presentation(text)
    last = chr(ord("A") + n_options - 1)

    m = LETTER_ONLY_RE.match(s)
    if m:
        letter = m.group(1).upper()
        return letter if letter <= last else None

    m = LEADING_LETTER_RE.match(s)     # "C) 3.14" -- letter plus the option's content
    if m:
        letter = m.group(1).upper()
        return letter if letter <= last else None

    # Fall back to a lone bracketed letter anywhere, e.g. "... so (B)."
    bracketed = re.findall(r"\(([A-J])\)", s.upper())
    if len(set(bracketed)) == 1:
        letter = bracketed[0]
        return letter if letter <= last else None
    return None


# Options as the datasets render them: "(A) text", "A) text", "A. text", one per line,
# under an "Options:" or "Choices:" heading when there is one.
_OPTION_LINE_RE = re.compile(r"^\(?([A-Z])[\)\.]\s*(.+)$", re.MULTILINE)


def option_texts(question: str) -> dict:
    """letter -> option text, parsed from the question's rendered choices.

    Restricted to the block after `Options:` / `Choices:` when one exists, because `(A)`
    also appears in ordinary prose and in chemistry stereo-descriptors.
    """
    if not question:
        return {}
    low = question.lower()
    body = question
    for marker in ("options:", "choices:"):
        if marker in low:
            body = question[low.index(marker) + len(marker):]
            break
    return {m.group(1).upper(): m.group(2).strip()
            for m in _OPTION_LINE_RE.finditer(body)}


def mcq_option_text_match(pred: str, gold_letter: str, question: str) -> bool:
    """Whether `pred` names the gold option by its TEXT rather than its letter.

    This mirrors `eval_for_multiple_choice` in Dynamic Cheatsheet, deliberately, so the two
    systems score multiple choice by the same standard and their accuracies are comparable.
    DC tests `gold_option_text in cleaned_answer`, i.e. substring containment, and that is
    reproduced here rather than tightened.

    The leniency is real and worth stating: a gold option of "4" is contained in an answer
    of "14", so a short numeric option can be credited on a wrong answer. Tightening it
    would make LeRe stricter than the baseline it is measured against, which would
    understate LeRe rather than inform anything, so the counters in `pipeline` record how
    often this path fires instead.
    """
    if not pred or not gold_letter or not question:
        return False
    text = (option_texts(question).get(gold_letter.upper()) or "").strip().lower()
    if not text:
        return False
    cleaned = pred.strip().lower().replace("`", "").replace("(", "").replace(")", "")
    return text in cleaned


def normalize_integer(text: str) -> int | None:
    """Extract an integer. AIME answers are 0-999, so leading zeros are stripped."""
    s = strip_presentation(text)
    m = NUMBER_RE.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        return int(float(raw)) if "." in raw else int(raw)
    except ValueError:
        return None


def normalize_float(text: str) -> float | None:
    s = strip_presentation(text)
    m = NUMBER_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# ------------------------------------------------------------------ MATH (LaTeX answers)

def math_500_equiv(pred: str, gold: str) -> bool:
    """MATH-500 answer equivalence. VERBATIM port of Dynamic Cheatsheet's
    `eval_for_math_500` so the two frameworks score MATH identically:

      1. strip all whitespace          (LaTeX spacing variants)
      2. strip \\text{...} wrappers   (gold \\text{Evelyn} vs pred Evelyn)
      3. strip degree / percent units  (90^\\circ vs 90, 10\\% vs 10)
      4. numeric float comparison      (8 vs 8.0, .35 vs 0.35)

    Deliberately NOT a symbolic (sympy) equivalence: DC's standard is string-level, and
    matching it matters more than catching \\frac{1}{2} == 0.5, which neither side does.
    """
    def normalize(x: str) -> str:
        return re.sub(r"\s+", "", (x or "").strip())

    def strip_text_cmd(x: str) -> str:
        return re.sub(r"\\text\{([^}]*)\}", lambda m: m.group(1), x)

    def strip_units(x: str) -> str:
        return re.sub(r"\^\{?\\circ\}?|\\circ|\\degree|\\%", "", x)

    def try_numeric(a: str, b: str) -> bool:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False

    n_p, n_g = normalize(str(pred)), normalize(str(gold))
    if n_p == n_g:
        return True
    if normalize(strip_text_cmd(n_p)) == normalize(strip_text_cmd(n_g)):
        return True
    if normalize(strip_units(n_p)) == normalize(strip_units(n_g)):
        return True
    return try_numeric(n_p, n_g)


def is_correct(pred: Any, gold: Any, answer_type: str,
               n_options: int = 10, rel_tol: float = 1e-3,
               question: str | None = None) -> bool:
    """Compare a prediction against gold under the dataset's answer convention.

    `question` enables the multiple-choice option-text fallback: a solver that answers
    "97" instead of "(A)" on an item whose first option is 97 is credited, matching
    Dynamic Cheatsheet's scoring. Without it, letters are the only accepted form, which
    is stricter than the baseline. See `mcq_option_text_match`.
    """
    if pred is None or gold is None:
        return False

    if answer_type == "math":
        return math_500_equiv(str(pred), str(gold))
    if answer_type == "mcq_letter":
        p = normalize_mcq(str(pred), n_options)
        g = normalize_mcq(str(gold), n_options)
        if g is None:
            return False
        if p is not None:
            return p == g
        # No letter in the prediction. Fall back to naming the option by its text.
        return mcq_option_text_match(str(pred), g, question or "")

    if answer_type == "integer":
        p = normalize_integer(str(pred))
        g = normalize_integer(str(gold))
        return p is not None and g is not None and p == g

    if answer_type == "float":
        p = normalize_float(str(pred))
        g = normalize_float(str(gold))
        if p is None or g is None:
            return False
        if g == 0.0:
            return abs(p) <= rel_tol
        return abs(p - g) / abs(g) <= rel_tol

    # free-form: case- and whitespace-insensitive exact match after stripping presentation
    return _canonical_free(str(pred)) == _canonical_free(str(gold))


def _canonical_free(text: str) -> str:
    s = strip_presentation(text).lower()
    return re.sub(r"\s+", " ", s).strip()


def canonical_answer(text: Any, answer_type: str, n_options: int = 10) -> str:
    """The canonical string form, used by the leakage guard and for majority voting."""
    if answer_type == "mcq_letter":
        return normalize_mcq(str(text), n_options) or ""
    if answer_type == "integer":
        v = normalize_integer(str(text))
        return "" if v is None else str(v)
    if answer_type == "float":
        v = normalize_float(str(text))
        return "" if v is None else repr(v)
    return _canonical_free(str(text))
