"""Leakage guard -- the countermeasure that makes streaming scope defensible.

In per-run streaming with ground truth, C3 sees the gold answer while writing entries that
later questions will retrieve. Nothing in the v0 design stopped it from writing the answer
into the book. For five of the six target datasets the answer is a single letter or a small
integer, so this is not a hypothetical failure mode.

Every rejection is returned with a reason so the rejection rate is reportable -- it is
evidence that the mechanism ran, which is what a reviewer will ask for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .answers import canonical_answer
from .schema import ProposedEntry

_WORD_RE = re.compile(r"[a-z0-9]+")

# "the answer is C", "answer: 204", "correct option is (B)"
# A prose answer claim, not any assignment that happens to use one of these nouns.
#
# The earlier form allowed a bare noun with `=`, so it matched `result = 0` inside a
# Python example and rejected the entry as answer leakage. C3's prompt actively asks for
# `tool.*` entries whose example is code, so the guard was rejecting a class of entry the
# design wants. A qualifier ("the", "final", "correct") admits `=`; a bare noun needs a
# prose connector.
_ANSWER_PHRASE_RE = re.compile(
    r"\b(?:the|final|correct)\s+(?:final\s+|correct\s+)?"
    r"(?:answer|option|choice|result)\b\s*(?:is|:|=)\s*\(?[A-J0-9]"
    r"|"
    r"\b(?:answer|option|choice|result)\b\s*(?:is|:)\s+\(?[A-J0-9]",
    re.IGNORECASE,
)

_CODE_MARKER_RE = re.compile(
    r"(?:\bdef\s|\bimport\s|\bprint\(|\breturn\s|\blambda\s|\bfor\s+\w+\s+in\b|;)")


def looks_like_code(text: str) -> bool:
    """Whether an `example` is a program rather than prose.

    `check_entry` already exempts `example` from the answer-LEAK check because worked
    numbers legitimately live there. The same reasoning applies to answer PHRASES: an
    assignment in a code example is not a claim about this problem's answer.
    """
    return bool(_CODE_MARKER_RE.search(text or ""))


@dataclass
class GuardResult:
    accepted: bool
    reasons: list[str]
    ngram_overlap: float = 0.0

    @property
    def reason_str(self) -> str:
        return ",".join(self.reasons) if self.reasons else "ok"


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


# Numbers with two or more digits. Single digits are too common to be evidence: an entry
# may legitimately say "modulo 7" or "the first 10 cases" about a question that also
# contains a 7 or a 10.
# Bounded by digits only, not by word characters: subscript notation is exactly where
# these leak. `$17_b$` must yield "17"; an earlier version treated the `_` as part of the
# token and matched nothing at all in the question, so the check never fired. The `.` in
# the lookbehind keeps `3.75` from contributing a spurious "75".
_DISTINCTIVE_NUMBER_RE = re.compile(r"(?<![\d.])(\d{2,})(?!\d)")


def distinctive_numbers(text: str) -> set:
    return set(_DISTINCTIVE_NUMBER_RE.findall(text or ""))


def shared_numbers(entry_text: str, question_text: str) -> set:
    """Multi-digit literals the entry and the question have in common.

    The n-gram check cannot see this class of restatement. The question writes
    `$17_b$ ... $97_b$` in LaTeX; a proposal wrote "to check if 17_b divides 97_b ...
    then check if 97 % 17 == 0" in prose. Word 8-grams overlap at 0.000 because the
    surface forms differ, and the guard accepted a verbatim copy of the graded item as an
    "illustrative example". The numbers are what survives the paraphrase.
    """
    return distinctive_numbers(entry_text) & distinctive_numbers(question_text)


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap(entry_text: str, question_text: str, n: int = 8) -> float:
    """Fraction of the entry's n-grams that also occur in the source question.

    High overlap means the entry restates the problem instead of abstracting it -- the
    signature of a memorized instance rather than a transferable skill.
    """
    e = _ngrams(_words(entry_text), n)
    if not e:
        return 0.0
    q = _ngrams(_words(question_text), n)
    if not q:
        return 0.0
    return len(e & q) / len(e)


def _contains_answer_token(text: str, answer: str) -> bool:
    """Whole-token containment, so gold '4' does not fire on 'factor' or '2024'."""
    if not answer:
        return False
    ans = answer.strip().lower()
    if not ans:
        return False
    if len(ans) == 1 and ans.isalpha():
        # A single MCQ letter needs a strict context to avoid firing on ordinary prose.
        return bool(re.search(rf"\(\s*{re.escape(ans)}\s*\)", text, re.IGNORECASE)) or bool(
            re.search(rf"\b(?:answer|option|choice)\b[^.\n]{{0,20}}\b{re.escape(ans)}\b",
                      text, re.IGNORECASE)
        )
    return bool(re.search(rf"(?<![\w.]){re.escape(ans)}(?![\w.])", text, re.IGNORECASE))


def check_entry(entry: ProposedEntry, question_text: str, gold_answer: str | None,
                answer_type: str, cfg: dict) -> GuardResult:
    """Screen one proposed entry. Rejection is the safe default for anything ambiguous."""
    reasons: list[str] = []
    entry_text = " ".join([entry.title, *entry.bullets, entry.example])

    if cfg.get("block_answer_leak", True) and gold_answer is not None:
        gold_canon = canonical_answer(gold_answer, answer_type)
        # The example field legitimately contains worked numbers; title and bullets do not.
        head_text = " ".join([entry.title, *entry.bullets])
        if _contains_answer_token(head_text, gold_canon):
            reasons.append("answer_leak")

    if cfg.get("block_answer_phrases", True):
        # Title and bullets are prose and always checked. The example is checked only when
        # it is prose too; see `looks_like_code`.
        phrase_text = " ".join([entry.title, *entry.bullets])
        if not looks_like_code(entry.example):
            phrase_text += " " + entry.example
        if _ANSWER_PHRASE_RE.search(phrase_text):
            reasons.append("answer_phrase")

    n = int(cfg.get("ngram_n", 8))
    overlap = ngram_overlap(entry_text, question_text, n)
    if overlap > float(cfg.get("max_ngram_overlap", 0.35)):
        reasons.append("problem_restatement")

    # Paraphrase-resistant restatement check; see `shared_numbers`.
    if cfg.get("block_shared_numbers", True):
        shared = shared_numbers(entry_text, question_text)
        if len(shared) > int(cfg.get("max_shared_numbers", 1)):
            reasons.append("shared_problem_numbers")

    if not entry.tags:
        reasons.append("no_valid_tags")
    if len(entry.bullets) < 2:
        reasons.append("too_few_bullets")

    return GuardResult(accepted=not reasons, reasons=reasons, ngram_overlap=overlap)
