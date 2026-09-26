"""Typed mirrors of the JSON contracts in schema/ and prompts/.

Anything an LLM emits passes through here before it touches the store. Coercion is
deliberate and always *counted*: silent normalization hides prompt regressions, so every
correction increments a violation counter that the run report surfaces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

# ---------------------------------------------------------------- vocabularies

MATH_DOMAINS = [
    "math.algebra", "math.number_theory", "math.combinatorics", "math.geometry",
    "math.probability", "math.precalculus", "math.calculus", "math.other",
]
BASE_DOMAINS = [
    "biology", "business", "chemistry", "computer_science", "economics",
    "engineering", "health", "history", "law", "philosophy", "physics",
    "psychology", "other",
]
DOMAINS = MATH_DOMAINS + BASE_DOMAINS

TAG_PREFIXES = ("strategy", "tool", "knowledge", "pitfall", "format")
TAG_RE = re.compile(r"^(strategy|tool|knowledge|pitfall|format)\.[a-z0-9_]+$")

# One example per prefix, kept here rather than in `prompts/taxonomy.md` so that the text
# shown to the model and the rule that coerces its output cannot drift apart.
TAG_EXAMPLES = {
    "strategy": ["strategy.casework", "strategy.invariant",
                 "strategy.equation_manipulation", "strategy.backward_induction"],
    "tool": ["tool.sympy", "tool.brute_force_check", "tool.numeric_search",
             "tool.exact_arithmetic"],
    "knowledge": ["knowledge.thermodynamics", "knowledge.stereochemistry"],
    "pitfall": ["pitfall.off_by_one", "pitfall.unit_conversion",
                "pitfall.degenerate_case"],
    "format": ["format.integer_0_999", "format.multiple_choice", "format.boxed"],
}


def vocabulary_block() -> str:
    """The closed vocabularies, rendered for a prompt.

    C1's and C3's prompts used to say "the closed list in `taxonomy.md`" while nothing
    ever put that list in front of the model, so it invented values -- `number_theory`
    for `math.number_theory`, `mathematics` for a domain, bare `divisibility` for a tag --
    every one of which was coerced to `other` or dropped, and an entry whose tags were all
    dropped was then rejected by the guard as `no_valid_tags`. Generated from `DOMAINS`
    and `TAG_RE` so the prompt cannot disagree with the coercion.
    """
    lines = ["`domain` — exactly one of these %d values, copied verbatim:" % len(DOMAINS),
             "", "```"]
    lines.append("  ".join(MATH_DOMAINS))
    for i in range(0, len(BASE_DOMAINS), 4):
        lines.append("  ".join(BASE_DOMAINS[i:i + 4]))
    lines += ["```", "",
              "A bare `math` or `mathematics` is NOT valid: pick the subdomain, or "
              "`math.other`.",
              "Anything outside this list becomes `other`, which makes the entry "
              "unretrievable by domain.",
              "",
              "`tags` — 1 to 4 values, each `prefix.snake_case` with the prefix from this "
              "closed set:", ""]
    for prefix in TAG_PREFIXES:
        lines.append("- `%s.` — e.g. %s"
                     % (prefix, ", ".join("`%s`" % t for t in TAG_EXAMPLES[prefix])))
    lines += ["",
              "The part after the dot is yours to choose but must be `snake_case` "
              "(lowercase letters, digits, underscores).",
              "A tag with no prefix, or a prefix outside that set, is DROPPED. An entry "
              "whose tags are all dropped is rejected."]
    return "\n".join(lines)


# C3's diagnosis of a failure. Closed because it is now load-bearing: it modulates how
# much blame a note takes when the answer was wrong (`curate.apply_attribution`), and a
# free-text value would silently fall through to the default factor.
ROOT_CAUSES = (
    "conceptual_gap", "computational_slip", "misread_question", "bad_reference",
    "format_error", "none",
)


def normalize_root_cause(value, violations=None) -> str:
    if isinstance(value, str):
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        if v in ROOT_CAUSES:
            return v
    if violations is not None:
        violations.bump("root_cause_out_of_vocab")
    return "none"


ATTRIBUTION_BUCKETS = (
    "used_positive", "used_negative", "unused_irrelevant", "unused_redundant",
)

AnswerType = Literal["integer", "float", "mcq_letter", "free"]


class VocabViolations(dict):
    """Counter of coercions applied while parsing model output."""

    def bump(self, key: str, n: int = 1) -> None:
        self[key] = self.get(key, 0) + n


def normalize_domain(value: Any, violations: VocabViolations | None = None) -> str:
    if isinstance(value, str):
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        if v in DOMAINS:
            return v
        # a bare "math" is a common model output; route it to the catch-all subdomain
        if v == "math" or v == "mathematics":
            if violations is not None:
                violations.bump("domain_bare_math")
            return "math.other"
    if violations is not None:
        violations.bump("domain_out_of_vocab")
    return "other"


def normalize_tags(value: Any, violations: VocabViolations | None = None) -> list[str]:
    if not isinstance(value, list):
        if violations is not None:
            violations.bump("tags_not_a_list")
        return []
    out: list[str] = []
    for raw in value:
        if not isinstance(raw, str):
            continue
        t = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if TAG_RE.match(t):
            if t not in out:
                out.append(t)
        elif violations is not None:
            violations.bump("tag_out_of_vocab")
    if len(out) > 4:
        if violations is not None:
            violations.bump("tags_truncated")
        out = out[:4]
    return out


def domain_affinity(a: str, b: str, partial_credit: float) -> float:
    """1.0 exact, `partial_credit` same top-level prefix, 0.0 otherwise."""
    if a == b:
        return 1.0
    top_a = a.split(".", 1)[0]
    top_b = b.split(".", 1)[0]
    if top_a == top_b:
        return partial_credit
    return 0.0


# ---------------------------------------------------------------- memory entry

@dataclass
class EntryMeta:
    created: str = ""
    source_queries: list[str] = field(default_factory=list)
    helpful: float = 0.0
    harmful: float = 0.0
    reliability: float = 0.5
    retrieved_count: int = 0
    last_used_step: int | None = None
    cluster_size: int = 1
    related_ids: list[str] = field(default_factory=list)

    def recompute_reliability(self) -> float:
        """Laplace-smoothed Beta posterior mean; exactly 0.5 with no evidence."""
        self.reliability = (self.helpful + 1.0) / (self.helpful + self.harmful + 2.0)
        return self.reliability

    @property
    def evidence(self) -> float:
        return self.helpful + self.harmful


@dataclass
class SkillEntry:
    id: str
    title: str
    bullets: list[str]
    example: str
    domain: str
    tags: list[str]
    status: str = "active"
    meta: EntryMeta = field(default_factory=EntryMeta)

    def skill_view(self) -> str:
        """The only text Es ever encodes: the entry's identity, without its steps.

        Bullets are excluded on purpose, and the same rendering serves both retrieval and
        consolidation. Two reasons they now agree:

        * Consolidation must not see bullets. Merging exists to union differing steps
          under one concept, so comparing bullet text prevents exactly the merges that
          matter -- two proposals of the same skill each contributing a different step
          score around 0.71, below any merge threshold, and both get written.
        * Retrieval indexes a short key and *presents* the full entry. That is standard
          dense-retrieval practice, and it makes the query and skill towers symmetric at
          concept level, which is the geometry CCQS is training.

        The cost is that this string is short and low-entropy, so sibling entries collide
        under a frozen base encoder. Separating them is precisely Es's job.
        """
        return f"{self.title} | domain: {self.domain} | skills: {', '.join(self.tags)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "SkillEntry":
        meta = EntryMeta(**d.get("meta", {}))
        return SkillEntry(
            id=d["id"], title=d["title"], bullets=list(d["bullets"]),
            example=d.get("example", ""), domain=d["domain"], tags=list(d["tags"]),
            status=d.get("status", "active"), meta=meta,
        )


# ---------------------------------------------------------------- C1 output

@dataclass
class PlannerOutput:
    semantic_context: str
    domain: str
    tags: list[str]
    retrieval_query: str = ""
    visual_context: str | None = None
    reason: str = ""
    tool_expected: bool = False

    def query_view(self) -> str:
        """Text encoded by Ep. Same template shape as SkillEntry.skill_view.

        Built here rather than taken from C1's `retrieval_query`, which is diagnostic
        only. Two reasons, both structural: a deterministic template guarantees Ep and Es
        see the *same* shape of string, which is the symmetry CCQS trains; and a
        model-authored key varies run to run, so `sim_threshold` would be calibrated
        against a moving target.

        `tool_expected` is deliberately absent. Tool intent already reaches retrieval
        through the `tool.*` tags, which are in this string -- adding the flag as well
        would double-count it and change every calibrated cosine for no new signal.
        """
        ctx = self.semantic_context
        if self.visual_context:
            ctx = f"{ctx} {self.visual_context}"
        return f"{ctx} | domain: {self.domain} | skills: {', '.join(self.tags)}"

    @staticmethod
    def parse(d: dict, violations: VocabViolations) -> "PlannerOutput":
        return PlannerOutput(
            semantic_context=str(d.get("semantic_context", "")).strip(),
            domain=normalize_domain(d.get("domain"), violations),
            tags=normalize_tags(d.get("tags"), violations),
            retrieval_query=str(d.get("retrieval_query", "") or "").strip(),
            visual_context=(d.get("visual_context") or None),
            reason=str(d.get("reason", "")).strip(),
            tool_expected=bool(d.get("tool_expected", False)),
        )


# ---------------------------------------------------------------- C2 output

@dataclass
class SolverOutput:
    answer: str
    reasoning_trajectory: str = ""
    coding: str = "N/A"
    coding_result: str = "N/A"
    used: list[dict] = field(default_factory=list)
    unused: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    tool_calls: int = 0
    tool_failures: int = 0
    tool_executed: bool = False   # True once real output replaced the model's claim

    def verdict_ids(self) -> list[str]:
        """Ids the solver named across used + unused, in order, duplicates included."""
        out = []
        for bucket in (self.used, self.unused):
            for entry in bucket:
                mid = entry.get("id")
                if isinstance(mid, str):
                    out.append(mid)
        return out

    @staticmethod
    def parse(d: dict, violations: VocabViolations) -> "SolverOutput":
        verdict = d.get("reference_verdict") or {}
        conf = d.get("confidence", 0.0)
        try:
            conf = min(1.0, max(0.0, float(conf)))
        except (TypeError, ValueError):
            violations.bump("solver_confidence_unparseable")
            conf = 0.0
        return SolverOutput(
            answer=str(d.get("answer", "")).strip(),
            reasoning_trajectory=str(d.get("reasoning_trajectory", "")),
            coding=str(d.get("coding", "N/A")),
            coding_result=str(d.get("coding_result", "N/A")),
            used=[x for x in (verdict.get("used") or []) if isinstance(x, dict)],
            unused=[x for x in (verdict.get("unused") or []) if isinstance(x, dict)],
            confidence=conf,
        )


# ---------------------------------------------------------------- C3 output

def audit_solver_verdict(solver: "SolverOutput", retrieved_ids: list[str],
                         violations: VocabViolations) -> None:
    """Count C2 breaking rule 5 of its own prompt: every note exactly once.

    Nothing downstream depends on the solver's verdict -- C3's attribution is what feeds
    evidence and CCQS labels -- so this cannot be enforced without discarding a solved
    item. It is counted instead: a rise in these numbers is a prompt regression, and
    without a counter it would be invisible.
    """
    named = solver.verdict_ids()
    seen = set()
    for mid in named:
        if mid in seen:
            violations.bump("solver_verdict_duplicate_id")
        seen.add(mid)
        if mid not in retrieved_ids:
            violations.bump("solver_verdict_hallucinated_id")
    for mid in retrieved_ids:
        if mid not in seen:
            violations.bump("solver_verdict_missing_id")


@dataclass
class ProposedEntry:
    title: str
    bullets: list[str]
    example: str
    domain: str
    tags: list[str]

    def skill_view(self) -> str:
        """See SkillEntry.skill_view -- identity without steps, for Es and consolidation."""
        return f"{self.title} | domain: {self.domain} | skills: {', '.join(self.tags)}"


@dataclass
class CuratorOutput:
    correct: bool | None
    reasoning_sound: bool
    verdict_source: str
    reason: str
    root_cause: str
    attribution: dict[str, list[str]]
    lesson: str
    sufficient: int
    proposed_entries: list[ProposedEntry]

    @staticmethod
    def parse(d: dict, retrieved_ids: list[str],
              violations: VocabViolations) -> "CuratorOutput":
        ver = d.get("verification") or {}
        raw_attr = d.get("attribution") or {}

        attribution: dict[str, list[str]] = {b: [] for b in ATTRIBUTION_BUCKETS}
        seen: set[str] = set()
        for bucket in ATTRIBUTION_BUCKETS:
            for item in raw_attr.get(bucket) or []:
                mid = item.get("id") if isinstance(item, dict) else item
                if not isinstance(mid, str):
                    continue
                if mid not in retrieved_ids:
                    violations.bump("attribution_hallucinated_id")   # id never shown to C3
                    continue
                if mid in seen:
                    violations.bump("attribution_duplicate_id")
                    continue
                seen.add(mid)
                attribution[bucket].append(mid)

        # An unclassified retrieved entry defaults to unused_irrelevant, never to a
        # crediting bucket -- omission must not become free positive evidence.
        for mid in retrieved_ids:
            if mid not in seen:
                violations.bump("attribution_missing_id")
                attribution["unused_irrelevant"].append(mid)

        proposals: list[ProposedEntry] = []
        for p in d.get("proposed_entries") or []:
            if not isinstance(p, dict):
                continue
            bullets = [str(b).strip() for b in (p.get("bullets") or [])
                       if isinstance(b, (str, int, float)) and str(b).strip()]
            title = str(p.get("title", "")).strip()
            if len(bullets) < 2 or len(title) < 8:
                violations.bump("proposal_underspecified")
                continue
            proposals.append(ProposedEntry(
                title=title[:120],
                bullets=bullets[:8],
                example=str(p.get("example", "")).strip()[:4000],
                domain=normalize_domain(p.get("domain"), violations),
                tags=normalize_tags(p.get("tags"), violations),
            ))

        sufficient = 1 if d.get("sufficient") in (1, "1", True) else 0
        if sufficient == 1 and proposals:
            violations.bump("sufficient_but_proposed")
            sufficient = 0

        correct = ver.get("correct")
        if not isinstance(correct, bool):
            correct = None

        return CuratorOutput(
            correct=correct,
            reasoning_sound=bool(ver.get("reasoning_sound", True)),
            verdict_source=str(ver.get("verdict_source", "self")),
            reason=str(ver.get("reason", "")),
            root_cause=normalize_root_cause(ver.get("root_cause"), violations),
            attribution=attribution,
            lesson=str(d.get("lesson", "none")),
            sufficient=sufficient,
            proposed_entries=proposals,
        )
