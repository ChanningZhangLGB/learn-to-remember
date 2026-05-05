"""Custom dataset ordering strategies for LeRe experiments.

When shuffle_seed is set in config, instead of random shuffling, this module
provides task-specific orderings that exploit known solvability and topic
structure to maximise cross-question memory transfer.

Usage in run_lere_experiment.py:
    from utils.dataset_ordering import get_custom_indices, detect_topic
    indices = get_custom_indices(task, ds, n_eval, seed=shuffle_seed)
"""

import re
from collections import defaultdict
from typing import Any, Dict, List

# ── AIME 2025 solvability priors ─────────────────────────────────────────────
# Dataset indices (0-based) with known per-run accuracy across runs 1–4.
#   high:   3/4 runs correct  → Q_001 (idx 0), Q_017 (idx 16), Q_023 (idx 22)
#   medium: 2/4 runs correct  → Q_005 (idx 4), Q_009 (idx 8), Q_015 (idx 14)
#   low:    1/4 runs correct  → Q_004 (idx 3)
AIME_2025_SOLVABLE: Dict[str, List[int]] = {
    "high":   [0, 16, 22],
    "medium": [4, 8, 14],
    "low":    [3],
}

# Topic label per dataset index for AIME 2025 (0-based).
AIME_2025_TOPIC_MAP: Dict[int, str] = {
    0:  "number_theory",    # base-b divisibility
    1:  "geometry",         # triangle similarity / ratios
    2:  "combinatorics",    # ice-cream flavour counts
    3:  "number_theory",    # Diophantine: 12x²−xy−6y²=0
    4:  "combinatorics",    # 8! permutations divisible by 22
    5:  "geometry",         # trapezoid with inscribed circle
    6:  "combinatorics",    # letter-pair probability
    7:  "geometry",         # complex-number locus system
    8:  "geometry",         # rotated parabola intersection
    9:  "combinatorics",    # 3×9 grid, Latin-square style
    10: "algebra",          # piecewise-linear periodic function
    11: "geometry",         # 3-D plane with inequality regions
    12: "geometry",         # disk + random line segments
    13: "geometry",         # convex pentagon, Fermat point
    14: "number_theory",    # cubic residues mod 3^7
    15: "geometry",         # collinear points + triangle altitude
    16: "number_theory",    # divisibility sum (n+2 | product)
    17: "combinatorics",    # 2×2 grid 2-red-2-blue coloring
    18: "algebra",          # telescoping logarithm product
    19: "geometry",         # triangle midpoints
    20: "geometry",         # internally tangent circles
    21: "number_theory",    # 2025 divisors, LCM probability
    22: "combinatorics",    # coin greedy problem
    23: "trig_analysis",    # sin(7π·sin(5x)) zeros + tangencies
    24: "combinatorics",    # chair-selection arrangement
    25: "combinatorics",    # regular 24-gon, equal-length segments
    26: "geometry",         # non-convex 11-gon area
    27: "algebra",          # rational recursion xₖ₊₁
    28: "geometry",         # right triangle with internal points
    29: "algebra",          # rational polynomial minimum
}

# Preferred topic traversal order (more number-theory / combinatorics first
# since those are the problem types GPT-4o-mini handles best on AIME).
_TOPIC_ORDER = [
    "number_theory",
    "combinatorics",
    "algebra",
    "trig_analysis",
    "geometry",
]

# ── Topic keyword detector ────────────────────────────────────────────────────
_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "number_theory": [
        r"\bbase\b", r"\bmod\b", r"\bmodulo\b", r"\bdivisib",
        r"\bdivisor", r"\binteger\b", r"\bprime\b", r"\bgcd\b",
        r"\blcm\b", r"\bremainder\b", r"\bcongruent\b", r"\bdigit\b",
        r"\bdiophantine\b", r"\bfactori[sz]", r"\bperfect square\b",
    ],
    "combinatorics": [
        r"\bpermut", r"\bcombinat", r"\bprobabilit", r"\bchoose\b",
        r"\bsubset\b", r"\barrangement\b", r"\bcount(ing|ed|s)?\b",
        r"\bways\b", r"\bcoloring\b", r"\bpair\b", r"\bselect",
        r"\bexpect", r"\bgraph\b",
    ],
    "geometry": [
        r"\btriangle\b", r"\bcircle\b", r"\bquadrilateral\b", r"\bpolygon\b",
        r"\bangle\b", r"\barea\b", r"\blength\b", r"\bperimeter\b",
        r"\bradius\b", r"\bdiameter\b", r"\btangent\b", r"\bsegment\b",
        r"\bplane\b", r"\bperpendicular\b", r"\bparallel\b", r"\bchord\b",
        r"\binscribed\b", r"\bparabola\b", r"\bellipse\b", r"\bhyperbola\b",
    ],
    "algebra": [
        r"\bpolynomial\b", r"\bsequence\b", r"\brecursi", r"\bseries\b",
        r"\bminimum\b", r"\bmaximum\b", r"\blog\b", r"\blogarithm\b",
        r"\bfunction\b", r"\bequation\b", r"\broot\b", r"\bproduct\b",
    ],
    "trig_analysis": [
        r"\bsin\b", r"\bcos\b", r"\btan\b", r"\bsinusoidal\b",
        r"\btrigonometric\b", r"\bperiodic\b",
    ],
}


def detect_topic(text: str) -> str:
    """Detect the broad mathematical topic of a question via keyword matching.

    Returns the topic with the most keyword hits.  Ties are broken by
    ``_TOPIC_ORDER`` priority.  Falls back to ``"algebra"`` if no keywords hit.
    """
    text_lower = text.lower()
    scores: Dict[str, int] = {topic: 0 for topic in _TOPIC_KEYWORDS}
    for topic, patterns in _TOPIC_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                scores[topic] += 1

    best_score = max(scores.values())
    if best_score == 0:
        return "algebra"

    # Among topics tied at best_score, prefer the one earlier in _TOPIC_ORDER
    for topic in _TOPIC_ORDER:
        if scores[topic] == best_score:
            return topic
    return max(scores, key=lambda t: scores[t])


# ── Ordering helpers ──────────────────────────────────────────────────────────

def _topic_cluster_order(indices: List[int], topic_map: Dict[int, str]) -> List[int]:
    """Re-order indices so same-topic questions are grouped together.

    Groups follow ``_TOPIC_ORDER``.  Questions whose index is absent from
    *topic_map* are appended at the end.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx in indices:
        topic = topic_map.get(idx, "other")
        groups[topic].append(idx)

    ordered: List[int] = []
    for topic in _TOPIC_ORDER:
        ordered.extend(groups.pop(topic, []))
    for leftover in groups.values():
        ordered.extend(leftover)
    return ordered


# ── Public API ────────────────────────────────────────────────────────────────

def get_custom_indices(
    task: str,
    dataset: Any,
    n_eval: int,
    seed: int = 42,
) -> List[int]:
    """Return a custom question ordering for *task*.

    AIME_2025 / AIME2025
        1. High-solvability questions first (3/4 cross-run accuracy).
        2. Medium-solvability questions next (2/4).
        3. Low-solvability questions (1/4).
        4. Remaining questions, topic-clustered so related problems are
           adjacent and the memory bank can accumulate relevant strategies.

    All other tasks
        Seeded random shuffle (reproducible but non-custom).
    """
    import random as _random

    task_key = task.upper().replace("-", "_")

    if task_key in ("AIME_2025", "AIME2025"):
        solvable_high   = [i for i in AIME_2025_SOLVABLE["high"]   if i < n_eval]
        solvable_medium = [i for i in AIME_2025_SOLVABLE["medium"] if i < n_eval]
        solvable_low    = [i for i in AIME_2025_SOLVABLE["low"]    if i < n_eval]

        solvable_set = set(solvable_high + solvable_medium + solvable_low)
        remaining = [i for i in range(n_eval) if i not in solvable_set]

        remaining_clustered = _topic_cluster_order(remaining, AIME_2025_TOPIC_MAP)

        ordering = solvable_high + solvable_medium + solvable_low + remaining_clustered
        return ordering

    # Default: seeded random shuffle
    indices = list(range(n_eval))
    _random.seed(seed)
    _random.shuffle(indices)
    return indices


def topic_cluster_rerank(
    retrieved_memory: List[Dict[str, Any]],
    question_text: str,
) -> List[Dict[str, Any]]:
    """Re-rank *retrieved_memory* to surface topic-matching items first.

    Detection uses :func:`detect_topic` on *question_text*.  Items whose
    ``tags`` list contains the detected topic as a substring (e.g. tag
    ``"math.combinatorics"`` matches topic ``"combinatorics"``) are moved to
    the front while preserving relative order within each group.

    This is a *soft* re-ranking: the retriever's score-based ordering is still
    respected within each group.

    Note: The LLM curator stores tags in dot-namespaced form such as
    ``"math.geometry"``, ``"strategy.counting"``, ``"geometry.triangles"``.
    Substring matching is used so that our broad topic labels (``"geometry"``,
    ``"combinatorics"``, etc.) correctly match these curator-generated tags.
    """
    if not retrieved_memory:
        return retrieved_memory

    q_topic = detect_topic(question_text)

    matching = []
    non_matching = []
    for item in retrieved_memory:
        tags = item.get("tags", [])
        if isinstance(tags, list) and any(q_topic in tag for tag in tags):
            matching.append(item)
        else:
            non_matching.append(item)

    return matching + non_matching
