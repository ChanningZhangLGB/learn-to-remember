"""Tests for the mechanisms v0 left undefined.

Each test names the v0 gap it closes, so a regression here is traceable to a design
decision rather than just a red line.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lere.answers import canonical_answer, is_correct, normalize_integer, normalize_mcq  # noqa: E402
from lere.ccqs import CCQSTrainer  # noqa: E402
from lere.curate import apply_attribution, consolidate_and_write, nearest_entry  # noqa: E402
from lere.embed import HashingEncoder, DualEncoder  # noqa: E402
from lere.guard import (check_entry, distinctive_numbers,  # noqa: E402
                        looks_like_code, ngram_overlap, shared_numbers)
from lere.llm import EchoLLM, extract_json, load_prompt  # noqa: E402
from lere.datasets import (DATASET_DIR, DatasetError, dump_jsonl,  # noqa: E402
                           load_aime, load_aime_2024, load_aime_2025,
                           load_gpqa_diamond, load_jsonl,
                           load_mathvista_testmini_250)
from lere.pipeline import Item, Pipeline  # noqa: E402
from lere.providers import (ComponentSpec, ParseFailure, ProviderError,  # noqa: E402
                            ProviderLLM, TransientProviderError, TransportReply,
                            image_data_uri, read_key_file, resolve_api_key,
                            scrub_secret, sniff_mime)
from lere.retrieve import Retriever  # noqa: E402
from lere.trace import (CallRecorder, JsonlWriter, TracedLLM,  # noqa: E402
                        TracingRetriever, head_stats, head_weights)
from lere.schema import (DOMAINS, ROOT_CAUSES, TAG_PREFIXES, CuratorOutput,  # noqa: E402
                         EntryMeta, PlannerOutput, ProposedEntry, SkillEntry,
                         VocabViolations, domain_affinity, normalize_domain,
                         normalize_root_cause, normalize_tags, vocabulary_block)
from lere.store import SkillBook  # noqa: E402
from lere.tools import (ToolConfig, ToolResult, ToolTranscript,  # noqa: E402
                        probe_modules, run_python)
from lere.verify import (AUTHORITATIVE_SOURCES, VerificationSignal,  # noqa: E402
                         resolve_verdict, signal_from_consistency, signal_from_exec,
                         signal_from_gt, signal_from_judge)

CFG = {
    "run": {"batch_size": 2, "write_enabled": True, "seed": 0},
    "retrieval": {"top_k": 3, "sim_threshold": 0.35, "domain_filter": "soft",
                  "domain_penalty": 0.25, "domain_partial_credit": 0.6,
                  "alpha": 0.7, "mmr_lambda": 0.7},
    "verification": {"source": "gt", "judge_confidence_cap": 0.6,
                     "used_but_wrong_factor": 0.5,
                     "gains": {"gt": {"positive": 1.0, "negative": 1.0},
                               "consistency": {"positive": 0.6, "negative": 1.0},
                               "judge": {"positive": 0.3, "negative": 0.6}}},
    "curation": {"merge_threshold": 0.90, "link_threshold": 0.80,
                 "max_bullets_after_merge": 8, "max_proposals_per_query": 3},
    "guard": {"max_ngram_overlap": 0.35, "ngram_n": 8,
              "block_answer_leak": True, "block_answer_phrases": True,
              "block_shared_numbers": True, "max_shared_numbers": 1},
    "pruning": {"max_entries": 512, "quarantine_reliability": 0.25,
                "quarantine_min_evidence": 4.0},
}



CFG_EXEC = {"gains": {"exec": {"positive": 0.5, "negative": 1.0}},
            "used_but_wrong_factor": 0.5}

@pytest.fixture
def book() -> SkillBook:
    return SkillBook(encoder=DualEncoder(HashingEncoder(dim=256)))


def make_proposal(title: str, bullets: list[str], domain: str = "math.number_theory",
                  tags: list[str] | None = None) -> ProposedEntry:
    return ProposedEntry(title=title, bullets=bullets, example="minimal example",
                         domain=domain, tags=tags or ["strategy.modular_arithmetic"])


# ----------------------------------------------------- answer normalization

class TestAnswers:
    def test_boxed_integer(self):
        assert normalize_integer(r"\boxed{204}") == 204
        assert normalize_integer("The answer is 204.") == 204
        assert normalize_integer("007") == 7          # AIME leading zeros

    def test_mcq_variants(self):
        assert normalize_mcq("C") == "C"
        assert normalize_mcq("(C)") == "C"
        assert normalize_mcq("C) 3.14") == "C"
        assert normalize_mcq("answer: b") == "B"

    def test_mcq_respects_option_count(self):
        assert normalize_mcq("J", n_options=10) == "J"
        assert normalize_mcq("J", n_options=4) is None   # GPQA-Diamond has A-D only

    def test_float_tolerance(self):
        assert is_correct("3.1416", "3.14159", "float", rel_tol=1e-3)
        assert not is_correct("3.20", "3.14159", "float", rel_tol=1e-3)

    def test_presentation_does_not_lose_points(self):
        assert is_correct(r"\boxed{42}", "42", "integer")
        assert is_correct("The final answer is (D)", "D", "mcq_letter")


# --------------------------------------------------- vocabulary normalization

class TestVocabulary:
    def test_bare_math_routed_not_dropped(self):
        v = VocabViolations()
        assert normalize_domain("Mathematics", v) == "math.other"
        assert v["domain_bare_math"] == 1

    def test_out_of_vocab_domain_becomes_other(self):
        v = VocabViolations()
        assert normalize_domain("astrophysics", v) == "other"
        assert v["domain_out_of_vocab"] == 1

    def test_unprefixed_tags_dropped_and_counted(self):
        v = VocabViolations()
        assert normalize_tags(["strategy.casework", "geometry"], v) == ["strategy.casework"]
        assert v["tag_out_of_vocab"] == 1

    def test_domain_affinity_is_prefix_aware(self):
        assert domain_affinity("math.geometry", "math.geometry", 0.6) == 1.0
        assert domain_affinity("math.geometry", "math.algebra", 0.6) == 0.6
        assert domain_affinity("math.geometry", "chemistry", 0.6) == 0.0


# -------------------------------------------------------- reliability update
# v0 gap: reliability was initialized to 0.5 and had no update rule.

class TestReliability:
    def test_zero_evidence_is_exactly_half(self):
        assert EntryMeta().recompute_reliability() == 0.5

    def test_moves_with_evidence_and_stays_bounded(self):
        m = EntryMeta(helpful=3.0, harmful=0.0)
        assert m.recompute_reliability() == pytest.approx(4 / 5)
        m = EntryMeta(helpful=0.0, harmful=3.0)
        assert m.recompute_reliability() == pytest.approx(1 / 5)

    def test_store_ignores_authored_reliability(self, book, tmp_path):
        # A model emitting reliability: 0.99 on an entry with 4.0 harm must not be believed.
        book.entries["m_001"] = SkillEntry(
            id="m_001", title="a title here", bullets=["aa bb", "cc dd"], example="",
            domain="physics", tags=["strategy.x"],
            meta=EntryMeta(helpful=0.0, harmful=4.0, reliability=0.99))
        path = tmp_path / "book.json"
        book.save(path)
        reloaded = SkillBook.load(path, book.encoder)
        assert reloaded.entries["m_001"].meta.reliability == pytest.approx(1 / 6)


# ------------------------------------------------------------------ retrieval

class TestRetrieval:
    def test_cold_start_returns_nothing(self, book):
        plan = PlannerOutput("counting lattice paths", "math.combinatorics",
                             ["strategy.casework"])
        assert Retriever(book, CFG["retrieval"]).retrieve(plan, 0) == []

    def test_relevance_floor_blocks_irrelevant_match(self, book):
        book.create(make_proposal("Stereochemistry priority rules",
                                  ["Assign CIP priorities", "Compare substituents"],
                                  domain="chemistry", tags=["knowledge.stereochemistry"]),
                    "Q_001", 0)
        plan = PlannerOutput("counting lattice paths under a divisibility constraint",
                             "math.combinatorics", ["strategy.casework"])
        # Nothing relevant exists, so returning nothing beats returning the least-bad entry.
        assert Retriever(book, CFG["retrieval"]).retrieve(plan, 1) == []

    def test_hard_domain_filter_drops_mismatch(self, book):
        book.create(make_proposal("Modular arithmetic for last digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_001", 0)
        plan = PlannerOutput("modular arithmetic for last digits", "chemistry",
                             ["strategy.modular_arithmetic"])
        cfg = {**CFG["retrieval"], "domain_filter": "hard", "sim_threshold": 0.0}
        assert Retriever(book, cfg).retrieve(plan, 1) == []

    def test_quarantined_entries_are_not_retrieved(self, book):
        e = book.create(make_proposal("Modular arithmetic for last digits",
                                      ["Reduce mod 10 early", "Use Euler totient"]),
                        "Q_001", 0)
        plan = PlannerOutput("modular arithmetic for last digits", "math.number_theory",
                             ["strategy.modular_arithmetic"])
        cfg = {**CFG["retrieval"], "sim_threshold": 0.0}
        assert len(Retriever(book, cfg).retrieve(plan, 1)) == 1
        e.status = "quarantined"
        assert Retriever(book, cfg).retrieve(plan, 2) == []

    def test_retrieve_does_not_write_during_the_read_phase(self, book):
        """SPEC section 9 promises the read phase sees a frozen snapshot.

        `retrieved_count` and `last_used_step` are writes. Incrementing them inside
        `retrieve()` is a read-modify-write against the shared book, which is a race the
        moment items run in parallel -- and MMLU-Pro (~12k items) requires that. The
        pipeline records the bookkeeping at the batch boundary instead.
        """
        e = book.create(make_proposal("Modular arithmetic for last digits",
                                      ["Reduce mod 10 early", "Use Euler totient"]),
                        "Q_001", 0)
        plan = PlannerOutput("modular arithmetic for last digits", "math.number_theory",
                             ["strategy.modular_arithmetic"])
        refs = Retriever(book, {**CFG["retrieval"], "sim_threshold": 0.0}).retrieve(plan, 7)
        assert len(refs) == 1
        assert e.meta.retrieved_count == 0 and e.meta.last_used_step is None

    def test_pipeline_records_bookkeeping_at_the_batch_boundary(self, book):
        book.create(make_proposal("Modular arithmetic for last digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_001", 0)
        cfg = {**CFG, "run": {**CFG["run"], "reset_per_run": False},
               "retrieval": {**CFG["retrieval"], "sim_threshold": 0.0}}
        items = [Item(id="Q_002", question="A question?", answer_type="mcq_letter",
                      gold="A", n_options=4)]
        Pipeline(book, EchoLLM(answer="A"), cfg).run(items)
        entry = book.entries["m_001"]
        assert entry.meta.retrieved_count == 1 and entry.meta.last_used_step == 0


# -------------------------------------------------------------- leakage guard
# v0 gap: nothing prevented an entry from encoding the gold answer.

class TestGuard:
    def test_rejects_integer_answer_leak(self):
        p = make_proposal("Sum of divisors shortcut",
                          ["Factor n first", "The result is 204 for this family"])
        r = check_entry(p, "Find the sum of divisors of N.", "204", "integer", CFG["guard"])
        assert not r.accepted and "answer_leak" in r.reasons

    def test_rejects_mcq_answer_phrase(self):
        p = make_proposal("Choosing among close options",
                          ["Eliminate dominated options", "The answer is C in such cases"],
                          domain="physics", tags=["format.multiple_choice"])
        r = check_entry(p, "Which of the following...", "C", "mcq_letter", CFG["guard"])
        assert not r.accepted
        assert {"answer_leak", "answer_phrase"} & set(r.reasons)

    def test_rejects_problem_restatement(self):
        question = ("Let A B C be a triangle inscribed in a circle of radius thirteen "
                    "with tangents meeting at point D and line A D intersecting the "
                    "circle again at point P where the ratio is required")
        p = make_proposal(
            "Triangle inscribed in a circle of radius thirteen with tangents",
            ["Let A B C be a triangle inscribed in a circle of radius thirteen with "
             "tangents meeting at point D and line A D intersecting the circle again "
             "at point P where the ratio is required",
             "Then compute the ratio"],
            domain="math.geometry", tags=["strategy.power_of_a_point"])
        r = check_entry(p, question, "113", "integer", CFG["guard"])
        assert not r.accepted and "problem_restatement" in r.reasons

    def test_accepts_transferable_entry(self):
        p = make_proposal("Power of a point for tangent-chord configurations",
                          ["Identify the tangent point and the secant through it",
                           "Apply the power of a point relation before assigning coordinates"],
                          domain="math.geometry", tags=["strategy.power_of_a_point"])
        r = check_entry(p, "Let ABC be inscribed in circle omega with radius 13...",
                        "113", "integer", CFG["guard"])
        assert r.accepted, r.reasons

    def test_single_letter_gold_does_not_fire_on_prose(self):
        # Gold "A" must not reject every entry containing the article "a".
        p = make_proposal("Balancing redox half reactions",
                          ["Split into a reduction and an oxidation half reaction",
                           "Balance oxygen with water and hydrogen with protons"],
                          domain="chemistry", tags=["strategy.stoichiometry"])
        r = check_entry(p, "Which of the following is the balanced equation?", "A",
                        "mcq_letter", CFG["guard"])
        assert r.accepted, r.reasons

    def test_ngram_overlap_bounds(self):
        assert ngram_overlap("", "anything") == 0.0
        text = "one two three four five six seven eight nine"
        assert ngram_overlap(text, text, 8) == pytest.approx(1.0)


# ---------------------------------------------------------- credit assignment
# v0 gap: C3's positive/negative verdict was consumed by nothing.

class TestCreditAssignment:
    def _curation(self, positive=(), negative=()) -> CuratorOutput:
        return CuratorOutput(
            correct=None, reasoning_sound=True, verdict_source="signal", reason="",
            root_cause="none",
            attribution={"used_positive": list(positive), "used_negative": list(negative),
                         "unused_irrelevant": [], "unused_redundant": []},
            lesson="", sufficient=1, proposed_entries=[])

    def test_correct_answer_credits_used_entry(self, book):
        e = book.create(make_proposal("t title here", ["aa bb", "cc dd"]), "Q_001", 0)
        sig = VerificationSignal(True, 1.0, "gt")
        apply_attribution(book, self._curation(positive=[e.id]), sig,
                          CFG["verification"], "Q_002", 1)
        assert e.meta.helpful == pytest.approx(1.0)
        assert e.meta.reliability == pytest.approx(2 / 3)

    def test_misleading_entry_takes_full_blame_when_wrong(self, book):
        e = book.create(make_proposal("t title here", ["aa bb", "cc dd"]), "Q_001", 0)
        sig = VerificationSignal(False, 1.0, "gt")
        apply_attribution(book, self._curation(negative=[e.id]), sig,
                          CFG["verification"], "Q_002", 1)
        assert e.meta.harmful == pytest.approx(1.0)

    def test_used_but_wrong_takes_full_blame(self, book):
        e = book.create(make_proposal("t title here", ["aa bb", "cc dd"]), "Q_001", 0)
        sig = VerificationSignal(False, 1.0, "gt")
        apply_attribution(book, self._curation(positive=[e.id]), sig,
                          CFG["verification"], "Q_002", 1)
        assert e.meta.harmful == pytest.approx(1.0)   # v2: confidence alone, no blame factor

    def test_label_free_positive_evidence_is_not_discounted(self, book):
        e = book.create(make_proposal("t title here", ["aa bb", "cc dd"]), "Q_001", 0)
        sig = VerificationSignal(True, 1.0, "consistency")
        apply_attribution(book, self._curation(positive=[e.id]), sig,
                          CFG["verification"], "Q_002", 1)
        assert e.meta.helpful == pytest.approx(1.0)   # v2: symmetric, gains all 1.0

    def test_label_free_negative_evidence_is_not_discounted(self, book):
        e = book.create(make_proposal("t title here", ["aa bb", "cc dd"]), "Q_001", 0)
        sig = VerificationSignal(False, 1.0, "consistency")
        apply_attribution(book, self._curation(negative=[e.id]), sig,
                          CFG["verification"], "Q_002", 1)
        assert e.meta.harmful == pytest.approx(1.0)

    def test_missing_attribution_never_becomes_positive_evidence(self):
        v = VocabViolations()
        out = CuratorOutput.parse(
            {"verification": {"correct": True}, "attribution": {}, "sufficient": 1},
            retrieved_ids=["m_001", "m_002"], violations=v)
        assert out.attribution["unused_irrelevant"] == ["m_001", "m_002"]
        assert out.attribution["used_positive"] == []
        assert v["attribution_missing_id"] == 2

    def test_hallucinated_ids_are_discarded(self):
        v = VocabViolations()
        out = CuratorOutput.parse(
            {"verification": {"correct": True},
             "attribution": {"used_positive": [{"id": "m_999"}]}, "sufficient": 1},
            retrieved_ids=["m_001"], violations=v)
        assert out.attribution["used_positive"] == []
        assert v["attribution_hallucinated_id"] == 1


# ------------------------------------------------------------- consolidation
# v0 gap: cluster_size implied merging, but no merge policy existed.

class TestConsolidation:
    def test_near_duplicate_merges_instead_of_creating(self, book):
        p1 = make_proposal("Modular arithmetic for final digits",
                           ["Reduce mod 10 before expanding",
                            "Use Euler totient for large exponents"])
        book.create(p1, "Q_001", 0)
        p2 = make_proposal("Modular arithmetic for final digits",
                           ["Reduce mod 10 before expanding",
                            "Check whether the modulus is prime first"])
        res = consolidate_and_write(book, [p2], "some unrelated question text", None,
                                    "integer", CFG, "Q_002", 1)
        assert res.merged and not res.created
        entry = book.entries[res.merged[0]]
        assert entry.meta.cluster_size == 2
        assert "Q_002" in entry.meta.source_queries
        assert len(entry.bullets) == 3          # union, not append-everything

    def test_distinct_proposal_creates_new_entry(self, book):
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_001", 0)
        p2 = make_proposal("Power of a point in tangent configurations",
                           ["Identify tangent and secant", "Apply the power relation"],
                           domain="math.geometry", tags=["strategy.power_of_a_point"])
        res = consolidate_and_write(book, [p2], "unrelated question", None, "integer",
                                    CFG, "Q_002", 1)
        assert res.created and not res.merged

    def test_merge_keeps_identity_stable(self, book):
        e = book.create(make_proposal("Modular arithmetic for final digits",
                                      ["Reduce mod 10 early", "Use Euler totient"]),
                        "Q_001", 0)
        original_id, original_title = e.id, e.title
        book.merge(e.id, make_proposal("Modular arithmetic for final digits",
                                       ["Reduce mod 10 early", "New bullet added"]),
                   "Q_002", 1)
        assert e.id == original_id and e.title == original_title

    def test_bullets_capped_on_repeated_merges(self, book):
        e = book.create(make_proposal("Modular arithmetic for final digits",
                                      ["b0 aaa", "b1 bbb"]), "Q_001", 0)
        for i in range(2, 14):
            book.merge(e.id, make_proposal("Modular arithmetic for final digits",
                                           [f"b{i} unique bullet text"]),
                       f"Q_{i:03d}", i, max_bullets=8)
        assert len(e.bullets) == 8

    def test_guard_rejection_prevents_write(self, book):
        p = make_proposal("Leaky entry title", ["Factor first", "The result is 204"])
        res = consolidate_and_write(book, [p], "Find N.", "204", "integer", CFG,
                                    "Q_001", 0)
        assert res.rejected and not res.created and len(book) == 0

    def test_merge_keys_on_concept_not_bullets(self, book):
        """Same skill, different steps, must merge.

        This is the case that keeps bullets out of `skill_view`: with bullets included
        these two score ~0.71 and both get written.
        """
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 before expanding",
                                   "Use Euler totient for large exponents"]), "Q_001", 0)
        p2 = make_proposal("Modular arithmetic for final digits",
                           ["Check whether the modulus is prime first",
                            "Fall back to Carmichael lambda when it is not"])
        _, concept_sim = nearest_entry(book, p2)
        assert concept_sim >= CFG["curation"]["merge_threshold"]
        res = consolidate_and_write(book, [p2], "unrelated question", None, "integer",
                                    CFG, "Q_002", 1)
        assert res.merged and len(book) == 1

    def test_unrelated_titles_in_one_domain_do_not_merge(self, book):
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_001", 0)
        p2 = make_proposal("Bounding solutions with the pigeonhole principle",
                           ["Count the containers before the objects",
                            "State the bound explicitly"],
                           tags=["strategy.pigeonhole"])
        _, sim = nearest_entry(book, p2)
        assert sim < CFG["curation"]["merge_threshold"]
        res = consolidate_and_write(book, [p2], "unrelated question", None, "integer",
                                    CFG, "Q_002", 1)
        assert res.created and len(book) == 2

    def test_nearest_entry_is_domain_scoped(self, book):
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_001", 0)
        p = make_proposal("Modular arithmetic for final digits",
                          ["Reduce mod 10 early", "Use Euler totient"],
                          domain="chemistry", tags=["knowledge.stereochemistry"])
        assert nearest_entry(book, p) == (None, 0.0)


# ------------------------------------------------------ quarantine and pruning

class TestMaintenance:
    def test_quarantine_needs_both_low_reliability_and_evidence(self, book):
        e = book.create(make_proposal("t title here", ["aa bb", "cc dd"]), "Q_001", 0)
        e.meta.harmful, e.meta.helpful = 2.0, 0.0     # reliability 0.25, evidence 2.0
        e.meta.recompute_reliability()
        assert book.quarantine_pass(CFG["pruning"], 1, "Q_002") == []

        e.meta.harmful = 5.0                          # evidence now above the floor
        e.meta.recompute_reliability()
        assert book.quarantine_pass(CFG["pruning"], 2, "Q_003") == [e.id]
        assert e.status == "quarantined"
        assert e.id in book.entries                   # retained for audit, not deleted

    def test_capacity_prune_drops_weakest_first(self, book):
        for i in range(5):
            e = book.create(make_proposal(f"Entry number {i} title",
                                          [f"bullet a{i}", f"bullet b{i}"]),
                            f"Q_{i:03d}", i)
            e.meta.helpful = float(i)
            e.meta.recompute_reliability()
        dropped = book.prune_to_capacity({"max_entries": 3}, 10, "Q_010")
        assert len(dropped) == 2
        assert all(book.entries[e].meta.helpful >= 2.0 for e in book.entries)


# --------------------------------------------------------- verification signal

class TestVerification:
    def test_gt_signal_is_authoritative(self):
        s = signal_from_gt(r"\boxed{204}", "204", "integer")
        assert s.correct and s.confidence == 1.0 and s.source == "gt"

    def test_consistency_confidence_is_the_agreement_rate(self):
        s = signal_from_consistency(["A", "A", "A", "B"], "mcq_letter")
        assert s.confidence == pytest.approx(0.75) and s.correct

    def test_consistency_flags_the_minority_sample(self):
        s = signal_from_consistency(["B", "A", "A", "A"], "mcq_letter")
        assert s.correct is False        # first sample disagrees with the majority

    def test_judge_confidence_is_capped(self):
        s = signal_from_judge(True, 0.99, CFG["verification"])
        assert s.confidence == pytest.approx(0.6)

    def test_gains_are_asymmetric_only_in_label_free_mode(self):
        gt = VerificationSignal(True, 1.0, "gt").gains(CFG["verification"])
        cons = VerificationSignal(True, 1.0, "consistency").gains(CFG["verification"])
        assert gt == (1.0, 1.0)
        assert cons[0] < cons[1]


# ---------------------------------------------------------- encoder invariant

class TestEncoderInvariant:
    def test_heads_start_as_identity_so_untrained_ccqs_is_the_frozen_base(self):
        """The `ccqs.enabled: false` arm must be an EXACT control, not an approximate one.

        Random init would make the untrained arm a different retriever, and any measured
        CCQS effect would confound "training helped" with "the projection changed".
        """
        enc = DualEncoder(HashingEncoder(dim=128))
        text = "identical text on both sides"
        assert float(enc.encode_query(text) @ enc.encode_skill([text])[0]) ==             pytest.approx(1.0, abs=1e-5)

    def test_head_update_invalidates_stored_vectors(self, book):
        """Scoring a fresh query against vectors from an older Es is a silent failure:
        the two sides come from different parameterizations and cosine means nothing."""
        entry = book.create(make_proposal("Modular arithmetic for last digits",
                                          ["reduce early"]), "q1", 0)
        book.vector(entry)
        assert book._vector_version == book.encoder.version
        assert entry.id in book._vectors

        book.encoder.bump_version()                    # what a CCQS update does
        book.vector(entry)                             # next read must re-project
        assert book._vector_version == book.encoder.version

    def test_query_and_skill_views_share_a_template(self):
        plan = PlannerOutput("modular arithmetic for last digits", "math.number_theory",
                             ["strategy.modular_arithmetic"])
        entry = SkillEntry(id="m_001", title="modular arithmetic for last digits",
                           bullets=["reduce early"], example="",
                           domain="math.number_theory",
                           tags=["strategy.modular_arithmetic"])
        for marker in ("| domain:", "| skills:"):
            assert marker in plan.query_view() and marker in entry.skill_view()


# ----------------------------------------------------------------------- CCQS

class TestCCQS:
    def _trainer(self, **over):
        cfg = {"enabled": True, "k_upd": 2, "min_pairs": 1, "batch_size": 8,
               "steps_per_update": 2, "lr": 0.01}
        cfg.update(over)
        return CCQSTrainer(DualEncoder(HashingEncoder(dim=64), proj_dim=32), cfg)

    def test_unused_redundant_is_not_a_negative(self):
        """It means relevant-but-already-covered. Repelling it teaches the wrong geometry."""
        t = self._trainer()
        views = {"m_1": "A | domain: x | skills: t", "m_2": "B | domain: x | skills: t",
                 "m_3": "C | domain: x | skills: t", "m_4": "D | domain: x | skills: t"}
        pair = t.observe("q | domain: x | skills: t", views, {
            "used_positive": ["m_1"], "used_negative": ["m_2"],
            "unused_irrelevant": ["m_3"], "unused_redundant": ["m_4"],
        }, step=0)
        assert pair.positives == [views["m_1"]]
        assert set(pair.hard_negatives) == {views["m_2"], views["m_3"]}
        assert views["m_4"] not in pair.hard_negatives

    def test_no_positive_means_no_pair(self):
        """An all-negative anchor has no InfoNCE numerator; it would only teach repulsion."""
        t = self._trainer()
        views = {"m_1": "A | domain: x | skills: t"}
        assert t.observe("q", views, {"used_negative": ["m_1"]}, step=0) is None
        assert len(t.buffer) == 0

    def test_hallucinated_ids_cannot_enter_the_buffer(self):
        t = self._trainer()
        assert t.observe("q", {}, {"used_positive": ["m_999"]}, step=0) is None

    def test_update_fires_on_schedule_and_moves_the_heads(self):
        t = self._trainer(k_upd=2)
        views = {"m_1": "modular arithmetic | domain: math.number_theory | skills: s.mod",
                 "m_2": "generating functions | domain: math.combinatorics | skills: s.gf"}
        attr = {"used_positive": ["m_1"], "unused_irrelevant": ["m_2"]}
        before = t.encoder.ep.apply(t.encoder.base_vectors(["q | domain: x | skills: t"]))
        for step in range(4):
            t.observe("q | domain: x | skills: t", views, attr, step)
            t.maybe_update(step)
        assert t.stats.updates == 2                     # steps 1 and 3
        assert t.stats.last_loss is not None
        after = t.encoder.ep.apply(t.encoder.base_vectors(["q | domain: x | skills: t"]))
        assert float(before[0] @ after[0]) < 1.0 - 1e-6

    def test_disabled_trainer_never_moves_the_heads(self):
        t = self._trainer(enabled=False)
        views = {"m_1": "A | domain: x | skills: t"}
        for step in range(6):
            t.observe("q", views, {"used_positive": ["m_1"]}, step)
            t.maybe_update(step)
        assert t.stats.updates == 0
        assert t.encoder.version == 0

    def test_reset_clears_buffer_and_returns_heads_to_identity(self):
        """Required between runs: otherwise run n retrieves using runs 1..n-1's labels."""
        t = self._trainer(k_upd=1)
        views = {"m_1": "A | domain: x | skills: t", "m_2": "B | domain: y | skills: u"}
        for step in range(4):
            t.observe("q | domain: x | skills: t", views,
                      {"used_positive": ["m_1"], "used_negative": ["m_2"]}, step)
            t.maybe_update(step)
        assert t.stats.updates > 0
        t.reset()
        assert len(t.buffer) == 0 and t.stats.updates == 0
        text = "identical text on both sides"
        assert float(t.encoder.encode_query(text) @ t.encoder.encode_skill([text])[0]) ==             pytest.approx(1.0, abs=1e-5)


# ------------------------------------------------------------------- plumbing

class TestPlumbing:
    def test_extract_json_survives_fences_and_preamble(self):
        assert extract_json('Here you go:\n```json\n{"a": 1}\n```')["a"] == 1
        assert extract_json('{"a": {"b": [1,2]}} trailing text')["a"]["b"] == [1, 2]

    def test_extract_json_ignores_braces_inside_strings(self):
        assert extract_json('{"a": "not } a brace"}')["a"] == "not } a brace"

    def test_pipeline_runs_end_to_end_offline(self, book, tmp_path):
        items = [Item(id=f"Q_{i:03d}", question=f"Question number {i}?",
                      answer_type="mcq_letter", gold="A", n_options=4,
                      dataset="smoke") for i in range(5)]
        report = Pipeline(book, EchoLLM(answer="A"), CFG).run(items)
        assert len(report.steps) == 5
        assert report.accuracy == 1.0
        assert report.summary()["errors"] == 0

    def test_reset_per_run_empties_the_book(self, book):
        """pass@1 over 5-10 passes means 5-10 INDEPENDENT passes.

        Resetting only Ep/Es left the book carried across passes, so run n answered items
        that runs 1..n-1 had already written entries about -- the prequential break the
        reset exists to prevent, and it failed silently by drifting accuracy upward.
        """
        book.create(make_proposal("Carried over from a previous pass",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_000", 0)
        assert len(book) == 1
        items = [Item(id="Q_001", question="A question?", answer_type="mcq_letter",
                      gold="A", n_options=4)]
        Pipeline(book, EchoLLM(answer="A"), CFG).run(items)
        assert not any(e.meta.created == "Q_000" for e in book.entries.values())

    def test_frozen_pass_keeps_its_prebuilt_book(self, book):
        """reset_per_run must not empty the book a frozen pass is there to read.

        Frozen mode reads a book built elsewhere; that book is the experiment. Emptying it
        would leave the frozen arm retrieving from nothing while still reporting a number.
        """
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_000", 0)
        cfg = {**CFG, "run": {**CFG["run"], "write_enabled": False,
                              "reset_per_run": True}}
        items = [Item(id="Q_001", question="A question?", answer_type="mcq_letter",
                      gold="A", n_options=4)]
        Pipeline(book, EchoLLM(answer="A"), cfg).run(items)
        assert len(book) == 1

    def test_frozen_mode_never_writes(self, book):
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_000", 0)
        before = len(book)
        cfg = {**CFG, "run": {**CFG["run"], "write_enabled": False}}
        items = [Item(id="Q_001", question="A question?", answer_type="mcq_letter",
                      gold="A", n_options=4)]
        report = Pipeline(book, EchoLLM(answer="A"), cfg).run(items)
        assert len(book) == before and len(report.steps) == 1
        assert not [r for r in book.write_log if r.query_id == "Q_001"]

    def test_a_failing_item_does_not_kill_the_run(self, book):
        class Boom(EchoLLM):
            def complete_json(self, prompt, *, component, image=None):
                if component == "c2":
                    raise RuntimeError("provider timeout")
                return super().complete_json(prompt, component=component, image=image)

        items = [Item(id=f"Q_{i}", question="q?", answer_type="mcq_letter", gold="A")
                 for i in range(3)]
        report = Pipeline(book, Boom(), CFG).run(items)
        assert len(report.steps) == 3 and report.summary()["errors"] == 3

    def test_report_metrics_are_computable(self, book):
        items = [Item(id=f"Q_{i:03d}", question=f"Q{i}?", answer_type="mcq_letter",
                      gold="A" if i % 2 == 0 else "B", n_options=4) for i in range(10)]
        report = Pipeline(book, EchoLLM(answer="A"), CFG).run(items)
        assert report.accuracy == pytest.approx(0.5)
        assert report.retrieval_precision == 0.0
        assert len(report.accumulation_curve(bucket=5)) == 2

    def test_round_trip_persistence(self, book, tmp_path):
        book.create(make_proposal("Modular arithmetic for final digits",
                                  ["Reduce mod 10 early", "Use Euler totient"]),
                    "Q_001", 0)
        path = tmp_path / "book.json"
        book.save(path)
        reloaded = SkillBook.load(path, book.encoder)
        assert len(reloaded) == 1
        assert reloaded.entries["m_001"].title == "Modular arithmetic for final digits"
        assert reloaded._next_id == 2       # ids do not collide after reload


# ------------------------------------------------------------------ tool calls
# The solver may write a program and reason from what it really printed. Before this
# existed, C2's prompt asked it to "reason from its result" and emit `coding_result` with
# nothing running the code -- the model invented its own program's output and C3 was shown
# that invention as if it were execution evidence.

class TestTools:
    def test_runs_code_and_returns_real_stdout(self):
        r = run_python("print(sum(range(101)))", ToolConfig())
        assert r.ok and r.stdout == "5050" and not r.error

    def test_syntax_error_is_reported_not_executed(self):
        r = run_python("print(", ToolConfig())
        assert not r.ok and r.error.startswith("syntax_error")

    def test_traceback_reaches_the_solver(self):
        """A failure is feedback. Hiding it behind "execution failed" throws away the
        entire benefit of having run the code."""
        r = run_python("raise ValueError('boom')", ToolConfig())
        assert not r.ok and "ValueError" in r.stderr and "boom" in r.render()

    def test_denied_imports_are_refused_before_running(self):
        for code in ("import socket", "from urllib.request import urlopen",
                     "import subprocess as sp"):
            r = run_python(code + "\nprint(1)", ToolConfig())
            assert not r.ok and r.error.startswith("denied_import"), code

    def test_dynamic_import_is_refused(self):
        """A denylist walked over static imports says nothing about __import__, so the
        whole program is refused rather than passed as if it had been checked."""
        r = run_python("m = __import__('socket')\nprint(m)", ToolConfig())
        assert not r.ok and "__import__" in r.error

    def test_timeout_is_bounded(self):
        r = run_python("while True:\n    pass", ToolConfig(timeout_s=2))
        assert not r.ok and r.error == "timeout"

    def test_output_is_capped(self):
        r = run_python("print('x' * 20000)", ToolConfig(max_output_chars=500))
        assert r.ok and r.truncated and len(r.stdout) <= 500

    def test_child_cannot_see_the_api_key(self, monkeypatch):
        """The parent holds the key that pays for the run and the child is executing text
        a language model wrote. The child gets an interpreter and nothing else."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-visible")
        r = run_python("import os\nprint(os.environ.get('OPENAI_API_KEY', 'ABSENT'))",
                       ToolConfig())
        assert r.ok and r.stdout == "ABSENT"


class TestSolverToolLoop:
    def test_tool_result_reaches_the_solver_and_is_recorded(self, book):
        items = [Item(id="Q_001", question="What is 2+2?", answer_type="integer",
                      gold="4")]
        llm = EchoLLM(answer="4", tool_code="print(2+2)")
        report = Pipeline(book, llm, CFG).run(items)
        step = report.steps[0]
        assert step.tool_calls == 1 and step.tool_failures == 0
        assert step.tool_expected is True

    def test_llm_calls_counts_the_tool_round_trip(self, book):
        """Accuracy at matched cost is the comparison the paper turns on, so a tool
        round-trip must show up in the denominator. It used to be hardcoded to 3."""
        items = [Item(id="Q_001", question="What is 2+2?", answer_type="integer",
                      gold="4")]
        plain = Pipeline(book, EchoLLM(answer="4"), CFG).run(items)
        assert plain.steps[0].llm_calls == 3          # C1 + C2 + C3

        book2 = SkillBook(encoder=DualEncoder(HashingEncoder(dim=256)))
        tooled = Pipeline(book2, EchoLLM(answer="4", tool_code="print(4)"), CFG).run(items)
        assert tooled.steps[0].llm_calls == 4          # C1 + C2(tool) + C2(answer) + C3

    def test_real_output_overwrites_the_models_claim(self, book):
        """A model that misreports what its own program printed must not be able to
        mislead C3, and the disagreement is counted rather than hidden."""
        items = [Item(id="Q_001", question="What is 2+2?", answer_type="integer",
                      gold="4")]
        llm = EchoLLM(answer="4", tool_code="print(4)", claim_result="999")
        report = Pipeline(book, llm, CFG).run(items)
        assert report.violations.get("solver_fabricated_coding_result") == 1

    def test_budget_forces_an_answer(self, book):
        """A model that only ever requests tools must still produce an answer, or the item
        fails for a reason unrelated to the question."""
        class AlwaysTool(EchoLLM):
            def complete_json(self, prompt, *, component, image=None):
                if component == "c2":
                    self.calls.append(component)
                    return {"action": "tool", "tool": "python", "code": "print(1)"}
                return super().complete_json(prompt, component=component, image=image)

        cfg = {**CFG, "tools": {"enabled": True, "max_calls_per_item": 2, "timeout_s": 10}}
        items = [Item(id="Q_001", question="Q?", answer_type="integer", gold="4")]
        report = Pipeline(book, AlwaysTool(answer="4"), cfg).run(items)
        assert report.steps[0].tool_calls == 2
        assert report.steps[0].error is None

    def test_tools_disabled_never_executes(self, book):
        cfg = {**CFG, "tools": {"enabled": False}}
        items = [Item(id="Q_001", question="Q?", answer_type="integer", gold="4")]
        llm = EchoLLM(answer="4", tool_code="print(2+2)")
        report = Pipeline(book, llm, cfg).run(items)
        assert report.steps[0].tool_calls == 0 and report.steps[0].llm_calls == 3


class TestVerificationSources:
    def test_exec_reuses_the_recorded_run(self):
        """Re-running the same program costs a second subprocess and can disagree with
        the first if the code is not deterministic, which would make the signal depend on
        which of two runs it happened to read."""
        rec = ToolResult(ok=True, stdout="204", code="print(204)")
        sig = signal_from_exec("print(999)", "204", "integer", recorded=rec)
        assert sig.correct is True
        assert "204" in sig.detail and "999" not in sig.detail   # the recorded run, not `code`

    def test_exec_reports_a_failed_run_as_negative_evidence(self):
        rec = ToolResult(ok=False, stderr="Traceback", error="exit_1", code="boom")
        sig = signal_from_exec("boom", "204", "integer", recorded=rec)
        assert sig.correct is False

    def test_consistency_actually_draws_samples(self, book):
        """`source: consistency` used to degrade silently to a judge signal carrying
        correct=None -- the supervision ablation looked like it ran, and had not."""
        cfg = {**CFG, "verification": {**CFG["verification"], "source": "consistency",
                                       "consistency_samples": 3}}
        items = [Item(id="Q_001", question="Q?", answer_type="mcq_letter", gold=None,
                      n_options=4)]
        report = Pipeline(book, EchoLLM(answer="A"), cfg).run(items)
        step = report.steps[0]
        assert step.signal_source == "consistency"
        assert step.llm_calls == 5          # C1 + 3 solver samples + C3

    def test_judge_source_carries_a_verdict(self, book):
        """It used to return correct=None for every item because the pipeline never
        passed judge_correct -- a source that moved no counter while config claimed it was
        active."""
        cfg = {**CFG, "verification": {**CFG["verification"], "source": "judge"}}
        items = [Item(id="Q_001", question="Q?", answer_type="mcq_letter", gold=None,
                      n_options=4)]
        report = Pipeline(book, EchoLLM(answer="A"), cfg).run(items)
        assert report.steps[0].signal_source == "judge"
        assert report.steps[0].signal_confidence <= 0.6


class TestIdentityInitIsExact:
    def test_null_proj_dim_preserves_the_base_geometry(self):
        """The ccqs.enabled:false arm is the only internal control isolating CCQS, so an
        untrained head must reproduce frozen-base retrieval EXACTLY. A smaller proj_dim
        makes the identity a truncation instead -- at 256 over 384 it shifts pairwise
        cosines by ~0.10, and the control becomes a different retriever."""
        base = HashingEncoder(dim=384)
        texts = ["modular arithmetic | domain: math.number_theory",
                 "counting lattice paths | domain: math.combinatorics",
                 "stereochemistry of chiral centres | domain: chemistry"]
        raw = base.encode(texts)
        exact = DualEncoder(base, proj_dim=None)
        got = exact.encode_skill(texts)
        assert np.allclose(raw @ raw.T, got @ got.T, atol=1e-6)

        truncated = DualEncoder(base, proj_dim=256)
        cut = truncated.encode_skill(texts)
        assert not np.allclose(raw @ raw.T, cut @ cut.T, atol=1e-3)


# ---------------------------------------------------------------- provider client

class _FakeTransport:
    """Stands in for the network. Scripted replies; an unscripted call is a test failure,
    so a client that calls more times than the test expects cannot pass quietly."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def send(self, *, model, messages, temperature, max_tokens):
        self.calls.append({"model": model, "messages": messages,
                           "temperature": temperature, "max_tokens": max_tokens})
        assert self.script, "transport called more times than the test scripted"
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, TransportReply):
            return item
        return TransportReply(text=item, prompt_tokens=100, completion_tokens=20)


def make_client(script: list, **kw) -> ProviderLLM:
    slept: list = []
    client = ProviderLLM(transport=_FakeTransport(script),
                         sleep=slept.append, **kw)
    client.slept = slept          # type: ignore[attr-defined]
    return client


class TestProviderClient:
    """The gap: the repo could not talk to a real model at all. `EchoLLM` proved the
    pipeline runs; none of these paths -- fences, rate limits, cost, images -- existed."""

    def test_fenced_output_is_recovered(self):
        """Models emit ```json fences despite instructions. Failing the item over
        formatting would confound the measurement with a parser bug."""
        c = make_client(['```json\n{"answer": "A"}\n```'])
        assert c.complete_json("p", component="c2") == {"answer": "A"}

    def test_transient_error_is_retried_with_backoff(self):
        """A 12k-item MMLU-Pro run hits 429s. Treating one as fatal loses the run."""
        c = make_client([TransientProviderError("429 rate limit"), '{"ok": true}'])
        assert c.complete_json("p", component="c1") == {"ok": True}
        assert len(c.transport.calls) == 2
        assert len(c.slept) == 1 and c.slept[0] > 0
        assert c.usage.by_component["c1"].retries == 1
        # Only the successful call is billed.
        assert c.usage.by_component["c1"].calls == 1

    def test_retry_after_header_is_honoured(self):
        c = make_client([TransientProviderError("429", retry_after=7.0), '{"ok": 1}'],
                        backoff_base_s=1.0)
        c.complete_json("p", component="c1")
        assert 7.0 <= c.slept[0] <= 7.5      # server's number, plus a little jitter

    def test_retries_are_bounded_then_raise(self):
        c = make_client([TransientProviderError("boom")] * 3, max_retries=2)
        with pytest.raises(TransientProviderError):
            c.complete_json("p", component="c2")
        assert len(c.transport.calls) == 3

    def test_non_retryable_error_is_not_retried(self):
        """A 400 or a bad key will fail identically forever; burning five retries on it
        just delays the traceback."""
        c = make_client([ProviderError("HTTP 400: bad request")])
        with pytest.raises(ProviderError):
            c.complete_json("p", component="c1")
        assert len(c.transport.calls) == 1

    def test_parse_failure_retries_twice_then_raises(self):
        c = make_client(["not json at all"] * 3, max_parse_retries=2)
        with pytest.raises(ParseFailure):
            c.complete_json("p", component="c3")
        assert len(c.transport.calls) == 3
        assert c.usage.by_component["c3"].parse_failures == 3

    def test_parse_retry_changes_the_conversation(self):
        """Resending the identical prompt at temperature 0 reproduces the identical
        malformed output, so the retry has to correct rather than repeat."""
        c = make_client(["sorry, no JSON", '{"answer": "B"}'])
        assert c.complete_json("p", component="c2") == {"answer": "B"}
        second = c.transport.calls[1]["messages"]
        assert len(second) == 3
        assert second[1]["role"] == "assistant" and second[1]["content"] == "sorry, no JSON"
        assert "JSON object" in second[2]["content"]

    def test_top_level_array_is_a_parse_failure(self):
        """`complete_json` promises a dict; a list would blow up inside `.parse` with a
        traceback that points at the schema rather than at the model."""
        c = make_client(["[1, 2, 3]"] * 3, max_parse_retries=2)
        with pytest.raises(ParseFailure):
            c.complete_json("p", component="c1")

    def test_components_route_to_different_models(self):
        """A small planner with a large solver is the cheap ablation the notes call for,
        and the pipeline already routes `component`."""
        c = make_client(['{"a": 1}', '{"a": 2}'], model="gpt-4o-mini",
                        components={"c2": {"model": "gpt-4o", "temperature": 0.7}})
        c.complete_json("p", component="c1")
        c.complete_json("p", component="c2")
        assert c.transport.calls[0]["model"] == "gpt-4o-mini"
        assert c.transport.calls[0]["temperature"] == 0.0
        assert c.transport.calls[1]["model"] == "gpt-4o"
        assert c.transport.calls[1]["temperature"] == 0.7

    def test_cost_is_metered_per_component(self):
        """Cost is a headline metric and cannot be reconstructed after a
        run, so it has to be recorded as the call happens."""
        reply = TransportReply(text='{"ok": 1}', prompt_tokens=1_000_000,
                               completion_tokens=1_000_000)
        c = make_client([reply, reply], model="gpt-4o-mini")
        c.complete_json("p", component="c1")
        c.complete_json("p", component="c3")
        s = c.usage.summary()
        assert s["calls"] == 2
        assert s["by_component"]["c1"]["cost_usd"] == pytest.approx(0.15 + 0.60)
        assert s["cost_usd"] == pytest.approx(2 * 0.75)
        assert s["priced"] is True

    def test_cached_prompt_tokens_bill_at_the_cached_rate(self):
        """Every prompt shares a long instruction prefix, so caching triggers hard on a
        12k-item run; billing cached tokens at full rate would overstate the bill."""
        c = make_client([TransportReply(text='{"ok": 1}', prompt_tokens=1000,
                                        cached_prompt_tokens=400, completion_tokens=0)],
                        model="gpt-4o-mini")
        c.complete_json("p", component="c2")
        expected = 600 / 1e6 * 0.15 + 400 / 1e6 * 0.075
        assert c.usage.total_cost_usd == pytest.approx(expected)

    def test_unpriced_model_reports_zero_not_a_guess(self):
        c = make_client(['{"ok": 1}'], model="some-open-weight-7b")
        c.complete_json("p", component="c1")
        assert c.usage.summary()["priced"] is False
        assert c.usage.total_cost_usd == 0.0

    def test_image_becomes_a_data_uri_part(self):
        png = b"\x89PNG\r\n\x1a\n" + b"payload"
        c = make_client(['{"ok": 1}'])
        c.complete_json("p", component="c2", image=png)
        content = c.transport.calls[0]["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "p"}
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_unknown_image_format_raises(self):
        """Guessing a mime type sends MathVista's images mislabelled, and a silently
        dropped image scores the item on a question the model never saw."""
        with pytest.raises(ProviderError):
            sniff_mime(b"\x00\x01\x02\x03not-an-image")
        assert sniff_mime(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
        assert sniff_mime(b"RIFF1234WEBPrest") == "image/webp"
        assert image_data_uri(b"GIF89a").startswith("data:image/gif;base64,")

    def test_text_only_model_refuses_an_image_instead_of_dropping_it(self):
        c = make_client([], supports_images=False)
        with pytest.raises(ProviderError):
            c.complete_json("p", component="c2", image=b"\x89PNG\r\n\x1a\n")
        assert c.transport.calls == []

    def test_consistency_at_temperature_zero_is_refused_at_construction(self):
        """`_consistency_samples` re-enters `solve()` with an identical prompt, so at
        temperature 0 the five votes are one answer repeated and the label-free signal is
        vacuous -- silently, which is why this fails at startup."""
        cfg = {"llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
               "verification": {"source": "consistency", "consistency_samples": 5}}
        with pytest.raises(ProviderError):
            ProviderLLM.from_config(cfg, transport=_FakeTransport([]))

        cfg["llm"]["components"] = {"c2": {"temperature": 0.8}}
        client = ProviderLLM.from_config(cfg, transport=_FakeTransport([]))
        assert client.specs["c2"].temperature == 0.8

    def test_key_file_is_parsed_and_never_echoed(self):
        """`lere/tools.py` keeps the key out of the executor's environment on purpose; a
        traceback that echoes it would put it straight back."""
        secret = "sk-proj-THISISTHESECRET"
        assert scrub_secret("Bearer " + secret + " failed", secret) == \
            "Bearer <redacted> failed"

        c = make_client([], model="gpt-4o-mini")
        assert secret not in repr(c)
        with pytest.raises(ProviderError) as exc:
            resolve_api_key("openai", key_file="/nonexistent/API_key.txt", env={})
        assert secret not in str(exc.value)

    def test_key_file_formats(self, tmp_path):
        p = tmp_path / "API_key.txt"
        p.write_text('# comment\nopenAI="sk-aaa"\nexport GEMINI_API_KEY=sk-bbb\nblank=\n')
        parsed = read_key_file(p)
        assert parsed["openAI"] == "sk-aaa"
        assert parsed["GEMINI_API_KEY"] == "sk-bbb"
        assert "blank" not in parsed
        assert resolve_api_key("openai", key_file=p, env={}) == "sk-aaa"
        assert resolve_api_key("gemini", key_file=p, env={}) == "sk-bbb"

    def test_configured_env_var_wins_over_the_file(self, tmp_path):
        p = tmp_path / "API_key.txt"
        p.write_text('openAI="from-file"\n')
        got = resolve_api_key("openai", key_file=p, env_var="MY_KEY",
                              env={"MY_KEY": "from-env"})
        assert got == "from-env"

    def test_satisfies_the_llm_protocol_the_pipeline_calls(self):
        """The whole interface is one method; if this drifts, nothing else matters."""
        c = make_client(['{"ok": 1}'])
        pipeline_style_call = ProviderLLM.complete_json
        assert pipeline_style_call(c, "prompt", component="c1", image=None) == {"ok": 1}

    def test_component_spec_cost_is_pure_arithmetic(self):
        spec = ComponentSpec(model="m", temperature=0.0, max_tokens=10,
                             supports_images=True, price_input=1.0,
                             price_cached_input=0.5, price_output=2.0)
        assert spec.cost(1_000_000, 0, 1_000_000) == pytest.approx(3.0)
        assert spec.cost(1_000_000, 1_000_000, 0) == pytest.approx(0.5)


# ------------------------------------------------------------- dataset loaders

SUBSET = Path(__file__).resolve().parent.parent / "dataset" / "subsets" / \
    "aime2025_first10.jsonl"
needs_data = pytest.mark.skipif(not (DATASET_DIR / "AIME_2025").is_dir(),
                                reason="dataset/ not present in this checkout")


class TestAimeLoader:
    """The gap: nothing had ever turned a file on disk into `Item`s, so every guarantee
    the pipeline assumes about `gold` and `answer_type` was untested."""

    @needs_data
    def test_first_ten_of_2025_are_the_first_ten_rows(self):
        """`offset`/`limit` slice before anything else, so 'the first 10' is a statement
        about the file and not about whatever survived a filter."""
        items = load_aime_2025(limit=10)
        assert len(items) == 10
        assert [i.meta["source_index"] for i in items] == list(range(10))
        assert [i.id for i in items] == ["AIME_2025-%04d" % i for i in range(10)]
        assert items == load_aime_2025(limit=10)          # deterministic

    @needs_data
    def test_gold_is_an_aime_integer_the_harness_can_score(self):
        for it in load_aime_2025():
            assert it.answer_type == "integer"
            assert it.image is None
            assert 0 <= int(it.gold) <= 999
            assert is_correct(it.gold, it.gold, it.answer_type)
            assert is_correct("\\boxed{%s}" % it.gold, it.gold, it.answer_type)

    @needs_data
    def test_worked_solutions_never_reach_the_item(self):
        """AIME_2024's metadata carries the answer with its derivation attached. `guard.py`
        protects the book from the question, not from the loader."""
        items = load_aime_2024()
        assert any(i.meta["solution_withheld"] for i in items)
        for it in items:
            assert "Solution" not in it.meta
            blob = it.question + repr(it.meta)
            assert "Denote $\\log_2(x)$" not in blob

    @needs_data
    def test_malformed_gold_raises_rather_than_being_coerced(self):
        """AIME_2020_2025 row 2022-II-8 is `080 or 081 (both were accepted)`. Coercing it
        to 80 invents a label; dropping it silently changes n."""
        path = DATASET_DIR / "AIME_2020_2025"
        with pytest.raises(DatasetError):
            load_aime(path)
        kept = load_aime(path, on_bad_gold="skip")
        assert len(kept) == 162
        assert all(0 <= int(i.gold) <= 999 for i in kept)

    @needs_data
    def test_asking_for_more_items_than_exist_raises(self):
        with pytest.raises(DatasetError):
            load_aime_2025(limit=99)

    @needs_data
    def test_ids_are_stable_across_the_two_2025_sources(self):
        """AIME_2025 has no metadata column, so the row index is its only identity; if
        that drifted, a frozen subset would silently name different problems."""
        assert load_aime_2025(limit=3)[2].id == "AIME_2025-0002"
        assert load_aime_2025(offset=2, limit=1)[0].id == "AIME_2025-0002"

    @needs_data
    def test_frozen_subset_on_disk_matches_the_loader(self):
        """The run is identified by this file, not by the phrase 'first 10 of AIME_2025'.
        If the directory is ever re-exported, this test is what notices."""
        assert SUBSET.is_file()
        frozen = load_jsonl(SUBSET)
        live = load_aime_2025(limit=10)
        assert [f.id for f in frozen] == [l.id for l in live]
        assert [f.gold for f in frozen] == [l.gold for l in live]
        assert [f.question for f in frozen] == [l.question for l in live]

    def test_jsonl_round_trip(self, tmp_path):
        items = [Item(id="a-1", question="q?", answer_type="integer", gold="007",
                      dataset="t", meta={"source_index": 0})]
        back = load_jsonl(dump_jsonl(items, tmp_path / "s.jsonl"))
        assert back == items

    def test_images_round_trip_through_a_sidecar(self, tmp_path):
        """MathVista items carry bytes. They are written beside the jsonl rather than
        base64'd inline: a 1.5 MB PNG per row would make the frozen subset unreadable,
        and being able to see which items a run used is the point of freezing it."""
        png = b"\x89PNG\r\n\x1a\n" + b"payload"
        items = [Item(id="mv/1", question="q?", answer_type="mcq_letter", gold="A",
                      n_options=4, image=png, dataset="mv")]
        p = dump_jsonl(items, tmp_path / "s.jsonl")
        side = tmp_path / "s_images"
        assert side.is_dir() and (side / "mv_1.png").read_bytes() == png   # id sanitised
        back = load_jsonl(p)
        assert back[0].image == png and back[0].n_options == 4

    def test_a_missing_sidecar_image_is_an_error_not_a_silent_none(self, tmp_path):
        """Scoring a vision item on its text alone would look like a wrong answer rather
        than a broken subset."""
        items = [Item(id="i-1", question="q?", image=b"\x89PNG\r\n\x1a\npayload")]
        p = dump_jsonl(items, tmp_path / "s.jsonl")
        (tmp_path / "s_images" / "i-1.png").unlink()
        with pytest.raises(DatasetError):
            load_jsonl(p)


# ---------------------------------------------------------------- instrumentation hooks

class TestObserverHooks:
    """The gap: a demo needs the book, the heads and the loss AFTER the write phase, and
    every prompt and raw response, none of which any structure kept. Both hooks are None
    by default, so an untraced run must be byte-identical to before."""

    def test_pipeline_on_step_fires_once_per_completed_step(self, book):
        cfg = dict(CFG)
        cfg["run"] = {"batch_size": 1, "write_enabled": True, "seed": None}
        pipe = Pipeline(book, EchoLLM(answer="7"), cfg)
        seen = []
        pipe.on_step = lambda p, report, loss: seen.append((p.record.step, p.item.id, loss))
        items = [Item(id="i%d" % i, question="q%d" % i, answer_type="integer", gold="7")
                 for i in range(3)]
        pipe.run(items)
        assert [s[0] for s in seen] == [0, 1, 2]
        assert [s[1] for s in seen] == ["i0", "i1", "i2"]

    def test_untraced_run_is_unchanged(self, book):
        """`on_step` defaults to None and must never be consulted."""
        cfg = dict(CFG)
        cfg["run"] = {"batch_size": 1, "write_enabled": True, "seed": None}
        pipe = Pipeline(book, EchoLLM(answer="7"), cfg)
        assert pipe.on_step is None
        rep = pipe.run([Item(id="i0", question="q", answer_type="integer", gold="7")])
        assert len(rep.steps) == 1

    def test_provider_on_call_sees_the_raw_text(self):
        """`extract_json` discards the raw string, and a fence or preamble is only
        visible there -- it is the first sign of a prompt regression."""
        seen = []
        c = ProviderLLM(transport=_FakeTransport(['```json\n{"a": 1}\n```']),
                        on_call=lambda **kw: seen.append(kw))
        c.complete_json("the prompt", component="c1")
        assert len(seen) == 1
        assert seen[0]["raw_text"] == '```json\n{"a": 1}\n```'
        assert seen[0]["parsed"] == {"a": 1}
        assert seen[0]["messages"][0]["content"] == "the prompt"
        assert seen[0]["attempts"] == 1

    def test_provider_on_call_reports_parse_retries(self):
        seen = []
        c = ProviderLLM(transport=_FakeTransport(["junk", '{"a": 1}']),
                        on_call=lambda **kw: seen.append(kw))
        c.complete_json("p", component="c2")
        assert [k["attempts"] for k in seen] == [2]      # fired only on the parse that won

    def test_call_recorder_advances_on_c1(self, tmp_path):
        """C1 runs exactly once per item and always first, so it is the item boundary;
        the driver cannot supply the step because `Pipeline.run` owns the loop."""
        rec = CallRecorder(JsonlWriter(tmp_path / "calls.jsonl"))
        stub = TracedLLM(EchoLLM(answer="1"), rec)
        for _ in range(2):
            stub.complete_json("p", component="c1")
            stub.complete_json("p", component="c2")
            stub.complete_json("p", component="c3")
        assert [c["step"] for c in rec.calls] == [0, 0, 0, 1, 1, 1]
        assert len(rec.slice_for_step(1)) == 3


class TestTracingRetriever:
    def test_records_candidates_the_retriever_dropped(self, book):
        """An empty result cannot say WHY. 'best candidate scored 0.43 against a 0.60
        floor' is the finding, and it is what threshold recalibration needs."""
        for i, title in enumerate(["Modular casework", "Chirality of stereocenters"]):
            book.create(make_proposal(title, ["b1", "b2"],
                                      domain="math.number_theory" if i == 0 else "chemistry"),
                        "q0", 0)
        cfg = {"top_k": 3, "sim_threshold": 0.99, "alpha": 0.7, "domain_filter": "soft",
               "domain_penalty": 0.25, "domain_partial_credit": 0.6, "mmr_lambda": 0.7}
        r = TracingRetriever(book, cfg, JsonlWriter("/dev/null"))
        plan = PlannerOutput.parse(
            {"semantic_context": "modular casework", "domain": "math.number_theory",
             "tags": ["strategy.casework"], "retrieval_query": "modular casework",
             "tool_expected": False}, VocabViolations())
        selected = r.retrieve(plan, 0)
        assert selected == []                              # floor of 0.99 admits nothing
        assert r.last["n_candidates"] == 2                 # but both were still recorded
        assert r.last["n_passed_floor"] == 0
        assert r.last["best_raw_sim"] is not None
        assert all(not c["selected"] for c in r.last["candidates"])
        assert r.last["scoring_drift"] == []
        assert len(r.last["ep_query_vector"]) == book.encoder.dim

    def test_selected_entries_are_flagged_and_consistent(self, book):
        book.create(make_proposal("Modular casework", ["b1"]), "q0", 0)
        cfg = {"top_k": 3, "sim_threshold": -1.0, "alpha": 0.7, "domain_filter": "soft",
               "domain_penalty": 0.25, "domain_partial_credit": 0.6, "mmr_lambda": 0.7}
        r = TracingRetriever(book, cfg, JsonlWriter("/dev/null"))
        plan = PlannerOutput.parse(
            {"semantic_context": "modular casework", "domain": "math.number_theory",
             "tags": ["strategy.casework"], "retrieval_query": "modular casework",
             "tool_expected": False}, VocabViolations())
        selected = r.retrieve(plan, 0)
        assert [s.id for s in selected] == r.last["selected_ids"]
        assert [c["selected"] for c in r.last["candidates"]] == [True]
        assert r.last["scoring_drift"] == []               # the duplicated math agrees
        assert r.last["encoder_version"] == book.encoder.version


class TestHeadGeometry:
    def test_heads_start_at_identity_and_movement_is_visible(self):
        """`DualEncoder.reset_heads()` bumps the version itself, so version > 0 is NOT
        evidence the heads moved. `delta_from_identity` is."""
        enc = DualEncoder(HashingEncoder(dim=32))
        s0 = head_stats(enc)
        assert s0["ep"]["delta_from_identity"] == 0.0
        assert s0["es"]["delta_from_identity"] == 0.0

        enc.reset_heads()
        assert head_stats(enc)["version"] > s0["version"]          # version moved
        assert head_stats(enc)["ep"]["delta_from_identity"] == 0.0  # the head did not

        w = head_weights(enc)["ep"]
        assert w.shape == (32, 32)
        assert np.allclose(w, np.eye(32), atol=1e-6)


class TestVocabularyReachesTheModel:
    """The gap the 1-item probe found: C1's and C3's prompts said `domain` must come from
    'the closed list in taxonomy.md' and nothing ever showed the model that list. It
    answered `number_theory` and `mathematics`, every tag was dropped, and the guard then
    rejected the entry as `no_valid_tags`."""

    def test_block_is_generated_from_the_enforcing_code(self):
        block = vocabulary_block()
        for d in DOMAINS:
            assert d in block, d
        for prefix in TAG_PREFIXES:
            assert "`%s.`" % prefix in block
        # The two coercions the probe actually triggered are called out by name.
        assert "math.other" in block and "mathematics" in block

    def test_both_prompts_carry_the_slot(self):
        for name in ("c1_planner.md", "c3_curator.md"):
            assert "{{vocabulary}}" in load_prompt(name), name

    def test_rendered_prompts_contain_the_domains(self, book):
        """Rendering is where the slot could silently go unfilled: `render` replaces
        unknown `{{slot}}` markers with empty string rather than raising."""
        pipe = Pipeline(book, EchoLLM(), CFG)
        item = Item(id="i", question="q?", answer_type="integer", gold="1")
        seen = {}

        class Spy:
            def complete_json(self, prompt, *, component, image=None):
                seen[component] = prompt
                return EchoLLM().complete_json(prompt, component=component, image=image)

        pipe.llm = Spy()
        pipe.plan(item, VocabViolations())
        assert "math.number_theory" in seen["c1"]
        assert "{{vocabulary}}" not in seen["c1"]


class TestAnswerPhraseGuard:
    """The other probe finding: the answer-phrase regex matched `result = 0` inside a
    Python example, so it rejected the `tool.*` entries C3's own prompt asks for."""

    def test_code_assignment_is_not_an_answer_claim(self):
        entry = ProposedEntry(
            title="Convert between bases",
            bullets=["Multiply each digit by the base to its position.",
                     "Accumulate across positions."],
            example="def conv(v, b): result = 0; return result",
            domain="math.number_theory", tags=["tool.exact_arithmetic"])
        res = check_entry(entry, "some question text", "70", "integer", CFG["guard"])
        assert "answer_phrase" not in res.reasons
        assert res.accepted

    def test_prose_answer_claims_are_still_blocked(self):
        for leak in ("The answer is C.", "the final answer is 204", "answer: B"):
            entry = ProposedEntry(
                title="A method", bullets=["Step one here.", leak],
                example="a minimal illustrative case",
                domain="math.number_theory", tags=["strategy.casework"])
            res = check_entry(entry, "q", None, "integer", CFG["guard"])
            assert "answer_phrase" in res.reasons, leak

    def test_a_prose_example_is_still_checked(self):
        """Only a code example is exempt; prose in `example` can still leak."""
        entry = ProposedEntry(
            title="A method", bullets=["Step one here.", "Step two here."],
            example="For this problem the answer is 204.",
            domain="math.number_theory", tags=["strategy.casework"])
        res = check_entry(entry, "q", None, "integer", CFG["guard"])
        assert "answer_phrase" in res.reasons


class TestToolRepeats:
    """The gap the live C2 trace exposed: the solver re-emitted byte-identical code three
    times, got the same output three times, and spent its whole budget. Nothing noticed,
    and each repeat cost a full LLM turn out of `llm_calls_per_item` -- the denominator
    the matched-cost comparison turns on."""

    def test_identical_program_is_not_re_executed(self):
        cfg = dict(CFG)
        cfg["run"] = {"batch_size": 1, "write_enabled": True, "seed": None}
        cfg["tools"] = {"enabled": True, "max_calls_per_item": 3, "timeout_s": 10,
                        "max_output_chars": 4000}
        book = SkillBook(encoder=DualEncoder(HashingEncoder(dim=256)))
        llm = EchoLLM(answer="7", tool_code="print(7)")
        pipe = Pipeline(book, llm, cfg)
        item = Item(id="i0", question="q?", answer_type="integer", gold="7")
        v = VocabViolations()
        plan = pipe.plan(item, v)
        solver, transcript, calls = pipe.solve(item, [], plan, v)

        # EchoLLM asks for the tool on its first C2 turn only, so exactly one execution.
        assert transcript.calls == 1
        assert transcript.repeats == 0
        assert "solver_repeated_identical_code" not in v

    def test_repeat_is_recorded_counted_and_flagged_to_the_model(self):
        t = ToolTranscript()
        first = ToolResult(ok=True, stdout="0", code="print(0)")
        t.add(first)
        assert t.find_repeat("print(0)") is first
        assert t.find_repeat("  print(0)\n") is first     # indentation / trailing newline
        assert t.find_repeat("print( 0 )") is None         # deliberately conservative
        assert t.find_repeat("print(1)") is None

        echoed = t.add_repeat(first)
        assert echoed.repeated is True
        assert echoed.stdout == "0"                        # the first run's real output
        assert first.repeated is False                     # the original is untouched
        assert t.calls == 2                                # the turn is still charged
        assert t.repeats == 1
        assert "NOT RE-RUN" in t.render()

    def test_repeat_still_counts_towards_the_budget(self):
        """If a repeat were free the model could loop on one program forever and the
        forced-commit turn would never fire."""
        t = ToolTranscript()
        r = ToolResult(ok=True, stdout="0", code="print(0)")
        t.add(r)
        t.add_repeat(r)
        t.add_repeat(r)
        assert t.calls == 3                                # budget of 3 is now exhausted

    def test_planner_hint_decays_after_the_first_run(self):
        cfg = dict(CFG)
        cfg["tools"] = {"enabled": True, "max_calls_per_item": 3, "timeout_s": 10}
        book = SkillBook(encoder=DualEncoder(HashingEncoder(dim=256)))
        pipe = Pipeline(book, EchoLLM(), cfg)
        plan = PlannerOutput.parse(
            {"semantic_context": "x", "domain": "other", "tags": ["strategy.casework"],
             "retrieval_query": "x", "tool_expected": True}, VocabViolations())

        empty = ToolTranscript()
        first = pipe._tool_guidance(3, empty, plan)
        assert "The planner judged" in first

        used = ToolTranscript()
        used.add(ToolResult(ok=True, stdout="0", code="print(0)"))
        later = pipe._tool_guidance(3, used, plan)
        assert "The planner judged" not in later          # the guess stops repeating
        assert "already run code 1 time(s)" in later
        assert "DIFFERENT program" in later


class TestExecIsNotAnIndependentCheck:
    """The live C2 trace made this concrete: the solver's program was wrong, it read its
    answer off that program, and `exec` certified the wrong answer at confidence 0.9 --
    higher than it gave disagreement. Agreement checks transcription; disagreement is the
    direction that carries information."""

    def test_agreement_is_weighted_below_disagreement(self):
        agree = signal_from_exec("", "0", "integer",
                                 recorded=ToolResult(ok=True, stdout="0", code="c"))
        disagree = signal_from_exec("", "70", "integer",
                                    recorded=ToolResult(ok=True, stdout="0", code="c"))
        assert agree.correct is True and disagree.correct is False
        assert agree.confidence < disagree.confidence
        assert agree.confidence == 0.5 and disagree.confidence == 0.9
        assert "transcription" in agree.detail
        assert "CONTRADICTS" in disagree.detail

    def test_evidence_weight_reflects_the_asymmetry(self):
        """confidence x gain is what actually reaches the book."""
        gains = CFG_EXEC["gains"]
        agree = signal_from_exec("", "0", "integer",
                                 recorded=ToolResult(ok=True, stdout="0", code="c"))
        disagree = signal_from_exec("", "70", "integer",
                                    recorded=ToolResult(ok=True, stdout="0", code="c"))
        gp, _ = agree.gains(CFG_EXEC)
        _, gn = disagree.gains(CFG_EXEC)
        assert agree.confidence * gp == pytest.approx(0.25)
        assert disagree.confidence * gn == pytest.approx(0.90)
        assert gains["exec"]["positive"] < gains["exec"]["negative"]

    def test_conflicting_successful_runs_discount_agreement(self):
        """`last_success` is the answer-producing program only by coincidence. Two
        successful runs printing different values means the source cannot say which one
        the answer came from."""
        runs = [ToolResult(ok=True, stdout="12", code="a"),
                ToolResult(ok=True, stdout="0", code="b")]
        sig = signal_from_exec("", "0", "integer", recorded=runs[-1], recorded_runs=runs)
        assert sig.correct is True
        assert sig.confidence == 0.3                    # discounted below the usual 0.5
        assert "CONFLICTING" in sig.detail and "'12'" in sig.detail

    def test_a_repeat_is_not_a_conflict(self):
        """An echoed repeat carries the first run's stdout, so it must not look like a
        second, disagreeing execution."""
        t = ToolTranscript()
        first = ToolResult(ok=True, stdout="0", code="print(0)")
        t.add(first)
        t.add_repeat(first)
        sig = signal_from_exec("", "0", "integer", recorded=t.last_success,
                               recorded_runs=t.successes)
        assert len(t.successes) == 2
        assert sig.confidence == 0.5                    # not discounted
        assert "CONFLICTING" not in sig.detail


class TestCuratorVerdictIsTheLabel:
    """The gap: `CuratorOutput.correct` was parsed and never read by anything, and C3's
    prompt told it to adopt the signal under `exec` -- the source we established is
    circular. C3 said `reasoning_sound: false` on a wrong answer and it changed nothing."""

    VCFG = {"judge_confidence_cap": 0.6,
            "gains": {"gt": {"positive": 1.0, "negative": 1.0},
                      "exec": {"positive": 0.5, "negative": 1.0}}}

    def test_gt_is_never_overridden(self):
        """Gold is ground truth. Letting C3 second-guess it would corrupt the supervised
        arm, which is the control the gt-vs-consistency ablation depends on."""
        assert AUTHORITATIVE_SOURCES == ("gt",)
        sig = signal_from_gt("0", "70", "integer")
        out, overridden = resolve_verdict(sig, curator_correct=True, cfg=self.VCFG)
        assert out is sig and overridden is False

    def test_curator_overrides_exec(self):
        sig = signal_from_exec("", "0", "integer",
                               recorded=ToolResult(ok=True, stdout="0", code="c"))
        assert sig.correct is True and sig.confidence == 0.5
        out, overridden = resolve_verdict(sig, curator_correct=False, cfg=self.VCFG)
        assert overridden is True
        assert out.correct is False
        assert out.source == "exec"          # the item WAS exec-supervised: ablation label
        assert out.gains_source == "judge"   # but a model produced the verdict: pricing
        assert "OVERRODE" in out.detail

    def test_agreement_is_not_an_override(self):
        sig = signal_from_exec("", "0", "integer",
                               recorded=ToolResult(ok=True, stdout="0", code="c"))
        out, overridden = resolve_verdict(sig, curator_correct=True, cfg=self.VCFG)
        assert out is sig and overridden is False

    def test_a_missing_curator_verdict_leaves_the_signal_alone(self):
        sig = signal_from_exec("", "0", "integer",
                               recorded=ToolResult(ok=True, stdout="0", code="c"))
        out, overridden = resolve_verdict(sig, curator_correct=None, cfg=self.VCFG)
        assert out is sig and overridden is False

    def test_an_override_never_raises_confidence(self):
        """An override is C3's judgement, not a stronger measurement."""
        sig = VerificationSignal(correct=True, confidence=0.95, source="exec", detail="x")
        out, _ = resolve_verdict(sig, curator_correct=False, cfg=self.VCFG)
        assert out.confidence == 0.6                   # capped at judge_confidence_cap
        low = VerificationSignal(correct=True, confidence=0.2, source="exec", detail="x")
        out2, _ = resolve_verdict(low, curator_correct=False, cfg=self.VCFG)
        assert out2.confidence == 0.2                  # and never raised


class TestCCQSRecordsTheVerdictWithoutGating:
    """A CCQS positive asserts *retrieval relevance* -- "this query should have surfaced
    this entry" -- not that the item was solved. A solver holding the right note and
    slipping on the arithmetic does not make the note less relevant, and gating on the
    outcome would discard pairs on exactly the items where the book is being built.
    Reliability is the consumer that legitimately keys on the outcome; the geometry is not.
    """

    def _trainer(self):
        enc = DualEncoder(HashingEncoder(dim=64))
        return CCQSTrainer(enc, {"enabled": True, "k_upd": 1, "min_pairs": 1})

    ATTR = {"used_positive": ["m_001"], "used_negative": [],
            "unused_irrelevant": ["m_002"], "unused_redundant": []}
    VIEWS = {"m_001": "casework | domain: math.number_theory",
             "m_002": "chirality | domain: chemistry"}

    def test_a_wrong_answer_still_yields_a_pair(self):
        t = self._trainer()
        pair = t.observe("q view", self.VIEWS, self.ATTR, 0, correct=False)
        assert pair is not None
        assert pair.positives == [self.VIEWS["m_001"]]
        assert pair.hard_negatives == [self.VIEWS["m_002"]]
        assert t.stats.pairs_seen == 1 and len(t.buffer) == 1

    def test_the_verdict_is_carried_on_the_pair(self):
        t = self._trainer()
        assert t.observe("q", self.VIEWS, self.ATTR, 0, correct=False).answer_correct is False
        assert t.observe("q", self.VIEWS, self.ATTR, 1, correct=True).answer_correct is True
        assert t.observe("q", self.VIEWS, self.ATTR, 2).answer_correct is None

    def test_pairs_from_failed_items_are_counted(self):
        """The share of training pairs drawn from wrong answers has to be measurable even
        though it is not gated on."""
        t = self._trainer()
        t.observe("q", self.VIEWS, self.ATTR, 0, correct=False)
        t.observe("q", self.VIEWS, self.ATTR, 1, correct=True)
        assert t.stats.pairs_from_wrong_answer == 1
        assert t.stats.summary()["pairs_from_wrong_answer"] == 1

    def test_the_counter_counts_pairs_not_steps(self):
        """On a cold-start step nothing is retrieved, so there is no pair to attribute to
        a wrong answer. An earlier version counted the step anyway and reported 7 affected
        pairs on a run that had none."""
        t = self._trainer()
        empty = {"used_positive": [], "used_negative": [],
                 "unused_irrelevant": [], "unused_redundant": []}
        assert t.observe("q", {}, empty, 0, correct=False) is None
        assert t.stats.pairs_from_wrong_answer == 0

    def test_no_positive_still_means_no_pair(self):
        """Unchanged: InfoNCE has no numerator without a positive."""
        t = self._trainer()
        attr = dict(self.ATTR, used_positive=[])
        assert t.observe("q", self.VIEWS, attr, 0, correct=True) is None


class TestBlameDependsOnRootCause:
    """A flat `used_but_wrong_factor` charged "the note was wrong" and "the solver slipped
    while holding a good note" identically. `root_cause` separates them. The adjustment is
    deliberately modest: it is one unvalidated categorical, and the outcome check is the
    only signal in this loop that does not come from the model."""

    BLAME = {"bad_reference": 1.0, "computational_slip": 0.25}

    def _cfg(self):
        return {"gains": {"exec": {"positive": 0.5, "negative": 1.0}},
                "used_but_wrong_factor": 0.5, "blame_by_root_cause": dict(self.BLAME)}

    def _apply(self, book, root_cause, correct):
        e = book.create(make_proposal("Modular casework", ["a", "b"]), "q0", 0)
        cur = CuratorOutput(correct=correct, reasoning_sound=False, verdict_source="self",
                            reason="r", root_cause=root_cause,
                            attribution={"used_positive": [e.id], "used_negative": [],
                                         "unused_irrelevant": [], "unused_redundant": []},
                            lesson="l", sufficient=0, proposed_entries=[])
        sig = VerificationSignal(correct=correct, confidence=1.0, source="exec", detail="")
        return apply_attribution(book, cur, sig, self._cfg(), "q0", 0)[e.id]

    def test_a_bad_reference_takes_full_blame(self, book):
        h, x = self._apply(book, "bad_reference", correct=False)
        assert h == 0.0 and x == pytest.approx(1.0)          # 1.0 conf * 1.0 gain * 1.0

    def test_a_solver_slip_blames_the_note_in_full(self, book):
        """v2: root_cause is recorded but no longer scales evidence."""
        h, x = self._apply(book, "computational_slip", correct=False)
        assert h == 0.0 and x == pytest.approx(1.0)

    def test_an_unlisted_cause_also_blames_in_full(self, book):
        h, x = self._apply(book, "conceptual_gap", correct=False)
        assert x == pytest.approx(1.0)

    def test_a_correct_answer_is_unaffected_by_root_cause(self, book):
        h, x = self._apply(book, "bad_reference", correct=True)
        assert x == 0.0 and h == pytest.approx(1.0)          # v2: 1.0 conf, gain 1.0

    def test_root_cause_is_a_closed_vocabulary(self):
        """It is load-bearing now, so a free-text value must not slip through to the
        default factor unnoticed."""
        v = VocabViolations()
        assert normalize_root_cause("Computational Slip", v) == "computational_slip"
        assert normalize_root_cause("bad-reference", v) == "bad_reference"
        assert dict(v) == {}
        assert normalize_root_cause("the solver was tired", v) == "none"
        assert v["root_cause_out_of_vocab"] == 1
        assert set(ROOT_CAUSES) == {"conceptual_gap", "computational_slip",
                                    "misread_question", "bad_reference", "format_error",
                                    "none"}



class TestParaphrasedRestatement:
    """The guard accepted an `example` that was the graded problem rewritten out of LaTeX.
    Word 8-gram overlap scored 0.000 because `$17_b$` and `17_b` tokenize the same but the
    surrounding prose does not. The multi-digit literals survive the paraphrase."""

    QUESTION = ("Find the sum of all integer bases $b>9$ for which $17_b$ is a divisor "
                "of $97_b.$")

    def _entry(self, example):
        return ProposedEntry(
            title="Base conversion for divisibility checks",
            bullets=["Convert each number from its base to decimal.",
                     "Check the divisibility condition on the decimal forms."],
            example=example, domain="math.number_theory",
            tags=["strategy.equation_manipulation"])

    def _check(self, example):
        return check_entry(self._entry(example), self.QUESTION, "70", "integer",
                           CFG["guard"])

    def test_the_real_leak_is_now_caught(self):
        """Verbatim from a run: the guard passed this at ngram_overlap 0.000."""
        leak = ("To check if 17_b divides 97_b for b=10, convert 17 and 97 from base 10 "
                "to decimal: 17_10 = 17 and 97_10 = 97. Then check if 97 % 17 == 0.")
        res = self._check(leak)
        assert res.ngram_overlap == 0.0            # the old check still sees nothing
        assert "shared_problem_numbers" in res.reasons
        assert not res.accepted

    def test_a_genuinely_different_example_passes(self):
        res = self._check("To check if 12 in base b divides 34 in base b, convert both "
                          "to decimal and test the remainder.")
        assert res.accepted, res.reasons

    def test_one_shared_number_is_not_enough(self):
        """An entry may legitimately mention a bound the question also mentions."""
        assert self._check("Search bases up to 97 before concluding none exist.").accepted

    def test_single_digits_are_never_evidence(self):
        """'modulo 7' on a question containing a 7 is a coincidence, not a restatement."""
        assert distinctive_numbers("check modulo 7 and 9 for b=5") == set()
        assert distinctive_numbers("bases 17 and 97 with 10 cases") == {"17", "97", "10"}

    def test_the_check_can_be_switched_off(self):
        cfg = dict(CFG["guard"], block_shared_numbers=False)
        leak = "convert 17 and 97 from base 10; check 97 % 17 == 0"
        res = check_entry(self._entry(leak), self.QUESTION, "70", "integer", cfg)
        assert "shared_problem_numbers" not in res.reasons


class TestOverridingAnAbsentSignal:
    """Found in the first 30-item runs: two of three `used_negative` events applied no
    evidence. `exec` had returned `correct=None` at confidence 0.0 ("recorded run printed
    nothing"), the override clamped to `min(0.0, cap)`, and the curator's verdict reached
    the book weightless."""

    VCFG = {"judge_confidence_cap": 0.6,
            "gains": {"exec": {"positive": 0.5, "negative": 1.0},
                      "judge": {"positive": 0.3, "negative": 0.6}},
            "used_but_wrong_factor": 0.5}

    def test_a_verdict_over_a_null_signal_carries_the_judge_cap(self):
        absent = VerificationSignal(correct=None, confidence=0.0, source="exec",
                                    detail="recorded run printed nothing")
        out, overridden = resolve_verdict(absent, curator_correct=False, cfg=self.VCFG)
        assert overridden is True
        assert out.correct is False
        assert out.confidence == 0.6              # was 0.0, which voided the evidence

    def test_disagreeing_with_a_real_measurement_still_cannot_raise_confidence(self):
        measured = VerificationSignal(correct=True, confidence=0.5, source="exec",
                                      detail="transcription check")
        out, _ = resolve_verdict(measured, curator_correct=False, cfg=self.VCFG)
        assert out.confidence == 0.5
        strong = VerificationSignal(correct=True, confidence=0.95, source="exec", detail="")
        out2, _ = resolve_verdict(strong, curator_correct=False, cfg=self.VCFG)
        assert out2.confidence == 0.6             # capped, not raised

    def test_the_evidence_now_actually_lands(self, book):
        """End to end: the exact shape of the two voided events."""
        e = book.create(make_proposal("Modular casework", ["a", "b"]), "q0", 0)
        absent = VerificationSignal(correct=None, confidence=0.0, source="exec", detail="")
        sig, _ = resolve_verdict(absent, curator_correct=False, cfg=self.VCFG)
        cur = CuratorOutput(correct=False, reasoning_sound=False, verdict_source="self",
                            reason="r", root_cause="conceptual_gap",
                            attribution={"used_positive": [], "used_negative": [e.id],
                                         "unused_irrelevant": [], "unused_redundant": []},
                            lesson="l", sufficient=0, proposed_entries=[])
        h, x = apply_attribution(book, cur, sig, self.VCFG, "q0", 0)[e.id]
        # v2: 0.6 capped confidence x gain 1.0. (an earlier version priced this at 0.6 gain -> 0.36.)
        assert h == 0.0 and x == pytest.approx(0.6)
        assert book.entries[e.id].meta.reliability < 0.5
        assert sig.source == "exec" and sig.gains_source == "judge"


class TestScoreTermsAreBothInRange:
    """`score = alpha*sim + (1-alpha)*r_hat` is documented as mixing two [0,1] terms.
    `r_hat` is a Beta posterior mean and is bounded by construction; `sim` is a cosine and
    is not. The floor hid that at every shipped threshold, but the domain penalty is
    multiplicative, so on a negative cosine it moved the score UP: a domain mismatch
    improved the entry. `sim_threshold: -1.0`, the natural way to disable the floor, would
    have silently inverted the domain filter."""

    def _retriever(self, book, threshold):
        return Retriever(book, {"top_k": 3, "sim_threshold": threshold, "alpha": 0.7,
                                "domain_filter": "soft", "domain_penalty": 0.25,
                                "domain_partial_credit": 0.6, "mmr_lambda": 0.7})

    def _plan(self, domain="math.number_theory"):
        return PlannerOutput.parse(
            {"semantic_context": "modular casework", "domain": domain,
             "tags": ["strategy.casework"], "retrieval_query": "modular casework",
             "tool_expected": False}, VocabViolations())

    def test_a_negative_cosine_never_produces_a_negative_sim_term(self, book):
        book.create(make_proposal("Chirality of stereocenters", ["a", "b"],
                                  domain="chemistry"), "q0", 0)
        refs = self._retriever(book, -1.0).retrieve(self._plan(), 0)
        for r in refs:
            # raw_sim may be negative; the mixed score must not be dragged below the
            # reliability floor of (1-alpha)*0.5 by it.
            assert r.score >= (1 - 0.7) * 0.5 - 1e-9

    def test_the_domain_penalty_can_only_reduce_the_score(self, book):
        """The bug, stated as an invariant: penalising a mismatch must never help."""
        for raw, aff in ((0.8, 1.0), (0.8, 0.0), (-0.5, 1.0), (-0.5, 0.0)):
            clamped = max(0.0, raw)
            penalised = clamped * (1.0 - 0.25 * (1.0 - aff))
            assert penalised <= clamped + 1e-12
            assert penalised >= 0.0

    def test_reliability_is_bounded_by_construction(self):
        m = EntryMeta()
        assert m.recompute_reliability() == 0.5
        for h, x in ((100.0, 0.0), (0.0, 100.0), (3.5, 2.5), (0.0, 0.0)):
            m.helpful, m.harmful = h, x
            assert 0.0 < m.recompute_reliability() < 1.0


class TestUpdateGateMatchesWhatTheLossNeeds:
    """`due()` counted anchors while InfoNCE needs anchors WITH negatives, so it could
    return True on an update `_loss` would then skip silently. With no negative the
    denominator collapses to the numerator, p = 1, the loss is 0 and the gradient is
    exactly zero -- and on L2-normalized vectors "pull together" with nothing pushing back
    has the trivial solution of collapsing every point onto one."""

    def _trainer(self, **over):
        cfg = {"enabled": True, "k_upd": 1, "min_pairs": 1}
        cfg.update(over)
        return CCQSTrainer(DualEncoder(HashingEncoder(dim=64)), cfg)

    VIEWS = {"a": "casework | domain: math.number_theory",
             "b": "chirality | domain: chemistry",
             "c": "invariants | domain: math.geometry"}

    def _attr(self, pos, neg=()):
        return {"used_positive": list(pos), "used_negative": [],
                "unused_irrelevant": list(neg), "unused_redundant": []}

    def test_a_lone_positive_only_pair_is_not_trainable(self):
        t = self._trainer()
        t.observe("q1", self.VIEWS, self._attr(["a"]), 0, correct=True)
        assert len(t.buffer) == 1
        assert t.trainable_pairs() == 0
        assert t.due(0) is False                 # would have been True on count alone
        assert t.maybe_update(0) is None

    def test_hard_negatives_make_a_single_pair_trainable(self):
        t = self._trainer()
        t.observe("q1", self.VIEWS, self._attr(["a"], ["b"]), 0, correct=True)
        assert t.trainable_pairs() == 1
        assert t.due(0) is True
        assert t.maybe_update(0) is not None     # a real gradient step

    def test_two_pairs_with_distinct_positives_are_trainable_in_batch(self):
        t = self._trainer()
        t.observe("q1", self.VIEWS, self._attr(["a"]), 0, correct=True)
        t.observe("q2", self.VIEWS, self._attr(["c"]), 1, correct=True)
        assert t.trainable_pairs() == 2          # each is the other's in-batch negative
        assert t.due(1) is True

    def test_two_pairs_sharing_one_positive_are_not(self):
        """The degenerate case: many pairs, no update."""
        t = self._trainer()
        t.observe("q1", self.VIEWS, self._attr(["a"]), 0, correct=True)
        t.observe("q2", self.VIEWS, self._attr(["a"]), 1, correct=True)
        assert len(t.buffer) == 2 and t.trainable_pairs() == 0
        assert t.due(1) is False

    def test_min_pairs_still_gates_independently(self):
        t = self._trainer(min_pairs=4)
        for i in range(3):
            t.observe("q%d" % i, self.VIEWS, self._attr(["a"], ["b"]), i, correct=True)
        assert t.trainable_pairs() == 3 and t.due(2) is False    # trainable but too few
        t.observe("q3", self.VIEWS, self._attr(["a"], ["b"]), 3, correct=True)
        assert t.due(3) is True

    def test_shipped_configs_do_not_relax_min_pairs(self):
        import yaml
        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent /
             "configs" / "lere.yaml").read_text(encoding="utf-8"))
        assert cfg["ccqs"]["min_pairs"] >= 4


class TestGpqaLoader:
    """`prompts/taxonomy.md` singles this dataset out: n_options is 4, not the 10 `Item`
    defaults to. A wrong value makes `normalize_mcq` accept letters that were never on
    offer, so a solver answering 'G' would be scored rather than rejected."""

    needs = pytest.mark.skipif(not (DATASET_DIR / "GPQA_Diamond").is_dir(),
                               reason="dataset/ not present in this checkout")

    @needs
    def test_shape_and_option_count(self):
        items = load_gpqa_diamond()
        assert len(items) == 198
        assert all(i.n_options == 4 for i in items)
        assert all(i.answer_type == "mcq_letter" for i in items)
        assert all(i.image is None for i in items)
        assert [i.id for i in items[:2]] == ["GPQA_Diamond-0000", "GPQA_Diamond-0001"]

    @needs
    def test_every_gold_is_one_of_the_four_letters(self):
        for it in load_gpqa_diamond():
            assert normalize_mcq(it.gold, it.n_options) in ("A", "B", "C", "D")
            assert is_correct(it.gold, it.gold, it.answer_type, n_options=it.n_options)
            assert is_correct(normalize_mcq(it.gold, 4), it.gold, it.answer_type,
                              n_options=it.n_options)          # bare letter also scores

    @needs
    def test_options_stay_inline_in_the_question(self):
        """C2 is shown the question verbatim; if the options were stripped it would be
        answering a multiple-choice item with no choices."""
        for it in load_gpqa_diamond()[:20]:
            assert "Options:" in it.question
            assert "(A)" in it.question and "(D)" in it.question

    @needs
    def test_slicing_is_stable(self):
        assert load_gpqa_diamond(limit=3)[2].id == "GPQA_Diamond-0002"
        assert load_gpqa_diamond(offset=2, limit=1)[0].id == "GPQA_Diamond-0002"
        with pytest.raises(DatasetError):
            load_gpqa_diamond(limit=999)


class TestThresholdSentinel:
    """`sim_threshold: false` means no floor. Distinct from 0.0, which still drops
    negative cosines, and clearer than the -1.0 magic number."""

    def test_parsing(self):
        from lere.retrieve import parse_threshold
        assert parse_threshold(False) is None
        assert parse_threshold(None) is None
        assert parse_threshold(0.6) == 0.6
        assert parse_threshold(0) == 0.0          # a real zero, not the sentinel
        assert parse_threshold("0.45") == 0.45

    def test_no_floor_admits_a_negative_cosine(self, book):
        book.create(make_proposal("Chirality of stereocenters", ["a", "b"],
                                  domain="chemistry"), "q0", 0)
        plan = PlannerOutput.parse(
            {"semantic_context": "modular casework", "domain": "math.number_theory",
             "tags": ["strategy.casework"], "retrieval_query": "modular casework",
             "tool_expected": False}, VocabViolations())
        cfg = {"top_k": 3, "alpha": 0.7, "domain_filter": "soft", "domain_penalty": 0.25,
               "domain_partial_credit": 0.6, "mmr_lambda": 0.7}
        assert Retriever(book, {**cfg, "sim_threshold": 0.6}).retrieve(plan, 0) == []
        no_floor = Retriever(book, {**cfg, "sim_threshold": False}).retrieve(plan, 0)
        assert len(no_floor) == 1
        assert no_floor[0].score >= 0.0            # the clamp still holds

    def test_tracing_retriever_agrees_with_the_real_one(self, book):
        book.create(make_proposal("Modular casework", ["a", "b"]), "q0", 0)
        plan = PlannerOutput.parse(
            {"semantic_context": "modular casework", "domain": "math.number_theory",
             "tags": ["strategy.casework"], "retrieval_query": "modular casework",
             "tool_expected": False}, VocabViolations())
        cfg = {"top_k": 3, "sim_threshold": False, "alpha": 0.7, "domain_filter": "soft",
               "domain_penalty": 0.25, "domain_partial_credit": 0.6, "mmr_lambda": 0.7}
        r = TracingRetriever(book, cfg, JsonlWriter("/dev/null"))
        sel = r.retrieve(plan, 0)
        assert r.last["sim_threshold"] is None
        assert all(c["passed_floor"] for c in r.last["candidates"])
        assert r.last["scoring_drift"] == []
        assert [s.id for s in sel] == r.last["selected_ids"]


class TestSympyIsAvailable:
    def test_the_probe_reports_it_and_it_runs(self):
        """C2's prompt advertises exactly what `probe_modules` finds, so this is the only
        thing standing between a `from sympy import ...` and a wasted turn."""
        cfg = ToolConfig.from_cfg({"enabled": True, "timeout_s": 30,
                                   "max_output_chars": 4000, "memory_mb": 2048})
        mods = probe_modules(cfg)
        assert "sympy" in mods and "numpy" in mods
        r = run_python("from sympy import symbols, solve, Eq\n"
                       "x = symbols('x')\nprint(solve(Eq(x**2 - 4, 0), x))", cfg)
        assert r.ok and r.stdout.strip() == "[-2, 2]"


class TestMathVistaLoader:
    """This subset is not what `prompts/taxonomy.md` leads you to expect: all 250 rows are
    multi_choice with answer_type 'text', `answer` is the choice STRING not a letter, and
    n_options runs 2..7. Each of those, taken on faith, produces a silently wrong gold."""

    needs = pytest.mark.skipif(
        not (DATASET_DIR / "MathVista_testmini_250").is_dir(),
        reason="dataset/ not present in this checkout")

    @needs
    def test_shape(self):
        items = load_mathvista_testmini_250()
        assert len(items) == 250
        assert {i.answer_type for i in items} == {"mcq_letter"}
        assert min(i.n_options for i in items) == 2
        assert max(i.n_options for i in items) == 7
        assert all(i.image for i in items)

    @needs
    def test_gold_is_the_letter_at_the_answer_index(self):
        """`answer` is the choice text; gold must be its position as a letter, and that
        letter must match the lettered options already rendered in the question."""
        import re
        for it in load_mathvista_testmini_250():
            assert it.gold in "ABCDEFG"
            assert ord(it.gold) - ord("A") < it.n_options
            # the question renders "(X) <choice text>" for the gold letter
            pat = r"\(%s\)\s*%s" % (it.gold, re.escape(it.meta["answer_text"]))
            assert re.search(pat, it.question), it.id
            assert is_correct(it.gold, it.gold, it.answer_type, n_options=it.n_options)

    @needs
    def test_n_options_is_per_item_not_a_constant(self):
        """A fixed n_options would let `normalize_mcq` accept letters never offered on
        the 2-choice items, or reject valid ones on the 7-choice items."""
        items = load_mathvista_testmini_250()
        assert len({i.n_options for i in items}) > 1
        two = [i for i in items if i.n_options == 2][0]
        assert normalize_mcq("C", two.n_options) is None       # never offered
        seven = [i for i in items if i.n_options == 7][0]
        assert normalize_mcq("G", seven.n_options) == "G"      # legitimately offered

    @needs
    def test_images_are_a_format_the_provider_can_send(self):
        from lere.providers import sniff_mime
        mimes = {sniff_mime(i.image) for i in load_mathvista_testmini_250()}
        assert mimes <= {"image/png", "image/jpeg", "image/webp"}


class TestAlternativePromptSet:
    """`prompts.dir` lets a revised prompt set be A/B'd against the shipped one without
    editing `prompts/` in place. These tests build their own set so they never depend on
    one existing."""

    def _make_set(self, tmp_path, marker="SENTINEL RULE"):
        d = tmp_path / "prompts_alt"
        d.mkdir()
        for name in ("c1_planner.md", "c2_solver.md", "c3_curator.md", "taxonomy.md"):
            text = load_prompt(name)
            if name == "c2_solver.md":
                text += "\n\n" + marker + "\n"
            (d / name).write_text(text, encoding="utf-8")
        return d

    def test_default_is_the_shipped_set(self):
        assert load_prompt("c2_solver.md") == load_prompt("c2_solver.md", None)
        assert "# C2 — Problem solver" in load_prompt("c2_solver.md")

    def test_an_absolute_dir_selects_that_set(self, tmp_path):
        d = self._make_set(tmp_path)
        alt = load_prompt("c2_solver.md", d)
        assert alt != load_prompt("c2_solver.md")
        assert "SENTINEL RULE" in alt

    def test_a_relative_dir_resolves_against_the_repo_root(self):
        assert load_prompt("c2_solver.md", "prompts") == load_prompt("c2_solver.md")

    def test_a_missing_prompt_raises_rather_than_rendering_empty(self):
        """`render` fills unknown slots with empty string, so a silently missing template
        would produce a blank prompt and a run of meaningless items."""
        with pytest.raises(FileNotFoundError):
            load_prompt("c2_solver.md", "prompts_does_not_exist")

    def test_pipeline_honours_the_config(self, book, tmp_path):
        d = self._make_set(tmp_path)
        cfg = dict(CFG)
        cfg["prompts"] = {"dir": str(d)}
        pipe = Pipeline(book, EchoLLM(), cfg)
        assert pipe.prompt_dir == str(d)
        assert "SENTINEL RULE" in pipe.p_c2
        assert Pipeline(book, EchoLLM(), CFG).prompt_dir is None

    def test_a_revised_set_must_keep_every_slot_the_pipeline_fills(self, tmp_path):
        """A revision that dropped a slot would render it as empty text and silently lose,
        say, the retrieved references or the closed vocabulary."""
        d = self._make_set(tmp_path)
        for name in ("c1_planner.md", "c2_solver.md", "c3_curator.md"):
            base = set(re.findall(r"\{\{([a-z_]+)\}\}", load_prompt(name)))
            alt = set(re.findall(r"\{\{([a-z_]+)\}\}", load_prompt(name, d)))
            assert base == alt, (name, base ^ alt)

