#!/usr/bin/env python3
"""
Simple integration test for LeRe pipeline.
Tests the core v6 functionality without needing the full environment.
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main.utils.lere_extractor import (
    extract_trajectory,
    extract_memory_audit,
    extract_answer,
    extract_all_from_generator,
    extract_reflection,
    extract_curation,
    parse_trajectory_steps,
)
from main.utils.lere_pipeline import LeRePipeline


def test_generator_extraction():
    """Test extraction of generator outputs (trajectory, memory audit, answer)."""
    print("\n" + "="*80)
    print("TEST 1: Generator Output Extraction")
    print("="*80)

    # Simulate a generator response with trajectory, memory_audit, and answer
    gen_response = """
Let me work through this step by step.

<trajectory>
<step id="1" type="analysis" memory_refs="[m_001, m_002]" timestamp="T1">
First, I need to understand the problem statement carefully.
</step>
<step id="2" type="strategy" memory_refs="[m_001]" timestamp="T2">
Based on the pattern, I'll use brute-force enumeration.
</step>
<step id="3" type="computation" memory_refs="[]" timestamp="T3">
Computing all possible combinations...
</step>
<step id="4" type="verification" memory_refs="[m_002]" timestamp="T4">
Verifying the result matches the expected format.
</step>
</trajectory>

<memory_audit>
<used>
<entry id="m_001">
This item helped me understand the problem structure.
</entry>
<entry id="m_002">
This pattern is directly applicable to the verification step.
</entry>
</used>
<unused>
<entry id="m_003" />
<entry id="m_005" />
</unused>
</memory_audit>

<answer>(C)</answer>
"""

    extracted = extract_all_from_generator(gen_response)

    print(f"✓ Trajectory extracted: {len(extracted['trajectory'])} chars")
    print(f"✓ Parsed steps: {len(extracted['parsed_steps'])} steps")
    for step in extracted['parsed_steps']:
        print(f"  - Step {step.get('id')}: {step.get('type')} (refs: {step.get('memory_refs')})")

    print(f"✓ Memory consulted: {extracted['memory_consulted']}")
    print(f"✓ Memory unused: {extracted['memory_unused']}")
    print(f"✓ Final answer: '{extracted['final_answer']}'")

    # Verify key fields
    assert len(extracted['parsed_steps']) == 4, "Should extract 4 trajectory steps"
    assert extracted['memory_consulted'] == ['m_001', 'm_002'], "Should identify consulted items"
    assert extracted['memory_unused'] == ['m_003', 'm_005'], "Should identify unused items"
    assert extracted['final_answer'] == '(C)', "Should extract answer"

    print("\n✅ Generator extraction test PASSED")
    return True


def test_reflector_extraction():
    """Test extraction of reflector outputs (reflection JSON)."""
    print("\n" + "="*80)
    print("TEST 2: Reflector Output Extraction")
    print("="*80)

    reflector_response = """
<reflection>
{
  "execution_status": "SUCCESS",
  "trajectory_analysis": {
    "critical_steps": [
      {
        "step_id": "2",
        "step_type": "strategy",
        "description": "Switched to brute-force enumeration",
        "impact": "POSITIVE",
        "reasoning": "This approach guaranteed correctness",
        "lesson_and_insights": "For combinatorial problems, exhaustive search is more reliable than heuristics",
        "memory_influence": ["m_001"]
      }
    ]
  },
  "memory_evaluation": [
    {
      "item_id": "m_001",
      "title": "Brute-force Enumeration Pattern",
      "usage_context": "Steps 1, 2",
      "verdict": "HELPFUL",
      "reason": "Pattern directly applied and worked"
    },
    {
      "item_id": "m_002",
      "title": "Verification Heuristic",
      "usage_context": "Step 4",
      "verdict": "NEUTRAL",
      "reason": "Item was consulted but had no discernible effect"
    }
  ]
}
</reflection>
"""

    reflection = extract_reflection(reflector_response)

    assert reflection is not None, "Should extract reflection"
    print(f"✓ Execution status: {reflection['execution_status']}")
    print(f"✓ Critical steps: {len(reflection['trajectory_analysis']['critical_steps'])}")
    print(f"✓ Memory evaluations: {len(reflection['memory_evaluation'])}")

    for eval_item in reflection['memory_evaluation']:
        print(f"  - {eval_item['item_id']}: {eval_item['verdict']}")

    # Verify structure
    assert reflection['execution_status'] == 'SUCCESS'
    assert len(reflection['trajectory_analysis']['critical_steps']) == 1
    assert len(reflection['memory_evaluation']) == 2

    print("\n✅ Reflector extraction test PASSED")
    return True


def test_curator_extraction():
    """Test extraction of curator outputs (new memory entries)."""
    print("\n" + "="*80)
    print("TEST 3: Curator Output Extraction")
    print("="*80)

    curator_response = """
```json
{
  "rationale": "The reflection shows the agent successfully used brute-force enumeration. This pattern should be documented for future similar problems.",
  "new_entries": [
    {
      "title": "Brute-force Enumeration for Game of 24",
      "bullets": [
        "For combinatorial problems, enumerate all possible operator and operand orderings",
        "Use itertools.permutations to generate combinations efficiently",
        "Always verify the final result numerically before reporting"
      ],
      "example": "For Game of 24: use itertools.permutations to try all (num1 op1 num2 op2 num3 op3 num4) combinations until one equals 24",
      "tags": ["math.combinatorics", "python.execution", "strategy.verification"],
      "scope": "Applies to combinatorial puzzle-solving problems where manual enumeration is infeasible",
      "meta": {
        "helpful": 0,
        "harmful": 0,
        "created": "Q_001",
        "source_queries": ["Q_001"],
        "reliability": 0.0,
        "retrieved_count": 0,
        "last_used": null
      },
      "id": "m_043"
    }
  ]
}
```
"""

    curation = extract_curation(curator_response)

    assert curation is not None, "Should extract curation"
    print(f"✓ Rationale: {curation['rationale'][:80]}...")
    print(f"✓ New entries: {len(curation['new_entries'])}")

    for entry in curation['new_entries']:
        print(f"  - {entry['id']}: {entry['title']}")
        print(f"    Bullets: {len(entry['bullets'])}")
        print(f"    Tags: {entry['tags']}")

    # Verify structure
    assert len(curation['new_entries']) == 1
    entry = curation['new_entries'][0]
    assert entry['id'] == 'm_043'
    assert len(entry['bullets']) == 3
    assert entry['meta']['created'] == 'Q_001'

    print("\n✅ Curator extraction test PASSED")
    return True


def test_pipeline_formatting():
    """Test that pipeline formatting methods work correctly."""
    print("\n" + "="*80)
    print("TEST 4: Pipeline Prompt Formatting")
    print("="*80)

    # Create a simple pipeline with dummy templates
    gen_template = "Question: [[REFERENCE_MEMORY]]\n[[QUESTION]]\n[[CONTEXT]]"
    ref_template = "Trajectory: [[TRAJECTORY]]\nAnswer: [[FINAL_ANSWER]]\nResult: [[EXECUTION_RESULT]]"
    cur_template = "Memory: [[CURRENT_MEMORY]]\nOutput: [[REFLECTOR_OUTPUT]]"

    pipeline = LeRePipeline(gen_template, ref_template, cur_template)

    # Test generator prompt formatting
    gen_prompt = pipeline.format_generator_prompt(
        question="What is 2+2?",
        memory_briefing="Memory Item 1: Basic arithmetic",
        context="Math question"
    )
    assert "What is 2+2?" in gen_prompt
    assert "Memory Item 1" in gen_prompt
    print(f"✓ Generator prompt formatted: {len(gen_prompt)} chars")

    # Test reflector prompt formatting
    ref_prompt = pipeline.format_reflector_prompt(
        question="What is 2+2?",
        retrieved_memory=[],
        trajectory="Step 1: Add numbers",
        final_answer="4",
        execution_result="SUCCESS"
    )
    assert "Step 1" in ref_prompt
    assert "SUCCESS" in ref_prompt
    print(f"✓ Reflector prompt formatted: {len(ref_prompt)} chars")

    # Test curator prompt formatting
    cur_prompt = pipeline.format_curator_prompt(
        memory_bank=[],
        reflector_output={"status": "success"},
        query_id="Q_001"
    )
    assert len(cur_prompt) > 0
    print(f"✓ Curator prompt formatted: {len(cur_prompt)} chars")

    print("\n✅ Pipeline formatting test PASSED")
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*80)
    print("LeRe PIPELINE INTEGRATION TEST")
    print("="*80)

    tests = [
        ("Generator Extraction", test_generator_extraction),
        ("Reflector Extraction", test_reflector_extraction),
        ("Curator Extraction", test_curator_extraction),
        ("Pipeline Formatting", test_pipeline_formatting),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, "PASS"))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, "FAIL"))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results:
        status = "✅" if result == "PASS" else "❌"
        print(f"{status} {test_name}: {result}")

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
