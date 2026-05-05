"""
LeRe Framework Extractors

Extracts structured data from:
- Generator: trajectory, memory audit, final answer
- Reflector: reflection JSON with trajectory analysis and memory evaluation
- Curator: curation JSON with new memory entries
"""

import json
import re
from typing import Dict, List, Optional


def _fix_json_escapes(json_str: str) -> str:
    """
    Fix invalid JSON escape sequences (common with LaTeX notation).
    Doubles backslashes that aren't valid JSON escapes.
    """
    def double_backslashes_in_strings(match):
        opening_quote = match.group(1)
        content = match.group(2)
        closing_quote = match.group(3)
        fixed_content = content.replace('\\', '\\\\')
        return f'{opening_quote}{fixed_content}{closing_quote}'

    pattern = r'(")([^"\\]*(?:\\.[^"\\]*)*)(")'
    return re.sub(pattern, double_backslashes_in_strings, json_str)


def extract_trajectory(response: str) -> str:
    """
    Extract trajectory from generator response.

    Format:
    <trajectory>
    <step id="1" type="analysis" memory_refs="[]" timestamp="T1">
    [content]
    </step>
    ...
    </trajectory>
    """
    if "<trajectory>" not in response:
        return ""

    try:
        trajectory = response.split("<trajectory>")[1].split("</trajectory>")[0].strip()
        return trajectory
    except (IndexError, AttributeError):
        return ""


def parse_trajectory_steps(trajectory_text: str) -> List[Dict]:
    """
    Parse trajectory XML into structured steps.
    """
    steps = []
    step_pattern = r'<step\s+([^>]*)>(.*?)</step>'

    for match in re.finditer(step_pattern, trajectory_text, re.DOTALL):
        attrs_str = match.group(1)
        content = match.group(2).strip()

        # Parse attributes
        step_dict = {"content": content}

        # Extract id, type, memory_refs, timestamp
        id_match = re.search(r'id=["\']([^\'"]+)["\']', attrs_str)
        if id_match:
            step_dict["id"] = id_match.group(1)

        type_match = re.search(r'type=["\']([^\'"]+)["\']', attrs_str)
        if type_match:
            step_dict["type"] = type_match.group(1)

        timestamp_match = re.search(r'timestamp=["\']([^\'"]+)["\']', attrs_str)
        if timestamp_match:
            step_dict["timestamp"] = timestamp_match.group(1)

        # Extract memory_refs
        refs_match = re.search(r'memory_refs=["\']([^\'"]*)["\']', attrs_str)
        if refs_match:
            refs_str = refs_match.group(1)
            # Parse [m_001, m_002] or empty []
            memory_refs = re.findall(r'm_\d+', refs_str)
            step_dict["memory_refs"] = memory_refs
        else:
            step_dict["memory_refs"] = []

        steps.append(step_dict)

    return steps


def extract_memory_audit(response: str) -> Dict[str, List[str]]:
    """
    Extract memory audit from generator response.

    Format:
    <memory_audit>
    <used>
    <entry id="m_042">
    [Why this item was helpful...]
    </entry>
    </used>
    <unused>
    <entry id="m_021" />
    </unused>
    </memory_audit>
    """
    result = {"used": [], "unused": []}

    if "<memory_audit>" not in response:
        return result

    try:
        audit_section = response.split("<memory_audit>")[1].split("</memory_audit>")[0].strip()
    except (IndexError, AttributeError):
        return result

    # Extract used items with impact type
    if "<used>" in audit_section:
        try:
            used_section = audit_section.split("<used>")[1].split("</used>")[0]
            # Match entries with impact attribute: <entry id="m_042" impact="...">
            used_entries = re.findall(
                r'<entry\s+id="(m_\d+)"\s+impact="([^"]*)"', used_section
            )
            # Also match entries without impact attribute (backwards compat)
            used_ids_no_impact = re.findall(
                r'<entry\s+id="(m_\d+)"\s*>', used_section
            )
            # Build structured list: [(id, impact), ...]
            used_with_impact = {eid: imp for eid, imp in used_entries}
            for eid in used_ids_no_impact:
                if eid not in used_with_impact:
                    used_with_impact[eid] = "unspecified"
            result["used"] = list(used_with_impact.keys())
            result["used_with_impact"] = used_with_impact  # {id: impact_type}
        except (IndexError, AttributeError):
            pass

    # Extract unused items
    if "<unused>" in audit_section:
        try:
            unused_section = audit_section.split("<unused>")[1].split("</unused>")[0]
            unused_ids = re.findall(r'<entry\s+id="(m_\d+)"\s*/>', unused_section)
            result["unused"] = unused_ids
        except (IndexError, AttributeError):
            pass

    return result


def _extract_mc_letter_fallback(response: str) -> str:
    """
    Fallback patterns for multiple-choice letter extraction when primary markers are absent.
    Tries patterns in order of specificity; returns the matched letter or empty string.
    """
    text = response.strip()

    # 1. **Answer**: A  /  Answer: A  (case-insensitive, optional markdown bold/italic)
    m = re.search(r'(?i)[*\_]{0,2}Answer[*\_]{0,2}\s*:\s*[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])', text)
    if m:
        return m.group(1)

    # 2. LaTeX boxed: \boxed{A} or \boxed{The answer is A}
    m = re.search(r'\\boxed\{[^}]*([A-Z])[^}]*\}', text)
    if m:
        return m.group(1)

    # 3. "answer is (A)" — with parenthesis
    m = re.search(r'(?i)answer\s+is\s+\(([A-Za-z])\)', text)
    if m:
        return m.group(1).upper()

    # 4. "answer is A" — natural language
    m = re.search(r'(?i)answer\s+is\s+([A-Za-z])(?![a-zA-Z])', text)
    if m:
        return m.group(1).upper()

    # 5. "A is the correct answer"
    m = re.search(r'([A-Z])\s+is\s+the\s+correct\s+answer', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 6. Choice format: "A) some text" — only if near end of response (last 300 chars)
    tail = text[-300:]
    m = re.search(r'([A-Z])\)\s*[^A-Z\n]{0,80}$', tail)
    if m:
        return m.group(1)

    # 7. Standalone letter at end of response
    m = re.search(r'([A-Z])\s*$', text)
    if m:
        return m.group(1)

    # 8. Letter followed by period at end
    m = re.search(r'([A-Z])\s*\.\s*$', text)
    if m:
        return m.group(1)

    # 9. Letter followed by non-word character at end
    m = re.search(r'([A-Z])\s*[^\w]\s*$', text)
    if m:
        return m.group(1)

    return ""


def extract_answer(response: str) -> str:
    """
    Extract final answer from generator response.

    Handles (in order):
    1. <answer>...</answer> tags (primary format)
    2. FINAL ANSWER: prefix with optional </answer> closing tag, \boxed{}, code blocks
    3. \boxed{...} standalone (LaTeX-style models)
    4. MC letter fallback patterns (Answer: X, answer is X, etc.)
    """
    if "<answer>" in response:
        try:
            txt = response.split("<answer>")[-1].strip()
            txt = txt.split("</answer>")[0].strip()
            return txt
        except Exception:
            return ""

    if "FINAL ANSWER" not in response:
        # Try \boxed{...} fallback
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        # Try MC letter fallback patterns
        letter = _extract_mc_letter_fallback(response)
        return letter if letter else ""

    try:
        tail = response.split("FINAL ANSWER")[-1].strip()
        if tail and tail[0] == ":":
            tail = tail[1:].strip()

        # Strip \boxed{...} wrapper
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', tail)
        if boxed_match:
            tail = boxed_match.group(1).strip()

        # Malformed: closing </answer> tag without opening — extract content before it
        if "</answer>" in tail:
            tail = tail.split("</answer>")[0].strip()
            if len(tail.split("\n")) <= 3 and len(tail) < 200:
                return tail

        # No code blocks — return plain text
        idx_1 = tail.find("'''")
        idx_2 = tail.find("```")
        if idx_1 == -1 and idx_2 == -1:
            lines = tail.split("\n")
            if len(lines) <= 5 and len(tail) < 500:
                return tail
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
            return tail

        # Content before the first code block
        pre_block = tail.split("```")[0] if "```" in tail else tail.split("'''")[0]
        pre_block = pre_block.strip()
        if pre_block:
            for line in pre_block.split("\n"):
                line = line.strip()
                if line and not line.startswith('#'):
                    return line
            return pre_block

        # Content inside code block
        if min(idx_1 if idx_1 != -1 else idx_2, idx_2 if idx_2 != -1 else idx_1) == idx_1 and idx_1 != -1:
            inner = tail.split("'''")[1].strip()
        else:
            inner = tail.split("```")[1].strip()
        if "```" in inner:
            inner = inner.split("```")[0].strip()
        elif "'''" in inner:
            inner = inner.split("'''")[0].strip()
        if inner and inner.split("\n")[0].strip().lower() == "python":
            inner = "\n".join(inner.split("\n")[1:]).strip()
        return inner if inner else ""
    except Exception:
        return ""


def extract_all_from_generator(response: str) -> Dict:
    """
    Extract all relevant information from v6 generator response.
    """
    trajectory = extract_trajectory(response)
    memory_audit = extract_memory_audit(response)
    parsed_steps = parse_trajectory_steps(trajectory)

    # Any entry in <memory_audit><used> is considered consulted — the impact attribute
    # distinguishes how it was used (critical_step / overall_reasoning / custom).
    memory_consulted = memory_audit["used"]           # list of IDs
    memory_consulted_impact = memory_audit.get("used_with_impact", {})  # {id: impact_type}

    return {
        "trajectory": trajectory,
        "parsed_steps": parsed_steps,
        "memory_consulted": memory_consulted,
        "memory_consulted_impact": memory_consulted_impact,
        "memory_unused": memory_audit["unused"],
        "final_answer": extract_answer(response),
        "full_response": response,
    }


def extract_reflection(response: str) -> Optional[Dict]:
    """
    Extract reflection from v6 reflector response.

    Format:
    <reflection>
    {
      "execution_status": "SUCCESS|FAILURE",
      "trajectory_analysis": {
        "critical_steps": [...]
      },
      "memory_evaluation": [...]
    }
    </reflection>
    """
    if "<reflection>" not in response:
        return None

    try:
        json_str = response.split("<reflection>")[1].split("</reflection>")[0].strip()
    except (IndexError, AttributeError):
        return None

    try:
        reflection = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            reflection = json.loads(_fix_json_escapes(json_str))
        except Exception:
            return None

    # Validate v6 structure
    if not isinstance(reflection, dict):
        return None

    required_fields = ["execution_status", "trajectory_analysis", "memory_evaluation"]
    for field in required_fields:
        if field not in reflection:
            return None

    return reflection


def extract_curation(response: str) -> Optional[Dict]:
    """
    Extract curation from v6 curator response.

    Format:
    {
      "rationale": "...",
      "new_entries": [
        {
          "title": "...",
          "bullets": [...],
          "example": "...",
          "tags": [...],
          "scope": "...",
          "meta": {...},
          "id": "m_XXX"
        }
      ]
    }
    """
    json_str = None

    # Try: wrapped in markdown code fences
    if "```json" in response:
        try:
            json_str = response.split("```json")[1].split("```")[0].strip()
        except (IndexError, AttributeError):
            pass

    # Try: plain JSON
    if json_str is None:
        txt = response.strip()
        if txt.startswith("{") and txt.endswith("}"):
            json_str = txt

    if json_str is None:
        return None

    try:
        curation = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            curation = json.loads(_fix_json_escapes(json_str))
        except Exception:
            return None

    # Validate v6 structure
    if not isinstance(curation, dict):
        return None

    if "rationale" not in curation or "new_entries" not in curation:
        return None

    if not isinstance(curation["new_entries"], list):
        return None

    return curation
