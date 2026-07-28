"""
Learning to Remember (LeRe) Pipeline Orchestrator

This module provides the main execution pipeline for Generator → Reflector → Curator workflow.
It handles:
- Prompt formatting for each stage
- Data extraction and validation
- Memory retrieval and formatting
- Pipeline orchestration

Pipeline Flow:
1. GENERATOR: Solve problem with retrieved memory items + trajectory logging
2. REFLECTOR: Analyze trajectory and evaluate memory usage
3. CURATOR: Update memory bank based on reflection
"""

from typing import Dict, List, Optional, Tuple
import json
import re

from .lere_extractor import (
    extract_all_from_generator,
    extract_reflection,
    extract_curation,
)
from .memory_formatter import (
    format_memory_bank_for_generator,
    format_memory_bank_for_reflector,
    format_memory_bank_for_curator,
    get_memory_items_by_ids,
    apply_curation_updates,
    get_memory_health_stats,
    calculate_reliability,
    apply_merge_operator
)


def _normalize_answer_for_comparison(ans: str) -> str:
    """Normalize an answer string for comparison by stripping wrappers and punctuation."""
    if not ans:
        return ""
    s = str(ans).strip()
    # Remove closing XML-style tags
    s = re.sub(r'</[A-Za-z]+>', '', s).strip()
    # Iteratively strip outer parentheses and angle brackets
    changed = True
    while changed:
        changed = False
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
            changed = True
        if s.startswith("<") and s.endswith(">"):
            s = s[1:-1].strip()
            changed = True
    # Remove common punctuation
    for ch in [",", ";", ":", ".", '"']:
        s = s.replace(ch, "")
    # Remove remaining parentheses (e.g. standalone "(70)" already handled, but also "70)")
    s = s.replace("(", "").replace(")", "")
    return s.strip().lower()


class LeRePipeline:
    """
    Orchestrates the LeRe learning pipeline.

    Workflow:
    1. Generator: Solve problem using retrieved memory items
    2. Reflector: Analyze trajectory and evaluate memory usage
    3. Curator: Update memory bank based on reflection
    """

    def __init__(
        self,
        generator_prompt_template: str,
        reflector_prompt_template: str,
        curator_prompt_template: str,
        synthesizer_prompt_template: Optional[str] = None,
    ):
        """
        Initialize the pipeline with prompt templates.

        Args:
            generator_prompt_template: Template with placeholders [[MEMORY_BANK]], [[QUESTION]], [[CONTEXT]]
            reflector_prompt_template: Template with placeholders for reflector inputs
            curator_prompt_template: Template with placeholders for curator inputs
            synthesizer_prompt_template: Optional template for synthesizer
        """
        self.generator_template = generator_prompt_template
        self.reflector_template = reflector_prompt_template
        self.curator_template = curator_prompt_template
        self.synthesizer_template = synthesizer_prompt_template

    @staticmethod
    def _is_deepseek_model(model_name: str) -> bool:
        """Check if the model is a DeepSeek model (no litellm cost support)."""
        if not model_name:
            return False
        return model_name.startswith("deepseek") or "/deepseek" in model_name.lower()

    def _estimate_usage(self, language_model, prompt: str, response: str) -> Dict:
        """Estimate token usage and cost for a stage."""
        prompt_tokens = language_model.count_tokens(prompt)
        completion_tokens = language_model.count_tokens(response)
        model_name = getattr(language_model, "model_name", None)
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated_cost_usd": None,
        }
        if not self._is_deepseek_model(model_name):
            try:
                import litellm
                usage["estimated_cost_usd"] = float(
                    litellm.completion_cost(
                        model=model_name,
                        prompt=prompt,
                        completion=response,
                    )
                )
            except Exception:
                pass
        return usage

    def format_generator_prompt(
        self,
        question: str,
        memory_briefing: str,
        context: str = "",
    ) -> str:
        """
        Format the generator prompt with question and memory briefing.

        The memory_briefing is the formatted memory items text.

        Args:
            question: The problem to solve
            memory_briefing: Formatted memory items text
            context: Optional additional context

        Returns:
            str: Formatted prompt ready for LLM
        """
        prompt = self.generator_template.replace("[[REFERENCE_MEMORY]]", memory_briefing)
        prompt = prompt.replace("[[QUESTION]]", question)
        prompt = prompt.replace("[[CONTEXT]]", context)

        return prompt

    def format_reflector_prompt(
        self,
        question: str,
        retrieved_memory: List[Dict],
        trajectory: str,
        final_answer: str,
        execution_result: str,
        ground_truth: str = "",
        impact_map: Dict = None
    ) -> str:
        """
        Format the reflector prompt with generator outputs.

        Args:
            question: Original question
            retrieved_memory: Consulted memory items (pre-filtered to <used> entries)
            trajectory: Extracted trajectory from generator
            final_answer: Extracted final answer
            execution_result: "SUCCESS" or "FAILURE"
            ground_truth: Ground truth answer (if available)
            impact_map: {id: impact_type} from generator's memory audit

        Returns:
            str: Formatted prompt ready for LLM
        """
        retrieved_memory_text = format_memory_bank_for_reflector(retrieved_memory, impact_map=impact_map)

        prompt = self.reflector_template.replace("[[QUESTION]]", question)
        prompt = prompt.replace("[[RETRIEVED_MEMORY]]", retrieved_memory_text)
        prompt = prompt.replace("[[TRAJECTORY]]", trajectory)
        prompt = prompt.replace("[[FINAL_ANSWER]]", final_answer)
        prompt = prompt.replace("[[EXECUTION_RESULT]]", execution_result)
        prompt = prompt.replace("[[GROUND_TRUTH]]", ground_truth)

        return prompt

    def format_curator_prompt(
        self,
        memory_bank: List[Dict],
        reflector_output: Dict,
        query_id: str,
        question: str = "",
        training_progress: str = "",
        memory_stats: str = "",
        retrieved_audit: str = "",
        question_context: str = "",
        consulted_items: List[Dict] = None,
        impact_map: Dict = None
    ) -> str:
        """
        Format the curator prompt with reflection and current memory.

        Args:
            memory_bank: Current memory bank (full)
            reflector_output: Parsed reflection JSON
            query_id: Current query identifier (e.g., "Q_045")
            question: Original question/query
            training_progress: Sample number and total (e.g., "Sample 8 of 30")
            memory_stats: Memory bank statistics text
            retrieved_audit: Audit of retrieved vs consulted items
            question_context: Domain-specific context
            consulted_items: Memory items the generator marked as used
            impact_map: {id: impact_type} from generator's memory audit

        Returns:
            str: Formatted prompt ready for LLM
        """
        memory_bank_text = format_memory_bank_for_curator(memory_bank)
        reflector_output_text = json.dumps(reflector_output, indent=2)
        consulted_memory_text = format_memory_bank_for_reflector(
            consulted_items or [], impact_map=impact_map or {}
        )

        prompt = self.curator_template.replace("[[QUERY_ID]]", query_id)
        prompt = prompt.replace("[[TRAINING_PROGRESS]]", training_progress)
        prompt = prompt.replace("[[MEMORY_BANK_STATS]]", memory_stats)
        prompt = prompt.replace("[[ORIGINAL_QUERY]]", question)
        prompt = prompt.replace("[[QUESTION_CONTEXT]]", question_context)
        prompt = prompt.replace("[[CONSULTED_MEMORY]]", consulted_memory_text)
        prompt = prompt.replace("[[CURRENT_MEMORY]]", memory_bank_text)
        prompt = prompt.replace("[[RETRIEVED_AUDIT]]", retrieved_audit)
        prompt = prompt.replace("[[REFLECTOR_OUTPUT]]", reflector_output_text)

        return prompt

    def run_generator_stage(
        self,
        language_model,
        question: str,
        memory_briefing: str,
        context: str = "",
        images: list = None,
        **generation_kwargs
    ) -> Tuple[Dict, str, Dict]:
        """
        Run the generator stage.

        Args:
            language_model: LLM with generate() method
            question: Problem to solve
            memory_briefing: Formatted retrieved memory items text
            context: Optional context
            images: Optional list of raw image bytes to include as multimodal input
            **generation_kwargs: Additional args for generation (temperature, max_tokens, etc.)

        Returns:
            Dict: Generator outputs including trajectory, self_assessment, final_answer
        """
        import base64 as _b64
        prompt = self.format_generator_prompt(question, memory_briefing, context)

        if images:
            content = []
            for img_bytes in images:
                b64 = _b64.b64encode(img_bytes).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            history = [{"role": "user", "content": content}]
        else:
            history = [{"role": "user", "content": prompt}]
        response = language_model.generate(history=history, **generation_kwargs)

        usage = self._estimate_usage(language_model, prompt, response)

        extracted = extract_all_from_generator(response)
        extracted["prompt"] = prompt  # capture formatted input for logging
        return extracted, response, usage

    def run_reflector_stage(
        self,
        language_model,
        question: str,
        retrieved_memory: List[Dict],
        generator_outputs: Dict,
        execution_result: str,
        ground_truth: str = "",
        images: list = None,
        **generation_kwargs
    ) -> Tuple[Optional[Dict], str, Dict]:
        """
        Run the reflector stage.

        Args:
            language_model: LLM with generate() method
            question: Original question
            retrieved_memory: Full memory items retrieved for generator (already filtered to consulted items for v6)
            generator_outputs: Dict from run_generator_stage
            execution_result: "SUCCESS" or "FAILURE"
            ground_truth: Ground truth answer
            **generation_kwargs: Additional args for generation

        Returns:
            Tuple[Optional[Dict], str]: Parsed reflection JSON (or None if invalid) and raw response text
        """
        prompt = self.format_reflector_prompt(
            question=question,
            retrieved_memory=retrieved_memory,
            trajectory=generator_outputs["trajectory"],
            final_answer=generator_outputs["final_answer"],
            execution_result=execution_result,
            ground_truth=ground_truth,
            impact_map=generator_outputs.get("memory_consulted_impact", {})
        )

        if images:
            import base64 as _b64
            content = []
            for img_bytes in images:
                b64 = _b64.b64encode(img_bytes).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            history = [{"role": "user", "content": content}]
        else:
            history = [{"role": "user", "content": prompt}]
        response = language_model.generate(history=history, **generation_kwargs)

        usage = self._estimate_usage(language_model, prompt, response)

        reflection = extract_reflection(response)
        return reflection, response, usage, prompt  # prompt captured for logging

    def run_curator_stage(
        self,
        language_model,
        memory_bank: List[Dict],
        reflection: Dict,
        query_id: str,
        question: str = "",
        training_progress: str = "",
        memory_stats: str = "",
        retrieved_audit: str = "",
        question_context: str = "",
        consulted_items: List[Dict] = None,
        impact_map: Dict = None,
        images: list = None,
        **generation_kwargs
    ) -> Tuple[Optional[Dict], str, Dict]:
        """
        Run the curator stage.

        Args:
            language_model: LLM with generate() method
            memory_bank: Current full memory bank
            reflection: Parsed reflection JSON
            query_id: Query identifier (e.g., "Q_045")
            question: Original question
            training_progress: Sample progress string
            memory_stats: Memory bank statistics text
            retrieved_audit: Audit of retrieved vs consulted items
            question_context: Domain-specific context
            consulted_items: Memory items the generator marked as used
            impact_map: {id: impact_type} from generator's memory audit
            **generation_kwargs: Additional args for generation

        Returns:
            Tuple[Optional[Dict], str]: Parsed curation JSON (or None if invalid) and raw response text
        """
        prompt = self.format_curator_prompt(
            memory_bank, reflection, query_id,
            question=question,
            training_progress=training_progress,
            memory_stats=memory_stats,
            retrieved_audit=retrieved_audit,
            question_context=question_context,
            consulted_items=consulted_items,
            impact_map=impact_map
        )

        if images:
            import base64 as _b64
            content = []
            for img_bytes in images:
                b64 = _b64.b64encode(img_bytes).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            history = [{"role": "user", "content": content}]
        else:
            history = [{"role": "user", "content": prompt}]
        response = language_model.generate(history=history, **generation_kwargs)

        usage = self._estimate_usage(language_model, prompt, response)

        curation = extract_curation(response)

        if curation is None:
            print(f"[DEBUG] Curator extraction failed. Raw response (first 1000 chars):\n{response[:1000]}")

        return curation, response, usage, prompt  # prompt captured for logging

    def run_synthesizer_stage(
        self,
        language_model,
        memory_bank: List[Dict],
        next_question: str,
        next_query_id: str,
        training_progress: str = "",
        question_context: str = "",
        images: list = None,
        **generation_kwargs
    ) -> Tuple[Optional[Dict], str, Dict, str]:
        """
        Look-ahead curator: given the next question, propose memory entries that
        will help the generator answer it before it is actually seen.

        Args:
            language_model: LLM with generate() method
            memory_bank: Current memory bank after answering the previous question
            next_question: The upcoming question text
            next_query_id: Query ID for the upcoming question (e.g., "Q_006")
            training_progress: Sample progress string
            question_context: Domain-specific context

        Returns:
            Tuple: (curation dict or None, raw response, usage dict, prompt str)
        """
        if self.synthesizer_template is None:
            return None, "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, ""

        domain_counts: Dict[str, int] = {}
        for item in memory_bank:
            for tag in item.get("tags", []):
                domain = tag.split(".")[0]
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if domain_counts:
            domain_dist = ", ".join(
                f"{d}: {c}" for d, c in sorted(domain_counts.items(), key=lambda x: -x[1])
            )
            memory_stats = f"{len(memory_bank)} entries. Domain distribution: {domain_dist}."
        else:
            memory_stats = f"{len(memory_bank)} entries. No domain tags yet."

        memory_bank_text = format_memory_bank_for_curator(memory_bank)

        prompt = self.synthesizer_template.replace("[[QUERY_ID]]", next_query_id)
        prompt = prompt.replace("[[TRAINING_PROGRESS]]", training_progress)
        prompt = prompt.replace("[[MEMORY_BANK_STATS]]", memory_stats)
        prompt = prompt.replace("[[NEXT_QUESTION]]", next_question)
        prompt = prompt.replace("[[QUESTION_CONTEXT]]", question_context)
        prompt = prompt.replace("[[CURRENT_MEMORY]]", memory_bank_text)

        if images:
            import base64 as _b64
            content = []
            for img_bytes in images:
                b64 = _b64.b64encode(img_bytes).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
            content.append({"type": "text", "text": prompt})
            history = [{"role": "user", "content": content}]
        else:
            history = [{"role": "user", "content": prompt}]
        response = language_model.generate(history=history, **generation_kwargs)

        usage = self._estimate_usage(language_model, prompt, response)
        curation = extract_curation(response)
        return curation, response, usage, prompt

    def run_full_pipeline(
        self,
        language_model,
        question: str,
        memory_bank: List[Dict],
        ground_truth: str,
        query_id: str,
        retrieved_memory: List[Dict],
        context: str = "",
        embedding_index: Optional[Dict] = None,
        training_progress: str = "",
        question_context: str = "",
        dedup_threshold: float = 0.85,
        prune_reliability_threshold: float = 0.2,
        temporal_decay_lambda_query: float = 0.02,
        pruning_mode: str = "AND",
        min_queries_before_operations: int = 0,
        images: list = None,
        memory_encoder=None,
        past_solutions_briefing: str = "",
        ephemeral_items: Optional[List[Dict]] = None,
        **generation_kwargs
    ) -> Tuple[str, List[Dict], Dict, Dict]:
        """
        Run the complete LeRe pipeline for one question.

        Pipeline: Generator → Reflector → Curator

        Args:
            language_model: LLM with generate() method
            question: Problem to solve
            memory_bank: Current full memory bank
            ground_truth: Correct answer for evaluation
            query_id: Query identifier (e.g., "Q_045")
            retrieved_memory: Pre-retrieved memory items for this question
            context: Optional additional context
            embedding_index: Optional dict mapping item_id -> embedding vector
            **generation_kwargs: Args for generation (temperature, max_tokens, etc.)

        Returns:
            Tuple[str, List[Dict], Dict, Dict]:
                - final_answer: The answer from generator
                - updated_memory_bank: Memory bank after curation
                - updated_embedding_index: Embedding index after curation
                - pipeline_outputs: Dict with all intermediate outputs for logging
        """
        pipeline_outputs = {
            "query_id": query_id,
            "question": question,
            "ground_truth": ground_truth,
            "image_count": len(images) if images else 0,
        }

        memory_briefing = format_memory_bank_for_generator(retrieved_memory)
        if past_solutions_briefing:
            sep = "\n\n" if memory_briefing and memory_briefing != "(No memory items available)" else ""
            memory_briefing = (memory_briefing + sep + past_solutions_briefing).strip()

        # Increment retrieved_count on memory_bank items by ID (retrieved_memory may be shallow copies)
        if retrieved_memory:
            retrieved_ids = {item.get("id") for item in retrieved_memory if item.get("id")}
            for item in memory_bank:
                if item.get("id") in retrieved_ids:
                    meta = item.setdefault("meta", {})
                    meta["retrieved_count"] = meta.get("retrieved_count", 0) + 1

        # Stage 1: Generator
        print(f"[{query_id}] Running Generator stage...")
        generator_outputs, generator_raw, generator_usage = self.run_generator_stage(
            language_model,
            question,
            memory_briefing,
            context,
            images=images,
            **generation_kwargs
        )
        pipeline_outputs["generator"] = generator_outputs
        pipeline_outputs["generator_raw"] = generator_raw
        final_answer = generator_outputs["final_answer"]

        print(f"  [Generator] Answer: {final_answer}")
        print(f"  [Generator] Trajectory steps: {len(generator_outputs['parsed_steps'])}")
        print(f"  [Generator] Memory consulted: {generator_outputs['memory_consulted']}")

        # Determine execution result (normalize to handle wrapper differences like "(70)" vs "70")
        norm_final = _normalize_answer_for_comparison(final_answer)
        norm_gt = _normalize_answer_for_comparison(ground_truth)
        execution_result = "SUCCESS" if (norm_final and norm_gt and norm_final == norm_gt) else "FAILURE"
        pipeline_outputs["execution_result"] = execution_result

        # Stage 2: Reflector
        # Only pass items actually cited in step memory_refs — no fallback to all retrieved.
        memory_consulted = generator_outputs.get("memory_consulted", [])
        consulted_items = [m for m in retrieved_memory if m.get("id") in memory_consulted]
        print(f"[{query_id}] Running Reflector stage...")
        reflection, reflection_raw, reflector_usage, reflector_prompt = self.run_reflector_stage(
            language_model,
            question,
            consulted_items,
            generator_outputs,
            execution_result,
            ground_truth,
            images=images,
            **generation_kwargs
        )
        pipeline_outputs["reflection"] = reflection
        pipeline_outputs["reflection_raw"] = reflection_raw
        pipeline_outputs["reflector_prompt"] = reflector_prompt

        if reflection:
            print(f"  [Reflector] Status: {reflection.get('execution_status')}")
            print(f"  [Reflector] Critical steps: {len(reflection.get('trajectory_analysis', {}).get('critical_steps', []))}")
            print(f"  [Reflector] Memory evaluations: {len(reflection.get('memory_evaluation', []))}")
        pipeline_outputs["token_usage"] = {
            "generator": generator_usage,
            "reflector": reflector_usage,
        }

        if not reflection:
            print(f"[{query_id}] Reflector failed - skipping Curator")
            pipeline_outputs["token_usage"]["total"] = {
                "prompt_tokens": generator_usage["prompt_tokens"] + reflector_usage["prompt_tokens"],
                "completion_tokens": generator_usage["completion_tokens"] + reflector_usage["completion_tokens"],
                "total_tokens": generator_usage["total_tokens"] + reflector_usage["total_tokens"],
                "estimated_cost_usd": (
                    (generator_usage.get("estimated_cost_usd") or 0.0) +
                    (reflector_usage.get("estimated_cost_usd") or 0.0)
                ),
            }
            return final_answer, memory_bank, embedding_index or {}, pipeline_outputs

        # --- Synthesizer v1 (v2-design) Admission Gate ---
        # If ephemeral items were prepended to retrieved_memory, gate them here:
        #   - HELPFUL → admit to bank (assign m_NNN, embed, add)
        #   - HARMFUL/NEUTRAL/unused → discard
        # Then translate IDs in reflection.memory_evaluation, retrieved_memory,
        # generator_outputs.memory_consulted so downstream stages see only bank IDs.
        ephemeral_items = ephemeral_items or []
        ephemeral_id_set = {e.get("id") for e in ephemeral_items if e.get("id")}
        admitted_eph_remap: Dict[str, str] = {}

        if ephemeral_id_set:
            helpful_eph_ids = {
                ev.get("item_id")
                for ev in reflection.get("memory_evaluation", [])
                if ev.get("item_id") in ephemeral_id_set and ev.get("verdict") == "HELPFUL"
            }
            helpful_eph_items = [e for e in ephemeral_items if e.get("id") in helpful_eph_ids]

            if helpful_eph_items:
                new_items = [{k: v for k, v in e.items() if k != "id"} for e in helpful_eph_items]
                old_ids_in_order = [e["id"] for e in helpful_eph_items]
                bank_size_before = len(memory_bank)
                _embed_fn = memory_encoder.get_base_embedding if memory_encoder is not None else None
                memory_bank, embedding_index = apply_curation_updates(
                    memory_bank,
                    {"new_items": new_items, "reliability_updates": [],
                     "updates_to_existing": [], "prune_candidates": []},
                    language_model,
                    embedding_index or {},
                    embed_fn=_embed_fn,
                )
                new_m_ids = [item["id"] for item in memory_bank[bank_size_before:]]
                admitted_eph_remap = dict(zip(old_ids_in_order, new_m_ids))
                print(f"[{query_id}] [SynthV1] Admitted {len(helpful_eph_items)} HELPFUL ephemeral(s): {old_ids_in_order} → {new_m_ids}")

            # Translate reflection.memory_evaluation: rewrite admitted, drop rejected
            translated_eval = []
            for ev in reflection.get("memory_evaluation", []):
                iid = ev.get("item_id")
                if iid in ephemeral_id_set:
                    if iid in admitted_eph_remap:
                        new_ev = ev.copy()
                        new_ev["item_id"] = admitted_eph_remap[iid]
                        translated_eval.append(new_ev)
                else:
                    translated_eval.append(ev)
            reflection["memory_evaluation"] = translated_eval

            # Translate retrieved_memory: rewrite admitted, drop rejected
            new_retrieved = []
            for m in retrieved_memory:
                mid = m.get("id")
                if mid in ephemeral_id_set:
                    if mid in admitted_eph_remap:
                        nm = m.copy()
                        nm["id"] = admitted_eph_remap[mid]
                        new_retrieved.append(nm)
                else:
                    new_retrieved.append(m)
            retrieved_memory = new_retrieved

            # Translate generator_outputs.memory_consulted
            consulted = generator_outputs.get("memory_consulted", []) or []
            translated_consulted = []
            for cid in consulted:
                if cid in ephemeral_id_set:
                    if cid in admitted_eph_remap:
                        translated_consulted.append(admitted_eph_remap[cid])
                else:
                    translated_consulted.append(cid)
            generator_outputs["memory_consulted"] = translated_consulted

        pipeline_outputs["effective_retrieved_memory"] = retrieved_memory
        pipeline_outputs["ephemeral_admitted_ids"] = list(admitted_eph_remap.values())
        pipeline_outputs["ephemeral_id_remap"] = admitted_eph_remap

        # Refresh local memory_consulted from possibly-translated generator_outputs
        memory_consulted = generator_outputs.get("memory_consulted", [])

        # Update memory meta from audit + reflector verdicts:
        # - last_used_query: query index for staleness-based temporal decay
        # - helpful: incremented when reflector verdict = HELPFUL
        # - harmful: incremented when reflector verdict = HARMFUL
        # - NEUTRAL: last_used_query still updated, counters unchanged
        try:
            current_query_index = int(query_id[2:].lstrip("0") or "0")
        except (ValueError, IndexError):
            current_query_index = None

        if memory_consulted:
            consulted_id_set = set(memory_consulted)
            verdict_map = {
                ev["item_id"]: ev["verdict"]
                for ev in reflection.get("memory_evaluation", [])
                if "item_id" in ev and "verdict" in ev
            }
            for item in memory_bank:
                item_id = item.get("id")
                if item_id in consulted_id_set:
                    meta = item.setdefault("meta", {})
                    if current_query_index is not None:
                        meta["last_used_query"] = current_query_index
                    verdict = verdict_map.get(item_id)
                    if verdict == "HELPFUL":
                        meta["helpful"] = meta.get("helpful", 0) + 1
                    elif verdict == "HARMFUL":
                        meta["harmful"] = meta.get("harmful", 0) + 1

        # Stage 3: Curator
        print(f"[{query_id}] Running Curator stage...")
        # Build memory stats and retrieved audit for curator context
        # Build memory stats: total count + domain category distribution from tags
        domain_counts = {}
        for item in memory_bank:
            for tag in item.get("tags", []):
                domain = tag.split(".")[0]  # e.g. "math" from "math.probability"
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if domain_counts:
            domain_dist = ", ".join(
                f"{d}: {c}" for d, c in sorted(domain_counts.items(), key=lambda x: -x[1])
            )
            memory_stats = f"{len(memory_bank)} entries in memory bank. Domain distribution: {domain_dist}."
        else:
            memory_stats = f"{len(memory_bank)} entries in memory bank. No domain tags yet."
        consulted_ids = generator_outputs.get("memory_consulted", [])
        # Unused = all retrieved items NOT in consulted (derived from retrieved list, not model's <unused> tag)
        retrieved_ids = [m.get("id") for m in retrieved_memory if m.get("id")]
        unused_ids = [mid for mid in retrieved_ids if mid not in consulted_ids]
        retrieved_audit = (
            f"Retrieved {len(retrieved_memory)} items. "
            f"Consulted: {consulted_ids if consulted_ids else 'none'}. "
            f"Unused: {unused_ids if unused_ids else 'none'}."
        )
        curation, curation_raw, curator_usage, curator_prompt = self.run_curator_stage(
            language_model,
            memory_bank,
            reflection,
            query_id,
            question=question,
            training_progress=training_progress,
            memory_stats=memory_stats,
            retrieved_audit=retrieved_audit,
            question_context=question_context,
            consulted_items=consulted_items,
            impact_map=generator_outputs.get("memory_consulted_impact", {}),
            images=images,
            **generation_kwargs
        )
        pipeline_outputs["curation"] = curation
        pipeline_outputs["curation_raw"] = curation_raw
        pipeline_outputs["curator_prompt"] = curator_prompt
        pipeline_outputs["token_usage"]["curator"] = curator_usage

        if curation:
            new_entries = curation.get("new_entries", [])
            print(f"  [Curator] New entries proposed: {len(new_entries)}")
            if new_entries:
                print(f"    - Entry examples: {', '.join([e.get('title', e.get('id', 'unknown')) for e in new_entries[:3]])}")
        else:
            print(f"[{query_id}] Curator failed - memory not updated")
            pipeline_outputs["token_usage"]["total"] = {
                "prompt_tokens": generator_usage["prompt_tokens"] + reflector_usage["prompt_tokens"] + curator_usage["prompt_tokens"],
                "completion_tokens": generator_usage["completion_tokens"] + reflector_usage["completion_tokens"] + curator_usage["completion_tokens"],
                "total_tokens": generator_usage["total_tokens"] + reflector_usage["total_tokens"] + curator_usage["total_tokens"],
                "estimated_cost_usd": (
                    (generator_usage.get("estimated_cost_usd") or 0.0) +
                    (reflector_usage.get("estimated_cost_usd") or 0.0) +
                    (curator_usage.get("estimated_cost_usd") or 0.0)
                ),
            }
            return final_answer, memory_bank, embedding_index or {}, pipeline_outputs

        # Convert curator output format (new_entries) to format expected by apply_curation_updates (new_items)
        curation_for_update = {
            "new_items": curation.get("new_entries", []),  # outputs new_entries, function expects new_items
            "reliability_updates": [],  # curator doesn't produce these
            "updates_to_existing": [],
            "prune_candidates": []
        }

        # Apply updates to memory bank (returns tuple with embedding_index)
        existing_ids = {item.get("id") for item in memory_bank}
        _embed_fn = (
            memory_encoder.get_base_embedding
            if memory_encoder is not None and getattr(memory_encoder, "embedding_model", "text-embedding-3-small") != "text-embedding-3-small"
            else None
        )
        updated_memory_bank, updated_embedding_index = apply_curation_updates(
            memory_bank,
            curation_for_update,
            language_model,
            embedding_index,
            embed_fn=_embed_fn,
        )

        # Report similarity of each new entry against the existing bank so dedup
        # behaviour is observable in the nohup log.
        new_ids = [item.get("id") for item in updated_memory_bank
                   if item.get("id") not in existing_ids]
        if new_ids and len(updated_embedding_index) > 1:
            import numpy as _np
            all_ids = list(updated_embedding_index.keys())
            all_embs = [_np.array(updated_embedding_index[k], dtype=_np.float32) for k in all_ids]
            # L2-normalise
            all_embs_norm = [e / (_np.linalg.norm(e) + 1e-8) for e in all_embs]
            id_to_idx = {k: i for i, k in enumerate(all_ids)}
            for new_id in new_ids:
                if new_id not in id_to_idx:
                    continue
                new_emb = all_embs_norm[id_to_idx[new_id]]
                others = [all_embs_norm[i] for i, k in enumerate(all_ids) if k != new_id]
                if not others:
                    continue
                sims = [float(_np.dot(new_emb, o)) for o in others]
                title = next((item.get("title", new_id) for item in updated_memory_bank
                              if item.get("id") == new_id), new_id)
                max_sim = max(sims)
                flag = " *** WILL DEDUP ***" if max_sim > dedup_threshold else ""
                print(f"  [Dedup Check] '{title}' vs bank({len(sims)}): "
                      f"max={max_sim:.3f}, mean={sum(sims)/len(sims):.3f}, "
                      f"min={min(sims):.3f} (threshold={dedup_threshold}){flag}")

        # Report reliability of all entries so prune candidates are visible before
        # apply_merge_operator acts. Uses the same Bayesian formula as the merge operator.
        prune_candidates_found = []
        for item in updated_memory_bank:
            meta = item.get("meta", {})
            helpful = meta.get("helpful", 0)
            harmful = meta.get("harmful", 0)
            reliability = (helpful + 1) / (helpful + harmful + 2)
            if reliability < prune_reliability_threshold:
                prune_candidates_found.append((item.get("title", item.get("id", "?")), reliability))
        if prune_candidates_found:
            for title, rel in prune_candidates_found:
                print(f"  [Prune Check] '{title}': reliability={rel:.3f} < threshold={prune_reliability_threshold} *** WILL PRUNE ***")
        else:
            print(f"  [Prune Check] All {len(updated_memory_bank)} entries above reliability threshold ({prune_reliability_threshold})")

        # Step 4: Apply merge operator
        # - Embedding-based deduplication
        # - Reliability-based pruning
        # - Replace LLM confidence with computed reliability
        updated_memory_bank, updated_embedding_index, merge_stats = apply_merge_operator(
            updated_memory_bank,
            updated_embedding_index,
            language_model,
            dedup_threshold=dedup_threshold,
            prune_reliability_threshold=prune_reliability_threshold,
            current_query=current_query_index,
            temporal_decay_lambda_query=temporal_decay_lambda_query,
            pruning_mode=pruning_mode,
            min_queries_before_operations=min_queries_before_operations,
            embed_fn=_embed_fn,
        )
        pipeline_outputs["merge_stats"] = merge_stats

        pipeline_outputs["memory_health"] = get_memory_health_stats(updated_memory_bank)
        pipeline_outputs["token_usage"]["total"] = {
            "prompt_tokens": generator_usage["prompt_tokens"] + reflector_usage["prompt_tokens"] + curator_usage["prompt_tokens"],
            "completion_tokens": generator_usage["completion_tokens"] + reflector_usage["completion_tokens"] + curator_usage["completion_tokens"],
            "total_tokens": generator_usage["total_tokens"] + reflector_usage["total_tokens"] + curator_usage["total_tokens"],
            "estimated_cost_usd": (
                (generator_usage.get("estimated_cost_usd") or 0.0) +
                (reflector_usage.get("estimated_cost_usd") or 0.0) +
                (curator_usage.get("estimated_cost_usd") or 0.0)
            ),
        }

        print(f"[{query_id}] Pipeline complete. Memory items: {len(memory_bank)} -> {len(updated_memory_bank)}")

        return final_answer, updated_memory_bank, updated_embedding_index, pipeline_outputs


def load_prompts_from_directory(prompts_dir: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Load the prompt templates from a directory.

    Args:
        prompts_dir: Path to directory containing generator_prompt.txt, reflector_prompt.txt,
                     curator_prompt.txt, and optionally synthesizer_prompt.txt

    Returns:
        Tuple[str, str, str, Optional[str]]: (generator, reflector, curator, synthesizer) templates
    """
    import os

    generator_path = os.path.join(prompts_dir, "generator_prompt.txt")
    reflector_path = os.path.join(prompts_dir, "reflector_prompt.txt")
    curator_path = os.path.join(prompts_dir, "curator_prompt.txt")
    synthesizer_path = os.path.join(prompts_dir, "synthesizer_prompt.txt")

    with open(generator_path, 'r') as f:
        generator_template = f.read()

    with open(reflector_path, 'r') as f:
        reflector_template = f.read()

    with open(curator_path, 'r') as f:
        curator_template = f.read()

    synthesizer_template = None
    if os.path.isfile(synthesizer_path):
        with open(synthesizer_path, 'r') as f:
            synthesizer_template = f.read()

    return generator_template, reflector_template, curator_template, synthesizer_template


def create_pipeline_from_directory(prompts_dir: str) -> LeRePipeline:
    """
    Create a LeRePipeline from a prompts directory.

    Args:
        prompts_dir: Path to directory with prompt files

    Returns:
        LeRePipeline: Configured pipeline instance
    """
    generator, reflector, curator, synthesizer = load_prompts_from_directory(prompts_dir)
    return LeRePipeline(generator, reflector, curator, synthesizer)
