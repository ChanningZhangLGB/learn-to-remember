import argparse
import json
import logging
import os
import pickle
import random
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


def _slugify_model_name(model_name: str) -> str:
    # "openai/gpt-4.1-mini" -> "gpt-4.1-mini"
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _next_run_id(task_dir: str) -> int:
    if not os.path.isdir(task_dir):
        return 1
    run_ids = []
    for entry in os.listdir(task_dir):
        if not entry.startswith("run_"):
            continue
        try:
            run_ids.append(int(entry.split("_", 1)[1]))
        except Exception:
            continue
    return (max(run_ids) + 1) if run_ids else 1


def _normalize_choice(ans: str) -> str:
    """
    Normalize answer text by removing common wrappers.

    Handles various formats:
    - (C) → C
    - <C> → C
    - (<C>) → C
    - <C>\n</C> → C
    - C → C
    """
    if ans is None:
        return ""
    s = str(ans).strip()
    if not s:
        return ""

    # Remove newlines and closing XML-style tags like </C>, </B>, </answer>
    s = s.replace("\n", " ").strip()
    import re
    s = re.sub(r'</[A-Za-z]+>', '', s).strip()

    # Iteratively strip outer parentheses and angle brackets
    changed = True
    while changed:
        changed = False
        # Strip parentheses: (X) → X
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
            changed = True
        # Strip angle brackets: <X> → X
        if s.startswith("<") and s.endswith(">"):
            s = s[1:-1].strip()
            changed = True

    return s.strip().upper()

import re as _re


def _retrieve_past_solutions(
    query_embedding: "np.ndarray",
    past_solutions_store: List[Dict],
    top_k: int = 3,
) -> str:
    """
    DR-style retrieval: cosine similarity between current query embedding and all
    stored past question embeddings. Returns top-k past Q&A pairs formatted as
    a PREVIOUS SOLUTIONS block for injection into the generator prompt.
    """
    if not past_solutions_store or top_k <= 0:
        return ""

    q = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    sims = []
    for entry in past_solutions_store:
        e = entry["embedding"]
        e_norm = e / (np.linalg.norm(e) + 1e-10)
        sims.append(float(np.dot(q, e_norm)))

    top_indices = sorted(range(len(sims)), key=lambda x: sims[x], reverse=True)[:top_k]
    # Present in ascending similarity order (most similar last, closest to the question)
    top_indices = list(reversed(top_indices))

    lines = ["#### PREVIOUS SOLUTIONS (START)"]
    for rank, idx in enumerate(top_indices, 1):
        entry = past_solutions_store[idx]
        lines.append(f"\n### Example {rank} (similarity={sims[idx]:.3f}) ###")
        lines.append(f"@ Input:\n{entry['question']}")
        lines.append(f"@ Output:\n{entry['answer']}")
    lines.append("\n#### PREVIOUS SOLUTIONS (END)")
    return "\n".join(lines)


def _remove_punctuation(text: str) -> str:
    # Only strip non-numeric punctuation; preserve '.' inside numbers (e.g. "550.0")
    markers = [",", ";", ":", '"']
    for marker in markers:
        text = text.replace(marker, "")
    # Remove standalone '.' that are not between digits
    text = _re.sub(r'(?<!\d)\.(?!\d)', '', text)
    return text


def _normalize_numeric(s: str) -> str:
    """Strip parens/whitespace and normalise numeric strings.
    '(550.0)' -> '550', '081' -> '81'"""
    s = s.strip().lstrip("(").rstrip(")")
    try:
        f = float(s)
        # If it's a whole number, return as int string to allow 081==81 etc.
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s.strip()


def _extract_or_candidates(target: str) -> list:
    """Split 'OR'-style ground truths like '080 or 081 (both were accepted)'.
    Returns list of candidate strings, or [target] if no OR present."""
    # Match patterns like "X or Y" or "X OR Y", optionally followed by parenthetical note
    if _re.search(r'\bor\b', target, flags=_re.IGNORECASE):
        # Strip trailing parenthetical note, e.g. "(both were accepted)"
        cleaned = _re.sub(r'\s*\(.*?\)\s*$', '', target).strip()
        parts = _re.split(r'\s+or\s+', cleaned, flags=_re.IGNORECASE)
        return [p.strip() for p in parts if p.strip()]
    return [target]


def _eval_for_exact_matching_with_no_punctuation(output: str, target: str) -> bool:
    """Robust numeric answer matching:
    - Handles float vs int (550.0 == 550)
    - Handles leading zeros (081 == 81)
    - Handles OR-style multi-accepted ground truths ('080 or 081 (both were accepted)')
    - Falls back to punctuation-stripped string comparison
    """
    output_norm = _normalize_numeric(output)
    candidates = _extract_or_candidates(target)

    for cand in candidates:
        cand_norm = _normalize_numeric(cand)
        # Numeric comparison
        try:
            if float(output_norm) == float(cand_norm):
                return True
        except ValueError:
            pass
        # String comparison after punctuation strip
        out_str = _remove_punctuation(output).replace("\n", " ").replace("(", "").replace(")", "").strip()
        cand_str = _remove_punctuation(cand).replace("\n", " ").replace("(", "").replace(")", "").strip()
        if out_str == cand_str:
            return True

    return False


def _eval_for_math_500(output: str, target: str) -> bool:
    """5-pass normalization: whitespace → \\text{} strip → \\left/\\right strip → unit suffix strip → numeric float."""
    def _norm(s: str) -> str:
        return _re.sub(r'\s+', '', s.strip())

    def _strip_text(s: str) -> str:
        return _re.sub(r'\\text\{([^}]*)\}', lambda m: m.group(1), s, flags=_re.IGNORECASE)

    def _strip_left_right(s: str) -> str:
        return _re.sub(r'\\(?:left|right)\s*[\(\)\[\]\{\}\|.]?', '', s, flags=_re.IGNORECASE)

    def _strip_units(s: str) -> str:
        return _re.sub(r'\^\{?\\circ\}?|\\circ|\\degree|\\%', '', s, flags=_re.IGNORECASE)

    def _strip_outer_parens(s: str) -> str:
        while s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        return s

    def _flatten_frac(s: str) -> str:
        """Replace \\frac{a}{b} with a/b (handles simple non-nested cases)."""
        return _re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'\1/\2', s, flags=_re.IGNORECASE)

    def _try_numeric(a: str, b: str) -> bool:
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (ValueError, TypeError):
            return False

    def _try_eval(s: str) -> float | None:
        try:
            val = eval(s, {"__builtins__": {}})  # noqa: S307
            return float(val)
        except Exception:
            return None

    def _full_norm(s: str) -> str:
        return _norm(_strip_outer_parens(_flatten_frac(_strip_units(_strip_left_right(_strip_text(s)))))).lower()

    n_out, n_tgt = _norm(output), _norm(target)
    if n_out == n_tgt:
        return True
    fn_out, fn_tgt = _full_norm(output), _full_norm(target)
    if fn_out == fn_tgt:
        return True
    if _try_numeric(n_out, n_tgt):
        return True
    # Numeric comparison on fully-normalized (frac-flattened) forms
    v_out, v_tgt = _try_eval(fn_out), _try_eval(fn_tgt)
    if v_out is not None and v_tgt is not None and abs(v_out - v_tgt) < 1e-6:
        return True
    return False


def _clean_output_for_game_of_24(output: str) -> str:
    if "=" in output:
        output = output.split("=")[0].strip()
    if "is" in output:
        output = output.split("is")[1].strip()
    if "equals" in output:
        output = output.split("equals")[0].strip()
    if "evaluates to" in output:
        output = output.split("evaluates to")[0].strip()
    return output


def _eval_for_game_of_24(input_text: str, output: str) -> bool:
    clean_output = _clean_output_for_game_of_24(output)
    clean_output = clean_output.replace("x", "*").strip()
    clean_output = clean_output.replace("×", "*").strip()
    clean_output = clean_output.replace("÷", "/").strip()
    try:
        value = eval(clean_output)
        if not (abs(value - 24) < 1e-3):
            return False

        input_digits = input_text.split(" ")
        replacements = ["+", "-", "*", "/", "÷", "(", ")"]
        for symbol in replacements:
            clean_output = clean_output.replace(symbol, " ")

        import re
        clean_output = re.sub(" +", " ", clean_output).strip()
        output_digits = clean_output.split(" ")
        input_digits.sort()
        output_digits.sort()
        return input_digits == output_digits
    except Exception:
        return False


def _eval_equation_balancer(output: str, target: str) -> bool:
    output = output.split("=")[0].strip()
    target_val = target.split("=")[1].strip()
    target_expr = target.split("=")[0].strip()
    output_nums = output.replace("+", "").replace("-", "").replace("*", "").replace("/", "").replace(" ", "").strip()
    target_nums = target_expr.replace("+", "").replace("-", "").replace("*", "").replace("/", "").replace(" ", "").strip()
    if output_nums != target_nums:
        return False
    try:
        output_value = eval(output)
        return abs(output_value - eval(target_val)) < 1e-6
    except Exception:
        return False


def _eval_for_multiple_choice(input_text: str, final_answer: str, target: str) -> bool:
    """
    Mirrors dynamic-cheatsheet MC matching behavior.
    """
    if not final_answer or not target:
        return False

    def clean_text(text: str) -> str:
        if not text:
            return ""
        return text.lower().strip().replace("`", "").replace("(", "").replace(")", "").strip()

    def extract_option_text(question_text: str, option_letter: str) -> str:
        try:
            options_section = ""
            if "options:" in question_text.lower():
                options_section = question_text.lower().split("options:")[1].strip()
            elif "choices:" in question_text.lower():
                options_section = question_text.lower().split("choices:")[1].strip()

            if not options_section:
                lines = question_text.lower().split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith(f"({option_letter})") or line.startswith(f"{option_letter})"):
                        return line.split(")", 1)[1].strip()

            for line in options_section.split("\n"):
                line = line.strip()
                if line.startswith(f"({option_letter})") or line.startswith(f"{option_letter})"):
                    return line.split(")", 1)[1].strip()
                if line.startswith(f"{option_letter}."):
                    return line.split(".", 1)[1].strip()
        except Exception:
            return ""
        return ""

    # Robust normalization precheck (e.g., "( A )" vs "(A)")
    if _normalize_choice(final_answer) and _normalize_choice(final_answer) == _normalize_choice(target):
        return True

    if final_answer == target:
        return True

    clean_answer = clean_text(final_answer)
    clean_target = clean_text(target)

    target_letter = ""
    if len(clean_target) == 1:
        target_letter = clean_target
    elif clean_target.endswith(")"):
        target_letter = clean_target[-2]
    else:
        last_char = clean_target[-1]
        if last_char in "abcdefghij":
            target_letter = last_char

    if len(clean_answer) == 1 and clean_answer in "abcdefghij" and clean_answer == target_letter:
        return True

    if clean_answer.startswith(target_letter) and (len(clean_answer) == 1 or (len(clean_answer) == 2 and clean_answer[1] == ".")):
        return True

    if clean_answer.endswith(target_letter) and (clean_answer[-2:] == f" {target_letter}" or clean_answer[-3:] == f" {target_letter}."):
        return True

    # Rule: target letter appears as a standalone token anywhere in the answer.
    # Handles multi-option answers like "(A), (F)" where GT is "(A)" (letter appears first).
    # Uses word boundary so "a" in "lateral" does NOT match.
    if target_letter and _re.search(r'(?<![a-z])' + _re.escape(target_letter) + r'(?![a-z])', clean_answer):
        return True

    target_text = extract_option_text(input_text, target_letter)
    if target_text and target_text in clean_answer:
        return True

    if target_letter.isdigit() and target_letter in clean_answer:
        return True

    return False


def _match_answer_to_option_letter(answer: str, question: str) -> str:
    """Scan all options in the question and return the letter whose text best matches the answer.

    Handles cases where the model outputs option text (e.g. '97') instead of a letter ('A').
    Uses numeric normalization so '97°' matches option '97' and vice versa.
    Returns the matched letter (uppercase) or '' if no match found.
    """
    if not answer or not question:
        return ""
    options = _re.findall(r'\(([A-J])\)\s*([^\n(]+)', question, _re.IGNORECASE)
    if not options:
        return ""

    def _num_norm(s: str) -> str:
        s = s.strip()
        try:
            return str(float(_re.sub(r'[^0-9.\-]', '', s)))
        except ValueError:
            return _re.sub(r'[^a-z0-9]', '', s.lower())

    fa_n = _num_norm(answer)
    if not fa_n:
        return ""
    for letter, opt_text in options:
        if fa_n == _num_norm(opt_text):
            return letter.upper()
    return ""


def _eval_mathvista(final_answer: str, ground_truth: str, question: str = "") -> bool:
    """Evaluate MathVista answers: letter match for multi_choice, numerical/text for free_form."""
    fa = str(final_answer).strip()
    gt = str(ground_truth).strip()

    # Multi-choice: ground truth is a single letter (A-J)
    if len(gt) == 1 and gt.upper() in "ABCDEFGHIJ":
        # If answer doesn't look like a letter, try to map it to an option letter first
        if question and not _re.match(r'^\(?[A-Ja-j]\)?\.?$', fa):
            matched = _match_answer_to_option_letter(fa, question)
            if matched:
                return matched == gt.upper()
        return _eval_for_multiple_choice(question, fa, gt)

    # Free-form numerical: try float comparison with tolerance
    try:
        fa_f = float(fa.replace(",", "").rstrip("."))
        gt_f = float(gt.replace(",", "").rstrip("."))
        if gt_f == 0:
            return abs(fa_f - gt_f) < 1e-6
        return abs(fa_f - gt_f) / max(abs(gt_f), 1e-8) < 0.02  # 2% relative tolerance
    except ValueError:
        pass

    # Free-form text: exact normalized match, or ground truth is contained in answer
    def _norm(s):
        return _re.sub(r"[^a-z0-9]", "", s.lower())
    nfa, ngt = _norm(fa), _norm(gt)
    if not nfa or not ngt:
        return False
    return nfa == ngt or ngt in nfa


def _is_correct_answer(task: str, question: str, final_answer: str, ground_truth: str) -> bool:
    if task in {"MATH_500", "MATH"}:
        return _eval_for_math_500(str(final_answer), str(ground_truth))

    if task in {"AIME_2025", "AIME_2024", "AIME_2020_2024"}:
        return _eval_for_exact_matching_with_no_punctuation(
            str(final_answer).lower(),
            str(ground_truth).lower(),
        )

    if task in {"GPQA_Diamond", "MMLU_Pro_Engineering", "MMLU_Pro_Physics", "MMLU_Pro_Engineering_250", "MMLU_Pro_Physics_250",
                "MMLU_Pro_Multi_6_300", "MMLU_Pro_Multi_6_600", "MMLU_Pro_Multi_14_700",
                "MMMU_Pro_standard_4_250", "MMMU_Pro_standard_10_250"}:
        return _eval_for_multiple_choice(question, final_answer, ground_truth)

    if task in {"MathVista_testmini_250"}:
        return _eval_mathvista(final_answer, ground_truth, question=question)

    if task == "GameOf24":
        return _eval_for_game_of_24(question, final_answer)

    if task == "MathEquationBalancer":
        return _eval_equation_balancer(final_answer, ground_truth)

    # Default fallback keeps original strict normalized equality.
    normalized_final = _normalize_choice(final_answer)
    normalized_gt = _normalize_choice(ground_truth)
    return bool(normalized_final) and bool(normalized_gt) and normalized_final == normalized_gt


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger(f"acecl:{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


@dataclass(frozen=True)
class ExperimentPaths:
    data_dir: str
    prompts_dir: str
    results_root: str


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_gpqa_dataset(data_dir: str):
    from datasets import load_from_disk

    ds = load_from_disk(data_dir)
    if hasattr(ds, "keys") and "train" in getattr(ds, "keys")():
        ds = ds["train"]
    return ds


def _load_mathvista_dataset(data_dir: str):
    """Load MathVista via pyarrow directly, bypassing load_from_disk (List type incompatibility)."""
    import pyarrow as pa

    arrow_path = os.path.join(data_dir, "data-00000-of-00001.arrow")
    reader = pa.ipc.open_stream(arrow_path)
    table = reader.read_all()

    class _TableData:
        def __init__(self, t):
            self.table = t

    class _ArrowDataset:
        def __init__(self, t):
            self.data = _TableData(t)
            self._table = t

        def __len__(self):
            return self._table.num_rows

    return _ArrowDataset(table)


def _preprocess_mmmu_pro_row(row: Dict, is_vision: bool = False) -> Dict:
    """Convert MMMU-Pro row to LeRe expected format with input/target fields and images list."""
    import ast as _ast, re as _re
    options_raw = row.get("options", "[]")
    try:
        options = _ast.literal_eval(options_raw) if isinstance(options_raw, str) else options_raw
    except Exception:
        options = []
    options_str = "\n".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(options))
    # Strip <image N> placeholders from question text (image content is passed separately as bytes)
    question_text = _re.sub(r"<image\s*\d+>", "", row.get("question", "") or "").strip()
    if is_vision:
        input_text = f"{question_text}\nOptions:\n{options_str}" if question_text else options_str
        img = row.get("image")
        images = [img["bytes"] if isinstance(img, dict) else img] if img is not None else []
    else:
        input_text = f"{question_text}\nOptions:\n{options_str}"
        images = []
        for k in [f"image_{n}" for n in range(1, 8)]:
            img = row.get(k)
            if img is not None:
                images.append(img["bytes"] if isinstance(img, dict) else img)
    return {
        "input": input_text,
        "target": row.get("answer", ""),
        "images": images,
    }


def _is_mmmu_pro_task(task: str) -> bool:
    return task in {"MMMU_Pro_standard_4_250", "MMMU_Pro_standard_10_250", "MMMU_Pro_vision_250"}


def _is_mmmu_pro_vision_task(task: str) -> bool:
    return task == "MMMU_Pro_vision_250"


def _is_mathvista_task(task: str) -> bool:
    return task in {"MathVista_testmini_250"}


def _preprocess_mathvista_row(row: Dict) -> Dict:
    """Convert MathVista row to LeRe expected format."""
    import io as _io
    from PIL import Image as _PILImage

    question = row.get("question", "") or ""
    choices = row.get("choices")  # list or None
    answer = str(row.get("answer", "") or "")
    q_type = row.get("question_type", "free_form")

    # Build input text: question + options if multi_choice
    if choices and q_type == "multi_choice":
        opts_str = "\n".join(f"({chr(65+j)}) {opt}" for j, opt in enumerate(choices))
        input_text = f"{question}\nOptions:\n{opts_str}"
        # MathVista stores answer as choice text; convert to letter for evaluation
        for j, opt in enumerate(choices):
            if str(opt).strip().lower() == answer.strip().lower():
                answer = chr(65 + j)
                break
    else:
        input_text = question

    # Extract image
    images = []
    decoded = row.get("decoded_image")
    if decoded is not None:
        raw = decoded.get("bytes") if isinstance(decoded, dict) else decoded
        if raw:
            images.append(raw)

    return {
        "input": input_text,
        "target": answer,
        "question_type": q_type,
        "images": images,
    }


def _init_components(cfg: Dict[str, Any]):
    from main.language_model_lere import LanguageModel
    from main.utils.lere_pipeline import create_pipeline_from_directory
    from main.ccme_encoder import QueryEncoder, MemoryEncoder, CCMERetriever

    llm_cfg = cfg["llm"]
    ccme_cfg = cfg["ccme"]
    paths_cfg = cfg["paths"]
    adapters_cfg = cfg.get("adapters", {})
    language_model = LanguageModel(model_name=llm_cfg["model_name"])
    pipeline = create_pipeline_from_directory(paths_cfg["prompts_dir"])

    adapters_enabled = adapters_cfg.get("enabled", False)
    adapter_hidden = adapters_cfg.get("hidden_dim", 256)
    # Per-component training toggles for ablation (default True when adapters enabled).
    # Adapter layers are still constructed (so retrieval uses pretrained projection),
    # but their parameters are frozen if the corresponding flag is False.
    ccme_training_enabled = cfg.get("ccme_training", {}).get("enabled", True)
    crte_training_enabled = cfg.get("crte_training", {}).get("enabled", True)
    # Encoder architecture: read from adapters block (shared across Eq, Em, Et, Er)
    # For multimodal tasks (MMMU-Pro, MathVista), switch to CLIP so both text and image
    # embeddings share the same contrastively-aligned 768d latent space.
    task = cfg["experiment"].get("task", "")
    _is_multimodal_task = _is_mmmu_pro_task(task) or _is_mathvista_task(task)
    if _is_multimodal_task:
        embedding_model = "clip"
        base_dim = 768
    else:
        embedding_model = adapters_cfg.get("embedding_model", ccme_cfg.get("embedding_model", "text-embedding-3-small"))
        base_dim = adapters_cfg.get("base_dim", ccme_cfg.get("base_dim", 1536))
    projection_dim = adapters_cfg.get("projection_dim", ccme_cfg.get("projection_dim", base_dim))

    embed_api_key = os.getenv("OPENAI_API_KEY_EMBED")
    query_encoder = QueryEncoder(
        embedding_model=embedding_model,
        base_dim=base_dim,
        projection_dim=projection_dim,
        trainable=adapters_enabled and ccme_training_enabled,
        adapter_enabled=adapters_enabled,
        adapter_hidden=adapter_hidden,
        api_key=embed_api_key,
    )
    memory_encoder = MemoryEncoder(
        embedding_model=embedding_model,
        base_dim=base_dim,
        projection_dim=projection_dim,
        trainable=adapters_enabled and ccme_training_enabled,
        adapter_enabled=adapters_enabled,
        adapter_hidden=adapter_hidden,
        api_key=embed_api_key,
    )
    retrieval_cfg = cfg.get("retrieval", {})
    retriever = CCMERetriever(
        query_encoder=query_encoder,
        memory_encoder=memory_encoder,
        alpha=retrieval_cfg.get("ccme_alpha", 0.8),
        top_k=retrieval_cfg.get("retrieve_top_k", 5),
        use_ann=ccme_cfg.get("use_ann", True),
    )
    return language_model, pipeline, query_encoder, memory_encoder, retriever


def _init_training_components(cfg: Dict[str, Any], query_encoder, memory_encoder):
    """Initialize CRTE encoders, losses, data collector, and online trainer."""
    from main.crte_encoder import TrajectoryEncoder, ReflectionEncoder, CRTELoss
    from main.ccme_encoder import CCMELoss
    from main.training_data import TrainingDataCollector
    from main.online_trainer import OnlineTrainer

    ccme_cfg = cfg["ccme"]
    crte_cfg = cfg["crte"]
    training_cfg = cfg.get("online_training", {})
    buffers_cfg = cfg.get("training_buffers", {})  # legacy fallback
    ccme_train_cfg = cfg.get("ccme_training", {})
    crte_train_cfg = cfg.get("crte_training", {})
    adapters_cfg = cfg.get("adapters", {})

    # Shared adapter + encoder architecture settings (Eq, Em, Et, Er all identical)
    adapters_enabled = adapters_cfg.get("enabled", False)
    adapter_hidden   = adapters_cfg.get("hidden_dim", 256)
    # Per-component training toggles for ablation
    crte_training_enabled = crte_train_cfg.get("enabled", True)
    embedding_model  = adapters_cfg.get("embedding_model", ccme_cfg.get("embedding_model", "text-embedding-3-small"))
    base_dim         = adapters_cfg.get("base_dim", ccme_cfg.get("base_dim", 1536))
    projection_dim   = adapters_cfg.get("projection_dim", ccme_cfg.get("projection_dim", 768))
    device = training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    embed_api_key = os.getenv("OPENAI_API_KEY_EMBED")
    # Initialize CRTE encoders (Et, Er)
    trajectory_encoder = TrajectoryEncoder(
        embedding_model=embedding_model,
        base_dim=base_dim,
        projection_dim=projection_dim,
        d_temporal=crte_cfg.get("d_temporal", 64),
        trainable=adapters_enabled and crte_training_enabled,
        adapter_enabled=adapters_enabled,
        adapter_hidden=adapter_hidden,
        api_key=embed_api_key,
    )
    reflection_encoder = ReflectionEncoder(
        embedding_model=embedding_model,
        base_dim=base_dim,
        projection_dim=projection_dim,
        trainable=adapters_enabled and crte_training_enabled,
        adapter_enabled=adapters_enabled,
        adapter_hidden=adapter_hidden,
        api_key=embed_api_key,
    )

    # Initialize losses
    ccme_loss = CCMELoss(
        temperature=ccme_cfg.get("temperature", 0.07),
        adaptive_temperature=ccme_cfg.get("adaptive_temperature", True),
    )
    crte_loss = CRTELoss(
        temperature=crte_cfg.get("temperature", 0.07),
        lambda_cluster=crte_cfg.get("lambda_cluster", 0.1),
        lambda_margin=crte_cfg.get("lambda_margin", 0.1),
        margin_eta=crte_cfg.get("margin_eta", 0.2),
    )

    # Initialize training data collector
    data_collector = TrainingDataCollector(
        ccme_buffer_size=ccme_train_cfg.get("buffer_size", buffers_cfg.get("ccme_buffer_size", 100)),
        crte_buffer_size=crte_train_cfg.get("buffer_size", buffers_cfg.get("crte_buffer_size", 100)),
        temporal_decay=training_cfg.get("temporal_decay", buffers_cfg.get("temporal_decay", 0.9)),
    )

    # Initialize online trainer (periodic refinement config)
    mem_ops_cfg = cfg.get("periodic_memory_refinement", {}) or cfg.get("memory_operations", {})
    online_trainer = OnlineTrainer(
        query_encoder=query_encoder,
        memory_encoder=memory_encoder,
        trajectory_encoder=trajectory_encoder,
        reflection_encoder=reflection_encoder,
        ccme_loss=ccme_loss,
        crte_loss=crte_loss,
        data_collector=data_collector,
        k_upd=training_cfg.get("k_upd", 5),
        num_train_steps=training_cfg.get("num_train_steps", 8),
        batch_size=training_cfg.get("batch_size", 4),
        learning_rate_ccme=ccme_train_cfg.get("learning_rate", 1e-4),
        learning_rate_crte=crte_train_cfg.get("learning_rate", 2e-5),
        lambda_crte=crte_train_cfg.get("lambda_crte", 1.0),
        ccme_stability_window=ccme_train_cfg.get("stability_window", training_cfg.get("stability_window", 10)),
        ccme_stability_threshold=ccme_train_cfg.get("stability_threshold", training_cfg.get("stability_threshold", 0.1)),
        crte_stability_window=crte_train_cfg.get("stability_window", training_cfg.get("stability_window", 10)),
        crte_stability_threshold=crte_train_cfg.get("stability_threshold", training_cfg.get("stability_threshold", 0.1)),
        device=device,
        min_ccme_positive=ccme_train_cfg.get("min_positive", 1),
        min_crte_positive=crte_train_cfg.get("min_positive", 1),
        refine_every_updates=mem_ops_cfg.get("refine_every_updates", 0),
        refinement_mode=mem_ops_cfg.get("refinement_mode", "fixed"),
        redundancy_threshold=mem_ops_cfg.get("redundancy_threshold", 0.65),
        min_items_for_refinement=mem_ops_cfg.get("min_items_for_refinement", 5),
        n_clusters_mode=mem_ops_cfg.get("n_clusters_mode", "heuristic"),
        n_clusters_fixed=mem_ops_cfg.get("n_clusters_fixed", 10),
        post_refinement_grace=mem_ops_cfg.get("post_refinement_grace", 0),
        ccme_prune_on_instability=ccme_train_cfg.get("prune_on_instability", False),
        ccme_false_negative_threshold=ccme_train_cfg.get("false_negative_threshold", 0.55),
        ccme_reliability_prune_threshold=ccme_train_cfg.get("reliability_prune_threshold", 0.35),
        crte_prune_on_instability=crte_train_cfg.get("prune_on_instability", False),
        crte_stale_positive_threshold=crte_train_cfg.get("stale_positive_threshold", 0.15),
        crte_trivial_negative_threshold=crte_train_cfg.get("trivial_negative_threshold", 0.05),
        memory_ops_cfg=mem_ops_cfg,
    )

    return trajectory_encoder, reflection_encoder, ccme_loss, crte_loss, data_collector, online_trainer


def _make_output_dir(cfg: Dict[str, Any]) -> Tuple[str, str, int, str]:
    exp = cfg["experiment"]
    model_slug = _slugify_model_name(cfg["llm"]["model_name"])
    task = exp["task"]

    results_root = cfg["paths"].get("results_root", "results")
    task_dir = os.path.join(results_root, model_slug, task)

    # Allow explicit override (e.g. ablation runs) via experiment.mode_dir
    mode_override = exp.get("mode_dir")
    if mode_override:
        retrieval_mode = mode_override
    else:
        ret = cfg.get("retrieval", {})
        _inj = ret.get("inject_past_solutions", False)
        _synth = ret.get("synthesizer", False)
        _synth_v1 = ret.get("synthesizer_v1", False)
        if _inj:
            if _synth_v1:
                retrieval_mode = "past_sol_plus_ccme_synth_v1"
            elif _synth:
                retrieval_mode = "past_sol_plus_ccme_synth"
            else:
                retrieval_mode = "past_sol_plus_ccme"
        else:
            if _synth_v1:
                retrieval_mode = "ccme_topk_synth_v1"
            elif _synth:
                retrieval_mode = "ccme_topk_synth"
            else:
                retrieval_mode = "ccme_topk"
    mode_dir = os.path.join(task_dir, retrieval_mode)

    run_id_cfg = exp.get("run_id")
    run_id = int(run_id_cfg) if isinstance(run_id_cfg, int) else _next_run_id(mode_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = os.path.join(mode_dir, f"run_{run_id}", timestamp)
    _ensure_dir(out_dir)
    _ensure_dir(os.path.join(out_dir, "checkpoints"))
    _ensure_dir(os.path.join(out_dir, "detailed_outputs"))
    _ensure_dir(os.path.join(out_dir, "input_output_log"))
    return out_dir, model_slug, run_id, timestamp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/gpt-4.1-mini_gpqa_diamond_query_bullets_preview.json",
        help="Path to experiment config JSON",
    )
    args = ap.parse_args()

    cfg = _load_config(args.config)
    out_dir, model_slug, run_id, timestamp = _make_output_dir(cfg)

    exp_name = cfg["experiment"]["name"]
    task = cfg["experiment"]["task"]

    main_log_path = os.path.join(out_dir, f"{exp_name}_{timestamp}_main.log")
    results_path = os.path.join(out_dir, f"{exp_name}_{timestamp}_results.jsonl")
    training_log_path = os.path.join(out_dir, f"{exp_name}_{timestamp}_training.log")
    training_losses_path = os.path.join(out_dir, f"{exp_name}_{timestamp}_training_losses.json")

    logger = _setup_logger(main_log_path)
    verbose = bool(cfg.get("experiment_control", {}).get("verbose", True))
    if not verbose:
        logger.setLevel(logging.WARNING)

    logger.info(f"Experiment initialized: {exp_name}")
    logger.info(f"Output directory: {out_dir}")
    logger.info("Initializing Language Model...")
    start_time = time.time()

    language_model, pipeline, query_encoder, memory_encoder, retriever = _init_components(cfg)
    logger.info(f"  Model: {cfg['llm']['model_name']}")
    logger.info("Initializing CCME (Contrastive Contextual Memory Encoder)...")
    logger.info(f"  CCMERetriever: alpha={retriever.alpha}, retrieve_top_k={cfg.get('retrieval', {}).get('retrieve_top_k', 5)}, ann_candidates=dynamic(floor={retriever.top_k * 4}, cap=50)")
    _ret_mode = "full_bank" if cfg.get("retrieval", {}).get("use_full_bank", False) else (
        "ccme_k0" if int(cfg.get("retrieval", {}).get("retrieve_top_k", 3)) == 0 else "ccme_topk"
    )
    if cfg.get("retrieval", {}).get("inject_past_solutions", False):
        _ret_mode = f"past_sol+{_ret_mode}" if _ret_mode != "ccme_k0" else "past_sol_only"
    if cfg.get("retrieval", {}).get("synthesizer_v1", False):
        _ret_mode = f"{_ret_mode}+synth_v1"
    elif cfg.get("retrieval", {}).get("synthesizer", False):
        _ret_mode = f"{_ret_mode}+synth"
    logger.info(f"  Retrieval mode: {_ret_mode}")

    # Initialize training components if training is enabled
    enable_training = cfg.get("online_training", {}).get("enable_training", False)
    training_components = None
    training_losses: Dict[str, List[float]] = {"ccme": [], "crte": []}
    success_history = deque(maxlen=50)
    # Cumulative counters for summary
    total_deduped = 0
    total_pruned = 0
    total_refinements = 0
    ccme_rollbacks = 0
    crte_rollbacks = 0

    if enable_training:
        logger.info("Initializing Online Training Components...")
        trajectory_encoder, reflection_encoder, ccme_loss, crte_loss, data_collector, online_trainer = \
            _init_training_components(cfg, query_encoder, memory_encoder)
        training_components = {
            "trajectory_encoder": trajectory_encoder,
            "reflection_encoder": reflection_encoder,
            "ccme_loss": ccme_loss,
            "crte_loss": crte_loss,
            "data_collector": data_collector,
            "online_trainer": online_trainer,
        }
        k_upd = cfg.get("online_training", {}).get("k_upd", 5)
        logger.info(f"  Online training enabled: k_upd={k_upd}")
        logger.info(f"  Adapters enabled: {cfg.get('adapters', {}).get('enabled', False)}")
    else:
        logger.info("  Online training disabled")

    data_dir = cfg["paths"]["data_dir"]
    logger.info(f"Loading dataset from {data_dir}...")
    if _is_mathvista_task(task):
        ds = _load_mathvista_dataset(data_dir)
    else:
        ds = _load_gpqa_dataset(data_dir)
    total_n = len(ds)
    logger.info(f"Loaded {total_n} samples")

    max_n_samples = int(cfg["experiment_control"].get("max_n_samples", -1))
    n_eval = total_n if max_n_samples < 0 else min(max_n_samples, total_n)

    # Custom / shuffled dataset ordering
    from main.utils.dataset_ordering import get_custom_indices, topic_cluster_rerank
    shuffle_seed = cfg["experiment_control"].get("shuffle_seed", None)
    if shuffle_seed is not None:
        task_name = cfg["experiment"].get("task", "")
        indices = get_custom_indices(task_name, ds, n_eval, seed=shuffle_seed)
        logger.info(
            f"Custom dataset ordering applied (task={task_name}, seed={shuffle_seed}): "
            f"first 10 indices = {indices[:10]}"
        )
    else:
        indices = list(range(n_eval))
        logger.info("Dataset order: sequential (no shuffle)")

    use_exposure_annealing = cfg.get("ccme", {}).get("use_exposure_annealing", True)
    use_topic_cluster_retrieval = cfg.get("retrieval", {}).get("use_topic_cluster_retrieval", False)
    # full_bank: pass entire memory bank to generator instead of top-k CCME retrieval
    use_full_bank = cfg.get("retrieval", {}).get("use_full_bank", False)
    # inject_past_solutions: DR-style cosine similarity over past Q&A pairs (independent of CCME)
    inject_past_solutions = cfg.get("retrieval", {}).get("inject_past_solutions", False)
    past_solutions_top_k = int(cfg.get("retrieval", {}).get("past_solutions_top_k", 3))
    # synthesizer: after answering Q_i, run a curator call that sees Q_{i+1}
    use_synthesizer = cfg.get("retrieval", {}).get("synthesizer", False)
    # synthesizer_v1: synthesized entries are EPHEMERAL — not added to bank.
    # Prepended to next query's retrieved_memory; admitted to bank only if reflector
    # marks them HELPFUL. CCME training only sees admitted (or originally-bank) items.
    use_synthesizer_v1 = cfg.get("retrieval", {}).get("synthesizer_v1", False)
    _run_synthesizer = use_synthesizer or use_synthesizer_v1
    # ephemeral items synthesized for the *next* query (assigned e_NNN IDs); cleared each cycle
    _pending_ephemeral_items: List[Dict] = []

    # Build query embeddings in *dataset* order (0..n_eval-1).
    # If a precomputed CSV exists, load from it to avoid redundant API calls.
    # Otherwise compute fresh via the embedding API.
    precomputed_csv = cfg.get("paths", {}).get("precomputed_embeddings_csv", None)
    query_embeddings_list = [None] * n_eval

    if precomputed_csv and os.path.isfile(precomputed_csv):
        import csv as _csv, ast as _ast
        logger.info(f"Loading query embeddings from precomputed CSV: {precomputed_csv}")
        with open(precomputed_csv, "r") as emb_f:
            emb_rows = list(_csv.DictReader(emb_f))
        _csv_is_mmmu = _is_mmmu_pro_task(cfg["experiment"].get("task", ""))
        _csv_is_vision = _is_mmmu_pro_vision_task(cfg["experiment"].get("task", ""))
        _csv_is_mathvista = _is_mathvista_task(cfg["experiment"].get("task", ""))
        for i in range(n_eval):
            if i < len(emb_rows):
                query_embeddings_list[i] = np.array(
                    _ast.literal_eval(emb_rows[i]["embedding"]), dtype=np.float32
                )
            else:
                # Fallback: compute fresh with same multimodal logic as main embed loop
                _fb_images = None
                if _csv_is_mmmu:
                    import re as _re
                    _fb_raw = ds.data.table.slice(i, 1).to_pydict()
                    _fb_row = {k: v[0] for k, v in _fb_raw.items()}
                    _fb_proc = _preprocess_mmmu_pro_row(_fb_row, is_vision=_csv_is_vision)
                    question = _fb_proc["input"]
                    _fb_images = _fb_proc.get("images") or None
                elif _csv_is_mathvista:
                    _fb_raw = ds.data.table.slice(i, 1).to_pydict()
                    _fb_row = {k: v[0] for k, v in _fb_raw.items()}
                    _fb_proc = _preprocess_mathvista_row(_fb_row)
                    question = _fb_proc["input"]
                    _fb_images = _fb_proc.get("images") or None
                else:
                    question = ds[i].get("input", "")
                # Store base 1536-dim embedding (NOT projected) — required by CCME training
                # buffer so Eq receives gradient during train_ccme_step. Retriever projects
                # internally when given a base-dim embedding.
                query_embeddings_list[i] = (
                    query_encoder.get_multimodal_embedding(question, _fb_images)
                    if _fb_images
                    else query_encoder.get_base_embedding(question)
                )
                logger.warning(f"  [EMBED] Index {i} missing from CSV — computed fresh")
        logger.info(f"Loaded {n_eval} query embeddings from CSV")
    else:
        logger.info(f"Computing embeddings for first {n_eval} queries...")
        _is_mmmu = _is_mmmu_pro_task(cfg["experiment"].get("task", ""))
        _is_vision = _is_mmmu_pro_vision_task(cfg["experiment"].get("task", ""))
        _is_mathvista = _is_mathvista_task(cfg["experiment"].get("task", ""))
        for i in range(n_eval):
            logger.info(f"  [EMBED] Dataset index {i + 1}/{n_eval}")
            _embed_images = None
            if _is_mmmu:
                import re as _re
                row_data = ds.data.table.slice(i, 1).to_pydict()
                import ast as _ast
                options_raw = (row_data.get("options", ["[]"]) or ["[]"])[0]
                try:
                    opts = _ast.literal_eval(options_raw) if isinstance(options_raw, str) else options_raw
                except Exception:
                    opts = []
                opts_str = "\n".join(f"({chr(65+j)}) {o}" for j, o in enumerate(opts))
                raw_q = row_data.get("question", [""])[0] or ""
                # Strip <image X> placeholders from question text
                q_clean = _re.sub(r"<image\s*\d+>", "", raw_q).strip()
                question = f"{q_clean}\nOptions:\n{opts_str}" if q_clean else opts_str
                _row = {k: v[0] for k, v in row_data.items()}
                _embed_images = _preprocess_mmmu_pro_row(_row, is_vision=_is_vision).get("images") or None
            elif _is_mathvista:
                row_data = ds.data.table.slice(i, 1).to_pydict()
                q = (row_data.get("question") or [""])[0] or ""
                choices = (row_data.get("choices") or [None])[0]
                q_type = (row_data.get("question_type") or ["free_form"])[0]
                if choices and q_type == "multi_choice":
                    opts_str = "\n".join(f"({chr(65+j)}) {o}" for j, o in enumerate(choices))
                    question = f"{q}\nOptions:\n{opts_str}"
                else:
                    question = q
                _row = {k: v[0] for k, v in row_data.items()}
                _embed_images = _preprocess_mathvista_row(_row).get("images") or None
            else:
                question = ds[i].get("input", "")
            if not question:
                raise ValueError(f"Empty question at index {i}; expected field 'input' to be populated.")
            # Store base 1536-dim embedding (NOT projected) — required by CCME training
            # buffer so Eq receives gradient during train_ccme_step. Retriever projects
            # internally when given a base-dim embedding.
            query_embeddings_list[i] = (
                query_encoder.get_multimodal_embedding(question, _embed_images)
                if _embed_images
                else query_encoder.get_base_embedding(question)
            )
        logger.info(f"Computed {n_eval} query embeddings")

    query_embeddings = (
        np.stack(query_embeddings_list, axis=0)
        if query_embeddings_list
        else np.zeros((0, 1536), dtype=np.float32)
    )

    per_query_ops = cfg.get("per_query_memory_operations", {}) or cfg.get("memory_operations", {})
    dedup_threshold = float(per_query_ops.get("dedup_threshold", 0.85))
    prune_reliability_threshold = float(per_query_ops.get("prune_reliability_threshold", 0.2))
    temporal_decay_lambda_query = float(per_query_ops.get("staleness_decay_lambda", per_query_ops.get("temporal_decay_lambda_query", 0.02)))
    pruning_mode = str(per_query_ops.get("pruning_mode", "AND"))
    min_queries_before_operations = int(per_query_ops.get("min_queries_before_operations", 0))

    # Check if detailed outputs should be saved
    save_detailed_outputs = cfg.get("experiment_control", {}).get("save_detailed_outputs", False)
    detailed_outputs_dir = os.path.join(out_dir, "detailed_outputs")
    io_log_dir = os.path.join(out_dir, "input_output_log")
    save_interval = int(cfg.get("experiment_control", {}).get("save_interval", 0))

    memory_bank = []
    embedding_index = {}
    # past_solutions store for "past_solutions" retrieval mode (DR-style)
    # Each entry: {"query_id": str, "question": str, "answer": str, "embedding": np.ndarray}
    past_solutions_store: List[Dict] = []
    correct = 0
    total = 0
    failed = False
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_estimated_cost = 0.0

    with open(results_path, "w") as results_f:
        try:
            for pos, i in enumerate(indices):
                logger.info("")
                logger.info("============================================================")
                logger.info(f"Query {pos + 1}/{n_eval} (dataset index {i})")
                logger.info("============================================================")

                if _is_mmmu_pro_task(task):
                    _raw = ds.data.table.slice(i, 1).to_pydict()
                    _raw_row = {k: v[0] for k, v in _raw.items()}
                    _processed = _preprocess_mmmu_pro_row(_raw_row, is_vision=_is_mmmu_pro_vision_task(task))
                    question = _processed["input"]
                    ground_truth = _processed["target"]
                    _images = _processed["images"]
                elif _is_mathvista_task(task):
                    _raw = ds.data.table.slice(i, 1).to_pydict()
                    _raw_row = {k: v[0] for k, v in _raw.items()}
                    _processed = _preprocess_mathvista_row(_raw_row)
                    question = _processed["input"]
                    ground_truth = _processed["target"]
                    _images = _processed["images"]
                else:
                    row = ds[i]
                    question = row.get("input", "") or ""
                    ground_truth = row.get("target", "") or ""
                    _images = None

                query_id = f"Q_{pos + 1:03d}"

                if memory_bank:
                    retriever.index_memory_bank(memory_bank)

                # CCME retrieval: score(x,m) = α·sim(Eq(x),Em(m)) + (1-α)·p̂(m)
                if use_full_bank:
                    retrieved_memory = list(memory_bank)
                    logger.info(f"  [full_bank] Passing all {len(retrieved_memory)} memory items to generator")
                else:
                    effective_top_k = int(cfg.get("retrieval", {}).get("retrieve_top_k", 3))
                    if effective_top_k == 0 or not memory_bank:
                        retrieved_memory = []
                    else:
                        retrieved_memory = retriever.retrieve(
                            query_embedding=query_embeddings[i],
                            memory_bank=memory_bank,
                            top_k=effective_top_k,
                            use_exposure_annealing=use_exposure_annealing,
                            use_source_query_sim=False,
                            current_query_base_emb=None,
                        )
                    logger.info(f"  Retrieved Memory Count: {len(retrieved_memory)}")
                    if retrieved_memory and hasattr(retriever, "last_retrieval_scores"):
                        for m in retrieved_memory:
                            mid = m.get("id", "?")
                            score = retriever.last_retrieval_scores.get(mid, float("nan"))
                            logger.info(f"    [ccme] {mid} | score={score:.4f} | {m.get('title', '')[:60]}")

                    # Topic-clustering re-rank: surface topic-matching memory items first
                    if use_topic_cluster_retrieval and retrieved_memory:
                        retrieved_memory = topic_cluster_rerank(retrieved_memory, question)
                        logger.info(f"  [TopicCluster] Reranked by topic: {[m.get('id', '?') for m in retrieved_memory]}")

                    # synthesizer_v1: prepend ephemeral entries (NOT in bank) to retrieved_memory.
                    # IDs are e_NNN namespace so no collision with bank m_NNN IDs.
                    # Pipeline's admission gate will admit HELPFUL ones to bank, drop the rest.
                    if use_synthesizer_v1 and _pending_ephemeral_items:
                        retrieved_memory = list(_pending_ephemeral_items) + retrieved_memory
                        logger.info(
                            f"  [SynthV1] Prepended {len(_pending_ephemeral_items)} ephemeral "
                            f"entr{'y' if len(_pending_ephemeral_items)==1 else 'ies'}: "
                            f"{[m.get('id','?') for m in _pending_ephemeral_items]}; "
                            f"total to generator: {len(retrieved_memory)}"
                        )

                training_progress = f"Sample {i + 1} of {n_eval}"
                question_context = cfg.get("experiment_control", {}).get("dataset_description", "")

                # Past-solutions retrieval (DR-style, independent of CCME): cosine similarity over stored Q&A pairs
                past_solutions_briefing = ""
                if inject_past_solutions and past_solutions_store:
                    past_solutions_briefing = _retrieve_past_solutions(
                        query_embedding=query_embeddings[i],
                        past_solutions_store=past_solutions_store,
                        top_k=past_solutions_top_k,
                    )
                    logger.info(f"  [past_solutions] Injecting top-{past_solutions_top_k} past solutions")
                    q_norm = query_embeddings[i] / (np.linalg.norm(query_embeddings[i]) + 1e-10)
                    ps_sims = []
                    for entry in past_solutions_store:
                        e = entry["embedding"] / (np.linalg.norm(entry["embedding"]) + 1e-10)
                        ps_sims.append((float(np.dot(q_norm, e)), entry["query_id"]))
                    for sim, qid in sorted(ps_sims, reverse=True)[:past_solutions_top_k]:
                        logger.info(f"    [past_sol] {qid} | sim={sim:.4f}")

                final_answer, updated_memory_bank, updated_embedding_index, pipeline_outputs = pipeline.run_full_pipeline(
                    language_model=language_model,
                    question=question,
                    memory_bank=memory_bank,
                    ground_truth=ground_truth,
                    query_id=query_id,
                    retrieved_memory=retrieved_memory,
                    embedding_index=embedding_index,
                    dedup_threshold=dedup_threshold,
                    prune_reliability_threshold=prune_reliability_threshold,
                    temporal_decay_lambda_query=temporal_decay_lambda_query,
                    pruning_mode=pruning_mode,
                    min_queries_before_operations=min_queries_before_operations,
                    training_progress=training_progress,
                    question_context=question_context,
                    past_solutions_briefing=past_solutions_briefing,
                    ephemeral_items=_pending_ephemeral_items if use_synthesizer_v1 else None,
                    temperature=float(cfg["llm"].get("temperature", 0.0)),
                    max_tokens=int(cfg["llm"].get("max_tokens", 4096)),
                    allow_code_execution=bool(cfg["llm"].get("execute_python_code", True)),
                    images=_images,
                    memory_encoder=memory_encoder,
                )

                # Reset ephemeral buffer — pipeline has either admitted HELPFUL items or discarded them
                if use_synthesizer_v1:
                    _pending_ephemeral_items = []
                    eph_remap = pipeline_outputs.get("ephemeral_id_remap") or {}
                    if eph_remap:
                        logger.info(f"  [SynthV1] Admitted {len(eph_remap)} ephemeral(s) to bank: {eph_remap}")

                retrieved_count = pipeline_outputs.get("retrieved_memory_count")
                memory_consulted = pipeline_outputs.get("generator", {}).get("memory_consulted", [])
                generator_success = bool(pipeline_outputs.get("generator", {}).get("final_answer", "").strip())
                reflector_success = pipeline_outputs.get("reflection") is not None
                curator_success = pipeline_outputs.get("curation") is not None
                usage_total = pipeline_outputs.get("token_usage", {}).get("total", {})
                total_prompt_tokens += int(usage_total.get("prompt_tokens", 0) or 0)
                total_completion_tokens += int(usage_total.get("completion_tokens", 0) or 0)
                total_estimated_cost += float(usage_total.get("estimated_cost_usd", 0.0) or 0.0)

                normalized_final = _normalize_choice(final_answer)
                normalized_gt = _normalize_choice(ground_truth)
                is_correct = _is_correct_answer(task, question, final_answer, ground_truth)

                # Override reflector's execution_status to match actual evaluator result.
                # The reflector LLM may mis-assess correctness (e.g. numeric answer matching
                # option text), so align it with is_correct to ensure training labels and
                # memory update signals are consistent.
                reflection = pipeline_outputs.get("reflection")
                if reflection is not None and isinstance(reflection, dict):
                    reflection["execution_status"] = "SUCCESS" if is_correct else "FAILURE"

                total += 1
                if is_correct:
                    correct += 1

                logger.info(f"  Generator: {'SUCCESS' if generator_success else 'FAILURE'}")
                logger.info(f"  Reflector: {'SUCCESS' if reflector_success else 'FAILURE'}")
                logger.info(f"  Curator: {'SUCCESS' if curator_success else 'FAILURE'}")
                logger.info(f"  Final Answer: ({normalized_final})" if normalized_final else f"  Final Answer: {final_answer}")
                logger.info(f"  Ground Truth: ({normalized_gt})" if normalized_gt else f"  Ground Truth: {ground_truth}")
                logger.info(f"  Correct: {is_correct}")
                logger.info(f"  Memory Bank Size: {len(updated_memory_bank)}")
                if retrieved_count is not None:
                    logger.info(f"  Retrieved Memory Count: {retrieved_count}")
                logger.info(f"  Memory Consulted Count: {len(memory_consulted)}")
                logger.info(f"  Running Accuracy: {correct}/{total} ({(100.0 * correct / total):.2f}%)")

                # --- Curation: new entries added ---
                curation = pipeline_outputs.get("curation")
                if curation:
                    new_entries = curation.get("new_entries", [])
                    if new_entries:
                        titles = [e.get("title", "?") for e in new_entries]
                        logger.info(f"  Curator Added: {len(new_entries)} entr{'y' if len(new_entries)==1 else 'ies'} → {titles}")
                    else:
                        logger.info(f"  Curator Added: 0 entries")

                # --- Per-query memory operations (dedup / prune / warm-up guard) ---
                merge_stats = pipeline_outputs.get("merge_stats", {})
                if merge_stats:
                    ops_skipped = merge_stats.get("ops_skipped")
                    if ops_skipped:
                        logger.info(f"  MemOps: skipped ({ops_skipped})")
                    else:
                        deduped = merge_stats.get("deduplicated", 0)
                        pruned  = merge_stats.get("pruned_low_reliability", 0)
                        before  = merge_stats.get("items_before", "?")
                        after   = merge_stats.get("items_after", "?")
                        logger.info(f"  MemOps: deduped={deduped}, pruned={pruned}, bank {before}→{after}")
                        total_deduped += deduped
                        total_pruned += pruned

                # --- Memory health ---
                health = pipeline_outputs.get("memory_health", {})
                if health:
                    avg_rel = health.get("avg_reliability", None)
                    avg_ret = health.get("avg_retrieved_count", None)
                    if avg_rel is not None:
                        logger.info(f"  Memory Health: avg_reliability={avg_rel:.3f}, avg_retrieved={avg_ret:.1f}")

                # --- Token cost ---
                cost = usage_total.get("estimated_cost_usd")
                if cost is not None:
                    logger.info(f"  Token Cost: prompt={usage_total.get('prompt_tokens',0)}, "
                                f"completion={usage_total.get('completion_tokens',0)}, "
                                f"cost=${cost:.4f} (cumulative=${total_estimated_cost:.4f})")

                result_row = {
                    "query_id": query_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "final_answer": final_answer,
                    "correct": is_correct,
                    "image_count": len(_images) if _images else 0,
                    "generator_success": generator_success,
                    "reflector_success": reflector_success,
                    "curator_success": curator_success,
                    "memory_bank_size": len(updated_memory_bank),
                    "retrieved_memory_count": retrieved_count,
                    "memory_consulted": memory_consulted,
                    "token_usage": pipeline_outputs.get("token_usage", {}),
                    "pipeline_outputs": pipeline_outputs,
                }
                results_f.write(json.dumps(result_row) + "\n")
                results_f.flush()

                if save_detailed_outputs:
                    generator_output = pipeline_outputs.get("generator", {}).get("full_response", "")
                    with open(os.path.join(detailed_outputs_dir, f"{query_id}_generator.txt"), "w") as f:
                        f.write(f"=== QUERY {i + 1} ===\n")
                        f.write(f"INPUT: {question}\n\n")
                        f.write(f"=== GENERATOR OUTPUT ===\n")
                        f.write(generator_output)

                    # Always serialize from the (possibly corrected) reflection dict
                    # so that execution_status in the file matches is_correct.
                    _refl_dict = pipeline_outputs.get("reflection")
                    if _refl_dict is not None:
                        reflector_output = json.dumps(_refl_dict, indent=2)
                    else:
                        reflector_output = pipeline_outputs.get("reflector", {}).get("full_response", "")
                    with open(os.path.join(detailed_outputs_dir, f"{query_id}_reflector.txt"), "w") as f:
                        f.write(f"=== QUERY {i + 1} ===\n")
                        f.write(f"INPUT: {question}\n\n")
                        f.write(f"=== REFLECTOR OUTPUT ===\n")
                        f.write(reflector_output)

                    curator_output = pipeline_outputs.get("curation_raw", "")
                    if not curator_output:
                        curation = pipeline_outputs.get("curation")
                        curator_output = json.dumps(curation, indent=2) if curation else ""
                    with open(os.path.join(detailed_outputs_dir, f"{query_id}_curator.txt"), "w") as f:
                        f.write(f"=== QUERY {i + 1} ===\n")
                        f.write(f"INPUT: {question}\n\n")
                        f.write(f"=== CURATOR OUTPUT ===\n")
                        f.write(curator_output)

                # Save input/output log per query
                query_log_dir = os.path.join(io_log_dir, f"query_{i + 1:03d}")
                _ensure_dir(query_log_dir)

                _img_note = f"IMAGES: {len(_images)} image(s) attached\n" if _images else ""

                # 1. query.txt — the question + ground truth
                with open(os.path.join(query_log_dir, "query.txt"), "w") as f:
                    f.write(f"=== QUERY {query_id} (index {i + 1}) ===\n\n")
                    if _img_note:
                        f.write(_img_note + "\n")
                    f.write(f"QUESTION:\n{question}\n\n")
                    f.write(f"\nGROUND TRUTH: {ground_truth}\n")
                    f.write(f"FINAL ANSWER: {final_answer}\n")
                    f.write(f"CORRECT: {is_correct}\n")

                # 2. generator_input.txt — formatted prompt sent to generator LLM
                gen_prompt = pipeline_outputs.get("generator", {}).get("prompt", "")
                with open(os.path.join(query_log_dir, "generator_input.txt"), "w") as f:
                    f.write(f"=== GENERATOR INPUT (formatted prompt) ===\n\n")
                    if _img_note:
                        f.write(_img_note + "\n")
                    f.write(gen_prompt)

                # 3. generator_output.txt — raw LLM response
                gen_output = pipeline_outputs.get("generator_raw", "")
                with open(os.path.join(query_log_dir, "generator_output.txt"), "w") as f:
                    f.write(f"=== GENERATOR OUTPUT (raw LLM response) ===\n\n")
                    f.write(gen_output)

                # 4. reflector_input.txt — formatted prompt sent to reflector LLM
                refl_prompt = pipeline_outputs.get("reflector_prompt", "")
                with open(os.path.join(query_log_dir, "reflector_input.txt"), "w") as f:
                    f.write(f"=== REFLECTOR INPUT (formatted prompt) ===\n\n")
                    if _img_note:
                        f.write(_img_note + "\n")
                    f.write(refl_prompt)

                # 5. reflector_output.txt — raw LLM response
                refl_output = pipeline_outputs.get("reflection_raw", "")
                with open(os.path.join(query_log_dir, "reflector_output.txt"), "w") as f:
                    f.write(f"=== REFLECTOR OUTPUT (raw LLM response) ===\n\n")
                    f.write(refl_output)

                # 6. curator_input.txt — formatted prompt sent to curator LLM
                cur_prompt = pipeline_outputs.get("curator_prompt", "")
                with open(os.path.join(query_log_dir, "curator_input.txt"), "w") as f:
                    f.write(f"=== CURATOR INPUT (formatted prompt) ===\n\n")
                    if _img_note:
                        f.write(_img_note + "\n")
                    f.write(cur_prompt)

                # 7. curator_output.txt — raw LLM response
                cur_output = pipeline_outputs.get("curation_raw", "")
                with open(os.path.join(query_log_dir, "curator_output.txt"), "w") as f:
                    f.write(f"=== CURATOR OUTPUT (raw LLM response) ===\n\n")
                    f.write(cur_output)

                # --- Synthesizer: see Q_{i+1} and pre-populate memory ---
                if _run_synthesizer and pos + 1 < len(indices):
                    next_i = indices[pos + 1]
                    next_query_id = f"Q_{pos + 2:03d}"
                    try:
                        _next_images = None
                        if _is_mmmu_pro_task(task):
                            _nxt_raw = ds.data.table.slice(next_i, 1).to_pydict()
                            _nxt_row = {k: v[0] for k, v in _nxt_raw.items()}
                            _nxt_processed = _preprocess_mmmu_pro_row(_nxt_row, is_vision=_is_mmmu_pro_vision_task(task))
                            next_question = _nxt_processed["input"]
                            _next_images = _nxt_processed.get("images") or None
                        elif _is_mathvista_task(task):
                            _nxt_raw = ds.data.table.slice(next_i, 1).to_pydict()
                            _nxt_row = {k: v[0] for k, v in _nxt_raw.items()}
                            _nxt_processed = _preprocess_mathvista_row(_nxt_row)
                            next_question = _nxt_processed["input"]
                            _next_images = _nxt_processed.get("images") or None
                        else:
                            next_question = ds[next_i].get("input", "") or ""

                        if next_question:
                            logger.info(f"  [Synthesizer] Running synthesizer for {next_query_id}...")
                            la_curation, la_curation_raw, la_curator_usage, la_curator_prompt = \
                                pipeline.run_synthesizer_stage(
                                    language_model=language_model,
                                    memory_bank=updated_memory_bank,
                                    next_question=next_question,
                                    next_query_id=next_query_id,
                                    training_progress=f"Sample {pos + 2} of {n_eval}",
                                    question_context=question_context,
                                    images=_next_images,
                                    temperature=float(cfg["llm"].get("temperature", 0.0)),
                                    max_tokens=int(cfg["llm"].get("max_tokens", 4096)),
                                )
                            if la_curation:
                                la_new = la_curation.get("new_entries", [])
                                if la_new:
                                    if use_synthesizer_v1:
                                        # v1 (v2-design): hold as ephemeral, do NOT add to bank yet.
                                        # Assigned e_NNN IDs (separate namespace from m_NNN bank IDs).
                                        _pending_ephemeral_items = []
                                        for idx, entry in enumerate(la_new):
                                            e = entry.copy()
                                            e["id"] = f"e_{idx + 1:03d}"
                                            _pending_ephemeral_items.append(e)
                                        titles = [e.get("title", "?") for e in la_new]
                                        logger.info(f"  [Synthesizer] Stored {len(la_new)} ephemeral entr{'y' if len(la_new)==1 else 'ies'} → {titles} (admission gated by next reflector verdict)")
                                    else:
                                        # v0 (original): add directly to bank
                                        from main.utils.memory_formatter import apply_curation_updates as _apply_curation
                                        updated_memory_bank, updated_embedding_index = \
                                            _apply_curation(
                                                updated_memory_bank,
                                                {"new_items": la_new, "reliability_updates": [],
                                                 "updates_to_existing": [], "prune_candidates": []},
                                                language_model,
                                                updated_embedding_index or {},
                                                embed_fn=memory_encoder.get_base_embedding,
                                            )
                                        titles = [e.get("title", "?") for e in la_new]
                                        logger.info(f"  [Synthesizer] Added {len(la_new)} entr{'y' if len(la_new)==1 else 'ies'} → {titles}")
                                else:
                                    logger.info(f"  [Synthesizer] Synthesizer proposed 0 entries")
                            else:
                                logger.info(f"  [Synthesizer] Synthesizer failed")

                            if save_detailed_outputs:
                                _next_img_note = f"IMAGES: {len(_next_images)} image(s) attached\n" if _next_images else ""
                                with open(os.path.join(query_log_dir, "synthesizer_input.txt"), "w") as f:
                                    f.write(f"=== SYNTHESIZER INPUT (for {next_query_id}) ===\n\n")
                                    if _next_img_note:
                                        f.write(_next_img_note + "\n")
                                    f.write(la_curator_prompt)
                                with open(os.path.join(query_log_dir, "synthesizer_output.txt"), "w") as f:
                                    f.write(f"=== SYNTHESIZER OUTPUT (for {next_query_id}) ===\n\n")
                                    f.write(la_curation_raw)
                    except Exception as _la_err:
                        logger.warning(f"  [Synthesizer] Error: {_la_err}")

                # Append to past-solutions store regardless of mode (cheap, always useful)
                if final_answer and final_answer.strip():
                    past_solutions_store.append({
                        "query_id": query_id,
                        "question": question,
                        "answer": pipeline_outputs.get("generator_raw", final_answer),
                        "embedding": query_embeddings[i],
                    })

                memory_bank = updated_memory_bank
                embedding_index = updated_embedding_index or {}

                if enable_training and training_components is not None:
                    success_history.append(float(is_correct))

                    generator_outputs = pipeline_outputs.get("generator", {})
                    parsed_steps = generator_outputs.get("parsed_steps", [])
                    memory_consulted = generator_outputs.get("memory_consulted", [])
                    reflection = pipeline_outputs.get("reflection")

                    execution_status = "SUCCESS" if is_correct else "FAILURE"

                    if reflection is not None and retrieved_memory:
                        data_collector = training_components["data_collector"]
                        trajectory_encoder = training_components["trajectory_encoder"]
                        reflection_encoder = training_components["reflection_encoder"]
                        online_trainer = training_components["online_trainer"]

                        try:
                            trajectory_embedding = trajectory_encoder.encode_trajectory(parsed_steps)
                            reflection_embedding = reflection_encoder.encode_reflection(reflection)
                        except Exception as e:
                            logger.warning(f"  Failed to encode trajectory/reflection: {e}")
                            trajectory_embedding = None
                            reflection_embedding = None

                        # In v1 (v2-design), pipeline returns effective_retrieved_memory:
                        # rejected ephemerals dropped, admitted ones renamed to m_NNN.
                        ccme_retrieved_memory = pipeline_outputs.get("effective_retrieved_memory", retrieved_memory)
                        try:
                            data_collector.collect_from_episode(
                                query=question,
                                query_embedding=query_embeddings[i],
                                retrieved_memory=ccme_retrieved_memory,
                                trajectory_steps=parsed_steps,
                                trajectory_embedding=trajectory_embedding,
                                reflection=reflection,
                                reflection_embedding=reflection_embedding,
                                execution_status=execution_status,
                                query_embedding_is_projected=False,
                                embedding_index=embedding_index,
                            )
                            buffer_stats = data_collector.get_buffer_stats()
                            logger.info(f"  Training buffers: CCME+={buffer_stats['ccme_positive']}, CCME-={buffer_stats['ccme_negative']}, "
                                        f"CRTE+={buffer_stats['crte_positive']}, CRTE-={buffer_stats['crte_negative']}")
                        except Exception as e:
                            logger.warning(f"  Failed to collect training data: {e}")

                        success_rate = np.mean(list(success_history)) if success_history else 0.5
                        try:
                            update_stats = online_trainer.step(success_rate=success_rate, memory_bank=memory_bank, retriever=retriever)
                            if update_stats is not None:
                                status = update_stats.get('status', '?')
                                logger.info(f"  Training update #{update_stats['update_count']}: "
                                            f"CCME loss={update_stats['avg_ccme_loss']:.4f}, "
                                            f"CRTE loss={update_stats['avg_crte_loss']:.4f} "
                                            f"stable={update_stats['is_stable']} status={status}")
                                if status in ('ccme_rolled_back', 'both_rolled_back'):
                                    logger.warning(f"  Rollback: CCME (Eq, Em) reverted to last stable checkpoint")
                                    ccme_rollbacks += 1
                                if status in ('crte_rolled_back', 'both_rolled_back'):
                                    logger.warning(f"  Rollback: CRTE (Et, Er) reverted to last stable checkpoint")
                                    crte_rollbacks += 1
                                refinement = update_stats.get("memory_refinement")
                                if refinement:
                                    logger.info(f"  Memory Refinement: {refinement.get('original_count','?')}→"
                                                f"{refinement.get('prototypes_kept','?')} items "
                                                f"({refinement.get('items_removed','?')} removed, "
                                                f"{refinement.get('clusters_found','?')} clusters)")
                                mean_sim = update_stats.get("refinement_mean_sim")
                                if mean_sim is not None:
                                    triggered = update_stats.get("refinement_triggered", False)
                                    tag = "TRIGGERED" if triggered else "check"
                                    logger.info(f"  Memory Refinement [{tag}]: mean_sim={mean_sim:.4f} "
                                                f"(threshold={update_stats.get('refinement_threshold','?')})")
                                training_losses["ccme"].append(update_stats["avg_ccme_loss"])
                                training_losses["crte"].append(update_stats["avg_crte_loss"])
                                total_loss = update_stats["avg_ccme_loss"] + update_stats["avg_crte_loss"]
                                with open(training_log_path, "a") as f:
                                    f.write(json.dumps({
                                        "step": update_stats["update_count"],
                                        "ccme_loss": update_stats["avg_ccme_loss"],
                                        "crte_loss": update_stats["avg_crte_loss"],
                                        "total_loss": total_loss,
                                        "timestamp": datetime.now().isoformat()
                                    }) + "\n")
                                    if update_stats.get("memory_refinement"):
                                        f.write(json.dumps({
                                            "event": {
                                                "type": "memory_refinement",
                                                "update_count": update_stats["update_count"],
                                                **update_stats["memory_refinement"]
                                            },
                                            "timestamp": datetime.now().isoformat()
                                        }) + "\n")
                                # Apply refined memory bank back if refinement produced one
                                if update_stats.get("refined_memory_bank") is not None:
                                    old_size = len(memory_bank)
                                    memory_bank = update_stats["refined_memory_bank"]
                                    # Rebuild embedding_index to only keep items still in the bank
                                    surviving_ids = {item.get("id") for item in memory_bank}
                                    embedding_index = {k: v for k, v in embedding_index.items() if k in surviving_ids}
                                    logger.info(f"  [Memory Refinement] Applied: {old_size} → {len(memory_bank)} items")
                                    total_refinements += 1
                        except Exception as e:
                            logger.warning(f"  Training step failed: {e}")

                if save_interval > 0 and (i + 1) % save_interval == 0:
                    try:
                        with open(os.path.join(out_dir, "checkpoints", f"memory_bank_q{i + 1}.json"), "w") as f:
                            json.dump(memory_bank, f, indent=2)
                    except Exception as e:
                        logger.warning(f"  Failed to save checkpoint: {e}")
        except Exception:
            failed = True
            logger.exception("Experiment failed")

    # Save artifacts to match existing folder structure
    config_out = dict(cfg)
    config_out["experiment"] = dict(cfg.get("experiment", {}))
    config_out["experiment"]["run_id"] = run_id
    config_out["experiment"]["timestamp"] = timestamp

    with open(os.path.join(out_dir, "experiment_config.json"), "w") as f:
        json.dump(config_out, f, indent=2)

    with open(os.path.join(out_dir, "final_memory_bank.json"), "w") as f:
        json.dump(memory_bank, f, indent=2)

    with open(os.path.join(out_dir, "embedding_index.pkl"), "wb") as f:
        pickle.dump(embedding_index, f)

    # Save training buffers
    if enable_training and training_components is not None:
        data_collector = training_components["data_collector"]

        def _ser(v):
            """Recursively make a value JSON-serializable."""
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, torch.Tensor):
                return v.detach().cpu().numpy().tolist()
            if isinstance(v, dict):
                return {kk: _ser(vv) for kk, vv in v.items()}
            if isinstance(v, list):
                return [_ser(x) for x in v]
            return v

        def _serialize_ccme(item: Dict, pair_idx: int) -> Dict:
            """Serialize one CCME buffer item with explicit classification fields."""
            pair_type = item.get("pair_type", "unknown")
            label = item.get("label", "unknown")
            if label == "positive":
                neg_category = None
                neg_hardness = None
            elif pair_type == "harmful":
                neg_category = "hard_negative"
                neg_hardness = "harmful"
            elif pair_type == "soft_neutral":
                neg_category = "soft_negative"
                neg_hardness = "neutral_verdict"
            elif pair_type == "soft_unused":
                neg_category = "soft_negative"
                neg_hardness = "not_evaluated"
            else:
                neg_category = "negative"
                neg_hardness = pair_type

            mem = item.get("memory_item", {})
            meta = mem.get("meta", {})
            return {
                "_pair_index": pair_idx,
                "_label": label,
                "_pair_type": pair_type,
                "_neg_category": neg_category,       # None | "hard_negative" | "soft_negative"
                "_neg_hardness": neg_hardness,        # None | "harmful" | "neutral_verdict" | "not_evaluated"
                "query": item.get("query", ""),
                "query_embedding_is_projected": item.get("query_embedding_is_projected", False),
                "memory_item": {
                    "id": mem.get("id"),
                    "title": mem.get("title"),
                    "bullets": mem.get("bullets", []),
                    "tags": mem.get("tags", []),
                    "meta": {
                        "helpful": meta.get("helpful", 0),
                        "harmful": meta.get("harmful", 0),
                        "retrieved_count": meta.get("retrieved_count", 0),
                        "reliability": meta.get("reliability", 0.5),
                        "created": meta.get("created"),
                        "source_queries": meta.get("source_queries", []),
                    },
                },
            }

        def _serialize_crte(item: Dict, pair_idx: int, role: str) -> Dict:
            """Serialize one CRTE buffer item with explicit classification fields.

            role: "positive" (SUCCESS episode) or "negative" (FAILURE episode).
            In-batch mismatch negatives are NOT stored — they are dynamically
            constructed from the positive set at training time (positive[j] where j≠i).
            Stored FAILURE episodes serve as hard (failed-episode) negatives.
            """
            steps = item.get("trajectory_steps", [])
            refl  = item.get("reflection", {})
            traj_emb  = _ser(item.get("trajectory_embedding"))
            refl_emb  = _ser(item.get("reflection_embedding"))
            return {
                "_pair_index": pair_idx,
                "_label": item.get("label", role),
                "_execution_status": item.get("execution_status", "UNKNOWN"),
                "_negative_role": (
                    None if role == "positive"
                    else "failed_episode_hard_negative"
                    # Note: in-batch mismatch negatives are built dynamically
                    # at train time from the positive pool — not stored here.
                ),
                "query": item.get("query", ""),
                "trajectory_steps_count": len(steps),
                "trajectory_steps": steps,
                "trajectory_embedding_dim": len(traj_emb) if traj_emb else None,
                "trajectory_embedding": traj_emb,
                "reflection": {
                    "execution_status": refl.get("execution_status"),
                    "critical_steps_count": len(
                        refl.get("trajectory_analysis", {}).get("critical_steps", [])
                    ),
                    "memory_evaluation": refl.get("memory_evaluation", []),
                    "trajectory_analysis": refl.get("trajectory_analysis", {}),
                },
                "reflection_embedding_dim": len(refl_emb) if refl_emb else None,
                "reflection_embedding": refl_emb,
            }

        # ── CCME structured folders ────────────────────────────────────────
        ccme_dir = os.path.join(out_dir, "buffers", "ccme")
        os.makedirs(os.path.join(ccme_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(ccme_dir, "negative", "hard"), exist_ok=True)
        os.makedirs(os.path.join(ccme_dir, "negative", "soft"), exist_ok=True)

        ccme_pos_items = data_collector.ccme_positive_buffer.get_all()
        ccme_neg_items = data_collector.ccme_negative_buffer.get_all()

        for idx, item in enumerate(ccme_pos_items):
            s = _serialize_ccme(item, idx)
            with open(os.path.join(ccme_dir, "positive", f"pair_{idx:04d}.json"), "w") as f:
                json.dump(s, f, indent=2)

        hard_idx = soft_idx = 0
        for idx, item in enumerate(ccme_neg_items):
            s = _serialize_ccme(item, idx)
            if s["_neg_category"] == "hard_negative":
                fpath = os.path.join(ccme_dir, "negative", "hard", f"pair_{hard_idx:04d}.json")
                hard_idx += 1
            else:
                fpath = os.path.join(ccme_dir, "negative", "soft", f"pair_{soft_idx:04d}.json")
                soft_idx += 1
            with open(fpath, "w") as f:
                json.dump(s, f, indent=2)

        pair_type_counts: Dict[str, int] = {}
        for item in ccme_neg_items:
            pt = item.get("pair_type", "unknown")
            pair_type_counts[pt] = pair_type_counts.get(pt, 0) + 1

        ccme_summary = {
            "total_positive": len(ccme_pos_items),
            "total_negative": len(ccme_neg_items),
            "negative_breakdown": {
                "hard_negative_harmful": pair_type_counts.get("harmful", 0),
                "soft_negative_neutral_verdict": pair_type_counts.get("soft_neutral", 0),
                "soft_negative_not_evaluated": pair_type_counts.get("soft_unused", 0),
            },
            "note": (
                "Positive = HELPFUL verdict. "
                "Hard negative = HARMFUL verdict. "
                "Soft negative (neutral_verdict) = NEUTRAL verdict, was consulted but had no signal. "
                "Soft negative (not_evaluated) = retrieved but absent from reflector memory_evaluation."
            ),
        }
        with open(os.path.join(ccme_dir, "summary.json"), "w") as f:
            json.dump(ccme_summary, f, indent=2)

        # ── CRTE structured folders ────────────────────────────────────────
        crte_dir = os.path.join(out_dir, "buffers", "crte")
        os.makedirs(os.path.join(crte_dir, "positive"), exist_ok=True)
        os.makedirs(os.path.join(crte_dir, "negative", "failed_episodes"), exist_ok=True)

        crte_pos_items = data_collector.crte_positive_buffer.get_all()
        crte_neg_items = data_collector.crte_negative_buffer.get_all()

        for idx, item in enumerate(crte_pos_items):
            s = _serialize_crte(item, idx, "positive")
            with open(os.path.join(crte_dir, "positive", f"pair_{idx:04d}.json"), "w") as f:
                json.dump(s, f, indent=2)

        for idx, item in enumerate(crte_neg_items):
            s = _serialize_crte(item, idx, "negative")
            with open(os.path.join(crte_dir, "negative", "failed_episodes", f"pair_{idx:04d}.json"), "w") as f:
                json.dump(s, f, indent=2)

        crte_summary = {
            "total_positive": len(crte_pos_items),
            "total_negative_failed_episodes": len(crte_neg_items),
            "note": (
                "Positive = SUCCESS episode: (Et(τ⁺), Er(u⁺)) paired. "
                "Also serves as in-batch mismatch negatives at training time (τⱼ/uⱼ where j≠i). "
                "Negative = FAILURE episode: stored as failed-episode hard negatives. "
                "In-batch mismatch negatives are built dynamically during train_crte_step "
                "and are NOT stored here — they come from the positive pool."
            ),
        }
        with open(os.path.join(crte_dir, "summary.json"), "w") as f:
            json.dump(crte_summary, f, indent=2)

        # ── legacy flat files (kept for backward compat) ──────────────────
        with open(os.path.join(out_dir, "ccme_buffer.json"), "w") as f:
            json.dump({
                "positive": [_serialize_ccme(x, i) for i, x in enumerate(ccme_pos_items)],
                "negative": [_serialize_ccme(x, i) for i, x in enumerate(ccme_neg_items)],
            }, f, indent=2)
        with open(os.path.join(out_dir, "crte_buffer.json"), "w") as f:
            json.dump({
                "positive": [_serialize_crte(x, i, "positive") for i, x in enumerate(crte_pos_items)],
                "negative": [_serialize_crte(x, i, "negative") for i, x in enumerate(crte_neg_items)],
            }, f, indent=2)

        logger.info(
            f"Saved training buffers → buffers/ccme/ and buffers/crte/\n"
            f"  CCME: {len(ccme_pos_items)} positive | "
            f"{pair_type_counts.get('harmful',0)} hard-neg (harmful) | "
            f"{pair_type_counts.get('soft_neutral',0)} soft-neg (neutral) | "
            f"{pair_type_counts.get('soft_unused',0)} soft-neg (unused)\n"
            f"  CRTE: {len(crte_pos_items)} positive (SUCCESS) | "
            f"{len(crte_neg_items)} negative (FAILURE/failed-episode)"
        )
    else:
        with open(os.path.join(out_dir, "ccme_buffer.json"), "w") as f:
            json.dump({"positive": [], "negative": []}, f)

        with open(os.path.join(out_dir, "crte_buffer.json"), "w") as f:
            json.dump({"positive": [], "negative": []}, f)

    with open(training_losses_path, "w") as f:
        json.dump(training_losses, f, indent=2)

    # Build training statistics for summary
    training_stats = {}
    if enable_training and training_components is not None:
        online_trainer = training_components["online_trainer"]
        data_collector = training_components["data_collector"]
        buffer_stats = data_collector.get_buffer_stats()
        training_stats = {
            "enabled": True,
            "total_updates": online_trainer.update_count,
            "final_step_count": online_trainer.step_count,
            "k_upd": online_trainer.k_upd,
            "num_train_steps": online_trainer.num_train_steps,
            "batch_size": online_trainer.batch_size,
            "buffer_stats": buffer_stats,
            "avg_ccme_loss": np.mean(training_losses["ccme"]) if training_losses["ccme"] else 0.0,
            "avg_crte_loss": np.mean(training_losses["crte"]) if training_losses["crte"] else 0.0,
            "ccme_rollbacks": ccme_rollbacks,
            "crte_rollbacks": crte_rollbacks,
            "total_refinements": total_refinements,
        }
    else:
        training_stats = {"enabled": False}

    exp_cfg = cfg.get("experiment", {})
    llm_cfg = cfg.get("llm", {})
    retrieval_cfg = cfg.get("retrieval", {})
    elapsed_seconds = time.time() - start_time
    summary = {
        "experiment_name": exp_cfg.get("name", ""),
        "approach_name": exp_cfg.get("approach_name", "LeRe"),
        "run_id": exp_cfg.get("run_id", None),
        "model_name": llm_cfg.get("model_name", ""),
        "total_queries": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "retrieval_mode": _ret_mode,
        "retrieve_top_k": retrieval_cfg.get("retrieve_top_k"),
        "ccme_alpha": retrieval_cfg.get("ccme_alpha"),
        "memory_bank_size": len(memory_bank),
        "memory_ops": {
            "total_deduped": total_deduped,
            "total_pruned": total_pruned,
        },
        "elapsed_seconds": elapsed_seconds,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "estimated_cost_usd": total_estimated_cost,
            "cost_method": "litellm.completion_cost",
        },
        "training": training_stats,
        "output_dir": out_dir,
        "output_files": {
            "main_log": os.path.basename(main_log_path),
            "results": os.path.basename(results_path),
            "config": "experiment_config.json",
            "memory_bank": "final_memory_bank.json",
            "embedding_index": "embedding_index.pkl",
            "ccme_buffer": "ccme_buffer.json",
            "crte_buffer": "crte_buffer.json",
            "training_log": os.path.basename(training_log_path),
            "training_losses": os.path.basename(training_losses_path),
            "detailed_outputs": "detailed_outputs" if save_detailed_outputs else None,
        },
    }
    with open(os.path.join(out_dir, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Total elapsed time: {elapsed_seconds:.2f}s")

    if failed:
        logger.info("Experiment failed after partial completion")
        return 1
    logger.info("Experiment completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
