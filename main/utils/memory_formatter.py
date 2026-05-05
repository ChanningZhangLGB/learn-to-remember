"""
Memory Formatting Utilities

This module provides formatting functions to convert memory items between:
- JSON format (used internally by Curator)
- Human-readable text format (used by Generator and Reflector)
- Calculation of reliability scores
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def format_memory_item_for_embedding(item: Dict) -> str:
    """
    Produce the full-entry text used for embedding a memory item (Em).
    Encodes all semantic fields: title, bullets, example, tags, scope.
    This must match MemoryEncoder.format_memory_item() in ccme_encoder.py.
    """
    parts = []
    if item.get("title"):
        parts.append(f"Title: {item['title']}")
    bullets = item.get("bullets", [])
    if bullets:
        parts.append("Insights:\n" + "\n".join(f"- {b}" for b in bullets))
    if item.get("example"):
        parts.append(f"Example: {item['example']}")
    tags = item.get("tags", [])
    if tags:
        parts.append(f"Tags: {', '.join(tags)}")
    if item.get("scope"):
        parts.append(f"Scope: {item['scope']}")
    return "\n".join(parts)


def calculate_reliability(helpful: int, harmful: int) -> float:
    """
    Calculate reliability score using Bayesian estimate.

    Formula: (helpful + 1) / (helpful + harmful + 2)

    This starts at 0.5 for new items (uniform prior) and adjusts based on outcomes.

    Args:
        helpful: Number of times marked helpful
        harmful: Number of times marked harmful

    Returns:
        float: Reliability score between 0 and 1
    """
    return (helpful + 1) / (helpful + harmful + 2)


def format_memory_item_for_generator(item: Dict) -> str:
    """
    Format a single memory item for display to the Generator.

    Args:
        item: Memory item dictionary with id, title, bullets, tags, meta

    Returns:
        str: Human-readable formatted memory item
    """
    item_id = item.get("id", "unknown")
    title = item.get("title", "Untitled")
    bullets = item.get("bullets", [])
    tags = item.get("tags", [])
    if isinstance(tags, list):
        flat_tags = []
        for tag in tags:
            if isinstance(tag, list):
                flat_tags.extend(str(t) for t in tag)
            else:
                flat_tags.append(str(tag))
        tags = flat_tags
    else:
        tags = [str(tags)]
    meta = item.get("meta", {})

    # Calculate reliability
    helpful = meta.get("helpful", 0)
    harmful = meta.get("harmful", 0)
    reliability = calculate_reliability(helpful, harmful)

    example = item.get("example", "")
    scope = item.get("scope", "")

    # Format output
    output = []
    output.append(f"ID: {item_id}")
    output.append(f"Title: {title}")
    output.append(f"Bullets:")
    for bullet in bullets:
        output.append(f"  - {bullet}")
    if example:
        output.append(f"Example: {example}")
    if scope:
        output.append(f"Scope: {scope}")
    output.append(f"Tags: {', '.join(tags)}")
    output.append(f"Meta:")
    output.append(f"  - helpful: {meta.get('helpful', 0)}, harmful: {meta.get('harmful', 0)}, used: {meta.get('used', 0)}")
    output.append(f"  - last_used_query: {meta.get('last_used_query', 'never')}")
    output.append(f"  - reliability: {meta.get('reliability', reliability):.2f}")
    output.append("")  # Blank line

    return "\n".join(output)


def format_memory_bank_for_generator(memory_items: List[Dict]) -> str:
    """
    Format entire memory bank for display to the Generator.

    Args:
        memory_items: List of memory item dictionaries

    Returns:
        str: Human-readable formatted memory bank
    """
    if not memory_items:
        return "(No memory items available)"

    output = []
    for item in memory_items:
        output.append(format_memory_item_for_generator(item))

    return "\n".join(output)


def format_memory_item_for_reflector(item: Dict) -> str:
    """
    Format a single memory item for display to the Reflector.

    Similar to Generator format but includes more context for evaluation.

    Args:
        item: Memory item dictionary

    Returns:
        str: Human-readable formatted memory item
    """
    # Reflector needs the same format as Generator
    return format_memory_item_for_generator(item)


def format_memory_bank_for_reflector(memory_items: List[Dict], impact_map: Dict = None) -> str:
    """
    Format memory bank for Reflector input.

    Args:
        memory_items: List of memory item dictionaries
        impact_map: Optional {id: impact_type} from generator's memory audit

    Returns:
        str: Human-readable formatted memory bank
    """
    if not memory_items:
        return "(No consulted memory items)"

    output = []
    for item in memory_items:
        item_id = item.get("id", "unknown")
        impact = (impact_map or {}).get(item_id, "")
        formatted = format_memory_item_for_generator(item)
        if impact:
            formatted = f"[Generator impact: {impact}]\n" + formatted
        output.append(formatted)
    return "\n".join(output)


def format_memory_bank_for_curator(memory_items: List[Dict]) -> str:
    """
    Format memory bank for Curator input (JSON format).

    Args:
        memory_items: List of memory item dictionaries

    Returns:
        str: JSON string of memory items
    """
    return json.dumps(memory_items, indent=2)


def format_memory_bank_for_synthesizer(retrieved_items: List[Dict]) -> str:
    """
    Format CCME-retrieved items for Synthesizer input.

    Uses a compact format (same as generator format) with reliability scores,
    since the synthesizer now receives only the top-k retrieved items,
    not the full memory bank.

    Args:
        retrieved_items: List of CCME-retrieved memory item dictionaries

    Returns:
        str: Human-readable formatted items with reliability scores
    """
    if not retrieved_items:
        return "(No retrieved items)"

    output = []
    for item in retrieved_items:
        output.append(format_memory_item_for_generator(item))
    return "\n".join(output)


def get_memory_items_by_ids(memory_bank: List[Dict], item_ids: List[str]) -> List[Dict]:
    """
    Retrieve specific memory items by their IDs.

    Args:
        memory_bank: Full list of memory items
        item_ids: List of item IDs to retrieve

    Returns:
        List[Dict]: List of matching memory items
    """
    id_set = set(item_ids)
    return [item for item in memory_bank if item.get("id") in id_set]


def apply_reliability_updates(memory_bank: List[Dict], updates: List[Dict]) -> List[Dict]:
    """
    Apply retrieval counter updates to memory bank.

    Args:
        memory_bank: Current memory bank
        updates: List of update dictionaries with:
                 - item_id: str
                 - action: "INCREMENT_USED"

    Returns:
        List[Dict]: Updated memory bank

    Note: helpful/harmful/last_used_query are now updated directly in ace_CL_pipeline.py
    from reflector verdicts and memory audit (not via this function).
    """
    # Create a copy to avoid mutation
    updated_bank = [item.copy() for item in memory_bank]

    # Index by ID for fast lookup
    id_to_item = {item["id"]: item for item in updated_bank}

    for update in updates:
        item_id = update.get("item_id")
        action = update.get("action")

        if item_id not in id_to_item:
            print(f"Warning: Item {item_id} not found in memory bank")
            continue

        item = id_to_item[item_id]
        meta = item.setdefault("meta", {})

        if action == "INCREMENT_USED":
            meta["used"] = meta.get("used", 0) + 1

    return updated_bank


def add_new_items_to_memory(memory_bank: List[Dict], new_items: List[Dict], next_id: Optional[int] = None) -> List[Dict]:
    """
    Add new memory items to the memory bank with auto-generated IDs.

    Args:
        memory_bank: Current memory bank
        new_items: List of new item dictionaries (without IDs)
        next_id: Optional starting ID number (auto-detected if None)

    Returns:
        List[Dict]: Updated memory bank with new items
    """
    # Find the highest existing ID
    if next_id is None:
        max_id = 0
        for item in memory_bank:
            item_id = item.get("id", "")
            if item_id.startswith("m_"):
                try:
                    num = int(item_id.split("_")[1])
                    max_id = max(max_id, num)
                except (IndexError, ValueError):
                    pass
        next_id = max_id + 1

    # Create a copy
    updated_bank = memory_bank.copy()

    # Add new items with auto-generated IDs
    for item in new_items:
        new_item = item.copy()
        new_item["id"] = f"m_{next_id:03d}"
        next_id += 1
        updated_bank.append(new_item)

    return updated_bank


def apply_item_updates(memory_bank: List[Dict], updates: List[Dict]) -> List[Dict]:
    """
    Apply updates to existing memory items (e.g., append bullets, refine, add tags).

    Args:
        memory_bank: Current memory bank
        updates: List of update dictionaries with:
                 - item_id: str
                 - update_type: "APPEND_BULLET" | "REFINE_BULLET" | "ADD_TAG"
                 - changes: dict with field, old_value, new_value

    Returns:
        List[Dict]: Updated memory bank
    """
    # Create a copy
    updated_bank = [item.copy() for item in memory_bank]

    # Index by ID
    id_to_item = {item["id"]: item for item in updated_bank}

    for update in updates:
        item_id = update.get("item_id")
        update_type = update.get("update_type")
        changes = update.get("changes", {})

        if item_id not in id_to_item:
            print(f"Warning: Item {item_id} not found for update")
            continue

        item = id_to_item[item_id]

        if update_type == "APPEND_BULLET":
            field = changes.get("field", "bullets")
            new_value = changes.get("new_value")
            if field == "bullets" and new_value:
                item.setdefault("bullets", []).append(new_value)

        elif update_type == "REFINE_BULLET":
            field = changes.get("field", "bullets")
            old_value = changes.get("old_value")
            new_value = changes.get("new_value")
            if field == "bullets" and old_value and new_value:
                bullets = item.get("bullets", [])
                try:
                    idx = bullets.index(old_value)
                    bullets[idx] = new_value
                except ValueError:
                    print(f"Warning: Old bullet value not found for replacement")

        elif update_type == "ADD_TAG":
            field = changes.get("field", "tags")
            new_value = changes.get("new_value")
            if field == "tags" and new_value:
                tags = item.setdefault("tags", [])
                if new_value not in tags:
                    tags.append(new_value)

    return updated_bank


def prune_items(memory_bank: List[Dict], prune_candidates: List[Dict]) -> List[Dict]:
    """
    Remove items from memory bank based on prune decisions.

    Args:
        memory_bank: Current memory bank
        prune_candidates: List of prune dictionaries with:
                         - item_id: str
                         - immediate_prune: bool

    Returns:
        List[Dict]: Memory bank with pruned items removed
    """
    # Get IDs to prune
    ids_to_prune = {
        candidate["item_id"]
        for candidate in prune_candidates
        if candidate.get("immediate_prune", False)
    }

    # Filter out pruned items
    return [item for item in memory_bank if item.get("id") not in ids_to_prune]


def apply_curation_updates(memory_bank: List[Dict], curation: Dict, language_model=None, embedding_index: Optional[Dict] = None, embed_fn=None) -> tuple:
    """
    Apply all curation updates to the memory bank and update embedding index.

    This is the main function that orchestrates all update types.

    Args:
        memory_bank: Current memory bank
        curation: Curation dictionary from Curator
        language_model: Optional LLM instance for generating bullet embeddings
        embedding_index: Optional dict mapping item_id -> embedding vector

    Returns:
        tuple: (updated_memory_bank, updated_embedding_index)
            - updated_memory_bank: List[Dict] - memory items WITHOUT embeddings
            - updated_embedding_index: Dict[str, List[float]] - separate ID->embedding mapping
    """
    updated_bank = memory_bank
    updated_index = embedding_index if embedding_index is not None else {}

    # 1. Apply reliability updates
    reliability_updates = curation.get("reliability_updates", [])
    updated_bank = apply_reliability_updates(updated_bank, reliability_updates)

    # 2. Apply updates to existing items
    existing_updates = curation.get("updates_to_existing", [])
    updated_bank = apply_item_updates(updated_bank, existing_updates)

    # 3. Add new items (assign IDs first, then generate embeddings)
    new_items = curation.get("new_items", [])
    # First add items to get auto-generated IDs
    updated_bank = add_new_items_to_memory(updated_bank, new_items)

    # Then generate embeddings for the newly added items (if language_model provided)
    # embed_fn takes priority over language_model.get_embedding so the embedding
    # space matches the configured encoder (e.g. CLIP 768-dim vs text-embedding-3-small 1536-dim)
    _embed = embed_fn if embed_fn is not None else (language_model.get_embedding if language_model else None)
    if _embed and len(new_items) > 0:
        # Find the newly added items (they're the last N items in the bank)
        newly_added_items = updated_bank[-len(new_items):]
        for item in newly_added_items:
            item_text = format_memory_item_for_embedding(item)
            if item_text:
                item_id = item.get("id")
                # Store embedding in separate index, NOT in the item itself
                updated_index[item_id] = _embed(item_text)
                # Remove bullet_embedding from meta if it exists (cleanup)
                if "meta" in item and "bullet_embedding" in item.get("meta", {}):
                    del item["meta"]["bullet_embedding"]

    # 4. Prune items (also remove from embedding index)
    prune_candidates = curation.get("prune_candidates", [])
    if prune_candidates:
        ids_to_prune = {
            candidate["item_id"]
            for candidate in prune_candidates
            if candidate.get("immediate_prune", False)
        }
        # Remove from embedding index
        for item_id in ids_to_prune:
            if item_id in updated_index:
                del updated_index[item_id]
    updated_bank = prune_items(updated_bank, prune_candidates)

    # 5. Handle deduplication (remove duplicate items and their embeddings)
    dedup_actions = curation.get("deduplication_actions", [])
    for action in dedup_actions:
        remove_ids = set(action.get("remove_items", []))
        # Remove from memory bank
        updated_bank = [item for item in updated_bank if item.get("id") not in remove_ids]
        # Remove from embedding index
        for item_id in remove_ids:
            if item_id in updated_index:
                del updated_index[item_id]

    return updated_bank, updated_index


def get_memory_health_stats(memory_bank: List[Dict]) -> Dict:
    """
    Calculate memory health statistics.

    Args:
        memory_bank: Current memory bank

    Returns:
        Dict: Health statistics including total items, avg reliability, domain distribution
    """
    if not memory_bank:
        return {
            "total_items": 0,
            "average_reliability": 0.0,
            "domain_distribution": {}
        }

    # Calculate average reliability
    reliabilities = []
    for item in memory_bank:
        meta = item.get("meta", {})
        helpful = meta.get("helpful", 0)
        harmful = meta.get("harmful", 0)
        reliabilities.append(calculate_reliability(helpful, harmful))

    avg_reliability = sum(reliabilities) / len(reliabilities) if reliabilities else 0.0

    # Domain distribution
    domain_counts = {}
    for item in memory_bank:
        tags = item.get("tags", [])
        for tag in tags:
            # Skip if tag is not a string (handle malformed data)
            if not isinstance(tag, str):
                continue
            # Get domain (first part before dot)
            domain = tag.split(".")[0] if "." in tag else tag
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    return {
        "total_items": len(memory_bank),
        "average_reliability": avg_reliability,
        "domain_distribution": domain_counts
    }


def apply_merge_operator(
    memory_bank: List[Dict],
    embedding_index: Dict,
    language_model=None,
    dedup_threshold: float = 0.85,
    prune_reliability_threshold: float = 0.2,
    current_query: Optional[int] = None,
    temporal_decay_lambda_query: float = 0.02,
    pruning_mode: str = "AND",
    min_queries_before_operations: int = 0,
    embed_fn=None,
) -> Tuple[List[Dict], Dict, Dict]:
    """
    Apply the ⊕ merge operator from paper (Lines 233-236).

    Performs:
    1. Replace LLM 'confidence' with computed reliability (always runs)
    2. Embedding-based deduplication (cosine similarity > threshold)
    3. Prune score pruning with two modes (controlled by pruning_mode)

    Steps 2 and 3 are skipped if current_query < min_queries_before_operations.
    This warm-up guard preserves early memory diversity before sufficient signal
    has accumulated to make deduplication and pruning statistically meaningful.

    Pruning modes:
    - "AND": prune only if BOTH prune_score < threshold AND reliability < threshold
             Conservative: decay rarely triggers; effectively reliability-gated.
             Recommended threshold: 0.2, lambda_q: 0.05
    - "OR":  prune if EITHER prune_score < threshold OR reliability < threshold
             Aggressive: stale neutral items are evicted independent of quality.
             Recommended threshold: 0.15, lambda_q: 0.02

    Args:
        memory_bank: Current memory bank
        embedding_index: Dict mapping item_id -> embedding vector
        language_model: LLM for computing embeddings if needed
        dedup_threshold: Cosine similarity threshold for deduplication (default 0.85)
        prune_reliability_threshold: Score threshold for pruning decisions (default 0.2)
        current_query: Current query index for staleness computation (None = no decay)
        temporal_decay_lambda_query: Decay rate per query gap
            "AND" default: 0.05 | "OR" default: 0.02
        pruning_mode: "AND" (conservative) or "OR" (aggressive)
        min_queries_before_operations: Minimum query index before dedup/pruning activate.
            Default 0 means operations run from the very first query.

    Returns:
        Tuple of (updated_memory_bank, updated_embedding_index, merge_stats)
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    merge_stats = {
        "items_before": len(memory_bank),
        "deduplicated": 0,
        "pruned_low_reliability": 0,
        "items_after": 0
    }

    if len(memory_bank) == 0:
        merge_stats["items_after"] = 0
        return memory_bank, embedding_index, merge_stats

    # Step 1: Replase LLM 'confidence' with computed reliability
    for item in memory_bank:
        meta = item.get("meta", {})
        helpful = meta.get("helpful", 0)
        harmful = meta.get("harmful", 0)
        # Compute Bayesian reliability: (helpful+1)/(helpful+harmful+2)
        computed_reliability = calculate_reliability(helpful, harmful)
        # Remove old 'confidence' field and add 'reliability'
        if "confidence" in meta:
            del meta["confidence"]
        meta["reliability"] = round(computed_reliability, 4)
        item["meta"] = meta

    # Warm-up guard: skip dedup and pruning until enough queries have been seen.
    # Reliability update (step 1) always runs — it is pure metadata maintenance.
    ops_allowed = (current_query is None) or (current_query >= min_queries_before_operations)
    if not ops_allowed:
        merge_stats["items_after"] = len(memory_bank)
        merge_stats["ops_skipped"] = f"warm-up (query {current_query} < min {min_queries_before_operations})"
        return memory_bank, embedding_index, merge_stats

    # Step 2: Embedding-based deduplication
    # Get all embeddings
    item_ids = [item.get("id") for item in memory_bank]
    embeddings = []
    items_with_embeddings = []

    for item in memory_bank:
        item_id = item.get("id")
        if item_id in embedding_index and embedding_index[item_id] is not None:
            embeddings.append(embedding_index[item_id])
            items_with_embeddings.append(item)
        else:
            # Compute embedding if missing — prefer embed_fn (encoder-aligned) over language_model
            _emb_fn = embed_fn if embed_fn is not None else (language_model.get_embedding if language_model else None)
            if _emb_fn:
                item_text = format_memory_item_for_embedding(item)
                if item_text:
                    emb = _emb_fn(item_text)
                    embedding_index[item_id] = emb
                    embeddings.append(emb)
                    items_with_embeddings.append(item)

    # Find duplicates using cosine similarity
    items_to_remove = set()
    if len(embeddings) > 1:
        embeddings_array = np.array(embeddings)
        sim_matrix = cosine_similarity(embeddings_array)

        # Find pairs with similarity above threshold
        for i in range(len(items_with_embeddings)):
            if items_with_embeddings[i].get("id") in items_to_remove:
                continue
            for j in range(i + 1, len(items_with_embeddings)):
                if items_with_embeddings[j].get("id") in items_to_remove:
                    continue
                if sim_matrix[i, j] > dedup_threshold:
                    # Keep the one with higher reliability
                    item_i = items_with_embeddings[i]
                    item_j = items_with_embeddings[j]
                    rel_i = item_i.get("meta", {}).get("reliability", 0.5)
                    rel_j = item_j.get("meta", {}).get("reliability", 0.5)

                    if rel_i >= rel_j:
                        items_to_remove.add(item_j.get("id"))
                    else:
                        items_to_remove.add(item_i.get("id"))

    merge_stats["deduplicated"] = len(items_to_remove)

    # Remove duplicates
    memory_bank = [item for item in memory_bank if item.get("id") not in items_to_remove]
    for item_id in items_to_remove:
        if item_id in embedding_index:
            del embedding_index[item_id]

    # Step 3: Prune score pruning: reliability × exp(-λ_q × query_gap)
    # Only prune items that have been retrieved at least once (so we have signal).
    # For items with no last_used_query, treat as recently active (no decay penalty).
    items_to_prune = set()
    for item in memory_bank:
        meta = item.get("meta", {})
        reliability = meta.get("reliability", 0.5)
        retrieved_count = meta.get("retrieved_count", 0)

        if retrieved_count == 0:
            continue  # No signal yet — never prune

        last_used_query = meta.get("last_used_query")
        if current_query is not None and last_used_query is not None:
            query_gap = max(0, current_query - last_used_query)
            decay = np.exp(-temporal_decay_lambda_query * query_gap)
        else:
            decay = 1.0  # No data yet — no penalty

        prune_score = reliability * decay

        if pruning_mode == "OR":
            # Aggressive: evict if stale OR low quality
            if prune_score < prune_reliability_threshold or reliability < prune_reliability_threshold:
                items_to_prune.add(item.get("id"))
        else:
            # AND (conservative): requires both poor prune_score AND low raw reliability
            if prune_score < prune_reliability_threshold and reliability < prune_reliability_threshold:
                items_to_prune.add(item.get("id"))

    merge_stats["pruned_low_reliability"] = len(items_to_prune)

    # Remove low-reliability items
    memory_bank = [item for item in memory_bank if item.get("id") not in items_to_prune]
    for item_id in items_to_prune:
        if item_id in embedding_index:
            del embedding_index[item_id]

    merge_stats["items_after"] = len(memory_bank)

    return memory_bank, embedding_index, merge_stats
