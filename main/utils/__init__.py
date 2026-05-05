"""
Utils package for LeRe Framework

Contains:
- Extractors (generator, reflector, curator)
- Memory formatting and management
- Learning to Remember (LeRe) pipeline
- Code execution utilities
"""

# Extractors
from .lere_extractor import (
    extract_trajectory,
    extract_memory_audit,
    extract_answer,
    extract_all_from_generator,
    extract_reflection,
    extract_curation,
    parse_trajectory_steps,
)

# Memory formatting utilities
from .memory_formatter import (
    calculate_reliability,
    format_memory_item_for_generator,
    format_memory_bank_for_generator,
    format_memory_bank_for_reflector,
    format_memory_bank_for_curator,
    get_memory_items_by_ids,
    apply_reliability_updates,
    add_new_items_to_memory,
    apply_item_updates,
    prune_items,
    get_memory_health_stats,
)

# Learning to Remember (LeRe) pipeline
from .lere_pipeline import (
    LeRePipeline,
    load_prompts_from_directory,
    create_pipeline_from_directory
)

# Code execution
from .execute_code import extract_and_run_python_code, execute_code_with_timeout

__all__ = [
    # Extractors
    'extract_trajectory',
    'extract_memory_audit',
    'extract_answer',
    'extract_all_from_generator',
    'extract_reflection',
    'extract_curation',
    'parse_trajectory_steps',

    # Memory formatters
    'calculate_reliability',
    'format_memory_item_for_generator',
    'format_memory_bank_for_generator',
    'format_memory_bank_for_reflector',
    'format_memory_bank_for_curator',
    'get_memory_items_by_ids',
    'apply_reliability_updates',
    'add_new_items_to_memory',
    'apply_item_updates',
    'prune_items',
    'get_memory_health_stats',

    # Pipeline
    'LeRePipeline',
    'load_prompts_from_directory',
    'create_pipeline_from_directory',

    # Code execution
    'extract_and_run_python_code',
    'execute_code_with_timeout',
]
