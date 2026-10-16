"""Controlled 2D scenes; these are a synthetic distribution, not CLEVR."""

from .scenes import (
    DEFAULT_SPLIT_COUNTS,
    DESCRIPTOR_WHITELIST,
    evaluate_program,
    generate_dataset,
    independent_answer,
    validate_dataset,
)

__all__ = [
    "DEFAULT_SPLIT_COUNTS", "DESCRIPTOR_WHITELIST", "evaluate_program",
    "generate_dataset", "independent_answer", "validate_dataset",
]
