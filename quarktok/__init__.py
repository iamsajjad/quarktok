"""
QuarkTok - Advanced BPE Tokenizer for LLM Training

A comprehensive, modular tokenization toolkit built on HuggingFace tokenizers.

Features:
    - Multiple pre-tokenization strategies
    - Advanced text normalization
    - Configurable BPE models with dropout
    - Task-specific special tokens
    - Vocabulary optimization tools
    - Multiple post-processing templates
    - Advanced decoding with metadata
    - Evaluation metrics
    - Streaming/incremental training
    - Multilingual support
    - Performance caching

Example:
    >>> from quarktok_lib import QuarkTok
    >>> tokenizer = QuarkTok(vocab_size=30000)
    >>> tokenizer.train("data.txt")
    >>> ids = tokenizer.encode("Hello, world!")
    >>> text = tokenizer.decode(ids)
    >>> tokenizer.save("tokenizer.json")

Author: QuarkTok Team
License: Apache 2.0
"""

__version__ = "2.0.0"
__author__ = "QuarkTok Team"
__license__ = "Apache 2.0"

# Core class
from .core import QuarkTok

# Constants
from .constants import (
    STANDARD_SPECIAL_TOKENS,
    CODE_SPECIAL_TOKENS,
    MATH_SPECIAL_TOKENS,
    MEDIA_SPECIAL_TOKENS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_MIN_FREQUENCY,
    DEFAULT_CACHE_SIZE,
    DEFAULT_BATCH_SIZE,
    build_special_tokens,
)

# Public API
__all__ = [
    # Main class
    "QuarkTok",
    # Constants
    "STANDARD_SPECIAL_TOKENS",
    "CODE_SPECIAL_TOKENS",
    "MATH_SPECIAL_TOKENS",
    "MEDIA_SPECIAL_TOKENS",
    "DEFAULT_VOCAB_SIZE",
    "DEFAULT_MIN_FREQUENCY",
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_BATCH_SIZE",
    "build_special_tokens",
]


def get_version() -> str:
    """
    Get the version of QuarkTok.

    Returns:
        Version string

    Example:
        >>> from quarktok_lib import get_version
        >>> print(get_version())
        '2.0.0'
    """
    return __version__
