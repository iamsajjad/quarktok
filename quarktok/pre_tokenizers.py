"""
QuarkTok Pre-tokenizers Module

Configures various pre-tokenization strategies for the tokenizer.
"""

import logging
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import (
    Whitespace,
    ByteLevel,
    Digits,
    Punctuation,
    Sequence as PreTokenizerSequence
)

logger = logging.getLogger(__name__)


def setup_pretokenizer(tokenizer: Tokenizer, pretokenizer_type: str) -> None:
    """
    Configure the pre-tokenization strategy for a tokenizer.

    Args:
        tokenizer: The tokenizer instance to configure
        pretokenizer_type: Type of pre-tokenization to use
            - "whitespace": Split on whitespace only
            - "byte_level": Byte-level BPE (like GPT-2)
            - "advanced": Digits + Punctuation + Whitespace
            - "custom": Allows custom pre-tokenization (must be set manually)

    Raises:
        ValueError: If invalid pretokenizer_type is provided

    Example:
        >>> from tokenizers import Tokenizer
        >>> from tokenizers.models import BPE
        >>> tokenizer = Tokenizer(BPE())
        >>> setup_pretokenizer(tokenizer, "advanced")
    """
    if pretokenizer_type == "whitespace":
        tokenizer.pre_tokenizer = Whitespace()
        logger.debug("Pre-tokenizer: Whitespace")

    elif pretokenizer_type == "byte_level":
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        logger.debug("Pre-tokenizer: ByteLevel")

    elif pretokenizer_type == "advanced":
        tokenizer.pre_tokenizer = PreTokenizerSequence([
            Digits(individual_digits=True),
            Punctuation(),
            Whitespace()
        ])
        logger.debug("Pre-tokenizer: Advanced (Digits + Punctuation + Whitespace)")

    elif pretokenizer_type == "custom":
        # Allow user to set custom pre-tokenizer later
        logger.debug("Pre-tokenizer: Custom (must be set manually)")

    else:
        raise ValueError(
            f"Invalid pretokenizer_type '{pretokenizer_type}'. "
            f"Options: 'whitespace', 'byte_level', 'advanced', 'custom'"
        )


def get_available_pretokenizers() -> list:
    """
    Get list of available pre-tokenizer types.

    Returns:
        List of available pre-tokenizer type names

    Example:
        >>> pretokenizers = get_available_pretokenizers()
        >>> "advanced" in pretokenizers
        True
    """
    return ["whitespace", "byte_level", "advanced", "custom"]
