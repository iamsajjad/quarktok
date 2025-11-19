"""
QuarkTok Normalizers Module

Configures text normalization pipelines for the tokenizer.
"""

import logging
from tokenizers import Tokenizer
from tokenizers.normalizers import (
    Lowercase,
    NFD,
    NFKC,
    StripAccents,
    Replace,
    Sequence as NormalizerSequence
)

logger = logging.getLogger(__name__)


def setup_normalizer(
    tokenizer: Tokenizer,
    normalize_unicode: bool = False,
    normalize_lowercase: bool = False,
    normalize_urls: bool = False,
    normalize_numbers: bool = False
) -> None:
    """
    Configure text normalization pipeline for a tokenizer.

    Args:
        tokenizer: The tokenizer instance to configure
        normalize_unicode: Apply NFKC Unicode normalization
        normalize_lowercase: Convert to lowercase
        normalize_urls: Replace URLs with [URL] token
        normalize_numbers: Replace numbers with [NUM] token

    Example:
        >>> from tokenizers import Tokenizer
        >>> from tokenizers.models import BPE
        >>> tokenizer = Tokenizer(BPE())
        >>> setup_normalizer(
        ...     tokenizer,
        ...     normalize_unicode=True,
        ...     normalize_urls=True
        ... )
    """
    normalizers = []

    # Unicode normalization
    if normalize_unicode:
        normalizers.append(NFKC())
        normalizers.append(NFD())
        normalizers.append(StripAccents())

    # Whitespace normalization (always applied)
    normalizers.append(Replace(r"\s+", " "))

    # URL normalization
    if normalize_urls:
        normalizers.append(Replace(r"https?://\S+", "[URL]"))
        normalizers.append(Replace(r"www\.\S+", "[URL]"))

    # Number normalization
    if normalize_numbers:
        normalizers.append(Replace(r"\b\d+\b", "[NUM]"))

    # Lowercase
    if normalize_lowercase:
        normalizers.append(Lowercase())

    if normalizers:
        tokenizer.normalizer = NormalizerSequence(normalizers)
        logger.debug(f"Normalizers configured: {len(normalizers)} normalizers active")


def get_normalization_config(
    normalize_unicode: bool = False,
    normalize_lowercase: bool = False,
    normalize_urls: bool = False,
    normalize_numbers: bool = False
) -> dict:
    """
    Get a dictionary describing the normalization configuration.

    Args:
        normalize_unicode: Unicode normalization enabled
        normalize_lowercase: Lowercase normalization enabled
        normalize_urls: URL normalization enabled
        normalize_numbers: Number normalization enabled

    Returns:
        Dictionary with normalization settings

    Example:
        >>> config = get_normalization_config(normalize_urls=True)
        >>> config['normalize_urls']
        True
    """
    return {
        "normalize_unicode": normalize_unicode,
        "normalize_lowercase": normalize_lowercase,
        "normalize_urls": normalize_urls,
        "normalize_numbers": normalize_numbers,
    }
