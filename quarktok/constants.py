"""
QuarkTok Constants Module

Defines special tokens and constants used throughout the tokenizer.
"""

from typing import List

# Predefined special token sets
STANDARD_SPECIAL_TOKENS: List[str] = [
    "[PAD]",    # Padding token
    "[UNK]",    # Unknown token
    "[CLS]",    # Classification/start token
    "[SEP]",    # Separator token
    "[MASK]",   # Mask token for masked language modeling
    "[BOS]",    # Beginning of sequence
    "[EOS]",    # End of sequence
]

CODE_SPECIAL_TOKENS: List[str] = [
    "[CODE]",     # Start of code block
    "[/CODE]",    # End of code block
    "[INDENT]",   # Indentation
    "[DEDENT]",   # De-indentation
    "[COMMENT]",  # Code comment
]

MATH_SPECIAL_TOKENS: List[str] = [
    "[MATH]",      # Start of math expression
    "[/MATH]",     # End of math expression
    "[EQUATION]",  # Start of equation
    "[/EQUATION]", # End of equation
]

MEDIA_SPECIAL_TOKENS: List[str] = [
    "[IMG]",   # Start of image
    "[/IMG]",  # End of image
    "[URL]",   # URL placeholder
    "[/URL]",  # End of URL
    "[NUM]",   # Number placeholder
]

# Default configuration values
DEFAULT_VOCAB_SIZE: int = 30000
DEFAULT_MIN_FREQUENCY: int = 2
DEFAULT_CACHE_SIZE: int = 10000
DEFAULT_BATCH_SIZE: int = 10000


def build_special_tokens(
    custom_tokens: List[str] = None,
    languages: List[str] = None,
    include_code: bool = True,
    include_math: bool = True,
    include_media: bool = True
) -> List[str]:
    """
    Build a complete list of special tokens.

    Args:
        custom_tokens: Custom special tokens provided by user
        languages: Language codes for multilingual support
        include_code: Include code-specific tokens
        include_math: Include math-specific tokens
        include_media: Include media-specific tokens

    Returns:
        Complete list of special tokens

    Example:
        >>> tokens = build_special_tokens(languages=["en", "es"])
        >>> "[EN]" in tokens
        True
    """
    if custom_tokens is not None:
        tokens = custom_tokens.copy()
    else:
        # Start with standard tokens
        tokens = STANDARD_SPECIAL_TOKENS.copy()

        # Add optional token sets
        if include_code:
            tokens.extend(CODE_SPECIAL_TOKENS)
        if include_math:
            tokens.extend(MATH_SPECIAL_TOKENS)
        if include_media:
            tokens.extend(MEDIA_SPECIAL_TOKENS)

    # Add language tokens for multilingual support
    if languages:
        for lang in languages:
            lang_token = f"[{lang.upper()}]"
            if lang_token not in tokens:
                tokens.append(lang_token)

    return tokens
