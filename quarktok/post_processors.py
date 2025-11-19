"""
QuarkTok Post-processors Module

Configures post-processing templates for adding special tokens.
"""

import logging
from typing import List
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

logger = logging.getLogger(__name__)


def get_token_id(token: str, special_tokens: List[str]) -> int:
    """
    Get the ID of a special token based on its position in the list.

    Args:
        token: Special token string
        special_tokens: List of special tokens

    Returns:
        Token ID (position in special_tokens list)

    Example:
        >>> tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        >>> get_token_id("[CLS]", tokens)
        2
    """
    try:
        return special_tokens.index(token)
    except ValueError:
        return 1  # Return [UNK] token ID if not found


def setup_post_processor(
    tokenizer: Tokenizer,
    template_type: str,
    special_tokens: List[str]
) -> None:
    """
    Configure post-processing template for adding special tokens.

    Args:
        tokenizer: The tokenizer instance to configure
        template_type: Template style ('bert', 'gpt', 't5', 'none')
        special_tokens: List of special tokens

    Raises:
        ValueError: If invalid template_type is provided

    Example:
        >>> from tokenizers import Tokenizer
        >>> from tokenizers.models import BPE
        >>> tokenizer = Tokenizer(BPE())
        >>> tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[EOS]"]
        >>> setup_post_processor(tokenizer, "bert", tokens)
    """
    if template_type == "bert":
        # BERT-style: [CLS] text [SEP]
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B [SEP]",
            special_tokens=[
                ("[CLS]", get_token_id("[CLS]", special_tokens)),
                ("[SEP]", get_token_id("[SEP]", special_tokens)),
            ],
        )
        logger.debug("Post-processor: BERT style")

    elif template_type == "gpt":
        # GPT-style: text [EOS]
        tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            pair="$A [EOS] $B [EOS]",
            special_tokens=[
                ("[EOS]", get_token_id("[EOS]", special_tokens)),
            ],
        )
        logger.debug("Post-processor: GPT style")

    elif template_type == "t5":
        # T5-style: text [EOS]
        tokenizer.post_processor = TemplateProcessing(
            single="$A [EOS]",
            pair="$A [EOS] $B [EOS]",
            special_tokens=[
                ("[EOS]", get_token_id("[EOS]", special_tokens)),
            ],
        )
        logger.debug("Post-processor: T5 style")

    elif template_type == "none":
        # No post-processing
        tokenizer.post_processor = None
        logger.debug("Post-processor: None")

    else:
        raise ValueError(
            f"Invalid template_type '{template_type}'. "
            f"Options: 'bert', 'gpt', 't5', 'none'"
        )


def get_available_templates() -> list:
    """
    Get list of available post-processing templates.

    Returns:
        List of available template type names

    Example:
        >>> templates = get_available_templates()
        >>> "bert" in templates
        True
    """
    return ["bert", "gpt", "t5", "none"]
