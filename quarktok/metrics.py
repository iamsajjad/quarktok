"""
QuarkTok Metrics Module

Provides evaluation metrics for tokenizer analysis and optimization.
"""

from typing import List, Dict, Any, Callable, Optional
from collections import Counter


def calculate_compression_ratio(
    text: str,
    encode_fn: Callable[[str, bool], List[int]]
) -> float:
    """
    Calculate the compression ratio of the tokenization.

    The compression ratio is the ratio of characters to tokens.
    Higher values indicate better compression (fewer tokens per character).

    Args:
        text: Input text to analyze
        encode_fn: Function that encodes text to token IDs

    Returns:
        Compression ratio (characters per token)

    Example:
        >>> def encode(text, add_special): return [1, 2, 3, 4]
        >>> ratio = calculate_compression_ratio("Hello world!", encode)
        >>> ratio
        3.0
    """
    char_count = len(text)
    token_count = len(encode_fn(text, False))

    if token_count == 0:
        return 0.0

    return char_count / token_count


def fertility_score(
    texts: List[str],
    encode_fn: Callable[[str, bool], List[int]]
) -> float:
    """
    Calculate the average fertility score (tokens per word).

    Lower fertility scores indicate more efficient tokenization.
    A score close to 1.0 means each word is typically one token.

    Args:
        texts: List of texts to analyze
        encode_fn: Function that encodes text to token IDs

    Returns:
        Average fertility score

    Example:
        >>> def encode(text, add_special): return [1, 2, 3]
        >>> texts = ["Hello world", "Machine learning"]
        >>> score = fertility_score(texts, encode)
        >>> score
        1.5
    """
    total_tokens = 0
    total_words = 0

    for text in texts:
        # Count words (simple whitespace split)
        words = text.split()
        total_words += len(words)

        # Count tokens (excluding special tokens)
        tokens = encode_fn(text, False)
        total_tokens += len(tokens)

    if total_words == 0:
        return 0.0

    return total_tokens / total_words


def measure_unk_rate(
    texts: List[str],
    encode_fn: Callable[[str, bool], List[int]],
    unk_id: int
) -> float:
    """
    Measure the rate of unknown tokens in the given texts.

    Args:
        texts: List of texts to analyze
        encode_fn: Function that encodes text to token IDs
        unk_id: ID of the unknown token

    Returns:
        Percentage of unknown tokens (0.0 to 100.0)

    Example:
        >>> def encode(text, add_special): return [1, 2, 1, 3]  # 1 is UNK
        >>> texts = ["Sample text"]
        >>> rate = measure_unk_rate(texts, encode, unk_id=1)
        >>> rate
        50.0
    """
    total_tokens = 0
    unk_tokens = 0

    for text in texts:
        ids = encode_fn(text, False)
        total_tokens += len(ids)
        unk_tokens += sum(1 for id in ids if id == unk_id)

    if total_tokens == 0:
        return 0.0

    return (unk_tokens / total_tokens) * 100.0


def analyze_vocab_coverage(
    texts: List[str],
    encode_fn: Callable[[str, bool], List[int]],
    vocab_size: int,
    unk_id: int
) -> Dict[str, Any]:
    """
    Analyze how well the vocabulary covers the given texts.

    Args:
        texts: List of texts to analyze
        encode_fn: Function that encodes text to token IDs
        vocab_size: Total vocabulary size
        unk_id: ID of the unknown token

    Returns:
        Dictionary with coverage statistics:
            - compression_ratio: Average characters per token
            - fertility_score: Average tokens per word
            - unk_rate: Percentage of unknown tokens
            - total_tokens: Total number of tokens
            - total_chars: Total number of characters
            - unique_tokens: Number of unique tokens used
            - vocab_utilization: Percentage of vocabulary used

    Example:
        >>> def encode(text, add_special): return [1, 2, 3]
        >>> texts = ["Sample text"]
        >>> stats = analyze_vocab_coverage(texts, encode, 1000, unk_id=1)
        >>> 'compression_ratio' in stats
        True
    """
    # Calculate metrics
    compression = sum(
        calculate_compression_ratio(text, encode_fn)
        for text in texts
    ) / len(texts)

    fertility = fertility_score(texts, encode_fn)
    unk_rate = measure_unk_rate(texts, encode_fn, unk_id)

    # Count tokens and characters
    total_tokens = 0
    total_chars = 0
    all_token_ids = set()

    for text in texts:
        total_chars += len(text)
        ids = encode_fn(text, False)
        total_tokens += len(ids)
        all_token_ids.update(ids)

    return {
        "compression_ratio": round(compression, 2),
        "fertility_score": round(fertility, 2),
        "unk_rate": round(unk_rate, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "unique_tokens": len(all_token_ids),
        "vocab_utilization": round((len(all_token_ids) / vocab_size) * 100, 2)
    }


def prune_vocab_analysis(
    texts: List[str],
    encode_fn: Callable[[str, bool], List[int]],
    vocab: Dict[str, int],
    special_tokens: List[str],
    min_usage_count: int = 1
) -> Dict[str, Any]:
    """
    Analyze vocabulary usage to identify tokens for pruning.

    Args:
        texts: Sample texts to analyze for token usage
        encode_fn: Function that encodes text to token IDs
        vocab: Vocabulary dictionary (token -> ID)
        special_tokens: List of special tokens (won't be pruned)
        min_usage_count: Minimum times a token must appear to be kept

    Returns:
        Dictionary with pruning analysis:
            - tokens_to_prune: Number of tokens that would be pruned
            - total_vocab_size: Total vocabulary size
            - prune_percentage: Percentage of tokens that would be pruned
            - usage_counts: Counter of token usage

    Example:
        >>> def encode(text, add_special): return [1, 2, 3]
        >>> vocab = {"hello": 1, "world": 2, "rare": 3}
        >>> special_tokens = ["[PAD]", "[UNK]"]
        >>> analysis = prune_vocab_analysis(["hello world"], encode, vocab, special_tokens)
        >>> 'tokens_to_prune' in analysis
        True
    """
    # Count token usage
    token_counts = Counter()
    for text in texts:
        ids = encode_fn(text, False)
        token_counts.update(ids)

    # Identify tokens to remove
    tokens_to_remove = []

    for token, token_id in vocab.items():
        # Don't remove special tokens
        if token in special_tokens:
            continue

        # Check usage
        if token_counts.get(token_id, 0) < min_usage_count:
            tokens_to_remove.append(token)

    return {
        "tokens_to_prune": len(tokens_to_remove),
        "total_vocab_size": len(vocab),
        "prune_percentage": round((len(tokens_to_remove) / len(vocab)) * 100, 2),
        "usage_counts": dict(token_counts.most_common(100))
    }
