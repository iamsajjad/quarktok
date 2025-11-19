"""
QuarkTok Core Module

Main QuarkTok tokenizer class that integrates all components.
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Union, Optional, Dict, Iterator, Any

from tokenizers import Tokenizer
from tokenizers.models import BPE

from . import constants
from . import pre_tokenizers
from . import normalizers
from . import post_processors
from . import metrics
from . import trainers

logger = logging.getLogger(__name__)


class QuarkTok:
    """
    Advanced BPE Tokenizer for training on large text corpora.

    This class provides comprehensive tokenization capabilities including:
    - Flexible pre-tokenization strategies
    - Advanced text normalization
    - Customizable BPE training
    - Vocabulary optimization
    - Evaluation metrics
    - Caching for performance

    Example:
        >>> tokenizer = QuarkTok(
        ...     vocab_size=30000,
        ...     pretokenizer_type="byte_level",
        ...     normalize_unicode=True
        ... )
        >>> tokenizer.train("data.txt")
        >>> tokens = tokenizer.encode("Hello, world!")
        >>> print(tokenizer.decode(tokens))
        'Hello, world!'

    Attributes:
        vocab_size (int): Target vocabulary size
        min_frequency (int): Minimum token frequency threshold
        special_tokens (List[str]): List of special tokens
        is_trained (bool): Whether the tokenizer has been trained
        tokenizer (Tokenizer): Underlying HuggingFace tokenizer
    """

    # Expose constants as class attributes
    STANDARD_SPECIAL_TOKENS = constants.STANDARD_SPECIAL_TOKENS
    CODE_SPECIAL_TOKENS = constants.CODE_SPECIAL_TOKENS
    MATH_SPECIAL_TOKENS = constants.MATH_SPECIAL_TOKENS
    MEDIA_SPECIAL_TOKENS = constants.MEDIA_SPECIAL_TOKENS

    def __init__(
        self,
        vocab_size: int = constants.DEFAULT_VOCAB_SIZE,
        min_frequency: int = constants.DEFAULT_MIN_FREQUENCY,
        special_tokens: Optional[List[str]] = None,
        pretokenizer_type: str = "whitespace",
        normalize_unicode: bool = False,
        normalize_lowercase: bool = False,
        normalize_urls: bool = False,
        normalize_numbers: bool = False,
        use_byte_level: bool = False,
        dropout: Optional[float] = None,
        languages: Optional[List[str]] = None,
        enable_caching: bool = True,
        cache_size: int = constants.DEFAULT_CACHE_SIZE
    ):
        """
        Initialize the QuarkTok tokenizer with advanced configuration options.

        Args:
            vocab_size: Size of the vocabulary to build (default: 30000)
            min_frequency: Minimum frequency for a token to be included (default: 2)
            special_tokens: List of special tokens. If None, uses standard tokens
            pretokenizer_type: Pre-tokenization strategy ('whitespace', 'byte_level', 'advanced', 'custom')
            normalize_unicode: Apply NFKC Unicode normalization (default: False)
            normalize_lowercase: Convert text to lowercase (default: False)
            normalize_urls: Normalize URLs to [URL] token (default: False)
            normalize_numbers: Normalize numbers to [NUM] token (default: False)
            use_byte_level: Use byte-level BPE model (default: False)
            dropout: BPE dropout rate for regularization (default: None)
            languages: List of language codes for multilingual support
            enable_caching: Enable LRU caching for encoding (default: True)
            cache_size: Maximum cache size for encoding (default: 10000)

        Raises:
            ValueError: If invalid configuration parameters are provided
        """
        # Validate parameters
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if min_frequency < 0:
            raise ValueError(f"min_frequency must be non-negative, got {min_frequency}")
        if dropout is not None and (dropout < 0 or dropout > 1):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.pretokenizer_type = pretokenizer_type
        self.use_byte_level = use_byte_level
        self.dropout = dropout
        self.languages = languages or []
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        # Build special tokens list
        self.special_tokens = constants.build_special_tokens(special_tokens, languages)

        # Create tokenizer with BPE model
        logger.info(f"Initializing QuarkTok with vocab_size={vocab_size}")
        self.tokenizer = Tokenizer(BPE(
            unk_token="[UNK]",
            dropout=dropout,
            byte_fallback=use_byte_level
        ))

        # Configure pre-tokenization
        pre_tokenizers.setup_pretokenizer(self.tokenizer, pretokenizer_type)

        # Configure normalization
        normalizers.setup_normalizer(
            self.tokenizer,
            normalize_unicode=normalize_unicode,
            normalize_lowercase=normalize_lowercase,
            normalize_urls=normalize_urls,
            normalize_numbers=normalize_numbers
        )

        # Default post-processor (can be changed with set_post_processor)
        post_processors.setup_post_processor(self.tokenizer, "bert", self.special_tokens)

        self.is_trained = False

        # Cache for encoding if enabled
        if enable_caching:
            self._encode_cache = {}

        logger.info(f"QuarkTok initialized with {len(self.special_tokens)} special tokens")

    def set_post_processor(self, template_type: str = "bert") -> None:
        """
        Change the post-processing template after initialization.

        Args:
            template_type: Template style ('bert', 'gpt', 't5', 'none')

        Example:
            >>> tokenizer = QuarkTok()
            >>> tokenizer.set_post_processor("gpt")
        """
        post_processors.setup_post_processor(self.tokenizer, template_type, self.special_tokens)
        logger.info(f"Post-processor set to: {template_type}")

    def train(
        self,
        files: Union[str, List[str]],
        show_progress: bool = True,
        max_token_length: Optional[int] = None,
        limit_alphabet: Optional[int] = None
    ) -> None:
        """
        Train the BPE tokenizer on text files.

        Args:
            files: Path to a single file or list of file paths
            show_progress: Whether to show training progress (default: True)
            max_token_length: Maximum length for merged tokens (default: None)
            limit_alphabet: Limit initial alphabet size (default: None)

        Raises:
            FileNotFoundError: If training file(s) don't exist
            RuntimeError: If training fails

        Example:
            >>> tokenizer = QuarkTok()
            >>> tokenizer.train("data.txt", max_token_length=15)
        """
        trainers.train_tokenizer(
            self.tokenizer,
            files,
            self.vocab_size,
            self.min_frequency,
            self.special_tokens,
            show_progress,
            max_token_length,
            limit_alphabet
        )
        self.is_trained = True

    def train_incremental(
        self,
        data_iterator: Iterator[str],
        batch_size: int = constants.DEFAULT_BATCH_SIZE,
        show_progress: bool = True
    ) -> None:
        """
        Train the tokenizer incrementally on batches of text.

        This method is useful for very large datasets that don't fit in memory.

        Args:
            data_iterator: Iterator yielding text strings
            batch_size: Number of texts to process per batch (default: 10000)
            show_progress: Whether to show progress (default: True)

        Example:
            >>> def data_gen():
            ...     for line in open("huge_file.txt"):
            ...         yield line
            >>> tokenizer.train_incremental(data_gen(), batch_size=5000)
        """
        trainers.train_incremental(
            self.tokenizer,
            data_iterator,
            self.vocab_size,
            self.min_frequency,
            self.special_tokens,
            batch_size,
            show_progress
        )
        self.is_trained = True

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True
    ) -> Union[List[int], List[List[int]]]:
        """
        Encode text into token IDs.

        Args:
            text: Single text string or list of text strings
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token IDs for single text, or list of lists for batch

        Raises:
            RuntimeError: If tokenizer hasn't been trained yet

        Example:
            >>> ids = tokenizer.encode("Hello world")
            >>> print(ids)
            [101, 7592, 2088, 102]
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding. Call train() first.")

        if isinstance(text, str):
            # Check cache
            if self.enable_caching:
                cache_key = (text, add_special_tokens)
                if cache_key in self._encode_cache:
                    return self._encode_cache[cache_key]

                # Encode
                encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
                result = encoded.ids

                # Cache result (with size limit)
                if len(self._encode_cache) < self.cache_size:
                    self._encode_cache[cache_key] = result

                return result
            else:
                encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
                return encoded.ids
        else:
            # Batch encoding
            encodings = self.tokenizer.encode_batch(text, add_special_tokens=add_special_tokens)
            return [enc.ids for enc in encodings]

    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True
    ) -> Union[str, List[str]]:
        """
        Decode token IDs back into text.

        Args:
            ids: List of token IDs or list of lists for batch decoding
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string for single input, or list of strings for batch

        Raises:
            RuntimeError: If tokenizer hasn't been trained yet

        Example:
            >>> text = tokenizer.decode([101, 7592, 2088, 102])
            >>> print(text)
            'Hello world'
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding. Call train() first.")

        # Check if this is a batch (list of lists) or single sequence
        if ids and isinstance(ids[0], list):
            return self.tokenizer.decode_batch(ids, skip_special_tokens=skip_special_tokens)
        else:
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_with_metadata(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> Dict[str, Any]:
        """
        Decode token IDs with detailed metadata about the tokens.

        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in text output

        Returns:
            Dictionary with text, tokens, token_types, and special_tokens_mask

        Example:
            >>> result = tokenizer.decode_with_metadata([101, 7592, 2088, 102])
            >>> print(result['tokens'])
            ['[CLS]', 'Hello', 'world', '[SEP]']
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding. Call train() first.")

        tokens = [self.id_to_token(i) for i in ids]
        special_tokens_mask = [token in self.special_tokens for token in tokens]

        token_types = []
        for token in tokens:
            if token in self.special_tokens:
                token_types.append("special")
            elif token == "[UNK]":
                token_types.append("unknown")
            else:
                token_types.append("normal")

        return {
            "text": self.decode(ids, skip_special_tokens=skip_special_tokens),
            "tokens": tokens,
            "token_types": token_types,
            "special_tokens_mask": special_tokens_mask
        }

    def partial_decode(
        self,
        ids: List[int],
        start: int,
        end: int,
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode a slice of token IDs.

        Args:
            ids: List of token IDs
            start: Start index (inclusive)
            end: End index (exclusive)
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text for the token slice

        Example:
            >>> text = tokenizer.partial_decode([101, 7592, 2088, 102], 1, 3)
            >>> print(text)
            'Hello world'
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before decoding. Call train() first.")

        if start < 0 or end > len(ids) or start >= end:
            raise ValueError(f"Invalid slice [{start}:{end}] for ids of length {len(ids)}")

        return self.decode(ids[start:end], skip_special_tokens=skip_special_tokens)

    def get_tokens(
        self,
        text: str,
        add_special_tokens: bool = True
    ) -> List[str]:
        """
        Get the actual token strings for a given text.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add special tokens

        Returns:
            List of token strings

        Example:
            >>> tokens = tokenizer.get_tokens("Hello world")
            >>> print(tokens)
            ['[CLS]', 'Hello', 'world', '[SEP]']
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before tokenizing. Call train() first.")

        encoded = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoded.tokens

    # Evaluation metrics methods
    def calculate_compression_ratio(self, text: str) -> float:
        """Calculate the compression ratio of the tokenization."""
        return metrics.calculate_compression_ratio(text, self.encode)

    def fertility_score(self, texts: List[str]) -> float:
        """Calculate the average fertility score (tokens per word)."""
        return metrics.fertility_score(texts, self.encode)

    def measure_unk_rate(self, texts: List[str]) -> float:
        """Measure the rate of unknown tokens in the given texts."""
        unk_id = self.token_to_id("[UNK]")
        return metrics.measure_unk_rate(texts, self.encode, unk_id)

    def analyze_vocab_coverage(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze how well the vocabulary covers the given texts."""
        unk_id = self.token_to_id("[UNK]")
        return metrics.analyze_vocab_coverage(
            texts,
            self.encode,
            self.get_vocab_size(),
            unk_id
        )

    def prune_vocab(self, texts: List[str], min_usage_count: int = 1) -> int:
        """Analyze vocabulary usage to identify tokens for pruning."""
        logger.warning("prune_vocab analyzes token usage for potential pruning")

        analysis = metrics.prune_vocab_analysis(
            texts,
            self.encode,
            self.get_vocab(),
            self.special_tokens,
            min_usage_count
        )

        logger.info(f"Identified {analysis['tokens_to_prune']} tokens for pruning")
        logger.warning("Note: Actual pruning requires rebuilding the tokenizer")

        return analysis['tokens_to_prune']

    # I/O methods
    def save(self, path: str, pretty: bool = True) -> None:
        """
        Save the trained tokenizer to a file.

        Args:
            path: Path where to save the tokenizer (JSON format)
            pretty: Whether to save in pretty-printed format

        Example:
            >>> tokenizer.save("my_tokenizer.json")
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained tokenizer. Call train() first.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        self.tokenizer.save(path, pretty=pretty)
        logger.info(f"Tokenizer saved to: {path}")

        # Save metadata
        metadata_path = path.replace('.json', '_metadata.json')
        metadata = {
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "pretokenizer_type": self.pretokenizer_type,
            "use_byte_level": self.use_byte_level,
            "dropout": self.dropout,
            "languages": self.languages,
            "special_tokens": self.special_tokens
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2 if pretty else None)

        logger.info(f"Metadata saved to: {metadata_path}")

    @classmethod
    def load(cls, path: str) -> 'QuarkTok':
        """
        Load a trained tokenizer from a file.

        Args:
            path: Path to the saved tokenizer file

        Returns:
            QuarkTok instance with loaded tokenizer

        Example:
            >>> tokenizer = QuarkTok.load("my_tokenizer.json")
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        # Try to load metadata
        metadata_path = path.replace('.json', '_metadata.json')
        metadata = {}

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from: {metadata_path}")

        # Create instance with metadata or defaults
        instance = cls(
            vocab_size=metadata.get('vocab_size', constants.DEFAULT_VOCAB_SIZE),
            min_frequency=metadata.get('min_frequency', constants.DEFAULT_MIN_FREQUENCY),
            special_tokens=metadata.get('special_tokens'),
            pretokenizer_type=metadata.get('pretokenizer_type', 'whitespace'),
            use_byte_level=metadata.get('use_byte_level', False),
            dropout=metadata.get('dropout'),
            languages=metadata.get('languages')
        )

        # Load the tokenizer
        instance.tokenizer = Tokenizer.from_file(path)
        instance.is_trained = True

        logger.info(f"Tokenizer loaded from: {path}")
        logger.info(f"Vocabulary size: {instance.tokenizer.get_vocab_size()}")

        return instance

    # Vocabulary methods
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return self.tokenizer.get_vocab_size()

    def get_vocab(self) -> Dict[str, int]:
        """Get the full vocabulary as a dictionary mapping tokens to IDs."""
        return self.tokenizer.get_vocab()

    def id_to_token(self, token_id: int) -> Optional[str]:
        """Convert a token ID to its string representation."""
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> Optional[int]:
        """Convert a token string to its ID."""
        return self.tokenizer.token_to_id(token)

    # Cache methods
    def clear_cache(self) -> None:
        """Clear the encoding cache."""
        if self.enable_caching:
            self._encode_cache.clear()
            logger.info("Encoding cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the encoding cache."""
        if self.enable_caching:
            return {
                "size": len(self._encode_cache),
                "capacity": self.cache_size,
                "hit_rate": "Not tracked"
            }
        else:
            return {
                "size": 0,
                "capacity": 0,
                "hit_rate": "Caching disabled"
            }
