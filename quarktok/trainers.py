"""
QuarkTok Trainers Module

Handles tokenizer training operations including standard and incremental training.
"""

import os
import logging
import tempfile
from typing import List, Union, Optional, Iterator
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer

logger = logging.getLogger(__name__)


def train_tokenizer(
    tokenizer: Tokenizer,
    files: Union[str, List[str]],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    show_progress: bool = True,
    max_token_length: Optional[int] = None,
    limit_alphabet: Optional[int] = None
) -> None:
    """
    Train the BPE tokenizer on text files.

    Args:
        tokenizer: The tokenizer instance to train
        files: Path to a single file or list of file paths
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency threshold
        special_tokens: List of special tokens
        show_progress: Whether to show training progress
        max_token_length: Maximum length for merged tokens
        limit_alphabet: Limit initial alphabet size

    Raises:
        FileNotFoundError: If training file(s) don't exist
        RuntimeError: If training fails

    Example:
        >>> from tokenizers import Tokenizer
        >>> from tokenizers.models import BPE
        >>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        >>> train_tokenizer(
        ...     tokenizer,
        ...     "data.txt",
        ...     vocab_size=1000,
        ...     min_frequency=2,
        ...     special_tokens=["[UNK]", "[PAD]"]
        ... )
    """
    # Convert single file path to list
    if isinstance(files, str):
        files = [files]

    # Validate files exist
    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")

    logger.info(f"Training BPE tokenizer on {len(files)} file(s)...")
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Minimum frequency: {min_frequency}")

    # Configure the trainer with advanced options
    trainer_kwargs = {
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "special_tokens": special_tokens,
        "show_progress": show_progress,
        "continuing_subword_prefix": "##",
    }

    if max_token_length is not None:
        trainer_kwargs["max_token_length"] = max_token_length
        logger.info(f"Max token length: {max_token_length}")

    if limit_alphabet is not None:
        trainer_kwargs["limit_alphabet"] = limit_alphabet
        logger.info(f"Alphabet limit: {limit_alphabet}")
    else:
        trainer_kwargs["initial_alphabet"] = []

    trainer = BpeTrainer(**trainer_kwargs)

    # Train the tokenizer
    try:
        tokenizer.train(files, trainer)
        logger.info("Training completed successfully!")
        logger.info(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise RuntimeError(f"Failed to train tokenizer: {e}") from e


def train_incremental(
    tokenizer: Tokenizer,
    data_iterator: Iterator[str],
    vocab_size: int,
    min_frequency: int,
    special_tokens: List[str],
    batch_size: int = 10000,
    show_progress: bool = True
) -> None:
    """
    Train the tokenizer incrementally on batches of text.

    This method is useful for very large datasets that don't fit in memory.
    It processes data in batches and trains the tokenizer incrementally.

    Args:
        tokenizer: The tokenizer instance to train
        data_iterator: Iterator yielding text strings
        vocab_size: Target vocabulary size
        min_frequency: Minimum token frequency threshold
        special_tokens: List of special tokens
        batch_size: Number of texts to process per batch
        show_progress: Whether to show progress

    Example:
        >>> from tokenizers import Tokenizer
        >>> from tokenizers.models import BPE
        >>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        >>> def data_gen():
        ...     for line in ["line 1", "line 2"]:
        ...         yield line
        >>> train_incremental(
        ...     tokenizer,
        ...     data_gen(),
        ...     vocab_size=1000,
        ...     min_frequency=2,
        ...     special_tokens=["[UNK]"],
        ...     batch_size=100
        ... )
    """
    logger.info("Starting incremental training...")
    temp_files = []
    batch = []
    batch_num = 0

    try:
        # Use tqdm if available and show_progress is True
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(data_iterator, desc="Processing batches")
            except ImportError:
                logger.warning("tqdm not available, progress bar disabled")
                iterator = data_iterator
        else:
            iterator = data_iterator

        for text in iterator:
            batch.append(text)

            if len(batch) >= batch_size:
                # Write batch to temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    delete=False,
                    suffix='.txt',
                    encoding='utf-8'
                )
                temp_file.write('\n'.join(batch))
                temp_file.close()
                temp_files.append(temp_file.name)
                batch = []
                batch_num += 1

                if show_progress:
                    logger.info(f"Processed batch {batch_num} ({batch_size} texts)")

        # Write remaining batch
        if batch:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                delete=False,
                suffix='.txt',
                encoding='utf-8'
            )
            temp_file.write('\n'.join(batch))
            temp_file.close()
            temp_files.append(temp_file.name)
            batch_num += 1

        # Train on all temporary files
        logger.info(f"Training on {batch_num} batches...")
        train_tokenizer(
            tokenizer,
            temp_files,
            vocab_size,
            min_frequency,
            special_tokens,
            show_progress=show_progress
        )

    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except OSError:
                pass
