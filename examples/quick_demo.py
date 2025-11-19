#!/usr/bin/env python3
"""
QuarkTok Quick Demo - Showcasing Advanced Features

This script demonstrates key features without requiring a large dataset.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quarktok import QuarkTok
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create a small sample dataset for demonstration."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models use tokenization for text processing.",
        "Python is a popular programming language for data science.",
        "Artificial intelligence is transforming many industries.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of training data.",
        "Tokenizers break text into smaller units called tokens.",
        "Byte pair encoding is an efficient tokenization algorithm.",
        "Large language models can generate human-like text.",
        "The Transformer architecture revolutionized NLP tasks.",
    ] * 100  # Repeat to have enough data

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write('\n'.join(sample_texts))
    temp_file.close()

    return temp_file.name


def demo_basic_features():
    """Demonstrate basic tokenization features."""
    logger.info("="*70)
    logger.info("DEMO 1: Basic Tokenization")
    logger.info("="*70)

    # Create sample data
    data_file = create_sample_data()

    try:
        # Create and train tokenizer
        logger.info("Creating tokenizer...")
        tokenizer = QuarkTok(vocab_size=1000, min_frequency=2)

        logger.info("Training tokenizer...")
        tokenizer.train(data_file, show_progress=False)

        # Test encoding/decoding
        test_text = "Hello, world! This is a test."
        logger.info(f"\nTest text: {test_text}")

        tokens = tokenizer.get_tokens(test_text, add_special_tokens=False)
        logger.info(f"Tokens: {tokens}")

        ids = tokenizer.encode(test_text, add_special_tokens=False)
        logger.info(f"Token IDs: {ids}")

        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        logger.info(f"Decoded: {decoded}")
        logger.info(f"Match: {decoded.strip() == test_text}")

    finally:
        os.unlink(data_file)


def demo_advanced_pretokenization():
    """Demonstrate different pre-tokenization strategies."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Advanced Pre-tokenization")
    logger.info("="*70)

    data_file = create_sample_data()

    try:
        test_text = "Python123: print('Hello, World!')"

        for pretokenizer in ["whitespace", "advanced"]:
            logger.info(f"\n{pretokenizer.upper()} pre-tokenization:")
            tokenizer = QuarkTok(
                vocab_size=1000,
                pretokenizer_type=pretokenizer
            )
            tokenizer.train(data_file, show_progress=False)

            tokens = tokenizer.get_tokens(test_text, add_special_tokens=False)
            logger.info(f"  Tokens: {tokens}")

    finally:
        os.unlink(data_file)


def demo_normalization():
    """Demonstrate text normalization features."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Text Normalization")
    logger.info("="*70)

    data_file = create_sample_data()

    try:
        test_text = "Visit https://example.com for 12345 items!"
        logger.info(f"Original text: {test_text}")

        # Without normalization
        logger.info("\nWithout normalization:")
        tokenizer = QuarkTok(vocab_size=1000)
        tokenizer.train(data_file, show_progress=False)
        tokens1 = tokenizer.get_tokens(test_text, add_special_tokens=False)
        logger.info(f"  Tokens: {tokens1}")

        # With URL and number normalization
        logger.info("\nWith URL and number normalization:")
        tokenizer = QuarkTok(
            vocab_size=1000,
            normalize_urls=True,
            normalize_numbers=True
        )
        tokenizer.train(data_file, show_progress=False)
        tokens2 = tokenizer.get_tokens(test_text, add_special_tokens=False)
        logger.info(f"  Tokens: {tokens2}")

    finally:
        os.unlink(data_file)


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 4: Evaluation Metrics")
    logger.info("="*70)

    data_file = create_sample_data()

    try:
        tokenizer = QuarkTok(vocab_size=1000)
        tokenizer.train(data_file, show_progress=False)

        test_texts = [
            "Machine learning is fascinating.",
            "Deep learning models are powerful.",
            "Natural language processing is important."
        ]

        # Compression ratio
        logger.info("\nCompression Ratios:")
        for text in test_texts:
            ratio = tokenizer.calculate_compression_ratio(text)
            logger.info(f"  {text[:40]:40s} -> {ratio:.2f} chars/token")

        # Fertility score
        fertility = tokenizer.fertility_score(test_texts)
        logger.info(f"\nFertility score: {fertility:.3f} tokens/word")

        # UNK rate
        unk_rate = tokenizer.measure_unk_rate(test_texts)
        logger.info(f"Unknown token rate: {unk_rate:.2f}%")

        # Comprehensive analysis
        logger.info("\nVocabulary Coverage Analysis:")
        coverage = tokenizer.analyze_vocab_coverage(test_texts)
        for key, value in coverage.items():
            logger.info(f"  {key:20s}: {value}")

    finally:
        os.unlink(data_file)


def demo_post_processors():
    """Demonstrate different post-processing templates."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 5: Post-Processing Templates")
    logger.info("="*70)

    data_file = create_sample_data()

    try:
        tokenizer = QuarkTok(vocab_size=1000)
        tokenizer.train(data_file, show_progress=False)

        test_text = "Hello world"
        logger.info(f"Text: {test_text}\n")

        for template in ["bert", "gpt", "t5", "none"]:
            tokenizer.set_post_processor(template)
            tokens = tokenizer.get_tokens(test_text, add_special_tokens=True)
            logger.info(f"{template.upper():8s}: {tokens}")

    finally:
        os.unlink(data_file)


def demo_metadata_decoding():
    """Demonstrate decoding with metadata."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 6: Advanced Decoding with Metadata")
    logger.info("="*70)

    data_file = create_sample_data()

    try:
        tokenizer = QuarkTok(vocab_size=1000)
        tokenizer.train(data_file, show_progress=False)

        test_text = "Machine learning is amazing!"
        ids = tokenizer.encode(test_text, add_special_tokens=True)

        metadata = tokenizer.decode_with_metadata(ids, skip_special_tokens=False)

        logger.info(f"Text: {test_text}")
        logger.info(f"\nTokens: {metadata['tokens']}")
        logger.info(f"Token types: {metadata['token_types']}")
        logger.info(f"Special tokens mask: {metadata['special_tokens_mask']}")

    finally:
        os.unlink(data_file)


def demo_special_tokens():
    """Demonstrate special tokens."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 7: Special Tokens")
    logger.info("="*70)

    logger.info(f"Standard tokens: {QuarkTok.STANDARD_SPECIAL_TOKENS}")
    logger.info(f"Code tokens: {QuarkTok.CODE_SPECIAL_TOKENS}")
    logger.info(f"Math tokens: {QuarkTok.MATH_SPECIAL_TOKENS}")
    logger.info(f"Media tokens: {QuarkTok.MEDIA_SPECIAL_TOKENS}")

    # Multilingual
    logger.info("\nMultilingual tokenizer:")
    tokenizer = QuarkTok(vocab_size=1000, languages=["en", "es", "fr"])
    logger.info(f"Total special tokens: {len(tokenizer.special_tokens)}")
    logger.info(f"Language tokens added: [EN], [ES], [FR]")


def demo_caching():
    """Demonstrate caching features."""
    logger.info("\n" + "="*70)
    logger.info("DEMO 8: Performance Caching")
    logger.info("="*70)

    data_file = create_sample_data()

    try:
        tokenizer = QuarkTok(vocab_size=1000, enable_caching=True, cache_size=100)
        tokenizer.train(data_file, show_progress=False)

        # Encode same text multiple times
        text = "This text will be cached"
        for i in range(5):
            _ = tokenizer.encode(text)

        stats = tokenizer.get_cache_stats()
        logger.info(f"Cache size: {stats['size']}/{stats['capacity']}")

        # Clear cache
        tokenizer.clear_cache()
        logger.info("Cache cleared")

        stats = tokenizer.get_cache_stats()
        logger.info(f"Cache size after clear: {stats['size']}/{stats['capacity']}")

    finally:
        os.unlink(data_file)


def main():
    """Run all demos."""
    logger.info("QuarkTok Feature Demonstration")
    logger.info("This demo showcases the advanced features without requiring large datasets\n")

    demo_basic_features()
    demo_advanced_pretokenization()
    demo_normalization()
    demo_evaluation_metrics()
    demo_post_processors()
    demo_metadata_decoding()
    demo_special_tokens()
    demo_caching()

    logger.info("\n" + "="*70)
    logger.info("All demos completed successfully!")
    logger.info("="*70)
    logger.info("\nFor more information, see README.md")
    logger.info("To train on your own data: python main.py --help")


if __name__ == "__main__":
    main()
