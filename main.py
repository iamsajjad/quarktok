"""
QuarkTok - Advanced BPE Tokenizer Training Pipeline

This script demonstrates the comprehensive features of QuarkTok, including:
- Advanced pre-tokenization strategies
- Unicode and URL/number normalization
- Multiple post-processing templates
- Vocabulary analysis and evaluation metrics
- Incremental training support
- Caching and performance optimization

Usage:
    python main.py                          # Train with default settings
    python main.py --pretokenizer advanced  # Use advanced pre-tokenization
    python main.py --byte-level             # Use byte-level BPE
    python main.py --normalize-all          # Enable all normalization
"""

from quarktok import QuarkTok
import os
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QuarkTok Advanced Tokenizer Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic configuration
    parser.add_argument(
        "--dataset",
        default="datasets/wikipedia-en-1gb.txt",
        help="Path to training dataset"
    )
    parser.add_argument(
        "--output",
        default="tokenizer.json",
        help="Output path for trained tokenizer"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=30000,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency"
    )

    # Pre-tokenization options
    parser.add_argument(
        "--pretokenizer",
        choices=["whitespace", "byte_level", "advanced"],
        default="whitespace",
        help="Pre-tokenization strategy"
    )

    # Normalization options
    parser.add_argument(
        "--normalize-unicode",
        action="store_true",
        help="Enable Unicode normalization (NFKC, NFD, strip accents)"
    )
    parser.add_argument(
        "--normalize-lowercase",
        action="store_true",
        help="Convert text to lowercase"
    )
    parser.add_argument(
        "--normalize-urls",
        action="store_true",
        help="Normalize URLs to [URL] token"
    )
    parser.add_argument(
        "--normalize-numbers",
        action="store_true",
        help="Normalize numbers to [NUM] token"
    )
    parser.add_argument(
        "--normalize-all",
        action="store_true",
        help="Enable all normalization options"
    )

    # BPE model options
    parser.add_argument(
        "--byte-level",
        action="store_true",
        help="Use byte-level BPE model"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="BPE dropout rate for regularization (0.0-1.0)"
    )

    # Training options
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=None,
        help="Maximum length for merged tokens"
    )
    parser.add_argument(
        "--limit-alphabet",
        type=int,
        default=None,
        help="Limit initial alphabet size"
    )

    # Post-processing
    parser.add_argument(
        "--post-processor",
        choices=["bert", "gpt", "t5", "none"],
        default="bert",
        help="Post-processing template style"
    )

    # Multilingual
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language codes for multilingual support (e.g., en es fr)"
    )

    # Performance
    parser.add_argument(
        "--disable-caching",
        action="store_true",
        help="Disable encoding cache"
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10000,
        help="Maximum cache size"
    )

    # Demo options
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip feature demonstration after training"
    )

    return parser.parse_args()


def demonstrate_features(tokenizer: QuarkTok):
    """
    Demonstrate all advanced features of the tokenizer.

    Args:
        tokenizer: Trained QuarkTok instance
    """
    logger.info("\n" + "="*70)
    logger.info("FEATURE DEMONSTRATION")
    logger.info("="*70)

    # Example texts covering various features
    sample_texts = [
        "Machine learning models use byte pair encoding for tokenization.",
        "Visit https://example.com for more information about 12345 items.",
        "The quick brown fox jumps over the lazy dog.",
        "Python code: def hello(): print('Hello, World!')",
        "Math equation: E = mc² and π ≈ 3.14159"
    ]

    # 1. Basic Tokenization
    logger.info("\n1. BASIC TOKENIZATION")
    logger.info("-" * 70)
    for i, text in enumerate(sample_texts[:3], 1):
        logger.info(f"\nExample {i}: {text}")

        # Get tokens
        tokens = tokenizer.get_tokens(text, add_special_tokens=False)
        logger.info(f"  Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")

        # Encode
        ids = tokenizer.encode(text, add_special_tokens=False)
        logger.info(f"  Token IDs ({len(ids)}): {ids[:15]}{'...' if len(ids) > 15 else ''}")

        # Decode
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        logger.info(f"  Decoded: {decoded}")
        logger.info(f"  Match: {'✓' if decoded.strip() == text else '✗'}")

    # 2. Advanced Decoding with Metadata
    logger.info("\n2. ADVANCED DECODING WITH METADATA")
    logger.info("-" * 70)
    test_text = sample_texts[0]
    ids = tokenizer.encode(test_text, add_special_tokens=True)
    metadata = tokenizer.decode_with_metadata(ids, skip_special_tokens=False)

    logger.info(f"Text: {test_text}")
    logger.info(f"Tokens: {metadata['tokens'][:10]}...")
    logger.info(f"Token types: {metadata['token_types'][:10]}...")
    logger.info(f"Special tokens mask: {metadata['special_tokens_mask'][:10]}...")

    # 3. Partial Decoding
    logger.info("\n3. PARTIAL DECODING")
    logger.info("-" * 70)
    full_ids = tokenizer.encode(test_text, add_special_tokens=False)
    if len(full_ids) >= 5:
        partial_text = tokenizer.partial_decode(full_ids, 0, min(5, len(full_ids)))
        logger.info(f"Full text: {test_text}")
        logger.info(f"First 5 tokens decoded: {partial_text}")

    # 4. Compression Ratio Analysis
    logger.info("\n4. COMPRESSION RATIO ANALYSIS")
    logger.info("-" * 70)
    for text in sample_texts[:3]:
        ratio = tokenizer.calculate_compression_ratio(text)
        logger.info(f"Text: {text[:50]}...")
        logger.info(f"  Compression ratio: {ratio:.2f} chars/token")
        logger.info(f"  Efficiency: {'High' if ratio > 4 else 'Medium' if ratio > 3 else 'Low'}")

    # 5. Fertility Score
    logger.info("\n5. FERTILITY SCORE (Tokens per Word)")
    logger.info("-" * 70)
    fertility = tokenizer.fertility_score(sample_texts)
    logger.info(f"Average fertility score: {fertility:.3f} tokens/word")
    logger.info(f"Interpretation: {'Excellent' if fertility < 1.2 else 'Good' if fertility < 1.5 else 'Needs improvement'}")

    # 6. Unknown Token Rate
    logger.info("\n6. UNKNOWN TOKEN RATE")
    logger.info("-" * 70)
    unk_rate = tokenizer.measure_unk_rate(sample_texts)
    logger.info(f"Unknown token rate: {unk_rate:.2f}%")
    logger.info(f"Status: {'Excellent' if unk_rate < 1 else 'Good' if unk_rate < 5 else 'Needs improvement'}")

    # 7. Comprehensive Vocabulary Coverage Analysis
    logger.info("\n7. VOCABULARY COVERAGE ANALYSIS")
    logger.info("-" * 70)
    coverage = tokenizer.analyze_vocab_coverage(sample_texts)
    for key, value in coverage.items():
        logger.info(f"  {key}: {value}")

    # 8. Post-processor Template Demonstration
    logger.info("\n8. POST-PROCESSOR TEMPLATES")
    logger.info("-" * 70)
    test_sentence = "Hello world"

    for template in ["bert", "gpt", "t5", "none"]:
        tokenizer.set_post_processor(template)
        tokens = tokenizer.get_tokens(test_sentence, add_special_tokens=True)
        logger.info(f"  {template.upper():8s}: {tokens}")

    # Reset to default
    tokenizer.set_post_processor("bert")

    # 9. Cache Statistics
    logger.info("\n9. CACHE STATISTICS")
    logger.info("-" * 70)
    cache_stats = tokenizer.get_cache_stats()
    for key, value in cache_stats.items():
        logger.info(f"  {key}: {value}")

    # 10. Special Tokens
    logger.info("\n10. SPECIAL TOKENS")
    logger.info("-" * 70)
    logger.info(f"Total special tokens: {len(tokenizer.special_tokens)}")
    logger.info(f"Standard tokens: {tokenizer.STANDARD_SPECIAL_TOKENS}")
    logger.info(f"Code tokens: {tokenizer.CODE_SPECIAL_TOKENS}")
    logger.info(f"Math tokens: {tokenizer.MATH_SPECIAL_TOKENS}")
    logger.info(f"Media tokens: {tokenizer.MEDIA_SPECIAL_TOKENS}")
    if tokenizer.languages:
        lang_tokens = [f"[{lang.upper()}]" for lang in tokenizer.languages]
        logger.info(f"Language tokens: {lang_tokens}")

    # 11. Vocabulary Statistics
    logger.info("\n11. VOCABULARY STATISTICS")
    logger.info("-" * 70)
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Total vocabulary size: {vocab_size:,}")
    logger.info(f"Target vocabulary size: {tokenizer.vocab_size:,}")
    logger.info(f"Match: {'✓' if vocab_size == tokenizer.vocab_size else '✗'}")

    # Sample tokens
    vocab = tokenizer.get_vocab()
    sample_tokens = list(vocab.items())[:20]
    logger.info(f"\nSample tokens (first 20):")
    for token, token_id in sample_tokens:
        logger.info(f"  '{token}': {token_id}")


def main():
    """
    Main entry point for QuarkTok training pipeline.
    """
    args = parse_args()

    logger.info("="*70)
    logger.info("QuarkTok Advanced BPE Tokenizer Training Pipeline")
    logger.info("="*70)

    # Configuration summary
    logger.info("\nCONFIGURATION:")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Vocabulary size: {args.vocab_size:,}")
    logger.info(f"  Min frequency: {args.min_frequency}")
    logger.info(f"  Pre-tokenizer: {args.pretokenizer}")
    logger.info(f"  Post-processor: {args.post_processor}")

    # Validate dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found at: {args.dataset}")
        logger.info("Please run 'python -m src.dataset' to download the dataset first.")
        return 1

    # Handle normalize-all flag
    if args.normalize_all:
        args.normalize_unicode = True
        args.normalize_lowercase = True
        args.normalize_urls = True
        args.normalize_numbers = True

    # Create tokenizer with configuration
    logger.info("\nInitializing QuarkTok tokenizer...")
    tokenizer = QuarkTok(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        pretokenizer_type=args.pretokenizer,
        normalize_unicode=args.normalize_unicode,
        normalize_lowercase=args.normalize_lowercase,
        normalize_urls=args.normalize_urls,
        normalize_numbers=args.normalize_numbers,
        use_byte_level=args.byte_level,
        dropout=args.dropout,
        languages=args.languages,
        enable_caching=not args.disable_caching,
        cache_size=args.cache_size
    )

    # Set post-processor
    tokenizer.set_post_processor(args.post_processor)

    # Train the tokenizer
    logger.info(f"\nTraining on: {args.dataset}")
    logger.info("This may take several minutes for large datasets...")

    tokenizer.train(
        args.dataset,
        show_progress=True,
        max_token_length=args.max_token_length,
        limit_alphabet=args.limit_alphabet
    )

    # Save the trained tokenizer
    logger.info(f"\nSaving tokenizer to: {args.output}")
    tokenizer.save(args.output)

    # Demonstrate features (unless skipped)
    if not args.skip_demo:
        demonstrate_features(tokenizer)

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Tokenizer saved to: {args.output}")
    logger.info(f"Metadata saved to: {args.output.replace('.json', '_metadata.json')}")
    logger.info(f"Final vocabulary size: {tokenizer.get_vocab_size():,}")
    logger.info("\nYou can now use the tokenizer:")
    logger.info("  from src.quarktok import QuarkTok")
    logger.info(f"  tokenizer = QuarkTok.load('{args.output}')")
    logger.info("  tokens = tokenizer.encode('Your text here')")
    logger.info("="*70)

    return 0


if __name__ == "__main__":
    exit(main())
