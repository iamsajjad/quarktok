# QuarkTok - Advanced BPE Tokenizer

A comprehensive, production-ready Byte Pair Encoding (BPE) tokenizer for training Large Language Models (LLMs), built on top of HuggingFace's tokenizers library with extensive customization options.

## Features

### Core Capabilities

- **Multiple Pre-tokenization Strategies**
  - Whitespace splitting
  - Byte-level BPE (GPT-2 style)
  - Advanced mode (Digits + Punctuation + Whitespace)
  - Custom pre-tokenization support

- **Advanced Text Normalization**
  - Unicode normalization (NFKC, NFD, accent stripping)
  - Lowercase conversion
  - URL normalization (`https://example.com` � `[URL]`)
  - Number normalization (`12345` � `[NUM]`)
  - Whitespace normalization

- **Flexible BPE Configuration**
  - Configurable vocabulary size
  - Minimum frequency thresholds
  - BPE dropout for regularization
  - Byte fallback for rare characters
  - Maximum token length limits
  - Alphabet size constraints

- **Task-Specific Special Tokens**
  - Standard tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`, `[BOS]`, `[EOS]`
  - Code tokens: `[CODE]`, `[/CODE]`, `[INDENT]`, `[DEDENT]`, `[COMMENT]`
  - Math tokens: `[MATH]`, `[/MATH]`, `[EQUATION]`, `[/EQUATION]`
  - Media tokens: `[IMG]`, `[/IMG]`, `[URL]`, `[/URL]`, `[NUM]`

- **Multiple Post-Processing Templates**
  - BERT style: `[CLS] text [SEP]`
  - GPT style: `text [EOS]`
  - T5 style: `text [EOS]`
  - None (no special token wrapping)

- **Advanced Decoding**
  - Standard decoding
  - Decoding with metadata (token types, special token masks)
  - Partial decoding (decode token slices)
  - Batch decoding support

- **Evaluation Metrics**
  - Compression ratio (chars/token)
  - Fertility score (tokens/word)
  - Unknown token rate
  - Vocabulary coverage analysis
  - Vocabulary utilization tracking

- **Vocabulary Optimization**
  - Vocabulary pruning
  - Usage analysis
  - Coverage statistics

- **Performance Features**
  - LRU encoding cache (configurable size)
  - Streaming/incremental training for large datasets
  - Batch encoding/decoding
  - Progress tracking

- **Multilingual Support**
  - Language-specific tokens
  - Multi-language training

## Project Structure

```
quarktok/  (project root)
├── quarktok/           # Main package
│   ├── __init__.py     # Public API
│   ├── core.py         # QuarkTok class
│   ├── constants.py    # Special tokens
│   ├── pre_tokenizers.py
│   ├── normalizers.py
│   ├── post_processors.py
│   ├── metrics.py      # Evaluation
│   ├── trainers.py     # Training
│   └── dataset.py      # Data utilities
├── examples/           # Example scripts
├── main.py            # CLI interface
├── README.md
└── pyproject.toml
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/quarktok.git
cd quarktok

# Install dependencies (using uv or pip)
uv sync
# or
pip install tokenizers datasets tqdm
```

## Quick Start

### Basic Usage

```python
from quarktok import QuarkTok

# Create and train tokenizer
tokenizer = QuarkTok(vocab_size=30000)
tokenizer.train("path/to/data.txt")

# Encode text
text = "Hello, world!"
token_ids = tokenizer.encode(text)
print(token_ids)

# Decode back
decoded = tokenizer.decode(token_ids)
print(decoded)

# Save tokenizer
tokenizer.save("my_tokenizer.json")

# Load tokenizer
tokenizer = QuarkTok.load("my_tokenizer.json")
```

### Advanced Configuration

```python
from quarktok import QuarkTok

# Create tokenizer with all features
tokenizer = QuarkTok(
    vocab_size=30000,
    min_frequency=2,
    pretokenizer_type="advanced",      # Use advanced pre-tokenization
    normalize_unicode=True,             # Enable Unicode normalization
    normalize_lowercase=False,          # Keep original casing
    normalize_urls=True,                # Normalize URLs to [URL]
    normalize_numbers=True,             # Normalize numbers to [NUM]
    use_byte_level=False,               # Use character-level (not byte-level)
    dropout=0.1,                        # Add 10% dropout for robustness
    languages=["en", "es", "fr"],       # Multilingual support
    enable_caching=True,                # Enable encoding cache
    cache_size=10000                    # Cache up to 10k entries
)

# Train with advanced options
tokenizer.train(
    "data.txt",
    max_token_length=15,    # Limit merged token length
    limit_alphabet=1000     # Limit initial alphabet size
)

# Save with metadata
tokenizer.save("advanced_tokenizer.json")
```

### Command-Line Training

```bash
# Basic training
python main.py

# Advanced pre-tokenization
python main.py --pretokenizer advanced

# Byte-level BPE (like GPT-2)
python main.py --byte-level

# Enable all normalizations
python main.py --normalize-all

# Custom configuration
python main.py \
    --vocab-size 50000 \
    --pretokenizer advanced \
    --normalize-unicode \
    --normalize-urls \
    --dropout 0.1 \
    --max-token-length 20 \
    --languages en es fr \
    --output multilingual_tokenizer.json

# See all options
python main.py --help
```

## Advanced Features

### 1. Pre-tokenization Strategies

```python
# Whitespace splitting (default)
tokenizer = QuarkTok(pretokenizer_type="whitespace")

# Byte-level (GPT-2 style)
tokenizer = QuarkTok(pretokenizer_type="byte_level")

# Advanced (Digits + Punctuation + Whitespace)
tokenizer = QuarkTok(pretokenizer_type="advanced")
```

### 2. Text Normalization

```python
# Enable various normalizations
tokenizer = QuarkTok(
    normalize_unicode=True,      # NFKC, NFD, strip accents
    normalize_lowercase=True,    # Convert to lowercase
    normalize_urls=True,         # URLs � [URL]
    normalize_numbers=True       # Numbers � [NUM]
)

# Example: "Visit https://example.com for 123 items"
# Becomes: "visit [URL] for [NUM] items"
```

### 3. Post-Processing Templates

```python
tokenizer = QuarkTok()
tokenizer.train("data.txt")

# BERT style: [CLS] text [SEP]
tokenizer.set_post_processor("bert")

# GPT style: text [EOS]
tokenizer.set_post_processor("gpt")

# T5 style: text [EOS]
tokenizer.set_post_processor("t5")

# No wrapping
tokenizer.set_post_processor("none")
```

### 4. Advanced Decoding

```python
# Decode with metadata
ids = tokenizer.encode("Hello world")
metadata = tokenizer.decode_with_metadata(ids)

print(metadata['text'])                  # Decoded text
print(metadata['tokens'])                # Token strings
print(metadata['token_types'])           # ['special', 'normal', ...]
print(metadata['special_tokens_mask'])   # Boolean mask

# Partial decoding
partial = tokenizer.partial_decode(ids, start=1, end=5)
```

### 5. Evaluation Metrics

```python
# Compression ratio (higher is better)
ratio = tokenizer.calculate_compression_ratio("Some text")
print(f"Compression: {ratio:.2f} chars/token")

# Fertility score (lower is better)
texts = ["Sample text 1", "Sample text 2"]
fertility = tokenizer.fertility_score(texts)
print(f"Fertility: {fertility:.2f} tokens/word")

# Unknown token rate
unk_rate = tokenizer.measure_unk_rate(texts)
print(f"UNK rate: {unk_rate:.2f}%")

# Comprehensive coverage analysis
coverage = tokenizer.analyze_vocab_coverage(texts)
print(coverage)
# {
#     'compression_ratio': 4.2,
#     'fertility_score': 1.1,
#     'unk_rate': 0.5,
#     'total_tokens': 1000,
#     'total_chars': 4200,
#     'unique_tokens': 500,
#     'vocab_utilization': 1.67
# }
```

### 6. Vocabulary Optimization

```python
# Analyze which tokens are rarely used
sample_texts = ["your", "corpus", "texts"]
pruned_count = tokenizer.prune_vocab(sample_texts, min_usage_count=5)
print(f"Would prune {pruned_count} tokens")
```

### 7. Incremental Training

```python
# For very large datasets that don't fit in memory
def data_generator():
    with open("huge_file.txt") as f:
        for line in f:
            yield line

tokenizer = QuarkTok()
tokenizer.train_incremental(
    data_generator(),
    batch_size=10000,
    show_progress=True
)
```

### 8. Caching for Performance

```python
# Enable caching (default)
tokenizer = QuarkTok(enable_caching=True, cache_size=10000)

# Check cache stats
stats = tokenizer.get_cache_stats()
print(stats)
# {'size': 150, 'capacity': 10000, 'hit_rate': 'Not tracked'}

# Clear cache
tokenizer.clear_cache()
```

### 9. Multilingual Support

```python
# Create multilingual tokenizer
tokenizer = QuarkTok(languages=["en", "es", "fr", "de"])

# Language tokens [EN], [ES], [FR], [DE] are automatically added
# Use them to mark language in your training data
```

### 10. Special Tokens

```python
# Access predefined token sets
print(QuarkTok.STANDARD_SPECIAL_TOKENS)
# ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[BOS]', '[EOS]']

print(QuarkTok.CODE_SPECIAL_TOKENS)
# ['[CODE]', '[/CODE]', '[INDENT]', '[DEDENT]', '[COMMENT]']

print(QuarkTok.MATH_SPECIAL_TOKENS)
# ['[MATH]', '[/MATH]', '[EQUATION]', '[/EQUATION]']

print(QuarkTok.MEDIA_SPECIAL_TOKENS)
# ['[IMG]', '[/IMG]', '[URL]', '[/URL]', '[NUM]']

# Or use custom tokens
custom_tokens = ["[SPECIAL1]", "[SPECIAL2]", "[UNK]"]
tokenizer = QuarkTok(special_tokens=custom_tokens)
```

## API Reference

### QuarkTok Class

#### Initialization Parameters

```python
QuarkTok(
    vocab_size: int = 30000,
    min_frequency: int = 2,
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
    cache_size: int = 10000
)
```

#### Core Methods

- `train(files, show_progress=True, max_token_length=None, limit_alphabet=None)` - Train the tokenizer
- `train_incremental(data_iterator, batch_size=10000, show_progress=True)` - Incremental training
- `encode(text, add_special_tokens=True)` - Encode text to token IDs
- `decode(ids, skip_special_tokens=True)` - Decode token IDs to text
- `save(path, pretty=True)` - Save tokenizer to file
- `load(path)` - Load tokenizer from file (class method)

#### Advanced Methods

- `decode_with_metadata(ids, skip_special_tokens=True)` - Decode with metadata
- `partial_decode(ids, start, end, skip_special_tokens=True)` - Decode token slice
- `get_tokens(text, add_special_tokens=True)` - Get token strings
- `set_post_processor(template_type="bert")` - Change post-processing template

#### Evaluation Methods

- `calculate_compression_ratio(text)` - Calculate compression ratio
- `fertility_score(texts)` - Calculate fertility score
- `measure_unk_rate(texts)` - Measure unknown token rate
- `analyze_vocab_coverage(texts)` - Comprehensive coverage analysis
- `prune_vocab(texts, min_usage_count=1)` - Identify tokens to prune

#### Utility Methods

- `get_vocab_size()` - Get vocabulary size
- `get_vocab()` - Get vocabulary dictionary
- `id_to_token(token_id)` - Convert ID to token
- `token_to_id(token)` - Convert token to ID
- `clear_cache()` - Clear encoding cache
- `get_cache_stats()` - Get cache statistics

## Examples

### Example 1: Training for Code

```python
from quarktok import QuarkTok

# Optimized for code
tokenizer = QuarkTok(
    vocab_size=50000,
    pretokenizer_type="advanced",  # Better for code punctuation
    normalize_lowercase=False,     # Code is case-sensitive
    special_tokens=QuarkTok.STANDARD_SPECIAL_TOKENS + QuarkTok.CODE_SPECIAL_TOKENS
)

tokenizer.train("code_dataset.txt")
tokenizer.save("code_tokenizer.json")
```

### Example 2: Multilingual Tokenizer

```python
from quarktok import QuarkTok

# Multilingual tokenizer
tokenizer = QuarkTok(
    vocab_size=100000,
    languages=["en", "es", "fr", "de", "zh", "ja"],
    normalize_unicode=True,
    pretokenizer_type="byte_level"
)

tokenizer.train(["en_data.txt", "es_data.txt", "fr_data.txt"])
tokenizer.save("multilingual_tokenizer.json")
```

### Example 3: Evaluation Pipeline

```python
from quarktok import QuarkTok

# Load tokenizer
tokenizer = QuarkTok.load("tokenizer.json")

# Evaluate on test set
test_texts = load_test_texts()

# Get comprehensive metrics
metrics = tokenizer.analyze_vocab_coverage(test_texts)
print(f"Compression ratio: {metrics['compression_ratio']}")
print(f"Fertility score: {metrics['fertility_score']}")
print(f"UNK rate: {metrics['unk_rate']}%")
print(f"Vocab utilization: {metrics['vocab_utilization']}%")
```

## Performance Tips

1. **Enable Caching**: For repeated encoding of the same texts
   ```python
   tokenizer = QuarkTok(enable_caching=True, cache_size=10000)
   ```

2. **Use Batch Encoding**: Faster than encoding one at a time
   ```python
   texts = ["text1", "text2", "text3"]
   ids = tokenizer.encode(texts)  # Batch mode
   ```

3. **Incremental Training**: For datasets larger than RAM
   ```python
   tokenizer.train_incremental(data_generator(), batch_size=10000)
   ```

4. **Limit Token Length**: Prevents very long merged tokens
   ```python
   tokenizer.train("data.txt", max_token_length=15)
   ```

## Project Structure

```
quarktok/
   src/
      __init__.py
      quarktok.py      # Main tokenizer implementation
      dataset.py       # Dataset download utilities
   datasets/            # Training data directory
   main.py             # Training pipeline with CLI
   README.md           # This file
   LICENSE             # Apache 2.0 License
   pyproject.toml      # Project dependencies
```

## Contributing

Contributions are welcome! Areas for improvement:

- Additional pre-tokenization strategies
- More evaluation metrics
- Custom merge selection algorithms
- Visualization tools
- Performance optimizations

## License

Apache License 2.0 - See LICENSE file for details.

## Citation

If you use QuarkTok in your research, please cite:

```bibtex
@software{quarktok2024,
  title={QuarkTok: Advanced BPE Tokenizer for LLM Training},
  author={QuarkTok Team},
  year={2024},
  url={https://github.com/yourusername/quarktok}
}
```

## Acknowledgments

Built on top of [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers), which provides the underlying BPE implementation.

## Support

For questions, issues, or feature requests, please open an issue on GitHub.
