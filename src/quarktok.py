from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Create a tokenizer with BPE model
tokenizer = Tokenizer(BPE())

# Set up pre-tokenization (splitting on whitespace)
tokenizer.pre_tokenizer = Whitespace()

# Configure the trainer
trainer = BpeTrainer(
    vocab_size=5000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Train on your data (you would normally use files)
sample_data = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models use byte pair encoding.",
    # ... more training data
]

tokenizer.train_from_iterator(sample_data, trainer=trainer)

# Now you can tokenize new text
encoded = tokenizer.encode("The quickest fox is jumping")
print(f"Tokens: {encoded.tokens}")
print(f"Token IDs: {encoded.ids}")
