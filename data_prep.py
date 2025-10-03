from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_texts(dataset):
    """
    A generator function to yield all English and Telugu texts 
    from the training dataset for the tokenizer.
    """
    for item in dataset['train']:
        yield item['translation']['en']
        yield item['translation']['te']

def main():
    """
    Downloads data and trains a custom BPE tokenizer.
    """
    # 1. Load the dataset
    print("Loading dataset...")
    raw_datasets = load_dataset("opus100", "en-te")
    print("Dataset loaded successfully.")

    # 2. Create and train the tokenizer
    print("Training tokenizer...")
    
    # Initialize a new BPE tokenizer
    bpe_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    bpe_tokenizer.pre_tokenizer = Whitespace()

    # Prepare the trainer with our desired vocabulary size and special tokens
    trainer = BpeTrainer(vocab_size=30000, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])

    # Train the tokenizer on our data
    bpe_tokenizer.train_from_iterator(get_all_texts(raw_datasets), trainer=trainer)
    
    # 3. Save the trained tokenizer
    tokenizer_path = "eng-telugu-bpe-tokenizer.json"
    bpe_tokenizer.save(tokenizer_path)
    print(f"Tokenizer trained and saved to {tokenizer_path}")


if __name__ == "__main__":
    main()