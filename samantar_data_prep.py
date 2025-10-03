import os
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# --- Configuration ---
TOKENIZER_PATH = "samanantar-bpe-tokenizer.json"
PROCESSED_DATA_PATH = "processed_samanantar_data"
SUBSET_SIZE = 1_000_000
MAX_LEN = 150
VOCAB_SIZE = 32000

def main():
    print("--- Starting Data Preparation for Samanantar ---")

    # 1. Load Raw Dataset
    print(f"Loading Samanantar dataset ({SUBSET_SIZE} pairs)...")
    # For this dataset version, the columns are 'src' and 'tgt'
    dataset = load_dataset('ai4bharat/samanantar', 'te', split=f'train[:{SUBSET_SIZE}]', trust_remote_code=True)
    
    # 2. Inspect the data structure
    print("\nInspecting a sample data point:")
    sample = next(iter(dataset))
    print(sample)
    
    # 3. Split the dataset
    split = dataset.train_test_split(test_size=0.01, seed=42)
    raw_datasets = split
    raw_datasets['validation'] = raw_datasets.pop('test')

    # 4. Train a new tokenizer if it doesn't exist
    if not os.path.exists(TOKENIZER_PATH):
        print(f"\nTraining new tokenizer and saving to {TOKENIZER_PATH}...")
        
        # CORRECTED: Use the correct keys 'src' and 'tgt'
        def get_all_texts(dataset_split):
            for item in dataset_split:
                yield item['src']
                yield item['tgt']

        bpe_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        bpe_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"])
        bpe_tokenizer.train_from_iterator(get_all_texts(raw_datasets['train']), trainer=trainer)
        bpe_tokenizer.save(TOKENIZER_PATH)
    else:
        print(f"\nLoading existing tokenizer from {TOKENIZER_PATH}...")
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    # 5. Filter long sequences
    # CORRECTED: Use the correct keys 'src' and 'tgt'
    def filter_long_sequences(example):
        src_len = len(tokenizer.encode(example['src']).ids)
        tgt_len = len(tokenizer.encode(example['tgt']).ids)
        return src_len <= MAX_LEN and tgt_len <= MAX_LEN

    print("\nFiltering long sequences...")
    # The columns to filter on are 'src' and 'tgt'
    processed_datasets = raw_datasets.filter(filter_long_sequences)
    print("Dataset size after filtering:")
    print(processed_datasets)

    # 6. Save the processed dataset to disk
    print(f"\nSaving processed dataset to '{PROCESSED_DATA_PATH}'...")
    processed_datasets.save_to_disk(PROCESSED_DATA_PATH)
    
    print("\n--- Data Preparation Complete ---")

if __name__ == "__main__":
    main()