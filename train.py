import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from model import Seq2SeqTransformer, generate_square_subsequent_mask


# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "eng-telugu-bpe-tokenizer.json"
NUM_EPOCHS = 20 # Your choice of 20 is perfect


BATCH_SIZE = 96
EMB_SIZE = 1024
NHEAD = 16
FFN_HID_DIM = 4096 # Standard practice is 4 * EMB_SIZE
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
SRC_VOCAB_SIZE = 30000
TGT_VOCAB_SIZE = 30000


# --- 2. Load Tokenizer and Dataset ---
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
raw_datasets = load_dataset("opus100", "en-te")

# --- NEW: Filter out long sequences ---
MAX_LEN = 150
def filter_long_sequences(example):
    # The tokenizer is used to check the length after tokenization
    en_len = len(tokenizer.encode(example['translation']['en']).ids)
    te_len = len(tokenizer.encode(example['translation']['te']).ids)
    return en_len <= MAX_LEN and te_len <= MAX_LEN

print(f"Original train dataset size: {len(raw_datasets['train'])}")
raw_datasets = raw_datasets.filter(filter_long_sequences)
print(f"Filtered train dataset size: {len(raw_datasets['train'])}")
# --- End of New Code ---

SOS_token = tokenizer.token_to_id("[SOS]")
EOS_token = tokenizer.token_to_id("[EOS]")
PAD_token = tokenizer.token_to_id("[PAD]")

# --- 3. Preprocessing Function (same as before) ---
def tokenize_and_prepare(batch):
    inputs = [ex['en'] for ex in batch['translation']]
    targets = [ex['te'] for ex in batch['translation']]
    input_encodings = tokenizer.encode_batch(inputs)
    target_encodings = tokenizer.encode_batch(targets)
    processed_inputs = [[SOS_token] + enc.ids + [EOS_token] for enc in input_encodings]
    processed_targets = [[SOS_token] + enc.ids + [EOS_token] for enc in target_encodings]
    return {"input_ids": processed_inputs, "target_ids": processed_targets}

# --- 4. Apply Preprocessing and Create DataLoader (same as before) ---
tokenized_datasets = raw_datasets.map(tokenize_and_prepare, batched=True, remove_columns=['translation'])
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'target_ids'])

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=PAD_token)
    padded_targets = pad_sequence(target_ids, batch_first=True, padding_value=PAD_token)
    return {'input_ids': padded_inputs, 'target_ids': padded_targets}

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- 5. New: Training and Evaluation Functions ---
def train_epoch(model, optimizer, loss_fn):
    model.train()
    losses = 0
    
    # Wrap the dataloader with tqdm for a progress bar
    for batch in tqdm(train_dataloader, desc="Training"):
        src = batch['input_ids'].to(DEVICE)
        tgt = batch['target_ids'].to(DEVICE)
        
        # Prepare target data for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        
        # Create masks
        tgt_seq_len = tgt_input.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
        src_padding_mask = (src == PAD_token)
        tgt_padding_mask = (tgt_input == PAD_token)
        memory_key_padding_mask = src_padding_mask

        # Forward pass
        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, tgt_mask)
        
        optimizer.zero_grad()
        
        # Calculate loss
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        
        # Clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        losses += loss.item()
    
    return losses / len(list(train_dataloader))

def evaluate(model, loss_fn):
    model.eval()
    losses = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            src = batch['input_ids'].to(DEVICE)
            tgt = batch['target_ids'].to(DEVICE)
            
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            tgt_seq_len = tgt_input.shape[1]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
            src_padding_mask = (src == PAD_token)
            tgt_padding_mask = (tgt_input == PAD_token)
            memory_key_padding_mask = src_padding_mask
            
            logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, tgt_mask)
            
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

    return losses / len(list(val_dataloader))


# --- 6. New: Main Training Block ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"PyTorch sees GPU: {torch.cuda.get_device_name(0)}")

    
    model = Seq2SeqTransformer(
        NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD,
        SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM
    ).to(DEVICE)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, loss_fn)
        val_loss = evaluate(model, loss_fn)
        
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        
        # Save the model if it has the best validation loss so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model!")