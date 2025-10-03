import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm import tqdm
import os

from model import Seq2SeqTransformer, generate_square_subsequent_mask

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "samanantar-bpe-tokenizer.json"
MODEL_SAVE_PATH = "best_model_samanantar.pt"
PROCESSED_DATA_PATH = "processed_samanantar_data"
NUM_EPOCHS = 20

# Data and Model Hyperparameters
BATCH_SIZE = 24
ACCUMULATION_STEPS = 4
SRC_VOCAB_SIZE = 32000
TGT_VOCAB_SIZE = 32000
EMB_SIZE = 1024
NHEAD = 16
FFN_HID_DIM = 4096
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# --- 2. Load Processed Data and Tokenizer ---
print("Loading pre-processed dataset and tokenizer...")
processed_datasets = load_from_disk(PROCESSED_DATA_PATH)
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
SOS_token = tokenizer.token_to_id("[SOS]")
EOS_token = tokenizer.token_to_id("[EOS]")
PAD_token = tokenizer.token_to_id("[PAD]")

# --- 3. Preprocessing and DataLoader ---
def tokenize_and_prepare(batch):
    # CORRECTED: Access the 'src' and 'tgt' columns
    inputs = batch["src"]
    targets = batch["tgt"]
    input_encodings = tokenizer.encode_batch(inputs)
    target_encodings = tokenizer.encode_batch(targets)
    processed_inputs = [[SOS_token] + enc.ids + [EOS_token] for enc in input_encodings]
    processed_targets = [[SOS_token] + enc.ids + [EOS_token] for enc in target_encodings]
    return {"input_ids": processed_inputs, "target_ids": processed_targets}

print("Tokenizing dataset for the from-scratch model...")
tokenized_datasets = processed_datasets.map(tokenize_and_prepare, batched=True, remove_columns=['src', 'tgt', 'idx'])
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'target_ids'])

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=PAD_token)
    padded_targets = pad_sequence(target_ids, batch_first=True, padding_value=PAD_token)
    return {'input_ids': padded_inputs, 'target_ids': padded_targets}

train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- 4. Training and Evaluation Functions ---
def train_epoch(model, optimizer, loss_fn):
    model.train()
    losses = 0
    for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        src, tgt = batch['input_ids'].to(DEVICE), batch['target_ids'].to(DEVICE)
        tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]
        tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1], DEVICE)
        src_padding_mask, tgt_padding_mask = (src == PAD_token), (tgt_input == PAD_token)
        logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, src_padding_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        if (i + 1) % ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
        losses += loss.item() * ACCUMULATION_STEPS
    return losses / len(list(train_dataloader))

def evaluate(model, loss_fn):
    model.eval()
    losses = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            src, tgt = batch['input_ids'].to(DEVICE), batch['target_ids'].to(DEVICE)
            tgt_input, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = generate_square_subsequent_mask(tgt_input.shape[1], DEVICE)
            src_padding_mask, tgt_padding_mask = (src == PAD_token), (tgt_input == PAD_token)
            logits = model(src, tgt_input, src_padding_mask, tgt_padding_mask, src_padding_mask, tgt_mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
    return losses / len(list(val_dataloader))

# --- 5. Main Training Block ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available(): print(f"PyTorch sees GPU: {torch.cuda.get_device_name(0)}")

    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    best_val_loss = float('inf')
    
    print("\nStarting training on Samanantar...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, loss_fn)
        val_loss = evaluate(model, loss_fn)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")