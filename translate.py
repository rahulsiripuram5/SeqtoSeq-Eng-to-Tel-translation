import torch
import torch.nn.functional as F

from model import Seq2SeqTransformer, generate_square_subsequent_mask
from tokenizers import Tokenizer

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "eng-telugu-bpe-tokenizer.json"
MODEL_PATH = "best_model.pt"
MAX_LEN = 150

# Model Hyperparameters
SRC_VOCAB_SIZE = 30000
TGT_VOCAB_SIZE = 30000
EMB_SIZE = 1024
NHEAD = 16
FFN_HID_DIM = 4096
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

# --- Load Tokenizer and Model ---
print("Loading model and tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# Define special token IDs
SOS_token = tokenizer.token_to_id("[SOS]")
EOS_token = tokenizer.token_to_id("[EOS]")
PAD_token = tokenizer.token_to_id("[PAD]")

def greedy_decode(model, src_tensor, max_len, device):
    src_padding_mask = (src_tensor == PAD_token).to(device)
    with torch.no_grad():
        memory = model.encode(src=src_tensor, src_mask=None, src_padding_mask=src_padding_mask)
    
    ys = torch.ones(1, 1).fill_(SOS_token).type(torch.long).to(device)
    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_seq_len = ys.shape[1]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
            tgt_padding_mask = (ys == PAD_token).to(device)
            out = model.decode(ys, memory, tgt_mask, tgt_padding_mask)
            prob = model.generator(out[:, -1])
        
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor).fill_(next_word)], dim=1)
        if next_word == EOS_token:
            break
    return ys

def beam_search_decode(model, src_tensor, beam_width, max_len, device):
    src_padding_mask = (src_tensor == PAD_token).to(device)
    with torch.no_grad():
        memory = model.encode(src=src_tensor, src_mask=None, src_padding_mask=src_padding_mask)

    # Start with SOS token and a score of 0
    sequences = [[torch.ones(1, 1).fill_(SOS_token).long().to(device), 0.0]]

    for _ in range(max_len - 1):
        all_candidates = []
        for seq_tensor, score in sequences:
            if seq_tensor[0, -1].item() == EOS_token:
                all_candidates.append([seq_tensor, score])
                continue

            with torch.no_grad():
                tgt_seq_len = seq_tensor.shape[1]
                tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
                tgt_padding_mask = (seq_tensor == PAD_token).to(device)
                out = model.decode(seq_tensor, memory, tgt_mask, tgt_padding_mask)
                prob = model.generator(out[:, -1])
            
            # Use log probabilities for numerical stability
            log_probs = F.log_softmax(prob, dim=-1)
            top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=1)

            for i in range(beam_width):
                next_token = top_k_indices[0][i].reshape(1, 1)
                new_score = score + top_k_log_probs[0][i].item()
                new_seq = torch.cat([seq_tensor, next_token], dim=1)
                all_candidates.append([new_seq, new_score])

        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]

        # Break if all top sequences have ended
        if all(s[0][0, -1].item() == EOS_token for s in sequences):
            break
            
    return sequences[0][0] # Return the sequence with the highest score

def translate(model: torch.nn.Module, src_sentence: str, beam_width: int = 5):
    model.eval()
    src_tokens = tokenizer.encode(src_sentence).ids
    src_tensor = torch.LongTensor([SOS_token] + src_tokens + [EOS_token]).unsqueeze(0).to(DEVICE)
    
    # --- Greedy Decoding ---
    greedy_result_tensor = greedy_decode(model, src_tensor, max_len=MAX_LEN, device=DEVICE)
    greedy_translation = " ".join([tokenizer.id_to_token(i) for i in greedy_result_tensor.squeeze().tolist()][1:]).replace(" ##", "")
    
    # --- Beam Search Decoding ---
    beam_result_tensor = beam_search_decode(model, src_tensor, beam_width=beam_width, max_len=MAX_LEN, device=DEVICE)
    beam_translation = " ".join([tokenizer.id_to_token(i) for i in beam_result_tensor.squeeze().tolist()][1:]).replace(" ##", "")
    
    return greedy_translation, beam_translation

# --- Main execution block ---
if __name__ == "__main__":
    sentences = [
        "Hello, how are you?",
        "What is your name?",
        "This is a test of the translation system.",
        "The weather in Bengaluru is pleasant today."
    ]
    
    for sentence in sentences:
        print("-" * 30)
        print(f"Input: {sentence}")
        greedy, beam = translate(model, sentence, beam_width=5)
        print(f"  Greedy -> {greedy}")
        print(f"  Beam (k=5) -> {beam}")