import torch
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import sacrebleu # Ensure this is the library being used

from model import Seq2SeqTransformer, generate_square_subsequent_mask

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER_PATH = "eng-telugu-bpe-tokenizer.json"
MODEL_PATH = "best_model.pt"
MAX_LEN = 150
BEAM_WIDTH = 5

# Model Hyperparameters
SRC_VOCAB_SIZE = 30000
TGT_VOCAB_SIZE = 30000
EMB_SIZE = 1024
NHEAD = 16
FFN_HID_DIM = 4096
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

def beam_search_decode(model, src_tensor, beam_width, max_len, device, sos_token, eos_token, pad_token):
    src_padding_mask = (src_tensor == pad_token).to(device)
    with torch.no_grad():
        memory = model.encode(src=src_tensor, src_mask=None, src_padding_mask=src_padding_mask)

    sequences = [[torch.ones(1, 1).fill_(sos_token).long().to(device), 0.0]]
    for _ in range(max_len - 1):
        all_candidates = []
        for seq_tensor, score in sequences:
            if seq_tensor[0, -1].item() == eos_token:
                all_candidates.append([seq_tensor, score])
                continue
            with torch.no_grad():
                tgt_seq_len = seq_tensor.shape[1]
                tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device)
                tgt_padding_mask = (seq_tensor == pad_token).to(device)
                out = model.decode(seq_tensor, memory, tgt_mask, tgt_padding_mask)
                prob = model.generator(out[:, -1])
            log_probs = F.log_softmax(prob, dim=-1)
            top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width, dim=1)
            for i in range(beam_width):
                next_token = top_k_indices[0][i].reshape(1, 1)
                new_score = score + top_k_log_probs[0][i].item()
                new_seq = torch.cat([seq_tensor, next_token], dim=1)
                all_candidates.append([new_seq, new_score])
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        sequences = ordered[:beam_width]
        if all(s[0][0, -1].item() == eos_token for s in sequences):
            break
            
    return sequences[0][0]

def main():
    print("--- Starting Evaluation ---")
    print(f"Using device: {DEVICE}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    sos_token = tokenizer.token_to_id("[SOS]")
    eos_token = tokenizer.token_to_id("[EOS]")
    pad_token = tokenizer.token_to_id("[PAD]")

    print("Loading test dataset...")
    test_dataset = load_dataset("opus100", "en-te", split='test')

    # ...existing code...
    predictions = []
    references = []

    for item in tqdm(test_dataset, desc="Generating translations"):
        src_sentence = item['translation']['en']
        ref_sentence = item['translation']['te']
        
        src_tokens = tokenizer.encode(src_sentence).ids
        src_tensor = torch.LongTensor([sos_token] + src_tokens + [eos_token]).unsqueeze(0).to(DEVICE)
        result_tensor = beam_search_decode(model, src_tensor, BEAM_WIDTH, MAX_LEN, DEVICE, sos_token, eos_token, pad_token)
        
        pred_tokens = result_tensor.squeeze().tolist()
        pred_sentence = tokenizer.decode(pred_tokens[1:])  # Skip SOS
        
        predictions.append(pred_sentence)
        references.append(ref_sentence)  # <-- Now a list of strings

    print("\nCalculating BLEU score...")
    bleu = sacrebleu.corpus_bleu(predictions, [references])  # <-- Note the brackets

    print("\n--- Evaluation Complete ---")
    print(f"Model: {MODEL_PATH}")
    print(f"BLEU Score: {bleu.score:.2f}")
    

if __name__ == "__main__":
    main()