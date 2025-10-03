import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "nllb-finetuned-en-te/final_model" 

# --- Load Model and Tokenizer ---
print("Loading fine-tuned NLLB model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

# --- Main Interactive Loop ---
print("\n--- Interactive Translator (Fine-Tuned NLLB Model) ---")
print("Type an English sentence or 'quit' to exit.")
while True:
    src_sentence = input("> ")
    if src_sentence.lower() == 'quit':
        break

    inputs = tokenizer(src_sentence, return_tensors="pt").to(DEVICE)
    
    translated_tokens = model.generate(
        **inputs, 
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"), 
        max_length=150
    )
    
    pred_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

    print(f"Translation: {pred_sentence}")