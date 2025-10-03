import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import sacrebleu

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "nllb-finetuned-en-te/final_model" 
PROCESSED_DATA_PATH = "processed_samanantar_data"

def main():
    print("--- Starting Evaluation for Fine-Tuned NLLB Model ---")
    
    # 1. Load Model and Tokenizer
    print(f"Loading model and tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    # 2. Load Test Dataset
    print(f"Loading test dataset from {PROCESSED_DATA_PATH}...")
    test_dataset = load_from_disk(PROCESSED_DATA_PATH)['validation']

    # 3. Generate Translations
    predictions = []
    references = []

    for item in tqdm(test_dataset, desc="Generating NLLB translations"):
        src_sentence = item['src']
        ref_sentence = item['tgt']
        
        inputs = tokenizer(src_sentence, return_tensors="pt").to(DEVICE)
        
        # --- THIS IS THE FINAL CORRECTED LINE ---
        # Using the fundamental method to get the token ID
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("tel_Telu"), 
            max_length=150
        )
        
        pred_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

        predictions.append(pred_sentence)
        references.append([ref_sentence])

    # 4. Calculate BLEU Score
    print("\nCalculating BLEU score...")
    bleu = sacrebleu.corpus_bleu(predictions, references)
    
    print("\n--- NLLB Evaluation Complete ---")
    print(f"Model: {MODEL_DIR}")
    print(f"BLEU Score: {bleu.score:.2f}")
    print(f"Signature: {bleu.signature()}")

if __name__ == "__main__":
    main()