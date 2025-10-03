# English-to-Telugu Neural Machine Translation & SOTA Benchmarking

This project is an end-to-end implementation and comparative analysis of a Transformer-based system for translating English to Telugu, built using PyTorch and the Hugging Face ecosystem.

## Project Overview

This project follows the complete lifecycle of an NMT system:
1.  **Baseline:** A from-scratch Transformer was built and trained on the `opus100` dataset.
2.  **Scaling:** The from-scratch model was scaled up and re-trained on a 1-million-sentence subset of the AI4Bharat Samanantar corpus.
3.  **Benchmarking:** A state-of-the-art NLLB model was fine-tuned on the same large dataset to provide a quantitative comparison against the from-scratch approach.

### Key Features
* **Custom Model:** A 6-layer Encoder-Decoder Transformer built from scratch in PyTorch.
* **SOTA Fine-Tuning:** Fine-tuned the `facebook/nllb-200-distilled-600M` model.
* **Data Pipeline:** Handled multiple datasets (`opus100`, `samanantar`), including training a custom BPE tokenizer from scratch for the project-specific corpus.
* **Evaluation:** Used the BLEU score for quantitative analysis and compared greedy vs. beam search for qualitative analysis.

## Results & Analysis

The final models were evaluated on a held-out test set to compare their performance.

### Quantitative Results (BLEU Score)

| Model | Training Method | Dataset (1M pairs) | Final BLEU Score |
| :--- | :--- | :--- | :--- |
| **Seq2Seq Transformer**| From Scratch | Samanantar | **[Your 20.56 Score]** |
| **NLLB-600M** | Fine-Tuning | Samanantar | **[Your 35.93 Score]** |

### Conclusion
The results clearly demonstrate the power of transfer learning. The fine-tuned NLLB model significantly outperformed the from-scratch model, proving that leveraging large pre-trained models is the most effective approach for achieving state-of-the-art results.

## How to Run

1.  Clone the repository and navigate into the directory.
2.  Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate nmt
    ```
3.  **Data Preparation:** Run the data preparation script to download and process the data.
    ```bash
    python samantar_data_prep.py
    ```
4.  **Training:** Run either the from-scratch training or the fine-tuning script.
    ```bash
    # To train the from-scratch model
    python train_samanantar.py

    # To fine-tune the NLLB model
    python finetune_nllb.py
    ```