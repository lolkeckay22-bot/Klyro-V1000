# Klyro-V1000

A small decoder-only Transformer language model trained from scratch on English and Russian text.

## Overview

- Model: Decoder-only Transformer with ~30‑50M parameters
- Vocabulary: ~16k tokens (custom BPE tokenizer)
- Context length: 1024
- Languages: English (~50%) and Russian (~50%)
- Training: Causal language modeling (next‑token prediction)
- Infrastructure: Designed for execution on Kaggle GPUs; checkpoints stored on Hugging Face Hub

## Repository Structure

```
Klyro-V1000/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ config/
│   └─ train_config.yaml
├─ src/
│   ├─ model.py
│   ├─ tokenizer.py
│   ├─ dataset.py
│   ├─ train.py
│   ├─ inference.py
│   ├─ hf_utils.py
│   └─ checkpoint.py
├─ notebooks/
│   └─ Klyro_V1000_Training.ipynb
└─ tests/
    ├─ test_model.py
    ├─ test_tokenizer.py
    └─ test_checkpoint.py
```

## Setup (local development only)

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run tiny smoke tests to verify code integrity:
   ```bash
   pytest tests/
   ```

**Important:** Actual model training, tokenizer training, and dataset processing must be performed inside a Kaggle notebook (see `notebooks/Klyro_V1000_Training.ipynb`). Local execution is limited to static checks and unit tests.

## Kaggle Training Notebook

The notebook `notebooks/Klyro_V1000_Training.ipynb` implements the full training pipeline:

1. GPU detection and VRAM reporting
2. Dependency installation
3. Configuration loading
4. Secure retrieval of `HF_TOKEN` from Kaggle Secrets
5. Hugging Face Hub connection and repo check
6. Fresh training or resume from latest checkpoint
7. Streaming load of English & Russian datasets
8. Preprocessing
9. Custom tokenizer training (or loading)
10. Model creation and parameter count output
11. Training loop with mixed precision, gradient accumulation, clipping, LR scheduler, warmup, checkpointing, and validation
12. Periodic checkpoint upload to Hugging Face
13. Final model saving and upload to Hugging Face
14. Sample generation in Russian and English

## Configuration

All major hyperparameters are defined in `config/train_config.yaml`:

- Model dimensions (layers, hidden size, attention heads)
- Vocabulary size
- Context length
- Batch size, gradient accumulation
- Learning rate, warmup steps
- Maximum training steps / tokens
- Checkpoint & evaluation intervals
- Russian/English dataset ratio
- Random seed
- `HF_REPO_ID`
- `START_FROM_SCRATCH` / `RESUME_FROM_CHECKPOINT` flags

## Hugging Face Integration

- Repository: set via `HF_REPO_ID` (e.g., `username/Klyro-V1000`)
- Token: must be supplied via Kaggle Secret `HF_TOKEN` (never hard‑coded)
- Utilities in `src/hf_utils.py` handle upload/download of tokenizer, checkpoints, final model, and config.

## License

This project is released under the MIT License.

## Disclaimer

Local machine usage is restricted to development and testing only. No full‑scale training, tokenizer training, or large dataset downloads should be performed locally. All resource‑intensive operations must be executed in the Kaggle environment.