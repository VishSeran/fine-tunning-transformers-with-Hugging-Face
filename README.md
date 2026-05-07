# Fine-Tuning Transformers with Hugging Face

A practical, lab-style Jupyter notebook that walks you through the full workflow of loading, fine-tuning, and evaluating pretrained Large Language Models (LLMs) using PyTorch and the Hugging Face ecosystem.

## Overview

This notebook covers two real-world fine-tuning scenarios:

1. **Supervised Fine-Tuning with PyTorch (from scratch)** — Fine-tune a BERT model for sentiment classification on the Yelp Review dataset, building the training loop manually without relying on the Hugging Face `Trainer` class.
2. **Conversational Model Fine-Tuning with SFTTrainer** — Fine-tune Meta's `facebook/opt-350m` decoder model on the OpenAssistant Guanaco dataset using the `SFTTrainer` from the `trl` library to build a question-answering conversational agent.

## What You'll Learn

- Load and run inference with pretrained LLMs from Hugging Face Hub
- Tokenize and preprocess datasets using `AutoTokenizer`
- Build a custom PyTorch training loop with `AdamW` optimizer and a learning rate scheduler
- Evaluate model performance using `torchmetrics`
- Use `SFTTrainer` and `DataCollatorForCompletionOnlyLM` for efficient supervised fine-tuning
- Compare pretrained vs. fine-tuned model outputs on domain-specific prompts

## Models & Datasets

| Component | Details |
|---|---|
| Classification model | `bert-base-cased` (HuggingFace) |
| Generative model | `facebook/opt-350m` |
| Classification dataset | `yelp_review_full` (6.9M reviews, 5-star ratings) |
| Conversational dataset | `timdettmers/openassistant-guanaco` |

## Tech Stack

- Python, Jupyter Notebook
- PyTorch
- Hugging Face `transformers`, `datasets`, `trl`
- `torchmetrics`, `matplotlib`, `tqdm`

## Notebook Structure

```
1. Setup — Install and import dependencies
2. Supervised Fine-Tuning with PyTorch
   ├── Dataset preparation & tokenization
   ├── DataLoader setup
   ├── Custom training loop (AdamW + LR scheduler)
   ├── Evaluation with torchmetrics
   └── Loading a saved checkpoint
3. Exercise: Conversational Model with SFTTrainer
   ├── Load OPT-350M + Guanaco dataset
   ├── Configure DataCollatorForCompletionOnlyLM
   ├── Train with SFTTrainer
   └── Compare pre- vs post-fine-tuning outputs
```


