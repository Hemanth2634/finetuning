# Fine-Tuning Gemma-2B
This Project focuses on supervised Fine-Tuning(SFT) of the Gemma-2b llm on the Abirate/english_quotes dataset.

To achieve parameter-efficient and memory-efficient training, we leverage:

Supervised Fine-Tuning (SFTTrainer)

LoRA (Low-Rank Adaptation)

Quantization-aware loading (4-bit / 8-bit)

HuggingFace Transformers + PEFT stack

# What is LoRA?

LoRA (Low-Rank Adaptation) fine-tunes LLMs by injecting low-rank trainable matrices into specific layers (usually attention projections), instead of updating all model parameters.
we have experimented on the rank-8


# What is Quantization?

Quantization reduces the numerical precision of model weights (e.g., from FP16 → INT8 or INT4), significantly reducing memory usage.

# Training Stack

Transformers – Model and tokenizer

PEFT – LoRA integration

TRL – SFTTrainer

BitsAndBytes – Quantization

Accelerate – Distributed & mixed precision training
