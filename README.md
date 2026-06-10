# Character-Level GPT From Scratch

A complete implementation of a **GPT-style Transformer Language Model** built from scratch using **PyTorch**. This project demonstrates the core concepts behind modern Large Language Models (LLMs) including self-attention, positional embeddings, transformer blocks, autoregressive text generation, and language model training on OpenWebText data.

---

## Features

* Character-level tokenization
* Transformer architecture
* Multi-Head Self Attention
* Positional Embeddings
* Feed Forward Networks
* Layer Normalization
* Residual Connections
* AdamW Optimizer
* Learning Rate Scheduling
* Gradient Clipping
* Autoregressive Text Generation
* Training / Validation Split
* OpenWebText Dataset Support
* Parallel Data Processing Pipeline

---

## Project Structure

```text
.
├── data-extract.py      # Dataset preprocessing script
├── v1.ipynb             # GPT model implementation and training  
├── vocab.txt            # Character vocabulary
└── README.md
```

---

## Architecture

The model follows the standard GPT pipeline:

```text
Input Text
     │
     ▼
Character Tokenization
     │
     ▼
Token Embeddings
     │
     ▼
Positional Embeddings
     │
     ▼
Transformer Blocks
     │
     ├── Multi-Head Attention
     ├── Feed Forward Network
     ├── Residual Connections
     └── Layer Normalization
     │
     ▼
Linear Projection
     │
     ▼
Next Character Prediction
```

---

## Dataset Processing

The preprocessing pipeline:

1. Reads compressed `.xz` files from OpenWebText
2. Extracts raw text
3. Creates training and validation datasets
4. Generates a character vocabulary
5. Supports parallel processing using multiple CPU cores

Dataset split:

* 90% Training
* 10% Validation

For faster experimentation, only 1% of the dataset is sampled during testing.

---

## Transformer Components

### Token Embeddings

Converts characters into dense vector representations.

### Positional Embeddings

Provides sequence order information to the model.

### Self-Attention

Allows each token to focus on relevant tokens in the context window.

### Multi-Head Attention

Multiple attention heads learn different relationships within text.

### Feed Forward Network

Processes information after attention computations.

### Residual Connections

Improves gradient flow and training stability.

### Layer Normalization

Normalizes activations for stable optimization.

---

## Training

The model is trained using:

* Cross Entropy Loss
* AdamW Optimizer
* Learning Rate Warmup
* Cosine Learning Rate Decay
* Gradient Clipping

Training objective:

```text
Given previous characters,
predict the next character.
```

Example:

```text
Input : hello worl
Target: d
```

---

## Text Generation

After training, the model generates text autoregressively.

Example:

```text
Prompt:
Hello

Generated:
Hello, how are you doing today...
```

Generation supports:

* Temperature Sampling
* Top-K Sampling

---

## Technologies Used

* Python
* PyTorch
* NumPy
* OpenWebText Dataset
* tqdm
* concurrent.futures
* lzma

---

## Learning Outcomes

This project helped me understand:

* Transformer Architecture
* GPT Models
* Self-Attention Mechanism
* Language Modeling
* Deep Learning Training Pipelines
* Text Generation
* Dataset Engineering
* PyTorch Model Development

---

## Future Improvements

* Byte Pair Encoding (BPE)
* Word-Level Tokenization
* Larger Context Windows
* Mixed Precision Training
* Checkpoint Saving & Loading
* GPU Distributed Training
* Fine-Tuning Support
* Chat Interface

---

## References

* Attention Is All You Need
* GPT (Generative Pretrained Transformer)
* freeCodeCamp-youtube
* PyTorch Documentation
* OpenWebText Dataset

---

## Author

Krishna Pathak

Built as a learning project to understand and implement GPT-style language models from scratch.
