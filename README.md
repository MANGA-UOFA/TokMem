# TokMem: Tokenized Procedural Memory for LLMs

This repository implements **TokMem**, a method that enables large language models (LLMs) to acquire and recall procedural knowledge through tokenized memory. By freezing the base LM and training only specialized memory token embeddings, TokMem provides a modular and efficient way to learn new tasks without catastrophic forgetting.

## Key Features

- **Frozen Base LM**: Only memory token embeddings and lightweight projections are trained, preserving the original model's capabilities.
- **Zero Interference**: New tasks or tools are assigned unique tokens, preventing them from interfering with previously learned knowledge.
- **High Efficiency**: Significantly smaller parameter updates compared to full fine-tuning or LoRA.
- **Modular & Scalable**: Easily add new tasks or tools by simply training new token embeddings without retraining the entire system.
- **Comprehensive Baselines**: Includes LoRA fine-tuning, replay memory, in-context learning (ICL), and zero-shot baselines for comparison.

---

## Experimental Tracks

### 1. Atomic Memory Recall (`atomic/`)
Focuses on learning individual, distinct tasks using reserved special tokens as unique task identifiers.
- **Dataset**: [Natural Instructions (v2.8)](https://github.com/allenai/natural-instructions).
- **Goal**: Train task-specific embeddings that encode the procedural knowledge required for hundreds of diverse NLP tasks.
- **Key Methods**:
  - `TokMem`: Training specialized embeddings for each task.
  - `LoRA Baseline`: Sequential training with LoRA and optional experience replay.
- **Quick Start**:
  ```bash
  cd atomic && bash main_tokmem.sh
  ```

### 2. Compositional Memory Recall (`compositional/`)
Evaluates the model's ability to learn and compose multiple tool-calling functions sequentially.
- **Dataset**: **XLAM (APIGen)**. Uses tools 1-50 for adaptation and tools 51-100 for evaluation.
- **Goal**: Assess how well the model can learn new tool-calling capabilities across disjoint training rounds.
- **Key Methods**:
  - `TokMem`: Sequential training with an initial adaptation phase.
  - `LoRA Baseline`: Standard sequential fine-tuning with optional replay buffers.
  - `ICL Baseline`: Zero-shot/Few-shot evaluation with RAG-based tool retrieval.
- **Quick Start**:
  ```bash
  cd compositional && bash run_n_rounds_main.sh
  ```

### 3. Embedding Capacity Ablation (`memorization/`)
Stress tests to compare **TokMem** against **Prefix memory embeddings**.
- **Experiments**:
  - **GSM8K Reasoning**: Testing multi-step reasoning and structured output maintenance.
  - **Memory Stress Test**: Using PG-19 and Fanfics datasets to measure token compression and reconstruction accuracy.
- **Metrics**: Reconstruction accuracy, perplexity.
- **Quick Start**:
  ```bash
  cd memorization && bash run_memorization_comparison.sh
  ```

---

## Getting Started

### Prerequisites
- Python 3.9+
- CUDA-enabled GPU (BF16/FP16 support recommended)
- `pip install -r requirements.txt`

### Supported Models
The codebase is tested for Hugging Face Transformers:
- **Llama 3.1** (8B Instruct)
- **Llama 3.2** (1B, 3B Instruct)
- **Qwen 2.5** (0.5B Instruct)

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2025tokmem,
  title={TokMem: Tokenized Procedural Memory for Large Language Models},
  author={Wu, Zijun and Hao, Yongchang and Mou, Lili},
  journal={arXiv preprint arXiv:2510.00444},
  year={2025}
}
```