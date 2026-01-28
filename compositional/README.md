# Compositional Memory Recall

This directory contains the compositional memory recall experiments.

## Experimental Setup

The experiments use tools extracted from the **XLAM** aka. APIGen dataset. A total of 100 tools are used:
- **Tools 1-50**: Used as **adaptation tools**. In the TokMem method, these tools are used in the first round to adapt the model to the tool-calling format and environment.
- **Tools 51-100**: Used for **evaluation** across all methods. Performance is measured on these tools to assess how well the model learns and retains new tool-calling capabilities.

## Methods & Scripts

### 1. TokMem (Main Method)
`run_n_rounds_main.sh`
This script implements the **TokMem** approach. It performs sequential training where the first round (tools 1-50) serves as an adaptation phase.
- **Key Feature**: Uses a specialized training loop in `main_sequential.py` that can freeze adapters after the initial adaptation to maintain stable memory while learning new tool distributions.
- **Usage**:
  ```bash
  bash run_n_rounds_main.sh
  ```

### 2. LoRA Baseline
`run_n_rounds_lora.sh`
This script provides a standard sequential fine-tuning baseline using LoRA.
- **Key Feature**: It uses standard LoRA fine-tuning (with optional reinitialization or replay buffers) via `lora_sequential.py`. 
- **Usage**:
  ```bash
  bash run_n_rounds_lora.sh
  ```

### 3. ICL Baseline
`icl_baseline.sh`
This script evaluates the model's zero-shot or few-shot capabilities using In-Context Learning (ICL).
- **Key Feature**: Evaluates the model directly on tools 51-100 without any fine-tuning. It supports RAG-based tool retrieval to fit relevant tool descriptions into the context window.
- **Usage**:
  ```bash
  bash icl_baseline.sh
  ```

## Key Components

- `xlam_datasets.py`: Handles data generation, tool extraction, and multi-tool call composition from the XLAM source.
- `model.py` & `training.py`: Core logic for model initialization and the training loops.
- `tool_retrieval.py`: Implementation of RAG for selecting relevant tools during ICL or evaluation.
- `replay_buffer.py`: Management of historical samples to mitigate catastrophic forgetting during sequential learning.
