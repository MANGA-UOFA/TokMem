# Atomic Memory Recall

Atomic task learning using reserved special tokens as task identifiers. Each token encodes a specific task's procedural knowledge.

## Dataset

The project uses the Natural Instructions (Super-NaturalInstructions) dataset. For instructions on how to download and set up the dataset, please see the [Dataset README](natural-instructions-2.8/README.md).

## Configuration

The primary entry point is `main_tokmem.sh`. You can configure the training and evaluation by modifying the following key arguments in the script:

- `--num_tasks`: Total number of tasks to load from the Natural Instructions dataset for training and testing.
- `--model_name`: The transformer model to use. Supports Llama 3 and Qwen 2.5 models (e.g., `meta-llama/Llama-3.2-3B-Instruct` or `Qwen/Qwen2.5-7B-Instruct`).
- `--train_size`, `--val_size`, `--test_size`: Number of instances per task for training, validation, and testing.

## Usage

### TokMem
Main experiment using special tokens:
```bash
bash main_tokmem.sh
```

### LoRA Baseline
Sequential training with LoRA and optional continual replay:
```bash
bash main_lora_baseline.sh
```
