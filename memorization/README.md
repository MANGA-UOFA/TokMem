# Embedding Capacity Ablation: TokMem vs Prefix Tuning

This directory contains ablation experiments comparing **TokMem** (special token-based memory) and **Prefix Tuning** approaches. These experiments evaluate embedding capacity, memorization performance, and generalization on structured reasoning tasks.

## Key Experiments


### 1. GSM8K Reasoning and Structured Output
The script `run_gsm8k_comparsion.sh` evaluates the performance of TokMem and Prefix Tuning on the GSM8K dataset.
- **Goal**: Test the models' ability to perform multi-step reasoning and maintain structured output formats when using learned task identifiers or prefixes.
- **Configuration**: Compares performance across different training sample sizes and prompt configurations.

### 2. Embedding Capacity Stress Test
Before running the embedding capacity stress tests, download the required datasets (PG-19 and Fanfics chunks):
```bash
cd memorization/memorization_data
bash download_data.sh
cd ..
```

The script `run_memorization_comparison.sh` is designed for stress testing the memorization capacity of the learned embeddings.
- **Goal**: Determine exactly how much information (how many tokens) can be compressed into and recalled from the learned embeddings.
- **Metrics**: It measures reconstruction accuracy, perplexity, and analyzes embedding drift (angle change and cosine similarity) over time.

## Usage

To run the GSM8K comparison:
```bash
cd memorization
bash run_gsm8k_comparsion.sh
```

To run the memorization capacity stress test:
```bash
cd memorization
bash run_memorization_comparison.sh
```