#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Parameters
TOOLS="51-100"
SAMPLES_PER_TOOL=50
TRAIN_SIZE=5000
TEST_SIZE=500
TRAIN_MAX_CALLS=4
TEST_MAX_CALLS=4
MODEL="meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE=16

# RAG Configuration
USE_RAG=true  # Set to true to enable RAG, false to use all tools
RETRIEVAL_K=5  # Number of tools to retrieve when RAG is enabled

# Create data directories
mkdir -p data/training data/test

# Data file paths
TRAIN_FILE="data/training/function_calling_train_tools${TOOLS}_${TRAIN_MAX_CALLS}calls.json"
TEST_FILE="data/test/function_calling_test_tools${TOOLS}_${TEST_MAX_CALLS}calls.json"
TOOL_DESC_FILE="data/tool_descriptions_tools${TOOLS}.json"

echo "ICL Baseline Evaluation"
echo "======================="
echo "Configuration:"
echo "  - Tools: $TOOLS"
echo "  - Model: $MODEL"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max samples: $MAX_SAMPLES"
if [ "$USE_RAG" = true ]; then
    echo "  - RAG: Enabled (retrieving top-$RETRIEVAL_K tools)"
else
    echo "  - RAG: Disabled (using all tools)"
fi
echo ""

# Step 1: Generate data if needed
if [ -f "$TEST_FILE" ] && [ -f "$TOOL_DESC_FILE" ]; then
    echo "Using existing test data..."
    test_count=$(python -c "import json; print(len(json.load(open('$TEST_FILE'))))" 2>/dev/null)
    echo "✓ Using: $TEST_FILE (${test_count} samples)"
else
    echo "Generating data for tools $TOOLS..."
    
    python xlam_datasets.py \
        --top_k "$TOOLS" \
        --max_samples_per_tool $SAMPLES_PER_TOOL \
        --train_size $TRAIN_SIZE \
        --test_size $TEST_SIZE \
        --train_max_function_calls $TRAIN_MAX_CALLS \
        --test_max_function_calls $TEST_MAX_CALLS \
        --train_multi_tool_ratios "0.5,0.5" \
        --test_multi_tool_ratios "0.5,0.5" \
        --output_dir "data"
    
    if [ $? -ne 0 ]; then
        echo "Error generating data"
        exit 1
    fi
    
    test_count=$(python -c "import json; print(len(json.load(open('$TEST_FILE'))))" 2>/dev/null)
    echo "✓ Generated: $TEST_FILE (${test_count} samples)"
fi

echo ""

# Step 2: Run ICL baseline
echo "Running ICL baseline evaluation..."
echo "=================================="

# Build command based on RAG configuration
CMD="python icl_baseline.py \
    --test_data \"$TEST_FILE\" \
    --tool_descriptions \"$TOOL_DESC_FILE\" \
    --model_name \"$MODEL\" \
    --batch_size $BATCH_SIZE \
    --output \"icl_results_tools${TOOLS}.json\""

# Add RAG flags if enabled
if [ "$USE_RAG" = true ]; then
    CMD="$CMD --use_rag --retrieval_k $RETRIEVAL_K"
fi

# Execute the command
eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ ICL baseline evaluation completed!"
    echo "Results saved to: icl_results_tools${TOOLS}.json"
else
    echo "✗ ICL baseline evaluation failed!"
    exit 1
fi