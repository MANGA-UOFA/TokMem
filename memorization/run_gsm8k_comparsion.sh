#!/bin/bash
# =============================================================================
# CONFIGURATION SECTION - Modify these variables as needed
# =============================================================================

# Hardware configuration
CUDA_DEVICE="6"

# Model configuration
MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
PROMPT_POSITION="infix" # infix: the task identifier is in the middle of the prompt, prefix: the task identifier is at the beginning of the prompt
USE_CHAT_TEMPLATE="--use_chat_template" # use the chat template for the prompt
BATCH_SIZE=4
LEARNING_RATE=5e-3

# Test configuration
PROMPT_LENGTH=1  # Fixed prompt length
EPOCHS=3  # Number of training epochs
TRAINING_SAMPLES=(100 500 1000 2000 5000)  # Different numbers of training samples to test
RESULTS_DIR="results"
SLEEP_BETWEEN_RUNS=3

# Log file configuration
LOG_PREFIX="training_samples_test"
INCLUDE_TIMESTAMP=true

# =============================================================================
# SCRIPT EXECUTION - No need to modify below this line
# =============================================================================

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Generate timestamp for log files if enabled
if [ "$INCLUDE_TIMESTAMP" = true ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    TIMESTAMP_SUFFIX="_${TIMESTAMP}"
else
    TIMESTAMP_SUFFIX=""
fi

# Extract model name for log file (remove path and special characters)
MODEL_SHORT=$(echo "$MODEL_NAME" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9._-]/_/g')

echo "Starting training samples test..."
echo "Testing training sample sizes: ${TRAINING_SAMPLES[@]}"
echo "Model: $MODEL_NAME"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Prompt Position: $PROMPT_POSITION"
echo "Fixed Prompt Length: $PROMPT_LENGTH"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Results Directory: $RESULTS_DIR"
echo "----------------------------------------"

# Function to run training with specific number of training samples
run_training() {
    local num_samples=$1
    local log_file="${RESULTS_DIR}/${LOG_PREFIX}_${MODEL_SHORT}_pos-${PROMPT_POSITION}_bs-${BATCH_SIZE}_lr-${LEARNING_RATE}_ep-${EPOCHS}_plen-${PROMPT_LENGTH}_train-${num_samples}${TIMESTAMP_SUFFIX}.log"
    
    echo "Running training with $num_samples training samples"
    echo "Log file: $log_file"
    
    python main_gsm8k.py \
        --prompt_position "$PROMPT_POSITION" \
        $USE_CHAT_TEMPLATE \
        --batch_size "$BATCH_SIZE" \
        --lr "$LEARNING_RATE" \
        --epochs "$EPOCHS" \
        --prompt_length "$PROMPT_LENGTH" \
        --train_size "$num_samples" \
        --model_name "$MODEL_NAME" \
        2>&1 | tee "$log_file"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✅ Training completed successfully for $num_samples training samples"
    else
        echo "❌ Training failed for $num_samples training samples"
    fi
    
    echo "----------------------------------------"
}

# Main execution loop
run_count=0
for num_samples in "${TRAINING_SAMPLES[@]}"; do
    echo "Starting run $((++run_count))/${#TRAINING_SAMPLES[@]}"
    run_training $num_samples
    
    # Add a configurable delay between runs
    sleep $SLEEP_BETWEEN_RUNS
done

echo "Training samples test completed!"
echo "Results saved in: $RESULTS_DIR/"

# Generate a simple summary
echo "Training Samples Summary:"
echo "========================="
for num_samples in "${TRAINING_SAMPLES[@]}"; do
    log_file="${RESULTS_DIR}/${LOG_PREFIX}_${MODEL_SHORT}_pos-${PROMPT_POSITION}_bs-${BATCH_SIZE}_lr-${LEARNING_RATE}_ep-${EPOCHS}_plen-${PROMPT_LENGTH}_train-${num_samples}${TIMESTAMP_SUFFIX}.log"
    if [ -f "$log_file" ]; then
        echo "Training Samples $num_samples:"
        # Extract final test accuracy if available
        grep "Final test accuracy:" "$log_file" | tail -1
        # Extract best validation accuracy if available
        grep "New best accuracy:" "$log_file" | tail -1
        echo ""
    fi
done 