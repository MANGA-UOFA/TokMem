#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Default parameters
NUM_ROUNDS=2
TOOLS_PER_ROUND=50
SAMPLES_PER_TOOL=50
EPOCHS_PER_ROUND=1
TRAIN_MAX_CALLS=4
TEST_MAX_CALLS=4
TRAIN_MAX_CALLS_PER_ROUND="4,4"
TEST_MAX_CALLS_PER_ROUND="4,4"
BATCH_SIZE=2
LR=5e-5
MODEL="meta-llama/Llama-3.2-3B-Instruct"
START_TOOL=1
TRAIN_SIZE=5000
TEST_SIZE=500

REINIT_LORA=true # true: reinitialize the LoRA adapters after each round
EVAL_ALL_PREVIOUS=false
SAVE_CHECKPOINTS=false

# Replay buffer parameters
USE_REPLAY_BUFFER=false
REPLAY_BUFFER_SIZE=1000  # Total samples to keep from all previous rounds
REPLAY_RATIO=0.2         # 20% replay, 80% new data in each batch

# All configuration is done via these variables - no command line arguments

if [ -n "$TRAIN_MAX_CALLS_PER_ROUND" ]; then
    IFS=',' read -ra TRAIN_MAX_CALLS_ARRAY <<< "$TRAIN_MAX_CALLS_PER_ROUND"
    for idx in "${!TRAIN_MAX_CALLS_ARRAY[@]}"; do
        TRAIN_MAX_CALLS_ARRAY[$idx]="${TRAIN_MAX_CALLS_ARRAY[$idx]//[[:space:]]/}"
    done
    TRAIN_MAX_CALLS_PER_ROUND=$(IFS=','; echo "${TRAIN_MAX_CALLS_ARRAY[*]}")
else
    TRAIN_MAX_CALLS_ARRAY=()
fi

if [ -n "$TEST_MAX_CALLS_PER_ROUND" ]; then
    IFS=',' read -ra TEST_MAX_CALLS_ARRAY <<< "$TEST_MAX_CALLS_PER_ROUND"
    for idx in "${!TEST_MAX_CALLS_ARRAY[@]}"; do
        TEST_MAX_CALLS_ARRAY[$idx]="${TEST_MAX_CALLS_ARRAY[$idx]//[[:space:]]/}"
    done
    TEST_MAX_CALLS_PER_ROUND=$(IFS=','; echo "${TEST_MAX_CALLS_ARRAY[*]}")
else
    TEST_MAX_CALLS_ARRAY=()
fi

# Build rounds specification
ROUNDS=""
for ((i=0; i<NUM_ROUNDS; i++)); do
    start=$((START_TOOL + i * TOOLS_PER_ROUND))
    end=$((start + TOOLS_PER_ROUND - 1))
    
    if [ $i -gt 0 ]; then
        ROUNDS="${ROUNDS},"
    fi
    ROUNDS="${ROUNDS}${start}-${end}:${EPOCHS_PER_ROUND}"
done

echo "LoRA Baseline - Flexible N-Round Sequential Training"
echo "=================================================="
echo "Configuration:"
echo "  - Number of rounds: $NUM_ROUNDS"
echo "  - Tools per round: $TOOLS_PER_ROUND"
echo "  - Max samples per tool: $SAMPLES_PER_TOOL"
echo "  - Epochs per round: $EPOCHS_PER_ROUND"
echo "  - Starting tool: $START_TOOL"
echo "  - Training rounds: $ROUNDS"
echo "  - Model: $MODEL"
echo "  - Eval all previous test sets: $EVAL_ALL_PREVIOUS"
echo "  - Reinit LoRA after each round: $REINIT_LORA"
if [ "$REINIT_LORA" = true ]; then
    echo "  - Method: Standard LoRA fine-tuning with reinitialization"
else
    echo "  - Method: Standard LoRA fine-tuning"
fi
echo "  - Save checkpoints: $SAVE_CHECKPOINTS"
if [ ${#TRAIN_MAX_CALLS_ARRAY[@]} -gt 0 ]; then
    echo "  - Train max calls per round: $TRAIN_MAX_CALLS_PER_ROUND"
else
    echo "  - Train max calls: $TRAIN_MAX_CALLS"
fi
if [ ${#TEST_MAX_CALLS_ARRAY[@]} -gt 0 ]; then
    echo "  - Test max calls per round: $TEST_MAX_CALLS_PER_ROUND"
else
    echo "  - Test max calls: $TEST_MAX_CALLS"
fi
echo ""

# Create checkpoint directory name with configuration
CHECKPOINT_DIR="checkpoints_lora_${NUM_ROUNDS}rounds_${TOOLS_PER_ROUND}tools_${SAMPLES_PER_TOOL}samples"
SAVE_CHECKPOINTS=true

# Create data directories
mkdir -p data/training data/test

# Step 1: Generate data for all rounds (same as main approach)
echo "Step 1: Generating data for all rounds..."
echo "========================================="

for ((i=0; i<NUM_ROUNDS; i++)); do
    start=$((START_TOOL + i * TOOLS_PER_ROUND))
    end=$((start + TOOLS_PER_ROUND - 1))
    tools="${start}-${end}"
    
    if [ ${#TRAIN_MAX_CALLS_ARRAY[@]} -gt 0 ]; then
        if [ $i -lt ${#TRAIN_MAX_CALLS_ARRAY[@]} ]; then
            train_max_calls=${TRAIN_MAX_CALLS_ARRAY[$i]}
        else
            train_max_calls=${TRAIN_MAX_CALLS_ARRAY[$(( ${#TRAIN_MAX_CALLS_ARRAY[@]} - 1 ))]}
        fi
    else
        train_max_calls=$TRAIN_MAX_CALLS
    fi

    if [ ${#TEST_MAX_CALLS_ARRAY[@]} -gt 0 ]; then
        if [ $i -lt ${#TEST_MAX_CALLS_ARRAY[@]} ]; then
            test_max_calls=${TEST_MAX_CALLS_ARRAY[$i]}
        else
            test_max_calls=${TEST_MAX_CALLS_ARRAY[$(( ${#TEST_MAX_CALLS_ARRAY[@]} - 1 ))]}
        fi
    else
        test_max_calls=$TEST_MAX_CALLS
    fi

    # Check if data files already exist
    train_file="data/training/function_calling_train_tools${tools}_${train_max_calls}calls.json"
    test_file="data/test/function_calling_test_tools${tools}_${test_max_calls}calls.json"
    
    if [ -f "$train_file" ] && [ -f "$test_file" ]; then
        echo ""
        echo "Round $((i+1)): Data files already exist for tools $tools, skipping generation..."
        train_count=$(python -c "import json; print(len(json.load(open('$train_file'))))" 2>/dev/null)
        test_count=$(python -c "import json; print(len(json.load(open('$test_file'))))" 2>/dev/null)
        echo "✓ Using existing: $train_file (${train_count} samples)"
        echo "✓ Using existing: $test_file (${test_count} samples)"
    else
        echo ""
        echo "Round $((i+1)): Generating data for tools $tools..."
        
        python xlam_datasets.py \
            --top_k "$tools" \
            --max_samples_per_tool $SAMPLES_PER_TOOL \
            --train_size $TRAIN_SIZE \
            --test_size $TEST_SIZE \
            --train_max_function_calls $train_max_calls \
            --test_max_function_calls $test_max_calls \
            --train_multi_tool_ratios "0.5,0.5" \
            --test_multi_tool_ratios "0.5,0.5" \
            --output_dir "data"
        
        if [ $? -ne 0 ]; then
            echo "Error generating data for tools $tools"
            exit 1
        fi
        
        # Verify files were created
        if [ -f "$train_file" ] && [ -f "$test_file" ]; then
            train_count=$(python -c "import json; print(len(json.load(open('$train_file'))))" 2>/dev/null)
            test_count=$(python -c "import json; print(len(json.load(open('$test_file'))))" 2>/dev/null)
            echo "✓ Generated: $train_file (${train_count} samples)"
            echo "✓ Generated: $test_file (${test_count} samples)"
        else
            echo "Error: Failed to create data files for tools $tools"
            exit 1
        fi
    fi
done

echo ""
echo "Data generation complete!"
echo ""

# Step 2: Run LoRA sequential training
echo "Step 2: Running ${NUM_ROUNDS}-round LoRA sequential training..."
echo "============================================================"

# Build command with optional eval_all_previous
CMD="python lora_sequential.py \
    --training_rounds \"$ROUNDS\" \
    --batch_size $BATCH_SIZE \
    --model_name \"$MODEL\" \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules \"q_proj,v_proj\" \
    --eval_after_each_round \
    --checkpoint_dir \"$CHECKPOINT_DIR\" \
    --lr $LR \
    --eval_batch_size 16 \
    --seed 42"

if [ ${#TRAIN_MAX_CALLS_ARRAY[@]} -gt 0 ]; then
    CMD="$CMD --train_max_function_calls_per_round $TRAIN_MAX_CALLS_PER_ROUND"
else
    CMD="$CMD --train_max_function_calls $TRAIN_MAX_CALLS"
fi

if [ ${#TEST_MAX_CALLS_ARRAY[@]} -gt 0 ]; then
    CMD="$CMD --test_max_function_calls_per_round $TEST_MAX_CALLS_PER_ROUND"
else
    CMD="$CMD --test_max_function_calls $TEST_MAX_CALLS"
fi

if [ "$SAVE_CHECKPOINTS" = true ]; then
    CMD="$CMD --save_checkpoints"
fi

if [ "$EVAL_ALL_PREVIOUS" = true ]; then
    CMD="$CMD --eval_all_previous"
fi

if [ "$REINIT_LORA" = true ]; then
    CMD="$CMD --reinit_lora_after_each_round"
fi

if [ "$USE_REPLAY_BUFFER" = true ]; then
    CMD="$CMD --use_replay_buffer --replay_buffer_size $REPLAY_BUFFER_SIZE --replay_ratio $REPLAY_RATIO"
fi

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ LoRA training completed successfully!"
    echo ""
    echo "Results:"
    if [ "$SAVE_CHECKPOINTS" = true ]; then
        echo "  - Checkpoints: ${CHECKPOINT_DIR}/"
        echo "  - Summary: ${CHECKPOINT_DIR}/training_summary.json"
    else
        echo "  - Checkpoints disabled (SAVE_CHECKPOINTS=false)"
    fi
    echo ""
    echo "Configuration used:"
    if [ "$REINIT_LORA" = true ]; then
        echo "  - Method: Standard LoRA fine-tuning with reinitialization"
    else
        echo "  - Method: Standard LoRA fine-tuning"
    fi
    echo "  - Rounds: $NUM_ROUNDS"
    echo "  - Tools per round: $TOOLS_PER_ROUND"  
    echo "  - Samples per tool: $SAMPLES_PER_TOOL"
    echo "  - Total tools covered: $((START_TOOL)) to $((START_TOOL + NUM_ROUNDS * TOOLS_PER_ROUND - 1))"
else
    echo ""
    echo "✗ LoRA training failed!"
    exit 1
fi
