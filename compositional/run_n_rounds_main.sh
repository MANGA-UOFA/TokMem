#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Default parameters
NUM_ROUNDS=2
TOOLS_PER_ROUND=50
SAMPLES_PER_TOOL=50
EPOCHS_PER_ROUND='1,3' # the number of epochs for each round, comma-separated list
TRAIN_MAX_CALLS=4
TEST_MAX_CALLS=4
TRAIN_MAX_CALLS_PER_ROUND="4,4"
TEST_MAX_CALLS_PER_ROUND="4,4"
BATCH_SIZE=2
LR=5e-3
LORA_LR=5e-5
MODEL="meta-llama/Llama-3.2-3B-Instruct"
START_TOOL=1
TRAIN_SIZE=5000
TEST_SIZE=500
CURRICULUM_LEARNING=false
EVAL_ALL_PREVIOUS=false
RENORM_ACTIVE_TOOLS=true
FREEZE_LORA_AFTER_FIRST=true #true: the first round serves as the adaptation phase
USE_LORA=true
SAVE_CHECKPOINTS=false


# Parse epochs per round (can be single value or comma-separated list)
IFS=',' read -ra EPOCHS_ARRAY <<< "$EPOCHS_PER_ROUND"

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
    
    # Get epochs for this round (use last value if array is shorter than rounds)
    if [ ${#EPOCHS_ARRAY[@]} -eq 1 ]; then
        # Single value for all rounds
        epochs=${EPOCHS_ARRAY[0]}
    elif [ $i -lt ${#EPOCHS_ARRAY[@]} ]; then
        # Use specific value for this round
        epochs=${EPOCHS_ARRAY[$i]}
    else
        # Use last value if we have more rounds than epoch values
        epochs=${EPOCHS_ARRAY[-1]}
    fi
    
    if [ $i -gt 0 ]; then
        ROUNDS="${ROUNDS},"
    fi
    ROUNDS="${ROUNDS}${start}-${end}:${epochs}"
done

echo "Flexible N-Round Sequential Training"
echo "====================================="
echo "Configuration:"
echo "  - Number of rounds: $NUM_ROUNDS"
echo "  - Tools per round: $TOOLS_PER_ROUND"
echo "  - Max samples per tool: $SAMPLES_PER_TOOL"
# Display epochs configuration
if [ ${#EPOCHS_ARRAY[@]} -eq 1 ]; then
    echo "  - Epochs per round: $EPOCHS_PER_ROUND"
else
    echo "  - Epochs per round: $EPOCHS_PER_ROUND (variable per round)"
fi
echo "  - Starting tool: $START_TOOL"
echo "  - Training rounds: $ROUNDS"
echo "  - Model: $MODEL"
echo "  - Renorm active tools: $RENORM_ACTIVE_TOOLS"
echo "  - Freeze LoRA after first: $FREEZE_LORA_AFTER_FIRST"
echo "  - Use LoRA: $USE_LORA"
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
CHECKPOINT_DIR="checkpoints_${NUM_ROUNDS}rounds_${TOOLS_PER_ROUND}tools_${SAMPLES_PER_TOOL}samples"

# Create data directories
mkdir -p data/training data/test

# Step 1: Generate data for all rounds
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

# Step 2: Run sequential training
echo "Step 2: Running ${NUM_ROUNDS}-round sequential training..."
echo "========================================================="

python main_sequential.py \
    --training_rounds "$ROUNDS" \
    --batch_size $BATCH_SIZE \
    $([ ${#TRAIN_MAX_CALLS_ARRAY[@]} -gt 0 ] && echo "--train_max_function_calls_per_round $TRAIN_MAX_CALLS_PER_ROUND" || echo "--train_max_function_calls $TRAIN_MAX_CALLS") \
    $([ ${#TEST_MAX_CALLS_ARRAY[@]} -gt 0 ] && echo "--test_max_function_calls_per_round $TEST_MAX_CALLS_PER_ROUND" || echo "--test_max_function_calls $TEST_MAX_CALLS") \
    --model_name "$MODEL" \
    $([ "$USE_LORA" = "true" ] && echo "--use_lora") \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --lora_target_modules "q_proj,v_proj" \
    $([ "$FREEZE_LORA_AFTER_FIRST" = "true" ] && echo "--freeze_lora_after_first") \
    --eval_after_each_round \
    --save_checkpoints \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --lr $LR \
    --lora_lr $LORA_LR \
    --eval_batch_size 16 \
    --seed 42 \
    $([ "$CURRICULUM_LEARNING" = "true" ] && echo "--curriculum_learning") \
    $([ "$EVAL_ALL_PREVIOUS" = "true" ] && echo "--eval_all_previous") \
    $([ "$RENORM_ACTIVE_TOOLS" = "true" ] && echo "--renorm_active_tools")

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================="
    echo "✓ Training completed successfully!"
    echo ""
    echo "Results:"
    echo "  - Checkpoints: ${CHECKPOINT_DIR}/"
    echo "