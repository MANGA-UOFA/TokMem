#!/bin/bash

# Memorization comparison script
# Runs multiple experiments with same settings in shared log files
# Calculates average metrics at the end

export CUDA_VISIBLE_DEVICES=6
set -e

# Configuration
CONFIGS=(
    "8 128 1"
    "8 128 2"
    "8 128 5"
)
POSITIONS=("before" "after") # before: the task identifier is before the embeddings, after: the task identifier is after the embeddings
NUM_RUNS=5
ID_LENGTH=1 # the length of the task identifier

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="mem_exp_results/memorization_results_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "======================================="
echo "MEMORIZATION EXPERIMENT COMPARISON"
echo "======================================="
echo "Total configs: ${#CONFIGS[@]}"
echo "Positions: ${POSITIONS[@]}"
echo "Runs per config: $NUM_RUNS"
echo "Results directory: $RESULTS_DIR"
echo "======================================="

# Function to extract final metrics from output
extract_metrics() {
    local output="$1"
    # Look for the pattern: "Single Model:  Loss: X.XXXX, Acc: X.XXXX, Perplexity: X.XXXX, Best at epoch: XXX"
    local accuracy=$(echo "$output" | grep "Single Model:" | tail -1 | grep -o "Acc: [0-9.]*" | grep -o "[0-9.]*")
    local perplexity=$(echo "$output" | grep "Single Model:" | tail -1 | grep -o "Perplexity: [0-9.]*" | grep -o "[0-9.]*")
    # Look for the pattern: "Prompt Embeddings - Initial Norm: X.XXXX, Final Norm: X.XXXX"
    local initial_norm=$(echo "$output" | grep "Prompt Embeddings - Initial Norm:" | tail -1 | grep -o "Initial Norm: [0-9.]*" | grep -o "[0-9.]*")
    local final_norm=$(echo "$output" | grep "Prompt Embeddings - Initial Norm:" | tail -1 | grep -o "Final Norm: [0-9.]*" | grep -o "[0-9.]*")
    # Look for the pattern: "Prompt Embedding Angle Change: X.XX° (X.XXXX radians)"
    local angle_degrees=$(echo "$output" | grep "Prompt Embedding Angle Change:" | tail -1 | grep -o "[0-9.]*\.[0-9]*°" | grep -o "[0-9.]*\.[0-9]*")
    # Look for the pattern: "Cosine Similarity (initial vs final): X.XXXX"
    local cosine_similarity=$(echo "$output" | grep "Cosine Similarity (initial vs final):" | tail -1 | grep -o "[0-9.]*\.[0-9]*")
    echo "$accuracy $perplexity $initial_norm $final_norm $angle_degrees $cosine_similarity"
}

# Function to calculate average
calculate_avg() {
    local sum=0
    local count=0
    for val in "$@"; do
        if [[ "$val" =~ ^[0-9]+\.?[0-9]*$ ]]; then
            sum=$(echo "$sum + $val" | bc -l)
            count=$((count + 1))
        fi
    done
    if [ $count -gt 0 ]; then
        echo "scale=4; $sum / $count" | bc -l
    else
        echo "N/A"
    fi
}

# Run experiments
total_experiments=0
success_count=0

for config in "${CONFIGS[@]}"; do
    read -r batch_size max_length prompt_length <<< "$config"
    
    for position in "${POSITIONS[@]}"; do
        log_file="$RESULTS_DIR/bs${batch_size}_ml${max_length}_pl${prompt_length}_${position}_id${ID_LENGTH}.log"
        
        echo "Running config: bs=$batch_size, ml=$max_length, pl=$prompt_length, pos=$position"
        echo "Log file: $log_file"
        
        # Initialize log file
        echo "=======================================" > "$log_file"
        echo "Configuration: batch_size=$batch_size, max_length=$max_length, prompt_length=$prompt_length, position=$position" >> "$log_file"
        echo "Number of runs: $NUM_RUNS" >> "$log_file"
        echo "=======================================" >> "$log_file"
        
        # Arrays to store metrics for averaging
        accuracies=()
        perplexities=()
        initial_norms=()
        final_norms=()
        angles=()
        cosine_similarities=()
        
        for run in $(seq 1 $NUM_RUNS); do
            total_experiments=$((total_experiments + 1))
            # Linear start index: 0, 100, 200, 300, 400...
            start_index=$(( (run - 1) * 100 ))
            
            echo "  Run $run/$NUM_RUNS (start_index=$start_index)"
            
            # Show the exact command being run
            cmd="python main_memorization.py $batch_size $max_length $start_index $prompt_length ids $ID_LENGTH random_words $position"
            echo "  Command: $cmd"
            
            # Run experiment
            echo "" >> "$log_file"
            echo "--- RUN $run (start_index=$start_index) ---" >> "$log_file"
            echo "Command: $cmd" >> "$log_file"
            echo "Started at: $(date)" >> "$log_file"
            
            # Run with visible output and capture simultaneously
            if python main_memorization.py $batch_size $max_length $start_index $prompt_length ids $ID_LENGTH random_words $position 2>&1 | tee -a "$log_file"; then
                echo "Completed at: $(date)" >> "$log_file"
                success_count=$((success_count + 1))
                
                # Get the output from the log file for metric extraction
                output=$(tail -n 100 "$log_file")
                
                # Extract metrics
                read -r accuracy perplexity initial_norm final_norm angle_degrees cosine_similarity <<< "$(extract_metrics "$output")"
                if [ -n "$accuracy" ]; then
                    accuracies+=("$accuracy")
                fi
                if [ -n "$perplexity" ]; then
                    perplexities+=("$perplexity")
                fi
                if [ -n "$initial_norm" ]; then
                    initial_norms+=("$initial_norm")
                fi
                if [ -n "$final_norm" ]; then
                    final_norms+=("$final_norm")
                fi
                if [ -n "$angle_degrees" ]; then
                    angles+=("$angle_degrees")
                fi
                if [ -n "$cosine_similarity" ]; then
                    cosine_similarities+=("$cosine_similarity")
                fi
                
                echo "    ✓ Success - Accuracy: $accuracy, Perplexity: $perplexity, Angle: ${angle_degrees}°, Cosine: $cosine_similarity"
            else
                echo "Failed at: $(date)" >> "$log_file"
                echo "    ✗ Failed"
            fi
        done
        
        # Calculate and append averages
        avg_accuracy=$(calculate_avg "${accuracies[@]}")
        avg_perplexity=$(calculate_avg "${perplexities[@]}")
        avg_initial_norm=$(calculate_avg "${initial_norms[@]}")
        avg_final_norm=$(calculate_avg "${final_norms[@]}")
        avg_angle=$(calculate_avg "${angles[@]}")
        avg_cosine=$(calculate_avg "${cosine_similarities[@]}")
        
        echo "" >> "$log_file"
        echo "=======================================" >> "$log_file"
        echo "SUMMARY FOR THIS CONFIGURATION" >> "$log_file"
        echo "=======================================" >> "$log_file"
        echo "Successful runs: ${#accuracies[@]}/$NUM_RUNS" >> "$log_file"
        echo "Average Test Accuracy: $avg_accuracy" >> "$log_file"
        echo "Average Test Perplexity: $avg_perplexity" >> "$log_file"
        echo "Average Initial Norm: $avg_initial_norm" >> "$log_file"
        echo "Average Final Norm: $avg_final_norm" >> "$log_file"
        echo "Average Angle Change: $avg_angle degrees" >> "$log_file"
        echo "Average Cosine Similarity: $avg_cosine" >> "$log_file"
        echo "Individual accuracies: ${accuracies[*]}" >> "$log_file"
        echo "Individual perplexities: ${perplexities[*]}" >> "$log_file"
        echo "Individual initial norms: ${initial_norms[*]}" >> "$log_file"
        echo "Individual final norms: ${final_norms[*]}" >> "$log_file"
        echo "Individual angles: ${angles[*]}" >> "$log_file"
        echo "Individual cosine similarities: ${cosine_similarities[*]}" >> "$log_file"
        echo "=======================================" >> "$log_file"
        
        echo "  Average Accuracy: $avg_accuracy, Average Perplexity: $avg_perplexity, Average Angle: ${avg_angle}°"
        echo ""
    done
done

# Generate overall summary
summary_file="$RESULTS_DIR/overall_summary.txt"
echo "=======================================" > "$summary_file"
echo "OVERALL EXPERIMENT SUMMARY" >> "$summary_file"
echo "=======================================" >> "$summary_file"
echo "Timestamp: $(date)" >> "$summary_file"
echo "Total experiments: $total_experiments" >> "$summary_file"
echo "Successful: $success_count" >> "$summary_file"
echo "Failed: $((total_experiments - success_count))" >> "$summary_file"
echo "Success rate: $(echo "scale=1; $success_count * 100 / $total_experiments" | bc -l)%" >> "$summary_file"
echo "" >> "$summary_file"

echo "Configuration Results:" >> "$summary_file"
for config in "${CONFIGS[@]}"; do
    read -r batch_size max_length prompt_length <<< "$config"
    
    for position in "${POSITIONS[@]}"; do
        log_file="$RESULTS_DIR/bs${batch_size}_ml${max_length}_pl${prompt_length}_${position}_id${ID_LENGTH}.log"
        
        if [ -f "$log_file" ]; then
            avg_accuracy=$(grep "Average Test Accuracy:" "$log_file" | tail -1 | awk '{print $4}')
            avg_perplexity=$(grep "Average Test Perplexity:" "$log_file" | tail -1 | awk '{print $4}')
            avg_initial_norm=$(grep "Average Initial Norm:" "$log_file" | tail -1 | awk '{print $4}')
            avg_final_norm=$(grep "Average Final Norm:" "$log_file" | tail -1 | awk '{print $4}')
            avg_angle=$(grep "Average Angle Change:" "$log_file" | tail -1 | awk '{print $4}')
            avg_cosine=$(grep "Average Cosine Similarity:" "$log_file" | tail -1 | awk '{print $4}')
            successful_runs=$(grep "Successful runs:" "$log_file" | tail -1 | awk '{print $3}')
            
            echo "  bs=$batch_size, ml=$max_length, pl=$prompt_length, pos=$position: $successful_runs runs, avg_acc=$avg_accuracy, avg_perp=$avg_perplexity, avg_angle=${avg_angle}°, avg_cosine=$avg_cosine" >> "$summary_file"
        fi
    done
done

echo ""
echo "======================================="
echo "EXPERIMENT COMPLETED"
echo "======================================="
echo "Total experiments: $total_experiments"
echo "Successful: $success_count"
echo "Failed: $((total_experiments - success_count))"
echo "Success rate: $(echo "scale=1; $success_count * 100 / $total_experiments" | bc -l)%"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Overall summary: $summary_file"
echo "Individual logs: $RESULTS_DIR/bs*_ml*_pl*_*.log"