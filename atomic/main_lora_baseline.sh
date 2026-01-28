#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python main_lora_baseline.py \
    --num_tasks 1000 \
    --train_size 500 \
    --val_size 10 \
    --test_size 50 \
    --model_name "meta-llama/Llama-3.2-3B-Instruct" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_length 1280 \
    --max_instruction_tokens 1024 \
    --eval_batch_size 8 \
    --validate_every_n_steps 1000 \
    --lr 5e-5 \
    --lora_r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --target_modules "q_proj,v_proj" \
    --save_path "lora_baseline_model" \
    --continual_replay \
    --block_size 10 \
    --continual_replay_ratio 0.1