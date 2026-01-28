#!/usr/bin/env python3
"""
Example script for Natural Instructions Task Learning

This script demonstrates how to train and evaluate a model with task tokens
for Natural Instructions tasks, where each task learns its own token.

The approach is similar to function calling, but instead of learning tool tokens
for different functions, we learn task tokens for different instruction tasks.

Usage:
    python example_natural_task_learning.py --num_tasks 5 --num_epochs 3
"""

import torch
from transformers import AutoTokenizer
import argparse
import os
import random
import numpy as np

# Import our custom modules
from task_model import TaskCallingModel, print_model_info
from task_dataset import (
    create_natural_instructions_dataloader, 
    sample_natural_instructions_tasks
)
from task_training import (
    train_task_calling_model,
    demo_task_calling,
    eval_task_calling,
    setup_logging
)

def add_reserved_special_tokens(tokenizer, num_of_tasks, device="cuda"):
    """Add reserved special tokens to the tokenizer"""
    start_idx = len([t for t in tokenizer.get_vocab() if t.startswith("<|reserved_special_token_")])

    if num_of_tasks <= start_idx:
        return tokenizer, False
    else:
        num_additional_tokens = num_of_tasks - start_idx
        new_tokens = [f"<|reserved_special_token_{i}|>" for i in range(start_idx, start_idx + num_additional_tokens)]
        added = tokenizer.add_special_tokens({'additional_special_tokens': new_tokens})
        assert added == num_additional_tokens, f"Expected to add {num_additional_tokens} tokens, but added {added}"

        return tokenizer, True

def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Make deterministic operations more deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")

def main():
    parser = argparse.ArgumentParser(description='Natural Instructions Task Learning')
    parser.add_argument('--tasks_dir', type=str, default='natural-instructions-2.8/tasks', 
                        help='Directory containing Natural Instructions task files')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help='HuggingFace model name')
    parser.add_argument('--num_tasks', type=int, default=5, help='Number of tasks to sample')
    # Remove ratio-based splitting in favor of absolute sizes
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--decouple_embeddings', action='store_true', 
                        help='Use separate input/output embeddings for task tokens')
    parser.add_argument('--max_instruction_tokens', type=int, default=1024, 
                        help='Maximum token length for instructions (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and only run evaluation')
    parser.add_argument('--demo', action='store_true', help='Only run demo on a few examples')
    parser.add_argument('--load_task_tokens', type=str, default=None, 
                        help='Path to saved task tokens file (for evaluation/inference)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    # Absolute per-task sizes (override ratios when provided)
    parser.add_argument('--train_size', type=int, default=None, help='Absolute number of training samples per task (overrides train_ratio)')
    parser.add_argument('--val_size', type=int, default=None, help='Absolute number of validation samples per task (overrides val_ratio)')
    parser.add_argument('--test_size', type=int, default=None, help='Absolute number of test samples per task (overrides test_ratio; test selected first deterministically)')
    parser.add_argument('--few_shot', action='store_true', help='Use few-shot instructions')
    parser.add_argument('--validate_every_n_steps', type=int, default=1000, 
                        help='Validate every n steps')
    args = parser.parse_args()
    
    # Set random seed first for full reproducibility
    set_random_seed(args.seed)
    print()
    
    print("=" * 60)
    print("NATURAL INSTRUCTIONS TASK LEARNING")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Number of tasks to sample: {args.num_tasks}")
    # Ratios removed; using sizes mode only
    print(f"Decouple embeddings: {args.decouple_embeddings}")
    if any(x is not None for x in [args.train_size, args.val_size, args.test_size]):
        print(f"Sizes mode per task - Train: {args.train_size}, Val: {args.val_size}, Test: {args.test_size} (test is selected first, stable)")
    print(f"Random seed: {args.seed}")
    print()
    
    # Set up logging
    print("Setting up logging...")
    training_logger, eval_logger, training_log, evaluation_log, timestamp = setup_logging()
    print(f"   Training log: {training_log}")
    print(f"   Evaluation log: {evaluation_log}")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    print()

    # Add reserved special tokens to the tokenizer
    tokenizer, is_extended = add_reserved_special_tokens(tokenizer, args.num_tasks)
    print(f"Tokenizer loaded with adjustments. Vocab size: {len(tokenizer)}")
    print()
    
    # Sample tasks from Natural Instructions dataset
    print(f"Sampling {args.num_tasks} tasks from Natural Instructions dataset...")
    print(f"   Max instruction length: {args.max_instruction_tokens} tokens")
    train_data, val_data, test_data, task_names = sample_natural_instructions_tasks(
        tasks_dir=args.tasks_dir,
        num_tasks=args.num_tasks,
        max_instruction_tokens=args.max_instruction_tokens,
        tokenizer=tokenizer,
        stable_test_split=True,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        few_shot=args.few_shot,
    )
    
    # Initialize model
    print("Initializing Task Calling Model...")
    model = TaskCallingModel(
        model_name=args.model_name,
        num_tasks=len(task_names),
        task_names=task_names,
        tokenizer=tokenizer,
        device=args.device,
        decouple_embeddings=args.decouple_embeddings,
        is_extended=is_extended,
    )
    
    print("\nModel Information:")
    print_model_info(model.model, "Base Model (Frozen)")
    print_model_info(model, "Task Model (Trainable Task Tokens)")
    print()
    
    # Load task tokens if specified
    if args.load_task_tokens:
        if os.path.exists(args.load_task_tokens):
            print(f"Loading task tokens from: {args.load_task_tokens}")
            model.load_task_tokens(args.load_task_tokens)
            print("Task tokens loaded successfully!")
        else:
            print(f"❌ Error: Task tokens file not found: {args.load_task_tokens}")
            return
        print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_dataloader, val_dataloader, test_dataloader, tokenizer, test_examples = create_natural_instructions_dataloader(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size
    )
    
    # Training
    if not args.skip_training and train_dataloader:
        print("Starting Training...")
        train_results = train_task_calling_model(
            model=model,
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=args.device,
            timestamp=timestamp,
            validate_every_n_steps=args.validate_every_n_steps
        )
        print(f"Training completed with average loss: {train_results['avg_total_loss']:.4f}")
        
        # Load best model state if validation was performed and best model was found
        if train_results['best_model_state'] is not None:
            print(f"Loading best model state (validation loss: {train_results['best_val_loss']:.4f})")
            best_state = train_results['best_model_state']
            # Load only the available keys (token embeddings) without requiring full state
            model.load_state_dict(best_state, strict=False)
        print()
    
    # Demo on a few examples
    if args.demo:
        print("Running demo on sample examples...")
        # Show up to 5 examples
        demo_examples = random.sample(test_examples, 5)
        demo_task_calling(model, tokenizer, demo_examples, device=args.device)
        print()
    
    # Evaluation
    if test_dataloader:
        print("Running comprehensive evaluation...")
        
        # Normal task prediction evaluation
        results = eval_task_calling(
            model=model,
            tokenizer=tokenizer,
            test_dataloader=test_dataloader,
            device=args.device,
            use_ground_truth_tasks=False
        )
        
        print("\n" + "=" * 50)
        print("FINAL RESULTS SUMMARY:")
        print(f"   Task Prediction Accuracy: {results['task_accuracy']:.3f}")
        print(f"   Exact Match Accuracy: {results['exact_accuracy']:.3f}")
        print(f"   Average Response Score: {results['avg_response_score']:.3f}")
        print("=" * 50)
        
        # # Optional: Ground truth task evaluation for comparison
        # print("\nRunning ground truth task evaluation for comparison...")
        # gt_results = eval_task_calling(
        #     model=model,
        #     tokenizer=tokenizer,
        #     test_dataloader=test_dataloader,
        #     device=args.device,
        #     use_ground_truth_tasks=True
        # )
        
        # print("\n" + "=" * 50)
        # print("COMPARISON RESULTS:")
        # print(f"   Task Prediction Mode - Exact Match: {results['exact_accuracy']:.3f}")
        # print(f"   Ground Truth Mode - Exact Match: {gt_results['exact_accuracy']:.3f}")
        # print(f"   Performance Gap: {gt_results['exact_accuracy'] - results['exact_accuracy']:.3f}")
        # print("=" * 50)
    
    print("\nTask learning pipeline completed!")


if __name__ == "__main__":
    main()