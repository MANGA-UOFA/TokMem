#!/usr/bin/env python3
"""
GSM8K Prompt Tuning Experiment

This script trains a prompt tuning model on GSM8K math word problems,
using questions as ID tokens and evaluating on the test set.
"""

import argparse
import torch
from transformers import AutoTokenizer

from model import QAPromptTuning, print_model_info
from dataset import load_gsm8k_data, load_gsm8k_with_split, create_batch
from training import set_seed, train_qa_model, evaluate_on_test_set

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="GSM8K Prompt Tuning Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--prompt_length", type=int, default=1,
                       help="Number of prompt tokens")
    parser.add_argument("--prompt_position", type=str, default="infix", choices=["prefix", "infix"],
                       help="Position of prompt tokens: 'prefix' ([P][User][Assistant]) or 'infix' ([User][P][Assistant])")
    parser.add_argument("--use_chat_template", action="store_true", default=False,
                       help="Enable chat template formatting (for instruct models)")
    parser.add_argument("--system_prompt", type=str, 
                       default="You are a helpful assistant.",
                       help="System prompt to use for the conversation")
    
    # Data arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Number of examples to train on simultaneously")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    parser.add_argument("--train_size", type=int, default=None,
                       help="Number of training samples (None for all available)")
    parser.add_argument("--val_size", type=int, default=500,
                       help="Number of validation samples")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Number of test samples to evaluate on (None for all)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of gradient accumulation steps (saves VRAM)")
    parser.add_argument("--validation_steps", type=int, default=1000,
                       help="Run validation every N steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Data type to use")
    
    return parser.parse_args()

def get_dtype(dtype_str):
    """Convert string to torch dtype"""
    if dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unknown dtype: {dtype_str}")

def main():
    """Main experiment function"""
    args = parse_args()
    
    # Set up device and dtype
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = get_dtype(args.dtype)
    torch.set_default_dtype(dtype)
    
    # Set seed
    set_seed(args.seed)
    
    print(f"GSM8K Prompt Tuning Experiment")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Batch size: {args.batch_size}, Max length: {args.max_length}")
    print(f"Prompt length: {args.prompt_length}, Position: {args.prompt_position}")
    print(f"System prompt: {args.system_prompt}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps} steps")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    print("Loading GSM8K dataset...")
    train_dataset, val_dataset = load_gsm8k_with_split(args.train_size, args.val_size, args.seed)
    test_dataset = load_gsm8k_data("test", args.test_size)
    
    print("\nDATASET INFORMATION:")
    print("-" * 40)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {(len(train_dataset) + args.batch_size - 1) // args.batch_size}")
    print(f"Prompt position: {args.prompt_position}")
    
    # Determine chat template usage
    use_chat_template = args.use_chat_template
    print(f"Chat template: {use_chat_template}")
    
    # Show sample examples
    print("\nSample examples:")
    for i in range(min(1, len(train_dataset))):
        example = train_dataset[i]
        print(f"  Example {i+1}:")
        print(f"    Q: {example['question'][:80]}...")
        print(f"    A: {example['answer'][:80]}...")
        if args.prompt_position == "prefix":
            print(f"    Format: [P] [User] [Assistant]")
        else:
            print(f"    Format: [User] [P] [Assistant]")
    print()
    
    # Create model
    print("Creating model...")
    set_seed(args.seed)
    qa_model = QAPromptTuning(
        model_name=args.model_name,
        prompt_length=args.prompt_length,
        prompt_position=args.prompt_position,
        use_chat_template=use_chat_template,
        device=device,
        dtype=dtype
    )
    
    print("\nMODEL INFORMATION:")
    print("-" * 40)
    print_model_info(qa_model, "GSM8K QA Model")
    print("=" * 60)
    
    # Train model
    print("\nTRAINING:")
    print("-" * 40)
    set_seed(args.seed)
    losses, best_loss, best_acc, best_epoch = train_qa_model(
        qa_model, tokenizer, train_dataset, val_dataset, args.batch_size, args.max_length,
        args.epochs, args.lr, args.gradient_accumulation_steps, args.validation_steps, device, dtype, args.system_prompt
    )
    
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET:")
    print("=" * 60)
    
    # Evaluate on test set
    print("Generating predictions on test set...")
    test_accuracy, correct, total, format_rate, total_format_compliant = evaluate_on_test_set(
        qa_model, tokenizer, test_dataset, 
        batch_size=4, max_length=args.max_length, device=device, system_prompt=args.system_prompt
    )
    
    print(f"\nFINAL RESULTS:")
    print("-" * 40)
    print(f"Test Set Accuracy: {test_accuracy:.4f} ({correct}/{total})")
    print(f"Format Compliance: {format_rate:.4f} ({total_format_compliant}/{total})")
    print(f"Training Loss: {best_loss:.4f}")
    print(f"Training Token Accuracy: {best_acc:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Total epochs trained: {len(losses)}")
    
    # Save results
    results = {
        'args': vars(args),
        'batch_size': args.batch_size,
        'prompt_length': args.prompt_length,
        'prompt_position': args.prompt_position,
        'training_accuracy': best_acc,
        'training_loss': best_loss,
        'test_accuracy': test_accuracy,
        'test_correct': correct,
        'test_total': total,
        'format_rate': format_rate,
        'total_format_compliant': total_format_compliant,
        'best_epoch': best_epoch,
        'total_epochs': len(losses)
    }
    
    return results

if __name__ == "__main__":
    results = main()