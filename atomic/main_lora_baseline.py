#!/usr/bin/env python3
"""
LoRA Baseline for Natural Instructions Task Learning

This script implements a baseline that finetunes the model with LoRA
on the SuperNatural Instructions dataset, without using task tokens.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse
import os
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Import data loading functions from existing modules
from task_dataset import (
    sample_natural_instructions_tasks,
    NaturalInstructionsTaskDataset
)
from task_training import setup_logging
from natural_instructions_eval import evaluate_predictions, print_evaluation_results

def set_random_seed(seed):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to: {seed}")

def lora_collate_fn(batch, tokenizer):
    """Custom collate function for LoRA training"""
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

def build_block_replay_sequence(train_data, block_size=10, replay_ratio=0.1, seed=42):
    """Reorder training data by tasks and interleave cumulative replay from all prior blocks.

    Args:
        train_data: List[dict] samples, each with 'tasks': [task_name]
        block_size: Number of tasks per block (e.g., 10)
        replay_ratio: Fraction of current-block samples to inject from previous block [0-1]
        seed: RNG seed for deterministic sampling

    Returns:
        List[dict]: Reordered training data with block-wise replay interleaved
    """
    if not train_data or block_size <= 0 or replay_ratio <= 0.0:
        return train_data

    rng = random.Random(seed)

    # Determine task order as they first appear in the data
    task_order = []
    seen = set()
    for item in train_data:
        t = item.get('tasks', ['unknown'])[0]
        if t not in seen:
            seen.add(t)
            task_order.append(t)

    # Group samples by task in original order
    task_to_samples = {t: [] for t in task_order}
    for item in train_data:
        t = item.get('tasks', ['unknown'])[0]
        if t in task_to_samples:
            task_to_samples[t].append(item)

    # Partition tasks into blocks
    blocks = [task_order[i:i+block_size] for i in range(0, len(task_order), block_size)]

    final_sequence = []
    cumulative_pool = []

    for block_index, block_tasks in enumerate(blocks):
        # Concatenate current block tasks' samples in task order
        current_block_samples = []
        for t in block_tasks:
            current_block_samples.extend(task_to_samples[t])

        if block_index == 0 or not cumulative_pool:
            # First block: no replay, initialize cumulative pool
            final_sequence.extend(current_block_samples)
            # Update cumulative pool with current block
            cumulative_pool.extend(current_block_samples)
            continue

        # Compute how many replay samples to insert
        num_current = len(current_block_samples)
        if num_current == 0:
            continue
        num_replay = int(round(num_current * replay_ratio))
        num_replay = max(0, min(num_replay, len(cumulative_pool)))

        if num_replay == 0:
            # Nothing to interleave
            final_sequence.extend(current_block_samples)
            # Always grow cumulative pool
            cumulative_pool.extend(current_block_samples)
            continue

        # Sample replay items from cumulative prior pool (without replacement)
        replay_items = rng.sample(cumulative_pool, num_replay)

        # Interleave replay items roughly evenly into current block
        # Place one replay after each chunk of size interval
        interval = max(1, num_current // (num_replay + 1))
        interleaved = []
        cur_idx = 0
        rep_idx = 0
        next_insert = interval

        while cur_idx < num_current:
            interleaved.append(current_block_samples[cur_idx])
            cur_idx += 1

            # If we've passed the next insert point and still have replay items, insert one
            if cur_idx >= next_insert and rep_idx < num_replay:
                interleaved.append(replay_items[rep_idx])
                rep_idx += 1
                next_insert += interval

        # If any replay items remain, append them at the end
        while rep_idx < num_replay:
            interleaved.append(replay_items[rep_idx])
            rep_idx += 1

        final_sequence.extend(interleaved)

        # Grow cumulative pool with current block for future replay
        cumulative_pool.extend(current_block_samples)

    return final_sequence

def create_lora_dataloaders(train_data, val_data, test_data, tokenizer, 
                           batch_size=4, eval_batch_size=16, max_length=1024, shuffle_train=False):
    """Create DataLoaders for LoRA training without task tokens"""
    
    class LoRAInstructionsDataset(NaturalInstructionsTaskDataset):
        """Dataset wrapper that removes task token logic"""
        def __init__(self, data, tokenizer, max_length=512):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __getitem__(self, idx):
            item = self.data[idx]
            instruction = item.get('instruction', '')
            query = item.get('query', '')
            response = item['responses'][0] if item['responses'] else ""
            
            # Detect model type for chat format
            is_qwen = 'qwen' in self.tokenizer.name_or_path.lower()
            
            # Simple format: instruction + query -> response (no few-shot)
            conversation_parts = []
            if is_qwen:
                # Qwen chat format (no system prompt to match main setup)
                conversation_parts.append(f"<|im_start|>user\n{instruction}\n\n{query}<|im_end|>\n")
                conversation_parts.append(f"<|im_start|>assistant\n{response}<|im_end|>\n")
            else:
                # Llama chat format
                conversation_parts.append("<|begin_of_text|>")
                conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n\n{query}<|eot_id|>")
                conversation_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>")
            
            text = "".join(conversation_parts)
            
            # Tokenize with left padding
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            input_ids = encoding.input_ids.squeeze()
            attention_mask = encoding.attention_mask.squeeze()
            
            # Create labels (mask out user portion and padding)
            labels = input_ids.clone()
            
            # Find where assistant response starts
            if is_qwen:
                assistant_token = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            else:
                assistant_token = self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
            
            assistant_start = None
            for i in range(len(input_ids) - len(assistant_token)):
                if input_ids[i:i+len(assistant_token)].tolist() == assistant_token:
                    assistant_start = i + len(assistant_token)
                    break
            
            if assistant_start:
                labels[:assistant_start] = -100  # Mask user portion
            
            # Mask padding tokens (important for left padding)
            labels[attention_mask == 0] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
    
    # Create datasets
    train_dataset = LoRAInstructionsDataset(train_data, tokenizer, max_length) if train_data else None
    val_dataset = LoRAInstructionsDataset(val_data, tokenizer, max_length) if val_data else None
    test_dataset = LoRAInstructionsDataset(test_data, tokenizer, max_length) if test_data else None
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=lambda batch: lora_collate_fn(batch, tokenizer)
    ) if train_dataset else None
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: lora_collate_fn(batch, tokenizer)
    ) if val_dataset else None
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=lambda batch: lora_collate_fn(batch, tokenizer)
    ) if test_dataset else None
    
    if train_dataset:
        print(f"Training dataset created: {len(train_dataset)} samples")
    if val_dataset:
        print(f"Validation dataset created: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test dataset created: {len(test_dataset)} samples")
    
    return train_dataloader, val_dataloader, test_dataloader, test_data if test_data else []

def train_lora_model(model, train_dataloader, val_dataloader=None, 
                    num_epochs=3, lr=1e-4, device="cuda", 
                    gradient_accumulation_steps=1, validate_every_n_steps=100,
                    save_path=None, logger=None):
    """Train LoRA model with logging"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Adjust learning rate for gradient accumulation
    if gradient_accumulation_steps > 1:
        lr = lr / gradient_accumulation_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Log training config
    if logger:
        logger.info(f"Training config: lr={lr}, epochs={num_epochs}, grad_accum={gradient_accumulation_steps}")
    
    model.train()
    total_steps = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        epoch_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                epoch_steps += 1
                total_steps += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
                
                # Log training progress
                if logger and total_steps % (validate_every_n_steps // 2) == 0:
                    current_loss = loss.item() * gradient_accumulation_steps
                    logger.info(f"Step {total_steps}: train_loss={current_loss:.4f}")
                
                # Validation
                if val_dataloader and total_steps % validate_every_n_steps == 0:
                    val_loss = evaluate_model(model, val_dataloader, device)
                    print(f"\n  Step {total_steps}: Validation Loss = {val_loss:.4f}")
                    
                    if logger:
                        logger.info(f"Step {total_steps}: val_loss={val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        print(f"New best validation loss: {val_loss:.4f}")
                        
                        if logger:
                            logger.info(f"Step {total_steps}: new_best_val_loss={val_loss:.4f}")
                    
                    model.train()
        
        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f"  Average epoch loss: {avg_epoch_loss:.4f}")
        
        if logger:
            logger.info(f"Epoch {epoch+1}: avg_loss={avg_epoch_loss:.4f}, steps={epoch_steps}")
    
    # Save model if path provided
    if save_path and best_model_state:
        model.load_state_dict(best_model_state)
        model.save_pretrained(save_path)
        print(f"Best model saved to {save_path}")
        
        if logger:
            logger.info(f"Model saved to {save_path}, final_loss={avg_epoch_loss:.4f}, best_val_loss={best_val_loss:.4f}")
    
    return {
        'avg_loss': avg_epoch_loss,
        'best_val_loss': best_val_loss,
        'best_model_state': best_model_state
    }

def evaluate_model(model, dataloader, device="cuda"):
    """Evaluate model and return average loss"""
    model.eval()
    total_loss = 0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_steps += 1
    
    return total_loss / total_steps if total_steps > 0 else 0

def evaluate_with_generation(model, tokenizer, test_examples, device="cuda", max_new_tokens=256, batch_size=8):
    """Evaluate model by generating responses and computing ROUGE scores"""
    model.eval()
    
    all_predictions = []
    all_references = []
    all_task_names = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(test_examples), batch_size), desc="Generating responses"):
        batch_examples = test_examples[i:i+batch_size]
        
        batch_texts = []
        for example in batch_examples:
            instruction = example.get('instruction', '')
            query = example.get('query', '')
            
            # Detect model type for chat format
            is_qwen = 'qwen' in tokenizer.name_or_path.lower()
            
            # Simple format: instruction + query (no few-shot)
            conversation_parts = []
            if is_qwen:
                # Qwen chat format
                conversation_parts.append(f"<|im_start|>user\n{instruction}\n\n{query}<|im_end|>\n")
                conversation_parts.append("<|im_start|>assistant\n")
            else:
                # Llama chat format
                conversation_parts.append("<|begin_of_text|>")
                conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n\n{query}<|eot_id|>")
                conversation_parts.append(f"<|start_header_id|>assistant<|end_header_id|>")
            
            text = "".join(conversation_parts)
            batch_texts.append(text)
        
        # Tokenize batch with left padding
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract responses
        for j, output in enumerate(outputs):
            generated = tokenizer.decode(output, skip_special_tokens=True)
            
            # Extract response (after the assistant header)
            if "assistant" in generated:
                response = generated.split("assistant")[-1].strip()
            else:
                # Fallback: take everything after the input
                response = generated[len(batch_texts[j]):].strip()
            
            # Store predictions and references
            all_predictions.append(response)
            all_references.append(batch_examples[j]['responses'])
            if 'tasks' in batch_examples[j]:
                all_task_names.append(batch_examples[j]['tasks'][0])
            else:
                all_task_names.append("unknown")
    
    # Evaluate using Natural Instructions metrics
    results = evaluate_predictions(
        predictions=all_predictions,
        references=all_references,
        task_names=all_task_names
    )
    
    return results, all_predictions

def generate_responses(model, tokenizer, test_examples, device="cuda", max_new_tokens=256):
    """Generate responses for test examples (demo)"""
    model.eval()
    results = []
    
    for example in tqdm(test_examples[:10], desc="Generating responses"):  # Limit to 10 for demo
        instruction = example.get('instruction', '')
        query = example.get('query', '')
        expected_response = example['responses'][0] if example['responses'] else ""
        
        # Detect model type for chat format
        is_qwen = 'qwen' in tokenizer.name_or_path.lower()
        
        # Simple format: instruction + query (no few-shot)
        conversation_parts = []
        if is_qwen:
            # Qwen chat format
            conversation_parts.append(f"<|im_start|>user\n{instruction}\n\n{query}<|im_end|>\n")
            conversation_parts.append("<|im_start|>assistant\n")
        else:
            # Llama chat format
            conversation_parts.append("<|begin_of_text|>")
            conversation_parts.append(f"<|start_header_id|>user<|end_header_id|>\n{instruction}\n\n{query}<|eot_id|>")
            conversation_parts.append(f"<|start_header_id|>assistant<|end_header_id|>")
        
        text = "".join(conversation_parts)
        
        # Tokenize with left padding
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract response (after the user message)
        if "assistant" in generated:
            response = generated.split("assistant")[-1].strip()
        else:
            response = generated[len(text):].strip()
        
        results.append({
            'instruction': instruction,
            'query': query,
            'expected': expected_response,
            'generated': response
        })
        
        print(f"\n📝 Instruction: {instruction[:100]}...")
        print(f"📝 Query: {query[:100]}...")
        print(f"✅ Expected: {expected_response[:100]}...")
        print(f"🤖 Generated: {response[:100]}...")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='LoRA Baseline for Natural Instructions')
    parser.add_argument('--tasks_dir', type=str, default='natural-instructions-2.8/tasks', 
                        help='Directory containing Natural Instructions task files')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help='HuggingFace model name')
    parser.add_argument('--num_tasks', type=int, default=10, help='Number of tasks to sample')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Evaluation batch size')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda", help='Device to use')
    parser.add_argument('--max_instruction_tokens', type=int, default=1024, 
                        help='Maximum token length for instructions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train_size', type=int, default=100, help='Training samples per task')
    parser.add_argument('--val_size', type=int, default=10, help='Validation samples per task')
    parser.add_argument('--test_size', type=int, default=50, help='Test samples per task')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help='Gradient accumulation steps')
    parser.add_argument('--validate_every_n_steps', type=int, default=100, 
                        help='Validate every n steps')
    parser.add_argument('--few_shot', action='store_true', help='Use few-shot instructions')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save trained model')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load trained model')
    parser.add_argument('--skip_training', action='store_true', help='Skip training')
    parser.add_argument('--demo', action='store_true', help='Run demo generation')
    parser.add_argument('--shuffle_train', action='store_true', help='Shuffle training data')
    # Simple toggle for dataset-level replay with defaults
    parser.add_argument('--continual_replay', action='store_true', help='Enable simple dataset-level replay (prev 10 tasks -> next 10). Disables shuffling.')
    parser.add_argument('--continual_replay_ratio', type=float, default=0.1, help='Fraction of current 10-task block to interleave from previous block [0-1]')
    parser.add_argument('--block_size', type=int, default=10, help='Block size for dataset-level block replay')
    # LoRA specific arguments
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--target_modules', type=str, default="q_proj,v_proj", 
                        help='Target modules for LoRA (comma-separated)')
    
    args = parser.parse_args()
    # Simplified mode: continual replay implies no shuffle
    if args.continual_replay:
        args.shuffle_train = False
    
    # Set random seed
    set_random_seed(args.seed)
    print()
    
    print("=" * 60)
    print("LORA BASELINE FOR NATURAL INSTRUCTIONS")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print(f"Number of tasks: {args.num_tasks}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Target modules: {args.target_modules}")
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
    # Set left padding for causal language models
    tokenizer.padding_side = "left"
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer)}, padding side: {tokenizer.padding_side}")
    print()
    
    # Sample tasks from Natural Instructions dataset
    print(f"Sampling {args.num_tasks} tasks from Natural Instructions dataset...")
    train_data, val_data, test_data, _ = sample_natural_instructions_tasks(
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
    
    # Load base model
    print("Loading base model...")
    if args.load_model:
        # Load fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.load_model)
        model = model.to(args.device)
        print(f"Loaded fine-tuned model from {args.load_model}")
    else:
        # Load base model and add LoRA
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map=args.device
        )
        
        # Configure LoRA
        target_modules = args.target_modules.split(',')
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        print("LoRA configuration applied")
    
    # Print model info
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    print()
    
    # Optionally reorder training data with dataset-level block replay for continual adaptation
    if (not args.shuffle_train) and args.continual_replay and train_data:
        print(f"Applying dataset-level block replay: block_size={args.block_size}, ratio={args.continual_replay_ratio}")
        train_data = build_block_replay_sequence(
            train_data=train_data,
            block_size=args.block_size,
            replay_ratio=args.continual_replay_ratio,
            seed=args.seed,
        )

    # Create dataloaders
    print("Creating data loaders...")
    train_dataloader, val_dataloader, test_dataloader, test_examples = create_lora_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_length=args.max_length,
        shuffle_train=args.shuffle_train
    )
    print()
    
    # Training
    if not args.skip_training and train_dataloader:
        print("Starting Training...")
        train_results = train_lora_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=args.device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            validate_every_n_steps=args.validate_every_n_steps,
            save_path=args.save_path or f"lora_model_{timestamp}",
            logger=training_logger
        )
        print(f"Training completed with average loss: {train_results['avg_loss']:.4f}")
        
        # Log peak memory usage and training summary
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
            print(f"Peak GPU memory usage: {peak_memory_mb:.2f} MB")
            training_logger.info(f"Training completed: peak_memory_mb={peak_memory_mb:.2f}, final_loss={train_results['avg_loss']:.4f}")
        else:
            training_logger.info(f"Training completed: final_loss={train_results['avg_loss']:.4f}")
        
        # Load best model if available
        if train_results['best_model_state'] is not None:
            print(f"Loading best model (validation loss: {train_results['best_val_loss']:.4f})")
            model.load_state_dict(train_results['best_model_state'])
        print()
    
    # Evaluation
    if test_examples:
        print("Running comprehensive evaluation on test set...")
        
        # Evaluate with generation and ROUGE scores
        eval_results, predictions = evaluate_with_generation(
            model=model,
            tokenizer=tokenizer,
            test_examples=test_examples,
            device=args.device,
            max_new_tokens=256,
            batch_size=args.eval_batch_size
        )
        
        # Print evaluation results
        print_evaluation_results(eval_results)
        
        # Log detailed evaluation results
        eval_logger.info("=" * 60)
        eval_logger.info("COMPREHENSIVE EVALUATION RESULTS")
        eval_logger.info("=" * 60)
        eval_logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        eval_logger.info(f"Model: {args.model_name}")
        eval_logger.info(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        eval_logger.info(f"Examples evaluated: {eval_results['num_examples']}")
        eval_logger.info("")
        
        # Log overall metrics
        eval_logger.info("OVERALL PERFORMANCE:")
        eval_logger.info(f"  Exact Match: {eval_results['exact_match']:.4f}%")
        eval_logger.info(f"  ROUGE-L:     {eval_results['rougeL']:.4f}%")
        eval_logger.info("")
        
        # Log per-task breakdown if available
        if 'per_task' in eval_results:
            eval_logger.info("PER-TASK PERFORMANCE:")
            for task_name, metrics in eval_results['per_task'].items():
                eval_logger.info(f"  {task_name}:")
                eval_logger.info(f"    Exact Match: {metrics['exact_match']:.4f}%")
                eval_logger.info(f"    ROUGE-L:     {metrics['rougeL']:.4f}%")
                eval_logger.info(f"    Examples:    {metrics['num_examples']}")
                eval_logger.info("")
        
        eval_logger.info("=" * 60)
        print()
    
    # Demo generation
    if args.demo and test_examples:
        print("Running demo generation on sample examples...")
        demo_results = generate_responses(model, tokenizer, test_examples[:5], args.device)
        print()
        
        # Calculate simple accuracy (exact match)
        exact_matches = sum(1 for r in demo_results if r['expected'].strip() == r['generated'].strip())
        print(f"Demo exact match: {exact_matches}/{len(demo_results)} = {exact_matches/len(demo_results):.2%}")
    
    print("\nLoRA baseline pipeline completed!")


if __name__ == "__main__":
    main()