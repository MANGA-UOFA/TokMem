#!/usr/bin/env python3
"""
LoRA Baseline - Sequential training script for function calling.
Uses standard LoRA fine-tuning instead of tool-specific embeddings.
"""

import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import os
import logging

from dataset import discover_available_tools
from replay_buffer import SimpleReplayBuffer


def collate_fn(batch):
    """Custom collate function to handle variable length sequences with left padding"""
    # Extract components
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    raw_data = [item['raw_data'] for item in batch]
    
    # Stack tensors (they should all be the same length due to max_length padding)
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'labels': torch.stack(labels),
        'raw_data': raw_data
    }


class FunctionCallingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, mode="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Create a consistent mapping of all tools to generic labels
        all_tools = set()
        for example in self.data:
            if 'tools' in example:
                all_tools.update(example['tools'])
        
        # Sort tools for consistent mapping across runs
        sorted_tools = sorted(list(all_tools))
        self.tool_mapping = {tool: f"tool_{i+1}" for i, tool in enumerate(sorted_tools)}
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        user_input = example['user_input']
        
        if self.mode == "train":
            # Training: include expected function calls in output
            tools = example.get('tools', [])
            function_calls = example.get('function_calls', [])
            
            conversation = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n"
            
            if tools and function_calls:
                # Use generic tool tokens for fair comparison
                for tool, func_call in zip(tools, function_calls):
                    generic_tool = self.tool_mapping.get(tool, "tool_unknown")
                    conversation += f"\n[{generic_tool}]{func_call}"  # Add generic tool marker
            
            conversation += "<|eot_id|>"
            
        else:
            # Evaluation: only user input
            conversation = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
            conversation += f"<|start_header_id|>assistant<|end_header_id|>\n"
        
        encoding = self.tokenizer(
            conversation,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        if self.mode == "train":
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            # Only train on assistant response
            assistant_start = "<|start_header_id|>assistant<|end_header_id|>\n"
            if assistant_start in conversation:
                prefix = conversation.split(assistant_start)[0] + assistant_start
                prefix_tokens = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                if len(prefix_tokens) < len(labels):
                    labels[:len(prefix_tokens)] = -100
        else:
            labels = torch.tensor(-100)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_data": example
        }


def create_lora_dataloader(train_data_path, test_data_path, tokenizer, batch_size=4, max_length=512, eval_batch_size=32):
    """Create dataloaders for LoRA training"""
    
    # Create datasets
    train_dataset = FunctionCallingDataset(train_data_path, tokenizer, max_length, "train")
    test_dataset = FunctionCallingDataset(test_data_path, tokenizer, max_length, "eval")
    
    # Create dataloaders with collate_fn
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_dataloader, test_dataloader


def create_mixed_dataloader_with_replay(train_dataset, replay_buffer, batch_size, replay_ratio):
    """
    Create a dataloader that mixes new training samples with replay samples.
    
    Args:
        train_dataset: Dataset with new training samples
        replay_buffer: SimpleReplayBuffer with previous samples
        batch_size: Total batch size
        replay_ratio: Proportion of batch that should be replay samples (0.0-1.0)
    """
    
    class MixedDataset(Dataset):
        def __init__(self, train_dataset, replay_samples, replay_ratio):
            self.train_dataset = train_dataset
            self.replay_samples = replay_samples
            self.replay_ratio = replay_ratio
            
            # Calculate how many samples of each type per epoch
            self.replay_per_batch = int(batch_size * replay_ratio)
            self.new_per_batch = batch_size - self.replay_per_batch
            
            # Create indices for sampling
            self.train_indices = list(range(len(train_dataset)))
            
        def __len__(self):
            return len(self.train_dataset)
            
        def __getitem__(self, idx):
            # This gets called by DataLoader, but we'll handle mixing in the dataloader
            return self.train_dataset[idx]
    
    # Get replay samples
    replay_samples = replay_buffer.get_all() if replay_buffer else []
    
    if not replay_samples or replay_ratio == 0.0:
        # No replay samples, return regular dataloader
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Create mixed dataset
    mixed_dataset = MixedDataset(train_dataset, replay_samples, replay_ratio)
    
    def mixed_collate_fn(batch):
        """Custom collate function that mixes new and replay samples"""
        replay_per_batch = int(len(batch) * replay_ratio)
        new_per_batch = len(batch) - replay_per_batch
        
        # Keep only new_per_batch new samples
        new_batch = batch[:new_per_batch] if new_per_batch > 0 else []
        
        # Sample replay_per_batch replay samples
        replay_batch = []
        if replay_per_batch > 0 and replay_samples:
            sampled_replay = random.sample(replay_samples, min(replay_per_batch, len(replay_samples)))
            for replay_sample in sampled_replay:
                # Convert replay sample back to dataset format
                replay_batch.append(replay_sample)
        
        # Combine and shuffle
        combined_batch = new_batch + replay_batch
        random.shuffle(combined_batch)
        
        # Use original collate function
        return collate_fn(combined_batch)
    
    return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, collate_fn=mixed_collate_fn)


def train_lora_model(model, train_dataloader, num_epochs=3, lr=5e-4, device="cuda"):
    """Train LoRA model using standard fine-tuning approach"""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    model.train()
    
    # Set up optimizer for LoRA parameters
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = len(train_dataloader) * num_epochs
    
    # Create linear learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% of steps for warmup
        num_training_steps=total_steps
    )
    
    print(f"Training LoRA for {num_epochs} epochs, {len(train_dataloader)} batches per epoch")
    print(f"Total steps: {total_steps}")
    print(f"Learning rate: {lr} (with linear schedule + warmup)")
    print(f"Warmup steps: {total_steps // 10}")
    
    total_loss = 0
    step_count = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            step_count += 1
            
            if (batch_idx + 1) % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed, Average Loss: {avg_epoch_loss:.4f}")
    
    avg_total_loss = total_loss / step_count
    print(f"Training completed! Average total loss: {avg_total_loss:.4f}")
    
    return {
        'avg_loss': avg_total_loss,
        'total_steps': step_count
    }


def extract_function_calls_from_text(text):
    """Extract function calls and tool tokens from generated text"""
    import re
    function_calls = []
    tools = []
    
    # Split text into lines and process each line
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for lines with tool markers: [tool_X]{json}
        tool_pattern = r'^\s*\[(tool_\d+)\](.+)$'
        tool_match = re.match(tool_pattern, line)
        
        if tool_match:
            tool = tool_match.group(1)
            json_str = tool_match.group(2).strip()
            
            try:
                # Validate it's proper JSON
                json.loads(json_str)
                function_calls.append(json_str)
                tools.append(tool)
            except json.JSONDecodeError:
                # Try to extract JSON from the string
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                json_matches = re.findall(json_pattern, json_str)
                for json_match in json_matches:
                    try:
                        json.loads(json_match)
                        function_calls.append(json_match)
                        tools.append(tool)
                        break  # Only take first valid JSON
                    except json.JSONDecodeError:
                        pass
        else:
            # Fallback: look for JSON without tool marker
            json_pattern = r'^\s*(\{.*\})\s*$'
            match = re.match(json_pattern, line)
            
            if match:
                json_candidate = match.group(1)
                try:
                    json.loads(json_candidate)
                    function_calls.append(json_candidate)
                    tools.append("tool_unknown")
                except json.JSONDecodeError:
                    pass
    
    return function_calls, tools


def eval_lora_model(model, tokenizer, test_dataloader, device="cuda"):
    """Evaluate LoRA model using function calling metrics"""
    import time
    from collections import defaultdict
    from eval import compare_function_calls_advanced
    
    model.eval()
    
    # Calculate total examples from dataloader
    total_examples = len(test_dataloader.dataset)
    print("🔄 Running LoRA model evaluation...")
    start_time = time.time()
    
    exact_matches = 0
    tool_correct = 0
    processed_examples = 0
    parse_errors = 0
    f1_scores = []
    precision_scores = []
    recall_scores = []
    tool_f1_scores = []
    tool_precision_scores = []
    tool_recall_scores = []
    call_count_breakdown = defaultdict(lambda: {
        'total': 0, 
        'exact_matches': 0, 
        'tool_correct': 0, 
        'f1_scores': [],
        'precision_scores': [],
        'recall_scores': [],
        'tool_f1_scores': [],
        'tool_precision_scores': [],
        'tool_recall_scores': [],
        'parse_errors': 0
    })
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_size = len(batch['raw_data'])
            processed_examples += batch_size
            
            if batch_idx % 10 == 0 or processed_examples == total_examples:
                print(f"   Progress: {processed_examples}/{total_examples} ({100 * processed_examples / total_examples:.1f}%)")
            
            # Generate responses
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                temperature=0.6,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # Process each example
            for i in range(batch_size):
                example = batch['raw_data'][i]
                expected_calls = example.get('function_calls', [])
                expected_tools = example.get('tools', [])
                expected_call_count = len(expected_calls)
                
                # Get dataset from dataloader to access tool mapping
                dataset = test_dataloader.dataset
                expected_generic_tools = [dataset.tool_mapping.get(tool, "tool_unknown") for tool in expected_tools]
                
                # Extract generated text
                generated_tokens = generated[i, input_ids.shape[1]:]
                generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                # Extract function calls and predicted tools
                predicted_calls, predicted_tools = extract_function_calls_from_text(generated_text)
                
                # Tool matching with generic labels - calculate F1 scores
                from collections import Counter
                from eval import calculate_f1_score
                
                # Calculate tool-level F1 scores
                tool_f1_result = calculate_f1_score(predicted_tools, expected_generic_tools)
                tool_f1_scores.append(tool_f1_result['f1_score'])
                tool_precision_scores.append(tool_f1_result['precision'])
                tool_recall_scores.append(tool_f1_result['recall'])
                
                # Keep binary tool matching for backward compatibility
                if predicted_tools and expected_generic_tools:
                    tool_match = Counter(predicted_tools) == Counter(expected_generic_tools)
                elif not predicted_tools and not expected_generic_tools:
                    tool_match = True
                else:
                    tool_match = False
                
                if tool_match:
                    tool_correct += 1
                
                # Use the same evaluation function as native function calling for fair comparison
                eval_result = compare_function_calls_advanced(
                    predicted_calls,
                    expected_calls,
                    ignore_order=True
                )
                
                if eval_result.exact_match:
                    exact_matches += 1
                
                f1_scores.append(eval_result.f1_score)
                precision_scores.append(eval_result.precision)
                recall_scores.append(eval_result.recall)
                
                # Track parse errors from eval_result
                if 'parse_errors' in eval_result.details:
                    current_parse_errors = eval_result.details['parse_errors']['outputs']
                    parse_errors += current_parse_errors
                else:
                    current_parse_errors = 0
                
                # Track by call count
                call_count_breakdown[expected_call_count]['total'] += 1
                if eval_result.exact_match:
                    call_count_breakdown[expected_call_count]['exact_matches'] += 1
                if tool_match:
                    call_count_breakdown[expected_call_count]['tool_correct'] += 1
                call_count_breakdown[expected_call_count]['f1_scores'].append(eval_result.f1_score)
                call_count_breakdown[expected_call_count]['precision_scores'].append(eval_result.precision)
                call_count_breakdown[expected_call_count]['recall_scores'].append(eval_result.recall)
                call_count_breakdown[expected_call_count]['tool_f1_scores'].append(tool_f1_result['f1_score'])
                call_count_breakdown[expected_call_count]['tool_precision_scores'].append(tool_f1_result['precision'])
                call_count_breakdown[expected_call_count]['tool_recall_scores'].append(tool_f1_result['recall'])
                call_count_breakdown[expected_call_count]['parse_errors'] += current_parse_errors
    
    end_time = time.time()
    eval_time = end_time - start_time
    
    # Calculate final metrics
    exact_accuracy = exact_matches / total_examples
    tool_accuracy = tool_correct / total_examples
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_tool_f1_score = sum(tool_f1_scores) / len(tool_f1_scores) if tool_f1_scores else 0.0
    avg_tool_precision = sum(tool_precision_scores) / len(tool_precision_scores) if tool_precision_scores else 0.0
    avg_tool_recall = sum(tool_recall_scores) / len(tool_recall_scores) if tool_recall_scores else 0.0
    parse_error_rate = parse_errors / total_examples
    
    # Print formatted results matching the style from training.py
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"📋 Dataset: {total_examples} examples")
    print(f"⏱️  Evaluation time: {eval_time:.2f} seconds")
    print(f"🔧 Mode: LoRA Baseline")
    print()
    
    print("🎯 RESULTS:")
    print(f"   Exact Match Accuracy:     {exact_accuracy:.3f} ({exact_matches}/{total_examples})")
    print(f"   Tool Prediction Accuracy: {tool_accuracy:.3f} ({tool_correct}/{total_examples})")
    print(f"   Average F1 Score:         {avg_f1_score:.3f}")
    print(f"   Average Precision:        {avg_precision:.3f}")
    print(f"   Average Recall:           {avg_recall:.3f}")
    print(f"   Average Tool F1 Score:    {avg_tool_f1_score:.3f}")
    print(f"   Average Tool Precision:   {avg_tool_precision:.3f}")
    print(f"   Average Tool Recall:      {avg_tool_recall:.3f}")
    print(f"   Parse Error Rate:         {parse_error_rate:.3f}")
    print("=" * 50)
    
    # Breakdown by function call count
    print("\n📊 EXACT MATCH ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        accuracy = stats['exact_matches'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {accuracy:.3f} ({stats['exact_matches']}/{stats['total']})")
    print("=" * 50)
    
    print("\n📊 TOOL PREDICTION ACCURACY:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        tool_accuracy = stats['tool_correct'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {tool_accuracy:.3f} ({stats['tool_correct']}/{stats['total']})")
    print("=" * 50)
    
    print("\n📊 AVERAGE F1 SCORE (Function Calls):")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_f1 = sum(stats['f1_scores']) / len(stats['f1_scores']) if stats['f1_scores'] else 0.0
        avg_prec = sum(stats['precision_scores']) / len(stats['precision_scores']) if stats['precision_scores'] else 0.0
        avg_rec = sum(stats['recall_scores']) / len(stats['recall_scores']) if stats['recall_scores'] else 0.0
        print(f"   {call_count} call(s): F1={avg_f1:.3f}, P={avg_prec:.3f}, R={avg_rec:.3f}")
    print("=" * 50)
    
    print("\n📊 AVERAGE TOOL F1 SCORE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        avg_tool_f1 = sum(stats['tool_f1_scores']) / len(stats['tool_f1_scores']) if stats['tool_f1_scores'] else 0.0
        avg_tool_prec = sum(stats['tool_precision_scores']) / len(stats['tool_precision_scores']) if stats['tool_precision_scores'] else 0.0
        avg_tool_rec = sum(stats['tool_recall_scores']) / len(stats['tool_recall_scores']) if stats['tool_recall_scores'] else 0.0
        print(f"   {call_count} call(s): Tool F1={avg_tool_f1:.3f}, P={avg_tool_prec:.3f}, R={avg_tool_rec:.3f}")
    print("=" * 50)
    
    print("\n📊 PARSE ERROR RATE:")
    print("-" * 50)
    for call_count in sorted(call_count_breakdown.keys()):
        stats = call_count_breakdown[call_count]
        error_rate = stats['parse_errors'] / stats['total'] if stats['total'] > 0 else 0.0
        print(f"   {call_count} call(s): {error_rate:.3f} ({stats['parse_errors']}/{stats['total']})")
    print("=" * 50)
    
    return {
        'exact_accuracy': exact_accuracy,
        'tool_accuracy': tool_accuracy, 
        'avg_f1_score': avg_f1_score,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_tool_f1_score': avg_tool_f1_score,
        'avg_tool_precision': avg_tool_precision,
        'avg_tool_recall': avg_tool_recall,
        'parse_error_rate': parse_error_rate,
        'total_examples': total_examples,
        'call_count_breakdown': dict(call_count_breakdown)
    }


def parse_training_rounds(rounds_str):
    """
    Parse training rounds specification.
    Format: "tools:epochs,tools:epochs,..."
    Example: "1-10:3,11-20:3,21-30:2"
    """
    rounds = []
    for round_spec in rounds_str.split(','):
        parts = round_spec.strip().split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid round specification: {round_spec}. Use 'tools:epochs' format.")
        
        tools = parts[0].strip()
        epochs = int(parts[1].strip())
        rounds.append({'tools': tools, 'epochs': epochs})
    
    return rounds


def main():
    parser = argparse.ArgumentParser(description="LoRA Baseline - Sequential Function Calling Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model name")
    
    # Sequential training arguments
    parser.add_argument("--training_rounds", type=str, required=True,
                        help="Training rounds specification (e.g., '1-10:3,11-20:3,21-30:2' for 3 rounds)")
    
    # Training arguments  
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate")
    
    # Evaluation arguments
    parser.add_argument("--eval_after_each_round", action="store_true",
                        help="Run evaluation after each training round")
    parser.add_argument("--eval_all_previous", action="store_true",
                        help="Evaluate on all previous test sets after each round")
    parser.add_argument("--eval_batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    
    # Replay buffer arguments
    parser.add_argument("--use_replay_buffer", action="store_true",
                        help="Use replay buffer to mitigate catastrophic forgetting")
    parser.add_argument("--replay_buffer_size", type=int, default=1000,
                        help="Size of replay buffer")
    parser.add_argument("--replay_ratio", type=float, default=0.2,
                        help="Ratio of replay samples in each batch (0.0-1.0)")
    parser.add_argument("--demo", type=int, default=None,
                        help="Number of demo examples to show after training")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling factor (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate (default: 0.1)")
    parser.add_argument("--lora_target_modules", type=str, 
                        default="q_proj,v_proj",
                        help="Target modules for LoRA")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the data files")
    parser.add_argument("--train_max_function_calls", type=int, default=4,
                        help="Maximum function calls used in data generation")
    parser.add_argument("--train_max_function_calls_per_round", type=str, default=None,
                        help="Comma-separated max function call limits per round (overrides --train_max_function_calls)")
    parser.add_argument("--test_max_function_calls", type=int, default=4,
                        help="Maximum function calls used in data generation")
    parser.add_argument("--test_max_function_calls_per_round", type=str, default=None,
                        help="Comma-separated max function call limits per round for evaluation (overrides --test_max_function_calls)")
    
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save model checkpoints after each round")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_lora",
                        help="Directory to save checkpoints")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file for evaluation results")
    parser.add_argument("--reinit_lora_after_each_round", action="store_true",
                        help="Reinitialize LoRA parameters after each training round")
    
    args = parser.parse_args()
    
    # Setup simple logging
    import sys
    from datetime import datetime
    
    # Create log directory first
    os.makedirs("log", exist_ok=True)
    
    # Configure logging - put logs in log/ directory
    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        # Ensure log file goes in log/ directory
        log_file_path = args.log_file if args.log_file.startswith('log/') else f"log/{os.path.basename(args.log_file)}"
        log_handlers.append(logging.FileHandler(log_file_path, mode='a'))
    else:
        # Create default log file in log/ directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_log_file = f"log/lora_sequential_training_{timestamp}.log"
        log_handlers.append(logging.FileHandler(default_log_file, mode='a'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=log_handlers
    )
    logger = logging.getLogger(__name__)
    
    # Log start time and configuration
    logger.info(f"=== LoRA Sequential Training Started at {datetime.now()} ===")
    logger.info(f"Configuration: model={args.model_name}, rounds={args.training_rounds}, batch_size={args.batch_size}")
    if args.reinit_lora_after_each_round:
        logger.info("LoRA reinitialization enabled: Parameters will be reset after each round")
    
    # Parse training rounds
    try:
        rounds = parse_training_rounds(args.training_rounds)
        print(f"Parsed {len(rounds)} training rounds:")
        for i, round_spec in enumerate(rounds, 1):
            print(f"  Round {i}: Tools {round_spec['tools']}, {round_spec['epochs']} epochs")
    except ValueError as e:
        parser.error(str(e))

    num_rounds = len(rounds)

    def expand_per_round_values(values_str, fallback, arg_label):
        if values_str is None:
            return [fallback] * num_rounds

        raw_values = [part.strip() for part in values_str.split(',')]
        parsed_values = []
        for value in raw_values:
            if not value:
                continue
            try:
                parsed_values.append(int(value))
            except ValueError:
                parser.error(f"Invalid value '{value}' for {arg_label}; expected integers.")

        if not parsed_values:
            parser.error(f"No valid values provided for {arg_label}.")

        if len(parsed_values) < num_rounds:
            parsed_values.extend([parsed_values[-1]] * (num_rounds - len(parsed_values)))
        elif len(parsed_values) > num_rounds:
            print(f"Warning: {arg_label} specified {len(parsed_values)} values; using first {num_rounds}.")
            parsed_values = parsed_values[:num_rounds]

        return parsed_values

    train_max_calls_by_round = expand_per_round_values(
        args.train_max_function_calls_per_round,
        args.train_max_function_calls,
        "train_max_function_calls_per_round"
    )

    test_max_calls_by_round = expand_per_round_values(
        args.test_max_function_calls_per_round,
        args.test_max_function_calls,
        "test_max_function_calls_per_round"
    )

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set dtype
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    print("\n=== LoRA Baseline - Sequential Function Calling Training ===")
    print(f"Model: {args.model_name}")
    print(f"Training rounds: {len(rounds)}")
    print(f"Reinit LoRA after each round: {args.reinit_lora_after_each_round}")
    print(f"Method: Standard LoRA fine-tuning{' with reinitialization' if args.reinit_lora_after_each_round else ''}")
    print()
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    tokenizer.padding_side = "left"  # Use left padding for decoder-only models
    
    # Create checkpoint directory if needed
    if args.save_checkpoints:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize base model (will be wrapped with LoRA)
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    
    # Parse target modules once
    target_modules = [mod.strip() for mod in args.lora_target_modules.split(',')]
    
    # Store results for all rounds
    all_results = []
    # Store test dataloaders for cumulative evaluation
    all_test_dataloaders = []
    
    # Initialize model variable
    model = None
    
    # Initialize replay buffer if enabled
    replay_buffer = None
    if args.use_replay_buffer:
        replay_buffer = SimpleReplayBuffer(max_size=args.replay_buffer_size)
        print(f"Initialized replay buffer with size {args.replay_buffer_size}, replay ratio {args.replay_ratio}")
    
    # Training loop for each round
    for round_idx, round_spec in enumerate(rounds):
        round_num = round_idx + 1
        tools_range = round_spec['tools']
        epochs = round_spec['epochs']
        
        print("\n" + "="*60)
        print(f"ROUND {round_num}/{len(rounds)}: Training on tools {tools_range}")
        print("="*60 + "\n")
        
        # Initialize or reinitialize LoRA for this round
        if round_idx == 0 or args.reinit_lora_after_each_round:
            if round_idx > 0:
                print("Reinitializing LoRA parameters for new round...")
                # Get the base model for reinitialization
                base_model = model.get_base_model()
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
            )
            
            # Apply LoRA to base model
            model = get_peft_model(base_model, lora_config)
            print("LoRA configuration applied!")
            model.print_trainable_parameters()
        
        # Construct data file names based on tool range
        train_max_calls = train_max_calls_by_round[round_idx]
        test_max_calls = test_max_calls_by_round[round_idx]
        train_data_file = os.path.join(
            args.data_dir, 
            f"training/function_calling_train_tools{tools_range}_{train_max_calls}calls.json"
        )
        test_data_file = os.path.join(
            args.data_dir,
            f"test/function_calling_test_tools{tools_range}_{test_max_calls}calls.json"
        )
        
        # Check if data files exist
        if not os.path.exists(train_data_file):
            print(f"Warning: Training data file not found: {train_data_file}")
            print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        if not os.path.exists(test_data_file):
            print(f"Warning: Test data file not found: {test_data_file}")
            print(f"Please generate it using: python xlam_datasets.py --top_k {tools_range}")
            continue
        
        # Discover tools for this round
        print(f"Discovering tools from round {round_num} dataset...")
        round_tools = discover_available_tools(train_data_file, test_data_file)
        print(f"Found {len(round_tools)} tools for round {round_num}: {round_tools[:5]}..." if len(round_tools) > 5 else f"Found {len(round_tools)} tools: {round_tools}")
        
        # Create datasets for this round
        print(f"Creating round {round_num} dataset...")
        train_dataloader, test_dataloader = create_lora_dataloader(
            train_data_path=train_data_file,
            test_data_path=test_data_file,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            eval_batch_size=args.eval_batch_size,
        )
        
        # Create mixed dataloader with replay buffer if enabled
        if args.use_replay_buffer and replay_buffer is not None:
            print(f"Creating mixed dataloader with {replay_buffer.size()} replay samples (ratio: {args.replay_ratio})")
            # Get the train dataset from the dataloader
            train_dataset = train_dataloader.dataset
            train_dataloader = create_mixed_dataloader_with_replay(
                train_dataset, replay_buffer, args.batch_size, args.replay_ratio
            )
        
        # Store test dataloader for cumulative evaluation
        all_test_dataloaders.append((tools_range, test_dataloader))
        
        # Train this round
        print(f"\nTraining round {round_num} for {epochs} epochs...")
        print("Training: LoRA fine-tuning")
        
        round_results = train_lora_model(
            model=model,
            train_dataloader=train_dataloader,
            num_epochs=epochs,
            lr=args.lr,
            device=args.device
        )
        
        print(f"Round {round_num} training completed! Average loss: {round_results['avg_loss']:.4f}")
        logger.info(f"\n[ROUND {round_num} RESULTS] Tools: {tools_range}, Epochs: {epochs}, Loss: {round_results['avg_loss']:.4f}")
        
        # Add samples to replay buffer after training
        if args.use_replay_buffer and replay_buffer is not None:
            # Load training data to add to replay buffer
            with open(train_data_file, 'r') as f:
                training_data = json.load(f)
            
            # Sample a subset for the replay buffer (don't add all samples)
            max_samples_to_add = min(len(training_data) // 2, 500)  # Add at most 500 samples per round
            samples_to_add = random.sample(training_data, max_samples_to_add)
            
            replay_buffer.add(samples_to_add)
            print(f"Added {len(samples_to_add)} samples to replay buffer. Buffer size: {replay_buffer.size()}")
        
        # Store results
        all_results.append({
            'round': round_num,
            'tools': tools_range,
            'epochs': epochs,
            'avg_loss': round_results['avg_loss'],
            'results': round_results
        })
        
        # Import necessary modules for output capture
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Evaluate this round
        if args.eval_after_each_round:
            print(f"\nEvaluating round {round_num} model...")
            
            # Capture stdout to get the formatted evaluation results
            captured_output = io.StringIO()
            with redirect_stdout(captured_output):
                eval_results = eval_lora_model(
                    model=model,
                    tokenizer=tokenizer,
                    test_dataloader=test_dataloader,
                    device=args.device
                )
            
            # Get the captured formatted output
            formatted_eval_output = captured_output.getvalue()
            
            # Print the formatted output to console as well
            print(formatted_eval_output)
            
            all_results[-1]['eval_results'] = eval_results
            
            # Log the formatted evaluation results to file
            if args.log_file:
                log_file_base = args.log_file if args.log_file.startswith('log/') else f"log/{os.path.basename(args.log_file)}"
                eval_log_file = log_file_base.replace('.log', '_eval_results.log')
            else:
                eval_log_file = default_log_file.replace('.log', '_eval_results.log')
            
            with open(eval_log_file, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"ROUND {round_num} EVALUATION - Tools: {tools_range}, Epochs: {epochs}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"{'='*60}\n")
                f.write(formatted_eval_output)
                f.write(f"\n{'='*60}\n\n")
            
            # Evaluate on all previous test sets if requested
            if args.eval_all_previous and round_num > 1:
                print(f"\nEvaluating on all previous test sets...")
                cumulative_results = {}
                # Evaluate on all previous rounds (current round is at the end, so [:-1] gives us all previous)
                for prev_tools, prev_test_dataloader in all_test_dataloaders[:-1]:
                    print(f"  Evaluating on tools {prev_tools}...")
                    
                    # Capture formatted output for previous evaluation
                    prev_captured_output = io.StringIO()
                    with redirect_stdout(prev_captured_output):
                        prev_eval_results = eval_lora_model(
                            model=model,
                            tokenizer=tokenizer,
                            test_dataloader=prev_test_dataloader,
                            device=args.device
                        )
                    
                    # Get the captured formatted output
                    prev_formatted_eval_output = prev_captured_output.getvalue()
                    
                    # Print the formatted output to console as well
                    print(prev_formatted_eval_output)
                    
                    cumulative_results[f"tools_{prev_tools}"] = prev_eval_results
                    
                    # Also log cumulative results to eval file
                    if args.log_file:
                        log_file_base = args.log_file if args.log_file.startswith('log/') else f"log/{os.path.basename(args.log_file)}"
                        eval_log_file = log_file_base.replace('.log', '_eval_results.log')
                    else:
                        eval_log_file = default_log_file.replace('.log', '_eval_results.log')
                        
                    with open(eval_log_file, 'a') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"ROUND {round_num} EVAL on tools {prev_tools}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"{'='*60}\n")
                        f.write(prev_formatted_eval_output)
                        f.write(f"\n{'='*60}\n\n")
                        
                all_results[-1]['cumulative_eval_results'] = cumulative_results
        
        # Save checkpoint if requested
        if args.save_checkpoints:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"round_{round_num}_tools_{tools_range.replace('-', '_')}")
            print(f"Saving checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
    
    # Final summary
    print("\n" + "="*60)
    print("LORA TRAINING SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"Round {result['round']} (tools {result['tools']}): "
              f"{result['epochs']} epochs, avg loss: {result['avg_loss']:.4f}")
        if 'eval_results' in result and result['eval_results']:
            print(f"  Evaluation accuracy: {result['eval_results'].get('exact_accuracy', 'N/A'):.3f}")
    
    print("\n" + "="*60)
    print("LoRA sequential training completed!")
    print(f"Trained {len(rounds)} rounds with different tool sets")
    if args.reinit_lora_after_each_round:
        print("Method: Standard LoRA fine-tuning with reinitialization after each round")
    else:
        print("Method: Standard LoRA fine-tuning")
    print("="*60)
    
    # Save final results summary
    if args.save_checkpoints:
        summary_path = os.path.join(args.checkpoint_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nTraining summary saved to {summary_path}")
    
    # Log completion
    logger.info(f"=== LoRA Sequential Training Completed at {datetime.now()} ===")


if __name__ == "__main__":
    main()
