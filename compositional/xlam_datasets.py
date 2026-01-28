#!/usr/bin/env python3
"""
Unified XLAM Dataset Processor and Multi-Tool Synthesizer
"""

from datasets import load_dataset
import json
import pandas as pd
import random
import os
from typing import Dict, List, Optional

def load_xlam_dataset():
    """Load and convert XLAM dataset to pandas DataFrame"""
    datasets = load_dataset("Salesforce/xlam-function-calling-60k")
    return datasets['train'].to_pandas()

def extract_tool_descriptions(df, tool_names):
    """
    Extract original tool descriptions for the given tool names.
    Returns a dictionary mapping tool names to their descriptions.
    """
    tool_descriptions = {}
    
    for idx, row in df.iterrows():
        try:
            tools_data = row['tools']
            tools_list = json.loads(tools_data) if isinstance(tools_data, str) else tools_data
            
            for tool in tools_list:
                tool_name = tool.get('name', '')
                if tool_name in tool_names and tool_name not in tool_descriptions:
                    tool_descriptions[tool_name] = {
                        'name': tool.get('name', ''),
                        'description': tool.get('description', ''),
                        'parameters': tool.get('parameters', {}),
                        'type': tool.get('type', '')
                    }
                    
        except (json.JSONDecodeError, TypeError):
            continue
        
        if len(tool_descriptions) >= len(tool_names):
            break
    
    missing_tools = set(tool_names) - set(tool_descriptions.keys())
    if missing_tools:
        print(f"Missing tool descriptions: {missing_tools}")
    
    return tool_descriptions

def save_tool_descriptions(tool_descriptions, filename="tool_descriptions.json"):
    """Save tool descriptions to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(tool_descriptions, f, indent=2)
    return filename

def extract_single_tool_data(df, k=10, max_samples_per_tool=None):
    """
    Extract data for top-k tools, keeping only single-tool samples (including same-tool multiple calls).
    Discards samples with multiple different tools since we'll synthesize those.
    
    Args:
        df: DataFrame with XLAM dataset
        k: Number of top tools to extract OR a tuple (start, end) for range (1-indexed, inclusive)
        max_samples_per_tool: Maximum number of samples per tool (None for no limit)
    """
    # Count tool frequency (only from single-tool samples)
    tool_frequency = {}
    for idx, row in df.iterrows():
        try:
            answers_data = row['answers']
            answers_list = json.loads(answers_data) if isinstance(answers_data, str) else answers_data
            
            # Get unique tools in this sample
            tools_in_sample = set()
            for answer in answers_list:
                tool_name = answer.get('name', '')
                if tool_name:
                    tools_in_sample.add(tool_name)
            
            # Only count tools from single-tool samples
            if len(tools_in_sample) == 1:
                tool_name = list(tools_in_sample)[0]
                tool_frequency[tool_name] = tool_frequency.get(tool_name, 0) + 1
                
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Get tools based on k parameter (either top-k or range)
    all_ranked_tools = [name for name, _ in sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True) 
                        if name != 'search']
    
    if isinstance(k, tuple):
        # Range mode: k is (start, end) 1-indexed, inclusive
        start, end = k
        # Convert to 0-indexed
        top_k_tools = all_ranked_tools[start-1:end]
        print(f"Using tools ranked {start} to {end} (found {len(top_k_tools)} tools)")
    else:
        # Top-k mode
        top_k_tools = all_ranked_tools[:k]
    
    # Extract single-tool samples (including same-tool multiple calls)
    tool_data = {tool: [] for tool in top_k_tools}
    
    for idx, row in df.iterrows():
        try:
            # tools_data = row['tools']
            # tools_list = json.loads(tools_data) if isinstance(tools_data, str) else tools_data
            
            answers_data = row['answers']
            answers_list = json.loads(answers_data) if isinstance(answers_data, str) else answers_data
            
            # Get unique tools used in answers
            tools_used = set(answer.get('name') for answer in answers_list if answer.get('name'))
            
            # Only keep samples that use exactly one type of tool (but possibly multiple calls)
            if len(tools_used) == 1:
                tool_name = list(tools_used)[0]
                if tool_name in tool_data:
                    # Check if we've reached the limit for this tool
                    if max_samples_per_tool is None or len(tool_data[tool_name]) < max_samples_per_tool:
                        # Get all calls for this tool
                        tool_calls = [answer for answer in answers_list if answer.get('name') == tool_name]
                        
                        if tool_calls:
                            tool_data[tool_name].append({
                                'id': row['id'],
                                'query': row['query'],
                                'calls': tool_calls  # All calls for this tool
                            })
        except (json.JSONDecodeError, TypeError):
            continue

    # Print summary statistics
    total_samples = sum(len(samples) for samples in tool_data.values())
    multi_call_samples = sum(sum(1 for s in samples if len(s['calls']) > 1) for samples in tool_data.values())
    
    if max_samples_per_tool is not None:
        print(f"Extracted {total_samples} single-tool samples ({multi_call_samples} with multiple calls)")
        print(f"Limited to max {max_samples_per_tool} samples per tool")
        for tool_name, samples in tool_data.items():
            print(f"  {tool_name}: {len(samples)} samples")
    else:
        print(f"Extracted {total_samples} single-tool samples ({multi_call_samples} with multiple calls)")
    
    return tool_data, top_k_tools

def split_single_tool_data(tool_data, train_ratio=0.9):
    """
    Split single-tool data into training and test sets.
    Ensures test queries are completely different from training queries.
    """
    train_tool_data = {}
    test_tool_data = {}
    total_train, total_test = 0, 0
    
    for tool_name, samples in tool_data.items():
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        train_count = int(len(shuffled_samples) * train_ratio)
        train_samples = shuffled_samples[:train_count]
        test_samples = shuffled_samples[train_count:]
        
        train_tool_data[tool_name] = train_samples
        test_tool_data[tool_name] = test_samples
        total_train += len(train_samples)
        total_test += len(test_samples)
    
    print(f"Split data: {total_train} train / {total_test} test samples")
    return train_tool_data, test_tool_data

def synthesize_multi_tool_data(tool_data, tool_names, num_samples=5000, ratios=None, split_name="", multi_call_probability=0.2, max_function_calls=None):
    """
    Synthesize multi-tool samples by combining single-tool samples.
    Note: Multiple calls of the same tool count as "one tool used" for categorization.
    Special handling: Include ALL single-tool samples, apply ratios only to multi-tool samples.
    
    Args:
        multi_call_probability: Probability (0.0-1.0) of using multi-call samples 
                               when creating multi-tool samples (default: 0.2)
        max_function_calls: Maximum number of function calls allowed per sample (None for no limit)
    """
    
    if split_name:
        print(f"Synthesizing {split_name} samples...")
    
    # Convert tool_data to format expected by synthesizer
    single_samples = {}
    multi_call_samples = {}  # Keep multi-call samples intact
    
    for tool_name, samples in tool_data.items():
        single_samples[tool_name] = []
        multi_call_samples[tool_name] = []
        
        for sample in samples:
            if len(sample['calls']) == 1:
                # Single call - add normally
                call = sample['calls'][0]
                if call.get('arguments'):
                    single_samples[tool_name].append({
                        'query': sample['query'],
                        'arguments': call['arguments'],
                        'has_multiple_calls': False
                    })
            else:
                # Multiple calls - keep as complete unit (counts as ONE tool used)
                multi_call_samples[tool_name].append({
                    'query': sample['query'],
                    'calls': sample['calls'],
                    'has_multiple_calls': True
                })
    
    synthesized = []
    
    # First, include ALL single-tool samples (num_tools=1)
    single_tool_count = 0

    if split_name == "test":
        # Skip single-call samples for test set
        pass
    elif split_name == "training":
        # Add all single-call samples (these always have 1 function call, so always within limit)
        for tool_name, samples in single_samples.items():
            for sample_data in samples:
                # Double-check that arguments exist (should already be filtered, but be safe)
                if sample_data.get('arguments'):
                    # Single-call samples always have exactly 1 function call, so they're always within any reasonable limit
                    if max_function_calls is None or 1 <= max_function_calls:
                        sample = {
                            "user_input": sample_data['query'],
                            "tools": [tool_name],
                            "function_calls": [json.dumps(sample_data['arguments'])],
                            "has_same_tool_multiple_calls": False
                        }
                        synthesized.append(sample)
                        single_tool_count += 1
                else:
                    pass  # Skip silently
        
        # Add all multi-call samples (they still count as single-tool usage)
        skipped_due_to_limit = 0
        for tool_name, samples in multi_call_samples.items():
            for sample_data in samples:
                # Filter to only include calls that have arguments, and match tools accordingly
                valid_calls = [call for call in sample_data['calls'] if call.get('arguments')]
                
                if valid_calls:  # Only add sample if there are valid calls
                    # Check if this sample would exceed the function call limit
                    if max_function_calls is not None and len(valid_calls) > max_function_calls:
                        skipped_due_to_limit += 1
                        continue  # Skip this sample to maintain query-call integrity
                    
                    tools_list = [tool_name] * len(valid_calls)
                    calls_list = [json.dumps(call['arguments']) for call in valid_calls]
                    
                    sample = {
                        "user_input": sample_data['query'],
                        "tools": tools_list,
                        "function_calls": calls_list,
                        "has_same_tool_multiple_calls": True
                    }
                    synthesized.append(sample)
                    single_tool_count += 1
                else:
                    pass  # Skip silently
        
        if max_function_calls is not None and skipped_due_to_limit > 0:
            print(f"Skipped {skipped_due_to_limit} samples due to function call limit ({max_function_calls})")
    
    # Calculate counts for multi-tool samples (excluding single-tool)
    multi_tool_ratios = {k: v for k, v in ratios.items() if k > 1}
    if multi_tool_ratios:
        # Normalize multi-tool ratios
        total_multi_ratio = sum(multi_tool_ratios.values())
        if total_multi_ratio > 0:
            normalized_multi_ratios = {k: v/total_multi_ratio for k, v in multi_tool_ratios.items()}
            
            # Calculate remaining samples for multi-tool
            remaining_samples = max(0, num_samples - single_tool_count)
            
            counts = {}
            allocated = 0
            
            # Sort ratios to ensure consistent allocation
            sorted_ratios = sorted(normalized_multi_ratios.items())
            
            for i, (num_tools, ratio) in enumerate(sorted_ratios):
                if i == len(sorted_ratios) - 1:
                    # Last ratio gets any remaining samples to ensure exact total
                    counts[num_tools] = remaining_samples - allocated
                else:
                    count = int(remaining_samples * ratio)
                    counts[num_tools] = count
                    allocated += count
            
            # Generate multi-tool samples
            for num_tools, count in counts.items():
                if count > 0:
                    for _ in range(count):
                        sample = create_sample(single_samples, multi_call_samples, tool_names, num_tools, multi_call_probability, max_function_calls)
                        if sample:
                            synthesized.append(sample)
    
    random.shuffle(synthesized)
    
    # Generate detailed statistics about function calls
    function_call_stats = {}
    tool_count_stats = {}
    
    for sample in synthesized:
        num_function_calls = len(sample['function_calls'])
        num_tools = len(set(sample['tools']))  # Count unique tools
        
        # Track function call distribution
        if num_function_calls not in function_call_stats:
            function_call_stats[num_function_calls] = 0
        function_call_stats[num_function_calls] += 1
        
        # Track tool count distribution 
        if num_tools not in tool_count_stats:
            tool_count_stats[num_tools] = 0
        tool_count_stats[num_tools] += 1
    
    if split_name:
        print(f"Generated {len(synthesized)} {split_name} samples")
        
        # Show function call distribution
        print(f"{split_name} function call distribution:")
        for num_calls in sorted(function_call_stats.keys()):
            count = function_call_stats[num_calls]
            percentage = (count / len(synthesized)) * 100
            print(f"  {num_calls} calls: {count} samples ({percentage:.1f}%)")
    
    return synthesized

def create_sample(single_samples, multi_call_samples, tool_names, num_tools=1, multi_call_probability=0.2, max_function_calls=None):
    """Create a sample with specified number of tools and optional function call limit"""
    available_tools = [t for t in tool_names if t in single_samples and single_samples[t]]
    if len(available_tools) < num_tools:
        return None
    
    # Retry logic to ensure we don't exceed function call limits
    max_retries = 50  # Reasonable limit to avoid infinite loops
    
    for attempt in range(max_retries):
        # Always select unique tools for multi-tool scenarios
        # Multiple calls for same tool only come from original data, not synthesis
        selected_tools = random.sample(available_tools, num_tools)
        
        # Multi-tool sample
        queries = []
        tools = []
        calls = []
        has_multi_calls = False
        
        base_connectors = [
            " Also, ", " Additionally, ", " Furthermore, ", " Moreover, ", 
            " In addition, ", " Besides, ", " Plus, ", " Next, ", 
            " Then, ", " After that, ", " On top of that, "
        ]
        final_connectors = [" Finally, ", " Lastly, ", " At last, "]
        
        for i, tool in enumerate(selected_tools):
            # Configurable chance to use multi-call sample if available 
            use_multi_call = (random.random() < multi_call_probability and 
                             tool in multi_call_samples and 
                             multi_call_samples[tool])
            
            if use_multi_call:
                # Use multi-call sample with ALL calls preserved
                sample_data = random.choice(multi_call_samples[tool])
                query = sample_data['query']
                
                if i > 0:
                    query = query.lower()
                    # Use final connectors only for the last position
                    if i == num_tools - 1 and num_tools >= 3:
                        connector = random.choice(final_connectors)
                    else:
                        connector = random.choice(base_connectors)
                    queries.append(connector + query)
                else:
                    queries.append(query)
                
                # Use ALL calls from the multi-call sample
                # First filter to calls that have arguments
                valid_calls = [call for call in sample_data['calls'] if call.get('arguments')]
                if valid_calls:
                    # Add all valid calls, not just one
                    for call in valid_calls:
                        tools.append(tool)
                        calls.append(json.dumps(call['arguments']))
                    has_multi_calls = True
                else:
                    # If no valid calls, skip this tool entirely and remove its query
                    queries.pop()  # Remove the query we just added
            else:
                sample_data = random.choice(single_samples[tool])
                
                # Safety check - should already be filtered but be safe
                if sample_data.get('arguments'):
                    query = sample_data['query']
                    if i > 0:
                        query = query.lower()
                        # Use final connectors only for the last position
                        if i == num_tools - 1 and num_tools >= 3:
                            connector = random.choice(final_connectors)
                        else:
                            connector = random.choice(base_connectors)
                        queries.append(connector + query)
                    else:
                        queries.append(query)
                    
                    tools.append(tool)
                    calls.append(json.dumps(sample_data['arguments']))
                else:
                    pass  # Skip silently
        
        # Final validation - ensure tools and function_calls have matching lengths
        if len(tools) != len(calls):
            min_len = min(len(tools), len(calls))
            tools = tools[:min_len]
            calls = calls[:min_len]
        
        # Check if this sample exceeds the function call limit
        if max_function_calls is not None and len(calls) > max_function_calls:
            # If exceeded, try again with different random choices
            continue
        
        # Only return sample if we have valid data
        if tools and calls:
            return {
                "user_input": "".join(queries),
                "tools": tools,
                "function_calls": calls,
                "has_same_tool_multiple_calls": has_multi_calls
            }
        else:
            # If no valid tools/calls, try again
            continue
    
    return None


def save_training_data(data, filename="function_calling_data.json", split_name=""):
    """Save data in format compatible with training pipeline"""
    
    # Enhanced analysis showing precise tool usage breakdown
    tool_count_stats = {1: 0, 2: 0, 3: 0, 4: 0}  # Support up to 4 tools if needed
    samples_with_multi_calls = 0
    
    for sample in data:
        tools = sample['tools']
        unique_tools = len(set(tools))  # Count unique tools
        
        # Count by unique tools (this is what the user wants to control)
        if unique_tools in tool_count_stats:
            tool_count_stats[unique_tools] += 1
        
        # Count samples that have multiple calls of the same tool
        if len(tools) > unique_tools:
            samples_with_multi_calls += 1
    
    total_samples = len(data)
    
    if split_name:
        print(f"\n{split_name} Dataset: {total_samples} samples, {samples_with_multi_calls} with multi-calls")
    
    # Remove metadata before saving
    clean_data = []
    for item in data:
        clean_item = {k: v for k, v in item.items() if k != 'has_same_tool_multiple_calls'}
        clean_data.append(clean_item)
    
    with open(filename, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    if split_name:
        print(f"Saved {split_name} data: {filename}")
    return filename

def verify_ratios(data, intended_ratios, split_name=""):
    """
    Verify that the generated data matches the intended approach.
    Single-tool: All available samples included
    Multi-tool: Ratios among multi-tool samples only
    """
    tool_count_stats = {1: 0, 2: 0, 3: 0}
    
    for sample in data:
        tools = sample['tools']
        unique_tools = len(set(tools))  # Count unique tools
        if unique_tools in tool_count_stats:
            tool_count_stats[unique_tools] += 1
    
    total_samples = len(data)
    single_tool_samples = tool_count_stats[1]
    multi_tool_samples = total_samples - single_tool_samples
    
    if split_name:
        print(f"{split_name} verification: {single_tool_samples} single-tool, {multi_tool_samples} multi-tool samples")
    
    # Return actual ratios for compatibility
    actual_ratios = {k: v/total_samples for k, v in tool_count_stats.items() if v > 0}
    return actual_ratios

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="XLAM Dataset Processor & Multi-Tool Synthesizer with Train/Test Split")
    parser.add_argument("--train_size", type=int, default=5000, help="Number of training samples to generate")
    parser.add_argument("--test_size", type=int, default=500, help="Number of test samples to generate")
    
    def parse_top_k(value):
        """Parse top_k argument which can be either a number or a range like '10-20'"""
        if '-' in value:
            parts = value.split('-')
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"Invalid range format: {value}. Use 'start-end' format.")
            try:
                start = int(parts[0])
                end = int(parts[1])
                if start < 1:
                    raise argparse.ArgumentTypeError(f"Invalid range: {value}. Start must be >= 1 (1-indexed), not {start}.")
                if end < start:
                    raise argparse.ArgumentTypeError(f"Invalid range: {value}. End ({end}) must be >= start ({start}).")
                return (start, end)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid range format: {value}. Both start and end must be integers.")
        else:
            try:
                k = int(value)
                if k < 1:
                    raise argparse.ArgumentTypeError(f"top_k must be >= 1, got {k}")
                return k
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid top_k value: {value}. Must be an integer or range like '10-20'.")
    
    parser.add_argument("--top_k", type=parse_top_k, default=20, 
                       help="Number of top tools to use (e.g., 20) or range (e.g., '10-20' for tools ranked 10-20)")
    parser.add_argument("--max_samples_per_tool", type=int, default=None, help="Maximum number of samples per tool (None for no limit)")
    
    def parse_ratios(ratio_string):
        """Parse comma-separated ratios like '0.6,0.4'"""
        try:
            ratios = [float(x.strip()) for x in ratio_string.split(',')]
            if len(ratios) != 2:
                raise ValueError(f"Expected exactly 2 ratios, got {len(ratios)}")
            if abs(sum(ratios) - 1.0) > 0.001:
                raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.3f}")
            return ratios
        except Exception as e:
            raise argparse.ArgumentTypeError(f"Invalid ratio format: {e}")
    
    parser.add_argument("--train_multi_tool_ratios", type=parse_ratios, default=[0.5,0.5],
                       help="Training multi-tool ratios as 'ratio2,ratio3' (e.g., '0.6,0.4')")
    parser.add_argument("--test_multi_tool_ratios", type=parse_ratios, default=[0.5,0.5],
                       help="Test multi-tool ratios as 'ratio2,ratio3' (e.g., '0.6,0.4')")
    parser.add_argument("--train_max_function_calls", type=int, default=4, help="Maximum number of function calls per sample (None for no limit)")
    parser.add_argument("--test_max_function_calls", type=int, default=4, help="Maximum number of function calls per sample (None for no limit)")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for generated files (default: current directory)")
    
    args = parser.parse_args()
    
    # Calculate train ratio
    total_samples = args.train_size + args.test_size
    train_ratio = args.train_size / total_samples
    
    # Handle ratios - single-tool samples are included completely, multi-tool ratios are normalized
    # Auto-adjust ratios if max function calls is too low for 3-tool samples
    train_ratios = {2: args.train_multi_tool_ratios[0], 3: args.train_multi_tool_ratios[1]}
    test_ratios = {2: args.test_multi_tool_ratios[0], 3: args.test_multi_tool_ratios[1]}
    
    if args.train_max_function_calls is not None and args.train_max_function_calls < 3:
        print(f"Warning: train_max_function_calls={args.train_max_function_calls} < 3, setting 3-tool ratio to 0")
        train_ratios = {2: 1.0, 3: 0.0}
    
    if args.test_max_function_calls is not None and args.test_max_function_calls < 3:
        print(f"Warning: test_max_function_calls={args.test_max_function_calls} < 3, setting 3-tool ratio to 0")
        test_ratios = {2: 1.0, 3: 0.0}
    
    print("Processing XLAM dataset...")
    
    # Set seed for data extraction and splitting (test-independent operations)
    random.seed(42)
    df = load_xlam_dataset()
    tool_data, tool_names = extract_single_tool_data(df, k=args.top_k, max_samples_per_tool=args.max_samples_per_tool)
    tool_descriptions = extract_tool_descriptions(df, tool_names)
    
    # Create tool description filename with range information
    if isinstance(args.top_k, tuple):
        tool_desc_filename = f"tool_descriptions_tools{args.top_k[0]}-{args.top_k[1]}.json"
    else:
        tool_desc_filename = f"tool_descriptions_top{args.top_k}.json"
    
    # Save in output directory
    tool_desc_path = os.path.join(args.output_dir, tool_desc_filename)
    save_tool_descriptions(tool_descriptions, tool_desc_path)
    print(f"Saved tool descriptions: {tool_desc_path}")
    
    train_tool_data, test_tool_data = split_single_tool_data(tool_data, train_ratio)
    
    # Use separate seeds for train and test synthesis to ensure test consistency
    random.seed(100)  # Training seed
    train_data = synthesize_multi_tool_data(train_tool_data, tool_names, args.train_size, train_ratios, "training", multi_call_probability=0.1, max_function_calls=args.train_max_function_calls)
    
    random.seed(200)  # Test seed - consistent regardless of training parameters
    test_data = synthesize_multi_tool_data(test_tool_data, tool_names, args.test_size, test_ratios, "test", multi_call_probability=0.1, max_function_calls=args.test_max_function_calls)
    
    verify_ratios(train_data, train_ratios, "Training")
    verify_ratios(test_data, test_ratios, "Test")
    
    # Create filenames based on max function calls and tool selection
    # Add tool range/top-k information to filename
    if isinstance(args.top_k, tuple):
        tool_suffix = f"_tools{args.top_k[0]}-{args.top_k[1]}"
    else:
        tool_suffix = f"_top{args.top_k}"
    
    # Create output directories
    train_dir = os.path.join(args.output_dir, "training")
    test_dir = os.path.join(args.output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    if args.train_max_function_calls:
        train_filename_base = f"function_calling_train{tool_suffix}_{args.train_max_function_calls}calls.json"
    else:
        train_filename_base = f"function_calling_train{tool_suffix}_unlimited.json"
    
    if args.test_max_function_calls:
        test_filename_base = f"function_calling_test{tool_suffix}_{args.test_max_function_calls}calls.json"
    else:
        test_filename_base = f"function_calling_test{tool_suffix}_unlimited.json"
    
    train_filename = save_training_data(train_data, os.path.join(train_dir, train_filename_base), "Training")
    test_filename = save_training_data(test_data, os.path.join(test_dir, test_filename_base), "Test")
    
    print(f"\nComplete!")
    print(f"  Training data: {train_filename}")
    print(f"  Test data: {test_filename}")
    print(f"  Tool descriptions: {tool_desc_path}")


if __name__ == "__main__":
    main()


