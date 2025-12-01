#!/usr/bin/env python3
"""
Script to extract X examples from Natural Questions dataset, skipping the first 3 examples
that are used for few-shot demonstrations.
"""

import json
import os
import argparse

def extract_nq_samples(num_samples=50, skip_first=3):
    """Extract num_samples examples from Natural Questions, skipping the first skip_first examples"""
    
    input_file = "biencoder-nq-dev.json"
    
    print(f"📚 Processing Natural Questions...")
    print(f"   Input: {input_file}")
    print(f"   Samples: {num_samples}")
    print(f"   Skipping first: {skip_first} examples (used for few-shot)")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    try:
        # Load the full dataset
        print(f"   Loading dataset...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   Total examples in dataset: {len(data)}")
        
        # Check if we have enough examples after skipping
        available_after_skip = len(data) - skip_first
        if available_after_skip <= 0:
            print(f"   ❌ Not enough examples after skipping first {skip_first}")
            return False
        
        # Extract examples starting from skip_first + 1
        start_idx = skip_first
        end_idx = start_idx + num_samples
        
        if end_idx > len(data):
            print(f"   ⚠️  Only {available_after_skip} examples available after skipping, extracting all of them")
            sample_data = data[start_idx:]
        else:
            sample_data = data[start_idx:end_idx]
        
        # Process the examples
        examples = []
        for i, example in enumerate(sample_data):
            # Extract question and answers
            question = example["question"]
            answers = example["answers"]
            
            # Extract positive contexts (relevant documents)
            relevant_docs = []
            for ctx in example["positive_ctxs"]:
                relevant_docs.append({
                    "doc_id": ctx["passage_id"],
                    "title": ctx["title"],
                    "text": ctx["text"],
                    "score": ctx["score"]
                })
            
            # Create example in format similar to existing NQ files
            formatted_example = {
                "query_id": start_idx + i + 1,  # Generate sequential IDs starting from skip_first + 1
                "question": question,
                "answers": answers,
                "relevant_docs": relevant_docs,
                "num_relevant_docs": len(relevant_docs),
                "dataset": example["dataset"]
            }
            
            examples.append(formatted_example)
            print(f"   Extracted example {start_idx + i + 1}: {question[:50]}...")
        
        # Create output filename
        start_example_num = start_idx + 1
        end_example_num = start_idx + len(examples)
        output_file = f"natural_questions_examples_{start_example_num}_to_{end_example_num}.json"
        
        # Save the sample dataset
        print(f"   Saving {len(examples)} examples...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Successfully saved {len(examples)} examples to {output_file}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error processing Natural Questions: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract X examples from Natural Questions dataset')
    parser.add_argument('--samples', type=int, default=50, 
                       help='Number of examples to extract (default: 50)')
    parser.add_argument('--skip-first', type=int, default=3,
                       help='Number of examples to skip from the beginning (used for few-shot) (default: 3)')
    
    args = parser.parse_args()
    
    print(f"🚀 Extracting {args.samples} examples from Natural Questions")
    print(f"📋 Skipping first {args.skip_first} examples (used for few-shot)")
    print("="*80)
    
    success = extract_nq_samples(args.samples, args.skip_first)
    
    print("="*80)
    if success:
        print("✅ Natural Questions processing completed successfully!")
    else:
        print("❌ Natural Questions processing failed!")

if __name__ == "__main__":
    main()
