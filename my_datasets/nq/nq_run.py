#!/usr/bin/env python3

import json
import gzip
import os

def extract_nq_examples(num_examples=25):
    """Extract first N examples from DPR Natural Questions dev dataset"""
    
    input_file = "biencoder-nq-dev.json.gz"
    
    print(f"Loading {input_file}...")
    print("This may take a moment as the file is large...")
    
    # Read the compressed JSON file
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from the dataset")
    
    # Extract first N examples
    examples = []
    for i in range(min(num_examples, len(data))):
        example = data[i]
        
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
        
        # Create example in format similar to MS MARCO
        formatted_example = {
            "query_id": i + 1,  # Generate sequential IDs
            "question": question,
            "answers": answers,
            "relevant_docs": relevant_docs,
            "num_relevant_docs": len(relevant_docs),
            "dataset": example["dataset"]
        }
        
        examples.append(formatted_example)
        print(f"Extracted example {i+1}: {question[:50]}...")
    
    return examples

def main():
    """Main function to extract and save examples"""
    
    # Create directory if it doesn't exist
    script_dir = os.path.dirname(__file__)
    os.makedirs(script_dir, exist_ok=True)
    
    # Extract first 3 examples
    examples = extract_nq_examples(25)
    
    # Save to JSON file
    output_file = os.path.join(script_dir, "natural_questions_first_25_examples.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ First 25 examples saved to: {output_file}")
    print(f"Total examples extracted: {len(examples)}")
    
    # Print summary
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Query ID: {example['query_id']}")
        print(f"  Question: {example['question']}")
        print(f"  Answers: {example['answers']}")
        print(f"  Relevant docs: {example['num_relevant_docs']}")
        print(f"  Dataset: {example['dataset']}")

if __name__ == "__main__":
    main()
