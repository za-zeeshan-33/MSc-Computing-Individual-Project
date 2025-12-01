import jsonlines
import json

def extract_hagrid_data(input_file, output_file):
    """
    Extract only HAGRID examples from the OOD test data
    """
    hagrid_examples = []
    
    with jsonlines.open(input_file) as reader:
        for line in reader:
            # Fix case sensitivity - look for HAGRID (uppercase)
            if line.get('src_dataset', '').upper().startswith('HAGRID'):
                hagrid_examples.append(line)
    
    print(f"Found {len(hagrid_examples)} HAGRID examples")
    
    # Save as JSONL for easy processing
    with jsonlines.open(output_file, 'w') as writer:
        for example in hagrid_examples:
            writer.write(example)
    
    print(f"Saved HAGRID data to {output_file}")
    return hagrid_examples

def create_generation_input(hagrid_file, output_file):
    """
    Create input file for answer generation with references
    """
    generation_inputs = []
    
    with jsonlines.open(hagrid_file) as reader:
        for line in reader:
            generation_inputs.append({
                'id': line['id'],
                'question': line['question'],
                'references': line.get('references', []),
                'src_dataset': line['src_dataset']
            })
    
    with jsonlines.open(output_file, 'w') as writer:
        for item in generation_inputs:
            writer.write(item)
    
    print(f"Created generation input file with references: {output_file}")

if __name__ == "__main__":
    # Modify these paths based on your setup
    input_file = "data/test_ood_all_subset_balanced.jsonl"  # Your OOD test file
    hagrid_output = "data/hagrid_test.jsonl"
    generation_input = "data/hagrid_questions.jsonl"
    
    # Extract HAGRID data
    extract_hagrid_data(input_file, hagrid_output)
    
    # Create generation input with references
    create_generation_input(hagrid_output, generation_input)