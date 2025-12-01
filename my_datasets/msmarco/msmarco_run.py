from datasets import load_dataset
import json
import os

dataset = load_dataset("microsoft/ms_marco", "v1.1", split = "validation", num_proc=1)

# Extract examples 4-53 (50 examples, skipping first 3)
examples_4_to_53 = []
for i in range(4, 103):
    examples_4_to_53.append(dataset[i])

# Save to JSON file in the same directory
output_file = os.path.join(os.path.dirname(__file__), "msmarco_examples_4_to_103.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(examples_4_to_53, f, indent=2, ensure_ascii=False)

print(f"Examples 4-53 (50 examples) saved to: {output_file}")
