from datasets import load_dataset
import json
import os

dataset = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/miracl/hagrid/resolve/main/hagrid-v1.0-en/dev.jsonl",
    split="train"  # JSON loader always names it 'train' unless you remap
)

# Extract examples 4-53 (50 examples, skipping first 3)
examples_4_to_53 = []
for i in range(4, 203):
    examples_4_to_53.append(dataset[i])

# Save to JSON file in the same directory
output_file = os.path.join(os.path.dirname(__file__), "hagrid_examples_4_to_203.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(examples_4_to_53, f, indent=2, ensure_ascii=False)

print(f"Examples 4-53 (50 examples) saved to: {output_file}")

