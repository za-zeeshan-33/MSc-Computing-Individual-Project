from datasets import Dataset
import pandas as pd
import json
import os

# Load the local parquet file
parquet_file = os.path.join(os.path.dirname(__file__), "validation-00000-of-00001.parquet")
df = pd.read_parquet(parquet_file)
dataset = Dataset.from_pandas(df)

# Extract first 10 examples
first_10_examples = []
for i in range(10):
    first_10_examples.append(dataset[i])

# Save to JSON file in the same directory
output_file = os.path.join(os.path.dirname(__file__), "msmarco_v2_1_first_10_examples.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(first_10_examples, f, indent=2, ensure_ascii=False)

print(f"First 10 examples saved to: {output_file}")
