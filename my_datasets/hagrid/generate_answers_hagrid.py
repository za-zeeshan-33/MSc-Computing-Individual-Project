import jsonlines
import openai  # or use transformers for local models
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse
import os

from dotenv import load_dotenv

load_dotenv()

class AnswerGenerator:
    def __init__(self, model_name, use_openai=True):
        self.model_name = model_name
        self.use_openai = use_openai
        
        if use_openai:
            # For GPT models via API
            self.client = openai.OpenAI()
        else:
            # For local models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,
                device_map="auto"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_answer_with_references(self, question, references, max_length=512):
        """Generate answer for a single question using provided references"""
        
        if self.use_openai:
            # Format references for the prompt
            references_text = ""
            if references:
                references_text = "\n\nReferences:\n"
                for i, ref in enumerate(references, 1):
                    references_text += f"[{i}] {ref}\n"
            
            # OpenAI API call with references
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions using ONLY the provided references. If the references don't contain enough information to answer the question completely, say so. Always cite your sources using [1], [2], etc. format."},
                    {"role": "user", "content": f"Question: {question}{references_text}\n\nPlease answer the question using the references above. Cite your sources using [1], [2], etc."}
                ],
                max_tokens=max_length,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        else:
            # Local model generation with references
            references_text = ""
            if references:
                references_text = "\n\nReferences:\n"
                for i, ref in enumerate(references, 1):
                    references_text += f"[{i}] {ref}\n"
            
            prompt = f"Question: {question}{references_text}\n\nAnswer using the references above and cite sources [1], [2], etc.:"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return generated_text.strip()

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

def main(args):
    # Initialize generator
    generator = AnswerGenerator(args.model_name, args.use_openai)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    generated_answers = []
    question_count = 0
    max_questions = args.max_questions  # Use command line argument
    
    print(f"Generating answers using {args.model_name}...")
    print(f"Processing first {max_questions} questions only...")
    
    with jsonlines.open(args.input_file) as reader:
        for line in tqdm(reader, desc="Generating answers"):
            # Stop after processing max_questions
            if question_count >= max_questions:
                break
                
            try:
                # Get references for this question
                references = line.get('references', [])
                
                # Generate answer using references
                answer = generator.generate_answer_with_references(
                    line['question'], 
                    references, 
                    args.max_length
                )
                
                result = {
                    'id': line['id'],
                    'question': line['question'],
                    'references': references,
                    'generated_answer': answer,
                    'src_dataset': line['src_dataset'],
                    'model_used': args.model_name
                }
                
                generated_answers.append(result)
                question_count += 1
                
            except Exception as e:
                print(f"Error generating answer for {line['id']}: {e}")
                continue
    
    # Save generated answers
    with jsonlines.open(args.output_file, 'w') as writer:
        for result in generated_answers:
            writer.write(result)
    
    print(f"Generated {len(generated_answers)} answers (limited to first {max_questions} questions)")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="data/hagrid_test.jsonl", 
                       help="File with HAGRID questions and references")
    parser.add_argument("--output_file", default="results/hagrid_generated_answers.jsonl",
                       help="Output file for generated answers")
    parser.add_argument("--model_name", default="gpt-3.5-turbo", 
                       help="Model to use for generation")
    parser.add_argument("--use_openai", action="store_true",
                       help="Use OpenAI API (for GPT models)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum length for generated answers")
    parser.add_argument("--max_questions", type=int, default=10,
                       help="Maximum number of questions to process")
    
    args = parser.parse_args()
    
    # First create the input file with references if it doesn't exist
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} not found. Creating it from HAGRID data...")
        create_generation_input("data/hagrid_test.jsonl", args.input_file)
    
    main(args)