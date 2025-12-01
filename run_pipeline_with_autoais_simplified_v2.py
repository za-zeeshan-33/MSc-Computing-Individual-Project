#!/usr/bin/env python3
"""
Run the simplified attribution pipeline (ANSWERS ONLY VERSION)
Uses the simplified prompt manager for cleaner, more focused prompts
Generates answers only - no evaluation
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from attribution_pipeline_improved import ImprovedAttributionPipeline
from simplified_prompt_manager import SimplifiedPromptManager, MethodType

# Load environment variables
load_dotenv()

# Dataset configurations
DATASETS = {
    'asqa': {
        'file': 'my_datasets/asqa/asqa_examples_4_to_103.json',
        'loader': 'asqa'
    },
    'natural_questions': {
        'file': 'my_datasets/dpr_nq/natural_questions_examples_4_to_103.json',
        'loader': 'natural_questions'
    },
    'eli5': {
        'file': 'my_datasets/eli5/eli5_examples_4_to_103.json',
        'loader': 'eli5'
    },
    'hagrid': {
        'file': 'my_datasets/hagrid/hagrid_filtered_quotes_max5.json',
        'loader': 'hagrid'
    },
    'msmarco': {
        'file': 'my_datasets/msmarco/msmarco_examples_4_to_103.json',
        'loader': 'msmarco'
    },
    'qampari': {
        'file': 'my_datasets/qampari/qampari_examples_4_to_103.json',
        'loader': 'qampari'
    }
}

APPROACHES = ['post-retrieval', 'post-generation-llm-short', 'post-generation-llm-long', 'post-generation-tfidf']

def run_simplified_pipeline_answers_only(datasets=None, approaches=None, num_few_shot=1, num_examples=None, model_name='gpt-3.5-turbo', seed=None, save_initial_answers=True):
    """Run simplified pipeline - ANSWERS ONLY (no evaluation)"""
    
    if datasets is None:
        datasets = list(DATASETS.keys())
    if approaches is None:
        approaches = APPROACHES
    
    # Load config
    config = {
        'model_name': model_name,
        'top_k': 5,
        'temperature': 0.5,  # Updated from 0.3 to 0.5 for all approaches
        'max_tokens': 500,
        'num_few_shot': num_few_shot,
        'seed': seed,
        # Add top_p configuration: 1.0 for OpenAI models, 0.95 for others
        'top_p': 1.0 if 'gpt' in model_name.lower() or 'o1' in model_name.lower() else 0.95
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean model name for folder (replace special characters)
    clean_model_name = model_name.replace('-', '_').replace('.', '_')
    
    # Add few-shot suffix to directory name
    few_shot_suffix = f"_few_shot_{num_few_shot}"
    
    # Add num_examples suffix if specified
    if num_examples is not None and num_examples > 0:
        num_examples_suffix = f"_num_examples_{num_examples}"
    else:
        num_examples_suffix = ""
    
    results_dir = f"results/results_simplified_answers_only_{timestamp}_{clean_model_name}{few_shot_suffix}{num_examples_suffix}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Few-shot examples: {num_few_shot}")
    if num_examples:
        print(f"Num examples: {num_examples}")
    print(f"ANSWERS ONLY MODE - No evaluation will be performed")
    
    # Initialize simplified prompt manager
    prompt_manager = SimplifiedPromptManager()
    
    all_results = {}
    
    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"Running {approach.upper()} approach - ANSWERS ONLY")
        print(f"Few-shot examples: {num_few_shot}")
        print(f"{'='*60}")
        
        config['approach'] = approach
        pipeline = ImprovedAttributionPipeline(config)
        
        # Replace the pipeline's prompt manager with our simplified one
        pipeline.prompt_manager = prompt_manager
        
        approach_results = {}
        
        for dataset_name in datasets:
            print(f"\nProcessing {dataset_name} dataset...")
            
            dataset_config = DATASETS[dataset_name]
            
            # Generate answers using pipeline
            output_file = f"{results_dir}/{dataset_name}_{approach}_results.json"
            
            # Use the pipeline's run_pipeline method
            pipeline.run_pipeline(dataset_name, dataset_config['file'], output_file, num_examples, save_initial_answers)
            
            # Load the results that were just saved
            with open(output_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            approach_results[dataset_name] = results
            
            print(f"   Generated {len(results)} answers")
            print(f"   Saved to: {output_file}")
        
        all_results[approach] = approach_results
    
    # Save comprehensive summary
    summary = {
        "timestamp": timestamp,
        "model_name": model_name,
        "config": config,
        "datasets": datasets,
        "approaches": approaches,
        "num_few_shot": num_few_shot,
        "num_examples": num_examples,
        "results_dir": results_dir,
        "mode": "answers_only",
        "total_examples_per_dataset": {ds: len(all_results[approaches[0]][ds]) for ds in datasets},
        "approaches_completed": list(all_results.keys())
    }
    
    summary_file = f"{results_dir}/simplified_answers_only_pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"ANSWERS GENERATION COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print(f"Summary saved to: {summary_file}")
    print(f"\nTo evaluate these results, run:")
    print(f"   python evaluation_autoais_v2.py --results_dir {results_dir}")
    print(f"{'='*60}")
    
    return results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run simplified attribution pipeline - ANSWERS ONLY')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASETS.keys()), 
                        help='Datasets to process (default: all)')
    parser.add_argument('--approaches', nargs='+', choices=APPROACHES,
                        help='Approaches to run (default: all)')
    parser.add_argument('--num_few_shot', type=int, default=1,
                        help='Number of few-shot examples (default: 1)')
    parser.add_argument('--num_examples', type=int, default=None,
                        help='Limit number of examples per dataset (default: all)')
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        help='Model to use (default: gpt-3.5-turbo)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--answers-only', action='store_true',
                        help='Generate answers only (no evaluation) - this is the default mode for this script')
    parser.add_argument('--save-initial-answers', action='store_true',
                        help='Save initial answers (before citation prompts) for post-generation approaches')
    
    args = parser.parse_args()
    
    if args.answers_only or True:  # This script is always answers-only
        print("Running in ANSWERS ONLY mode")
        results_dir = run_simplified_pipeline_answers_only(
            datasets=args.datasets,
            approaches=args.approaches,
            num_few_shot=args.num_few_shot,
            num_examples=args.num_examples,
            model_name=args.model,
            seed=args.seed,
            save_initial_answers=args.save_initial_answers
        )
    else:
        print("Error: This script only supports --answers-only mode. Use run_pipeline_with_autoais_simplified.py for full pipeline.")
