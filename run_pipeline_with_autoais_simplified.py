#!/usr/bin/env python3
"""
Run the simplified attribution pipeline with AutoAIS evaluation
Uses the simplified prompt manager for cleaner, more focused prompts
"""

import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from attribution_pipeline_improved import ImprovedAttributionPipeline
from evaluation_autoais import AutoAISCitationEvaluator
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

def run_simplified_pipeline_with_autoais(datasets=None, approaches=None, num_few_shot=1, num_examples=None, capture_logs=True, model_name='gpt-3.5-turbo', seed=None, space_slug='za-zeeshan-33/true-model', hf_token=None, save_initial_answers=True):
    """Run simplified pipeline with AutoAIS evaluation"""
    
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
    
    results_dir = f"results/results_simplified_autoais_{timestamp}_{clean_model_name}{few_shot_suffix}{num_examples_suffix}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Results directory: {results_dir}")
    print(f"Few-shot examples: {num_few_shot}")
    if num_examples:
        print(f"Num examples: {num_examples}")
    
    # Initialize simplified prompt manager
    prompt_manager = SimplifiedPromptManager()
    
    all_results = {}
    
    for approach in approaches:
        print(f"\n{'='*60}")
        print(f"Running {approach.upper()} approach with SIMPLIFIED prompts + AutoAIS evaluation")
        print(f"Few-shot examples: {num_few_shot}")
        print(f"{'='*60}")
        
        config['approach'] = approach
        pipeline = ImprovedAttributionPipeline(config)
        
        # Replace the pipeline's prompt manager with our simplified one
        pipeline.prompt_manager = prompt_manager
        
        # Always use AutoAIS evaluator with Space access
        # Use provided token or fall back to environment variable
        if hf_token is None:
            hf_token = os.getenv('HF_TOKEN')
        
        evaluator = AutoAISCitationEvaluator(
            space_slug=space_slug,
            hf_token=hf_token,
            batch_size=3,
            delay_seconds=0.5,
            entailment_delay=0.5,
            capture_logs=capture_logs,
            enable_citations=True,
            enable_qa=True,
            enable_text_similarity=True
        )
        
        approach_results = {}
        
        for dataset_name in datasets:
            if dataset_name not in DATASETS:
                print(f"Warning: Unknown dataset: {dataset_name}")
                continue
                
            dataset_info = DATASETS[dataset_name]
            print(f"\nProcessing {dataset_name}...")
            
            try:
                # Run pipeline
                output_file = os.path.join(results_dir, f"{dataset_name}_{approach}_results.json")
                results = pipeline.run_pipeline(
                    dataset_info['loader'],
                    dataset_info['file'],
                    output_file,
                    num_examples,
                    save_initial_answers
                )
                
                # Evaluate results
                eval_output = os.path.join(results_dir, f"{dataset_name}_{approach}_evaluation.json")
                metrics = evaluator.evaluate_dataset(output_file, dataset_info['file'])
                evaluator.generate_report(metrics, eval_output)
                
                approach_results[dataset_name] = {
                    'results_file': output_file,
                    'evaluation_file': eval_output,
                    'metrics': metrics,
                    'num_examples': len(results),
                    'num_few_shot': num_few_shot
                }
                
                print(f"✓ {dataset_name} completed - {len(results)} examples processed")
                
            except Exception as e:
                print(f"✗ Error processing {dataset_name}: {e}")
                approach_results[dataset_name] = {
                    'error': str(e),
                    'num_examples': 0,
                    'num_few_shot': num_few_shot
                }
        
        all_results[approach] = approach_results
    
    # Save summary
    summary_file = os.path.join(results_dir, "simplified_autoais_pipeline_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'evaluation_type': 'autoais',
            'prompt_type': 'simplified',
            'num_few_shot': num_few_shot,
            'approaches': all_results,
            'config': config
        }, f, indent=2)
    
    # Generate comparison report
    generate_comparison_report(all_results, results_dir, 'simplified_autoais', num_few_shot)
    
    print(f"\n{'='*60}")
    print(f"SIMPLIFIED AUTOAIS PIPELINE COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"Summary: {summary_file}")
    print(f"Few-shot examples used: {num_few_shot}")
    
    return all_results

def generate_comparison_report(all_results, results_dir, eval_type='simplified_autoais', num_few_shot=1):
    """Generate comparison report between approaches"""
    
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_type': eval_type,
        'prompt_type': 'simplified',
        'num_few_shot': num_few_shot,
        'approaches': {},
        'summary': {}
    }
    
    for approach, datasets in all_results.items():
        approach_summary = {
            'approach': approach,
            'datasets_processed': [],
            'total_examples': 0,
            'dataset_results': {},
            'evaluation_summary': {}
        }
        
        for dataset_name, dataset_info in datasets.items():
            if 'error' not in dataset_info:
                approach_summary['datasets_processed'].append(dataset_name)
                approach_summary['total_examples'] += dataset_info['num_examples']
                approach_summary['dataset_results'][dataset_name] = dataset_info['num_examples']
                
                # Extract AutoAIS metrics
                metrics = dataset_info['metrics']
                approach_summary['evaluation_summary'][dataset_name] = {
                    'citation_f1': metrics.get('citation_f1_autoais_mean', 0),
                    'citation_precision': metrics.get('citation_precision_autoais_mean', 0),
                    'citation_recall': metrics.get('citation_recall_autoais_mean', 0),
                    'semantic_similarity': metrics.get('semantic_similarity_mean', -1),
                    'answer_completeness': metrics.get('answer_completeness_mean', 0)
                }
        
        comparison['approaches'][approach] = approach_summary
    
    # Save comparison
    comparison_file = os.path.join(results_dir, f"{eval_type}_approach_comparison.json")
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SIMPLIFIED AUTOAIS PIPELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Few-shot examples: {num_few_shot}")
    
    for approach, summary in comparison['approaches'].items():
        print(f"\n{approach.upper()} Approach:")
        print(f"  Total examples: {summary['total_examples']}")
        print(f"  Datasets: {', '.join(summary['datasets_processed'])}")
        
        print(f"  AutoAIS Citation Quality (F1):")
        for dataset, metrics in summary['evaluation_summary'].items():
            f1 = metrics['citation_f1']
            print(f"    {dataset}: {f1:.3f}")
        
        # Calculate average F1
        f1_scores = [m['citation_f1'] for m in summary['evaluation_summary'].values()]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        print(f"    Average: {avg_f1:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Run simplified attribution pipeline with AutoAIS evaluation')
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                       help='Datasets to process (default: all)')
    parser.add_argument('--approaches', nargs='+', default=APPROACHES,
                       help='Approaches to run (default: all)')
    parser.add_argument('--few-shot', type=int, default=1,
                       help='Number of few-shot examples to use (default: 1)')
    parser.add_argument('--num-examples', type=int, default=None,
                       help='Number of examples to process from each dataset (default: all)')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo',
                       help='Model name to use (default: gpt-3.5-turbo)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible results (default: None)')
    parser.add_argument('--hf-token', type=str, help='Hugging Face token for AutoAIS (or set HF_TOKEN env var)')
    parser.add_argument('--space-slug', type=str, default='za-zeeshan-33/true-model', help='AutoAIS Space slug (default: za-zeeshan-33/true-model)')
    parser.add_argument('--no-logs', action='store_true', help='Disable detailed evaluation logging (cleaner output)')
    parser.add_argument('--save-initial-answers', action='store_true',
                        help='Save initial answers (before citation prompts) for post-generation approaches')
    
    args = parser.parse_args()
    
    # Set HF_TOKEN if provided
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    
    # Check AutoAIS requirements
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("Error: HF_TOKEN not set. Please provide --hf-token or set HF_TOKEN environment variable")
        print("Note: You can get a Hugging Face token from: https://huggingface.co/settings/tokens")
        return
    
    print("Using SIMPLIFIED prompts with AutoAIS evaluation")
    print("Model:", args.model)
    print("Few-shot examples:", args.few_shot)
    if args.seed is not None:
        print("Seed:", args.seed, "(reproducible mode)")
    else:
        print("Seed: None (non-deterministic mode)")
    if args.num_examples:
        print(f"Processing {args.num_examples} examples per dataset")
    else:
        print("Processing all examples in each dataset")
    print("Note: This will be slower due to API calls")
    print("Clean, focused prompting enabled for all datasets")
    
    run_simplified_pipeline_with_autoais(
        datasets=args.datasets,
        approaches=args.approaches,
        num_few_shot=args.few_shot,
        num_examples=args.num_examples,
        capture_logs=not args.no_logs,
        model_name=args.model,
        seed=args.seed,
        space_slug=args.space_slug,
        hf_token=args.hf_token,
        save_initial_answers=args.save_initial_answers
    )

if __name__ == "__main__":
    main()
