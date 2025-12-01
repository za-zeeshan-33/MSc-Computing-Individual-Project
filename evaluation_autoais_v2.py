#!/usr/bin/env python3
"""
AutoAIS-based evaluation module for citation quality assessment (EVALUATION ONLY VERSION)
Evaluates pre-generated results from run_pipeline_with_autoais_simplified_v2.py
"""

import json
import os
import argparse
import glob
import tempfile
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

# Import evaluation components from original file
from evaluation_autoais import AutoAISCitationEvaluator

def load_results_from_directory(results_dir: str) -> Dict[str, Dict[str, List[Dict]]]:
    """Load all result files from a results directory"""
    
    if not os.path.exists(results_dir):
        raise ValueError(f"Results directory not found: {results_dir}")
    
    print(f"Loading results from: {results_dir}")
    
    # Find all result files
    result_files = glob.glob(os.path.join(results_dir, "*_results.json"))
    
    if not result_files:
        raise ValueError(f"No result files found in: {results_dir}")
    
    results = defaultdict(dict)
    
    for file_path in result_files:
        filename = os.path.basename(file_path)
        
        # Parse filename: {dataset}_{approach}_results.json
        # Handle known dataset names that contain underscores
        known_datasets = ['natural_questions']
        
        # Find the dataset name by matching against known datasets
        dataset = None
        approach = None
        
        for known_dataset in known_datasets:
            if filename.startswith(known_dataset + '_'):
                dataset = known_dataset
                # Remove dataset name and _results.json to get approach
                approach = filename.replace(known_dataset + '_', '').replace('_results.json', '')
                break
        
        if dataset is None:
            # Fallback to original logic for unknown datasets
            parts = filename.replace('_results.json', '').split('_')
            if len(parts) < 2:
                print(f"Warning: Skipping file with unexpected format: {filename}")
                continue
            dataset = parts[0]
            approach = '_'.join(parts[1:])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results[approach][dataset] = data
        print(f"   Loaded {dataset}_{approach}: {len(data)} examples")
    
    # Convert to regular dict for easier handling
    return dict(results)

def create_simplified_autoais_approach_comparison(evaluation_results: Dict[str, Dict[str, Dict]], results_dir: str) -> Dict:
    """Create approach comparison in the same format as original pipeline"""
    
    # Extract metadata from results directory name
    dir_name = os.path.basename(results_dir)
    parts = dir_name.split('_')
    
    # Try to extract few_shot number from directory name
    num_few_shot = 0
    for i, part in enumerate(parts):
        if part == "shot" and i > 0:
            try:
                num_few_shot = int(parts[i-1])
                break
            except ValueError:
                pass
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "simplified_autoais",
        "prompt_type": "simplified", 
        "num_few_shot": num_few_shot,
        "approaches": {}
    }
    
    # Organize by approach in original format
    for approach, datasets in evaluation_results.items():
        total_examples = 0
        datasets_processed = []
        dataset_results = {}
        evaluation_summary = {}
        
        for dataset_name, eval_result in datasets.items():
            if 'error' not in eval_result:
                datasets_processed.append(dataset_name)
                
                # Count examples from one of the metrics arrays if available
                if 'citation_precision_autoais_per_example' in eval_result:
                    dataset_count = len(eval_result['citation_precision_autoais_per_example'])
                    dataset_results[dataset_name] = dataset_count
                    total_examples += dataset_count
                
                # Create evaluation summary for this dataset
                evaluation_summary[dataset_name] = {
                    "citation_f1": eval_result.get('citation_f1_autoais_mean', 0),
                    "citation_precision": eval_result.get('citation_precision_autoais_mean', 0),
                    "citation_recall": eval_result.get('citation_recall_autoais_mean', 0),
                    "semantic_similarity": eval_result.get('semantic_similarity_mean', 0),
                    "answer_completeness": eval_result.get('answer_completeness_mean', 0)
                }
        
        comparison["approaches"][approach] = {
            "approach": approach,
            "datasets_processed": datasets_processed,
            "total_examples": total_examples,
            "dataset_results": dataset_results,
            "evaluation_summary": evaluation_summary
        }
    
    return comparison

def create_simplified_autoais_pipeline_summary(evaluation_results: Dict, results_dir: str) -> Dict:
    """Create pipeline summary in the same format as original pipeline"""
    
    # Extract timestamp from results directory name
    dir_name = os.path.basename(results_dir)
    parts = dir_name.split('_')
    
    # Try to extract timestamp and few_shot from directory name
    timestamp = "unknown"
    num_few_shot = 0
    
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():  # Date format YYYYMMDD
            if i + 1 < len(parts) and len(parts[i+1]) == 6 and parts[i+1].isdigit():  # Time format HHMMSS
                timestamp = f"{part}_{parts[i+1]}"
        elif part == "shot" and i > 0:
            try:
                num_few_shot = int(parts[i-1])
            except ValueError:
                pass
    
    summary = {
        "timestamp": timestamp,
        "evaluation_type": "autoais",
        "prompt_type": "simplified",
        "num_few_shot": num_few_shot,
        "approaches": {}
    }
    
    # Organize by approach, then by dataset in original format
    for approach, datasets in evaluation_results.items():
        summary["approaches"][approach] = {}
        
        for dataset_name, eval_result in datasets.items():
            if 'error' not in eval_result:
                # Create file paths (even though they might not exist in this context)
                results_file = f"{results_dir}/{dataset_name}_{approach}_results.json"
                evaluation_file = f"{results_dir}/{dataset_name}_{approach}_evaluation.json"
                
                summary["approaches"][approach][dataset_name] = {
                    "results_file": results_file,
                    "evaluation_file": evaluation_file,
                    "metrics": eval_result  # Include all metrics from evaluation
                }
    
    return summary

def evaluate_results_directory(results_dir: str, space_slug: str = 'za-zeeshan-33/true-model', 
                              hf_token: str = None, batch_size: int = 5, delay_seconds: float = 2.0, 
                              entailment_delay: float = 1.0, capture_logs: bool = True):
    """Evaluate all results in a directory"""
    
    print(f"\nStarting AutoAIS Evaluation")
    print(f"Results directory: {results_dir}")
    print("=" * 60)
    
    # Load all results
    all_results = load_results_from_directory(results_dir)
    
    if not all_results:
        raise ValueError("No valid results found to evaluate")
    
    # Initialize evaluator
    if hf_token is None:
        hf_token = os.getenv('HF_TOKEN')
    
    evaluator = AutoAISCitationEvaluator(
        space_slug=space_slug,
        hf_token=hf_token,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        entailment_delay=entailment_delay,
        capture_logs=capture_logs,
        enable_citations=True,
        enable_qa=True,
        enable_text_similarity=True
    )
    
    evaluation_results = {}
    
    # Evaluate each approach-dataset combination
    for approach, datasets in all_results.items():
        print(f"\nEvaluating {approach.upper()} approach")
        print("-" * 50)
        
        evaluation_results[approach] = {}
        
        for dataset_name, results in datasets.items():
            print(f"\nEvaluating {dataset_name} dataset...")
            print(f"   {len(results)} examples to evaluate")
            
            try:
                # Create temporary file for results since evaluator expects file paths
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_results:
                    json.dump(results, temp_results, indent=2)
                    temp_results_path = temp_results.name
                
                # We need the original dataset file path - try to infer it
                dataset_paths = {
                    'asqa': 'my_datasets/asqa/asqa_eval_gtr_top100_reranked_oracle.json',
                    'hagrid': 'my_datasets/hagrid/hagrid_filtered_quotes_max5.json', 
                    'msmarco': 'my_datasets/msmarco/msmarco_examples_4_to_53.json',
                    'eli5': 'my_datasets/eli5/eli5_eval_bm25_top100_reranked_oracle.json',
                    'natural_questions': 'my_datasets/dpr_nq/natural_questions_examples_4_to_53.json',
                    'qampari': 'my_datasets/qampari/qampari_eval_gtr_top100_reranked_oracle.json'
                }
                
                dataset_file = dataset_paths.get(dataset_name)
                if not dataset_file or not os.path.exists(dataset_file):
                    print(f"   Warning: Dataset file not found for {dataset_name}, skipping...")
                    continue
                
                # Perform evaluation
                eval_results = evaluator.evaluate_dataset(temp_results_path, dataset_file)
                evaluation_results[approach][dataset_name] = eval_results
                
                # Clean up temp file
                os.unlink(temp_results_path)
                
                # Save individual evaluation results
                eval_file = os.path.join(results_dir, f"{dataset_name}_{approach}_evaluation.json")
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_results, f, indent=2, ensure_ascii=False)
                
                print(f"   Evaluation complete")
                print(f"   Saved to: {eval_file}")
                
                # Print key metrics
                if 'evaluation_metrics' in eval_results:
                    metrics = eval_results['evaluation_metrics']
                    print(f"   Citation Recall: {metrics.get('citation_recall_autoais_mean', 'N/A'):.3f}")
                    print(f"   Citation Precision: {metrics.get('citation_precision_autoais_mean', 'N/A'):.3f}")
                    print(f"   Citation F1: {metrics.get('citation_f1_autoais_mean', 'N/A'):.3f}")
                
            except Exception as e:
                print(f"   Evaluation failed: {e}")
                evaluation_results[approach][dataset_name] = {"error": str(e)}
    
    # Create comparison in original format
    comparison_results = create_simplified_autoais_approach_comparison(evaluation_results, results_dir)
    
    # Save comparison results with original filename
    comparison_file = os.path.join(results_dir, "simplified_autoais_approach_comparison.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    # Create pipeline summary in original format
    pipeline_summary = create_simplified_autoais_pipeline_summary(evaluation_results, results_dir)
    
    # Save pipeline summary with original filename
    summary_file = os.path.join(results_dir, "simplified_autoais_pipeline_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE!")
    print(f"All evaluation files saved to: {results_dir}")
    print(f"Comparison saved to: {comparison_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*60}")
    
    # Perform initial vs final answer comparison if initial answer files exist
    perform_initial_final_comparison(results_dir)
    
    return evaluation_results

def strip_citations_and_normalize(text: str) -> str:
    """
    Strip citations and normalize text for comparison.
    
    Args:
        text: Text with potential citations like [1], [2], etc.
    
    Returns:
        Normalized text without citations and extra spaces
    """
    if not text:
        return ""
    
    import re
    # Remove citation patterns like [1], [2], [3], etc. along with any surrounding spaces
    # This handles cases like "word [2], word" -> "word, word" (no extra space)
    text = re.sub(r'\s*\[\d+\]\s*', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove trailing punctuation that might be left after citation removal
    text = re.sub(r'[.,;]+$', '', text)
    
    return text

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using SequenceMatcher.
    
    Args:
        text1: First text
        text2: Second text
    
    Returns:
        Similarity score between 0 and 1
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

def compare_initial_final_answers(initial_answer: str, final_answer: str) -> Dict[str, any]:
    """
    Compare initial and final answers comprehensively.
    
    Args:
        initial_answer: Answer before citation prompt
        final_answer: Answer after citation prompt
    
    Returns:
        Comprehensive comparison metrics
    """
    # Strip citations from final answer for fair comparison
    final_stripped = strip_citations_and_normalize(final_answer)
    initial_clean = strip_citations_and_normalize(initial_answer)
    
    # Calculate similarity
    similarity = calculate_text_similarity(initial_clean, final_stripped)
    
    # Calculate length metrics
    initial_len = len(initial_clean)
    final_len = len(final_stripped)
    length_change_percent = ((final_len - initial_len) / initial_len * 100) if initial_len > 0 else 0
    
    # Determine if answers are identical after citation stripping
    are_identical = initial_clean == final_stripped
    
    comparison = {
        'text_similarity': similarity,
        'are_identical_after_stripping': are_identical,
        'initial_length': initial_len,
        'final_length': final_len,
        'length_change_percent': length_change_percent,
        'initial_original': initial_answer,
        'final_original': final_answer,
        'initial_stripped': initial_clean,
        'final_stripped': final_stripped
    }
    
    return comparison

def perform_initial_final_comparison(results_dir: str):
    """
    Perform comparison between initial and final answers if initial answer files exist.
    
    Args:
        results_dir: Path to results directory
    """
    print(f"\nChecking for initial answer files...")
    
    # Find all initial answer files
    initial_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_initial_answers.json'):
            initial_files.append(filename)
    
    if not initial_files:
        print("   No initial answer files found. Skipping initial vs final comparison.")
        return
    
    print(f"   Found {len(initial_files)} initial answer files")
    
    comparison_results = {}
    overall_stats = {
        'total_comparisons': 0,
        'total_identical': 0,
        'all_similarities': [],
        'all_length_changes': []
    }
    
    # Compare each dataset
    for initial_file in initial_files:
        # Extract dataset and approach from filename
        # e.g., "qampari_post-generation-llm-long_results_initial_answers.json"
        base_name = initial_file.replace('_initial_answers.json', '')
        
        # Remove '_results' suffix if it exists (from the base name)
        if base_name.endswith('_results'):
            base_name = base_name[:-8]  # Remove '_results'
        
        parts = base_name.split('_')
        
        # Find the approach part (last few parts)
        if len(parts) >= 2:
            dataset = parts[0]
            approach = '_'.join(parts[1:])
            
            # Look for corresponding final file
            final_file = f"{dataset}_{approach}_results.json"
            final_path = os.path.join(results_dir, final_file)
            initial_path = os.path.join(results_dir, initial_file)
            
            if os.path.exists(final_path):
                print(f"   Comparing {dataset} {approach}...")
                
                try:
                    # Load both files
                    with open(initial_path, 'r', encoding='utf-8') as f:
                        initial_data = json.load(f)
                    with open(final_path, 'r', encoding='utf-8') as f:
                        final_data = json.load(f)
                    
                    # Create lookup dictionaries by ID
                    initial_lookup = {item['id']: item for item in initial_data}
                    final_lookup = {item['id']: item for item in final_data}
                    
                    comparisons = []
                    
                    # Compare items that exist in both files
                    for item_id in initial_lookup:
                        if item_id in final_lookup:
                            initial_item = initial_lookup[item_id]
                            final_item = final_lookup[item_id]
                            
                            comparison = compare_initial_final_answers(
                                initial_item['initial_answer'],
                                final_item['generated_answer']
                            )
                            
                            comparison['id'] = item_id
                            comparison['question'] = initial_item['question']
                            comparisons.append(comparison)
                    
                    # Calculate aggregate metrics
                    if comparisons:
                        similarities = [c['text_similarity'] for c in comparisons]
                        length_changes = [c['length_change_percent'] for c in comparisons]
                        identical_count = sum(1 for c in comparisons if c['are_identical_after_stripping'])
                        
                        aggregate_metrics = {
                            'total_comparisons': len(comparisons),
                            'identical_after_stripping': identical_count,
                            'identical_percentage': (identical_count / len(comparisons)) * 100,
                            'avg_similarity': sum(similarities) / len(similarities),
                            'min_similarity': min(similarities),
                            'max_similarity': max(similarities),
                            'avg_length_change_percent': sum(length_changes) / len(length_changes),
                            'min_length_change_percent': min(length_changes),
                            'max_length_change_percent': max(length_changes)
                        }
                        
                        key = f"{dataset}_{approach}"
                        comparison_results[key] = {
                            'aggregate_metrics': aggregate_metrics,
                            'individual_comparisons': comparisons
                        }
                        
                        # Update overall stats
                        overall_stats['total_comparisons'] += len(comparisons)
                        overall_stats['total_identical'] += identical_count
                        overall_stats['all_similarities'].extend(similarities)
                        overall_stats['all_length_changes'].extend(length_changes)
                        
                        print(f"      {len(comparisons)} comparisons, {identical_count} identical ({aggregate_metrics['identical_percentage']:.1f}%)")
                        print(f"      Avg similarity: {aggregate_metrics['avg_similarity']:.3f}")
                        print(f"      Avg length change: {aggregate_metrics['avg_length_change_percent']:.1f}%")
                    
                except Exception as e:
                    print(f"      Error comparing {dataset} {approach}: {e}")
    
    # Save comparison results (summary only for large datasets)
    if comparison_results:
        # Create summary-only version
        summary_overall_stats = overall_stats.copy()
        # Remove large arrays from summary
        if 'all_similarities' in summary_overall_stats:
            del summary_overall_stats['all_similarities']
        if 'all_length_changes' in summary_overall_stats:
            del summary_overall_stats['all_length_changes']
        
        summary_results = {
            'overall_statistics': summary_overall_stats,
            'dataset_breakdown': {},
            'dataset_comparisons': {}
        }
        
        # Include dataset breakdown summary (like terminal output) - sorted logically
        # Sort by dataset first, then by approach
        sorted_keys = sorted(comparison_results.keys(), 
                            key=lambda x: (x.split('_')[0], x.split('_', 1)[1] if '_' in x else x))
        
        for key in sorted_keys:
            comparison = comparison_results[key]
            metrics = comparison['aggregate_metrics']
            summary_results['dataset_breakdown'][key] = {
                'comparisons': metrics['total_comparisons'],
                'identical': metrics['identical_after_stripping'],
                'identical_percentage': round(metrics['identical_percentage'], 1),
                'avg_similarity': round(metrics['avg_similarity'], 3),
                'avg_length_change_percent': round(metrics['avg_length_change_percent'], 1)
            }
            
            # Include only aggregate metrics for each dataset, not individual comparisons
            summary_results['dataset_comparisons'][key] = {
                'aggregate_metrics': comparison['aggregate_metrics']
            }
        
        summary_results['analysis_timestamp'] = datetime.now().isoformat()
        
        comparison_file = os.path.join(results_dir, "initial_final_comparison_summary.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nInitial vs Final comparison summary saved to: {comparison_file}")
        print("Note: Use --verbose flag in compare_initial_final_answers.py for detailed individual comparisons")
        
        # Print summary
        if overall_stats['all_similarities']:
            overall_identical_pct = (overall_stats['total_identical'] / overall_stats['total_comparisons'] * 100)
            avg_similarity = sum(overall_stats['all_similarities']) / len(overall_stats['all_similarities'])
            avg_length_change = sum(overall_stats['all_length_changes']) / len(overall_stats['all_length_changes'])
            
            print(f"\nOVERALL INITIAL vs FINAL SUMMARY:")
            print(f"   Total comparisons: {overall_stats['total_comparisons']}")
            print(f"   Identical after citation stripping: {overall_stats['total_identical']} ({overall_identical_pct:.1f}%)")
            print(f"   Average text similarity: {avg_similarity:.3f}")
            print(f"   Average length change: {avg_length_change:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pre-generated attribution pipeline results')
    parser.add_argument('--results_dir', required=True,
                        help='Directory containing result files from run_pipeline_with_autoais_simplified_v2.py')
    parser.add_argument('--space_slug', default='za-zeeshan-33/true-model',
                        help='Hugging Face Space slug for AutoAIS evaluation')
    parser.add_argument('--hf_token', default=None,
                        help='Hugging Face token (default: from HF_TOKEN env var)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for evaluation (default: 5)')
    parser.add_argument('--delay_seconds', type=float, default=2.0,
                        help='Delay between API calls in seconds (default: 2.0)')
    parser.add_argument('--entailment_delay', type=float, default=1.0,
                        help='Delay between entailment checks in seconds (default: 1.0)')
    parser.add_argument('--no_capture_logs', action='store_true',
                        help='Disable log capture during evaluation')
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        exit(1)
    
    # Check for HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("Error: HF_TOKEN not found. Please set HF_TOKEN environment variable or use --hf_token")
        exit(1)
    
    print(f"HF_TOKEN found")
    
    try:
        evaluate_results_directory(
            results_dir=args.results_dir,
            space_slug=args.space_slug,
            hf_token=hf_token,
            batch_size=args.batch_size,
            delay_seconds=args.delay_seconds,
            entailment_delay=args.entailment_delay,
            capture_logs=not args.no_capture_logs
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        exit(1)
