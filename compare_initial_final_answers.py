#!/usr/bin/env python3
"""
Compare initial answers (before citation prompts) with final answers (after citation stripping)
to measure how much content the citation prompt adds or changes.
"""

import json
import os
import re
import argparse
from typing import Dict, List, Tuple
from difflib import SequenceMatcher
import numpy as np
from datetime import datetime

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
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_preservation_levenshtein(text_x: str, text_y: str) -> float:
    """
    Calculate Preservation Levenshtein (PresLev) metric.
    
    PresLev(x, y) = max(1 - Lev(x, y) / length(x), 0)
    
    Where:
    - Lev(x, y) is the Levenshtein edit distance between x and y
    - length(x) is the length of the original text x
    - max(..., 0) ensures the result is never negative
    
    Args:
        text_x: Original text (initial answer)
        text_y: Modified text (final answer after citation stripping)
    
    Returns:
        PresLev score between 0 and 1
        - 1.0: Perfect preservation (x and y are identical)
        - 0.0: No preservation (y completely overwrites x)
    """
    if not text_x:
        return 0.0
    
    # Calculate Levenshtein distance
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Calculate edit distance and normalize
    edit_distance = levenshtein_distance(text_x, text_y)
    length_x = len(text_x)
    
    # Apply PresLev formula: max(1 - Lev(x, y) / length(x), 0)
    preservation_score = max(1.0 - (edit_distance / length_x), 0.0)
    
    return preservation_score

def calculate_length_metrics(initial: str, final: str) -> Dict[str, float]:
    """
    Calculate various length-based metrics.
    
    Args:
        initial: Initial answer text
        final: Final answer text (after citation stripping)
    
    Returns:
        Dictionary with length metrics
    """
    initial_len = len(initial)
    final_len = len(final)
    
    metrics = {
        'initial_length': initial_len,
        'final_length': final_len,
        'length_difference': final_len - initial_len,
        'length_ratio': final_len / initial_len if initial_len > 0 else 0,
        'length_change_percent': ((final_len - initial_len) / initial_len * 100) if initial_len > 0 else 0
    }
    
    return metrics

def analyze_content_changes(initial: str, final: str) -> Dict[str, any]:
    """
    Analyze what content changed between initial and final answers.
    
    Args:
        initial: Initial answer text
        final: Final answer text (after citation stripping)
    
    Returns:
        Dictionary with content change analysis
    """
    initial_words = set(initial.lower().split())
    final_words = set(final.lower().split())
    
    added_words = final_words - initial_words
    removed_words = initial_words - final_words
    common_words = initial_words & final_words
    
    analysis = {
        'initial_word_count': len(initial.split()),
        'final_word_count': len(final.split()),
        'added_words': list(added_words),
        'removed_words': list(removed_words),
        'common_words_count': len(common_words),
        'added_words_count': len(added_words),
        'removed_words_count': len(removed_words),
        'word_overlap_ratio': len(common_words) / len(initial_words) if initial_words else 0
    }
    
    return analysis

def compare_answers(initial_answer: str, final_answer: str) -> Dict[str, any]:
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
    
    # Calculate similarity metrics
    similarity = calculate_text_similarity(initial_clean, final_stripped)
    preservation_levenshtein = calculate_preservation_levenshtein(initial_clean, final_stripped)
    
    # Calculate length metrics
    length_metrics = calculate_length_metrics(initial_clean, final_stripped)
    
    # Analyze content changes
    content_analysis = analyze_content_changes(initial_clean, final_stripped)
    
    # Determine if answers are identical after citation stripping
    are_identical = initial_clean == final_stripped
    
    comparison = {
        'text_similarity': similarity,
        'preservation_levenshtein': preservation_levenshtein,
        'are_identical_after_stripping': are_identical,
        'length_metrics': length_metrics,
        'content_analysis': content_analysis,
        'initial_original': initial_answer,
        'final_original': final_answer,
        'initial_stripped': initial_clean,
        'final_stripped': final_stripped
    }
    
    return comparison

def load_initial_answers_file(file_path: str) -> List[Dict]:
    """Load initial answers file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_final_answers_file(file_path: str) -> List[Dict]:
    """Load final answers file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compare_dataset_files(initial_file: str, final_file: str) -> Dict[str, any]:
    """Compare initial and final answer files for a dataset"""
    initial_data = load_initial_answers_file(initial_file)
    final_data = load_final_answers_file(final_file)
    
    # Create lookup dictionaries by ID
    initial_lookup = {item['id']: item for item in initial_data}
    final_lookup = {item['id']: item for item in final_data}
    
    comparisons = []
    missing_in_final = []
    missing_in_initial = []
    
    # Compare items that exist in both files
    for item_id in initial_lookup:
        if item_id in final_lookup:
            initial_item = initial_lookup[item_id]
            final_item = final_lookup[item_id]
            
            comparison = compare_answers(
                initial_item['initial_answer'],
                final_item['generated_answer']
            )
            
            comparison['id'] = item_id
            comparison['question'] = initial_item['question']
            comparisons.append(comparison)
        else:
            missing_in_final.append(item_id)
    
    # Find items in final but not in initial
    for item_id in final_lookup:
        if item_id not in initial_lookup:
            missing_in_initial.append(item_id)
    
    # Calculate aggregate metrics
    if comparisons:
        similarities = [c['text_similarity'] for c in comparisons]
        preservation_scores = [c['preservation_levenshtein'] for c in comparisons]
        length_changes = [c['length_metrics']['length_change_percent'] for c in comparisons]
        identical_count = sum(1 for c in comparisons if c['are_identical_after_stripping'])
        
        aggregate_metrics = {
            'total_comparisons': len(comparisons),
            'identical_after_stripping': identical_count,
            'identical_percentage': (identical_count / len(comparisons)) * 100,
            'avg_similarity': np.mean(similarities),
            'median_similarity': np.median(similarities),
            'min_similarity': np.min(similarities),
            'max_similarity': np.max(similarities),
            'avg_preservation_levenshtein': np.mean(preservation_scores),
            'median_preservation_levenshtein': np.median(preservation_scores),
            'min_preservation_levenshtein': np.min(preservation_scores),
            'max_preservation_levenshtein': np.max(preservation_scores),
            'avg_length_change_percent': np.mean(length_changes),
            'median_length_change_percent': np.median(length_changes),
            'missing_in_final': len(missing_in_final),
            'missing_in_initial': len(missing_in_initial)
        }
    else:
        aggregate_metrics = {
            'total_comparisons': 0,
            'error': 'No comparisons could be made'
        }
    
    return {
        'aggregate_metrics': aggregate_metrics,
        'individual_comparisons': comparisons,
        'missing_in_final': missing_in_final,
        'missing_in_initial': missing_in_initial
    }

def analyze_results_directory(results_dir: str) -> Dict[str, any]:
    """
    Analyze all initial vs final answer comparisons in a results directory.
    
    Args:
        results_dir: Path to results directory
    
    Returns:
        Comprehensive analysis results
    """
    if not os.path.exists(results_dir):
        return {'error': f'Results directory not found: {results_dir}'}
    
    # Find all initial answer files
    initial_files = []
    final_files = []
    
    for filename in os.listdir(results_dir):
        if filename.endswith('_initial_answers.json'):
            initial_files.append(filename)
        elif filename.endswith('_results.json') and '_initial_answers' not in filename:
            final_files.append(filename)
    
    dataset_comparisons = {}
    overall_stats = {
        'total_datasets': 0,
        'total_comparisons': 0,
        'total_identical': 0,
        'all_similarities': [],
        'all_preservation_levenshtein': [],
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
                print(f"Comparing {dataset} {approach}...")
                comparison = compare_dataset_files(initial_path, final_path)
                
                key = f"{dataset}_{approach}"
                dataset_comparisons[key] = comparison
                
                # Update overall stats
                if 'aggregate_metrics' in comparison:
                    metrics = comparison['aggregate_metrics']
                    overall_stats['total_datasets'] += 1
                    overall_stats['total_comparisons'] += metrics['total_comparisons']
                    overall_stats['total_identical'] += metrics['identical_after_stripping']
                    
                    # Add individual similarities, preservation scores, and length changes
                    for comp in comparison.get('individual_comparisons', []):
                        overall_stats['all_similarities'].append(comp['text_similarity'])
                        overall_stats['all_preservation_levenshtein'].append(comp['preservation_levenshtein'])
                        overall_stats['all_length_changes'].append(
                            comp['length_metrics']['length_change_percent']
                        )
    
    # Calculate overall statistics
    if overall_stats['all_similarities']:
        overall_stats['avg_similarity'] = np.mean(overall_stats['all_similarities'])
        overall_stats['median_similarity'] = np.median(overall_stats['all_similarities'])
        overall_stats['min_similarity'] = np.min(overall_stats['all_similarities'])
        overall_stats['max_similarity'] = np.max(overall_stats['all_similarities'])
        
        overall_stats['avg_preservation_levenshtein'] = np.mean(overall_stats['all_preservation_levenshtein'])
        overall_stats['median_preservation_levenshtein'] = np.median(overall_stats['all_preservation_levenshtein'])
        overall_stats['min_preservation_levenshtein'] = np.min(overall_stats['all_preservation_levenshtein'])
        overall_stats['max_preservation_levenshtein'] = np.max(overall_stats['all_preservation_levenshtein'])
        
        overall_stats['avg_length_change'] = np.mean(overall_stats['all_length_changes'])
        overall_stats['median_length_change'] = np.median(overall_stats['all_length_changes'])
        
        overall_stats['overall_identical_percentage'] = (
            overall_stats['total_identical'] / overall_stats['total_comparisons'] * 100
        )
    
    return {
        'overall_statistics': overall_stats,
        'dataset_comparisons': dataset_comparisons,
        'analysis_timestamp': datetime.now().isoformat()
    }

def print_summary_report(analysis_results: Dict[str, any]):
    """
    Print a summary report of the analysis.
    
    Args:
        analysis_results: Results from analyze_results_directory
    """
    print("\n" + "="*80)
    print("INITIAL vs FINAL ANSWER COMPARISON REPORT")
    print("="*80)
    
    if 'error' in analysis_results:
        print(f"Error: {analysis_results['error']}")
        return
    
    overall = analysis_results['overall_statistics']
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total datasets analyzed: {overall['total_datasets']}")
    print(f"  Total comparisons: {overall['total_comparisons']}")
    print(f"  Identical after citation stripping: {overall['total_identical']}")
    
    if 'overall_identical_percentage' in overall:
        print(f"  Overall identical percentage: {overall['overall_identical_percentage']:.1f}%")
        print(f"  Average text similarity: {overall['avg_similarity']:.3f}")
        print(f"  Median text similarity: {overall['median_similarity']:.3f}")
        print(f"  Similarity range: {overall['min_similarity']:.3f} - {overall['max_similarity']:.3f}")
        print(f"  Average preservation (PresLev): {overall['avg_preservation_levenshtein']:.3f}")
        print(f"  Median preservation (PresLev): {overall['median_preservation_levenshtein']:.3f}")
        print(f"  Preservation range: {overall['min_preservation_levenshtein']:.3f} - {overall['max_preservation_levenshtein']:.3f}")
        print(f"  Average length change: {overall['avg_length_change']:.1f}%")
        print(f"  Median length change: {overall['median_length_change']:.1f}%")
    
    print(f"\nDATASET BREAKDOWN:")
    for key, comparison in analysis_results['dataset_comparisons'].items():
        if 'aggregate_metrics' in comparison:
            metrics = comparison['aggregate_metrics']
            print(f"  {key}:")
            print(f"    Comparisons: {metrics['total_comparisons']}")
            print(f"    Identical: {metrics['identical_after_stripping']} ({metrics['identical_percentage']:.1f}%)")
            print(f"    Avg similarity: {metrics['avg_similarity']:.3f}")
            print(f"    Avg preservation (PresLev): {metrics['avg_preservation_levenshtein']:.3f}")
            print(f"    Avg length change: {metrics['avg_length_change_percent']:.1f}%")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description='Compare initial answers with final answers after citation stripping'
    )
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Path to results directory containing initial and final answer files')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for detailed comparison results (JSON format). If not specified, saves to initial_final_comparison_detailed.json in results directory.')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed individual comparisons')
    
    args = parser.parse_args()
    
    # Analyze the results directory
    print(f"Analyzing results directory: {args.results_dir}")
    analysis_results = analyze_results_directory(args.results_dir)
    
    # Print summary report
    print_summary_report(analysis_results)
    
    # Save results (summary only by default, detailed if verbose)
    if args.output_file:
        # Use user-specified output file
        output_path = args.output_file
    else:
        # Use default filename in results directory
        output_path = os.path.join(args.results_dir, "initial_final_comparison_summary.json")
    
    # Create summary-only version for large datasets
    overall_stats = analysis_results['overall_statistics'].copy()
    # Remove large arrays from summary
    if 'all_similarities' in overall_stats:
        del overall_stats['all_similarities']
    if 'all_preservation_levenshtein' in overall_stats:
        del overall_stats['all_preservation_levenshtein']
    if 'all_length_changes' in overall_stats:
        del overall_stats['all_length_changes']
    
    summary_results = {
        'overall_statistics': overall_stats,
        'dataset_breakdown': {},
        'dataset_comparisons': {}
    }
    
    # Include dataset breakdown summary (like terminal output) - sorted logically
    # Sort by dataset first, then by approach
    sorted_keys = sorted(analysis_results['dataset_comparisons'].keys(), 
                        key=lambda x: (x.split('_')[0], x.split('_', 1)[1] if '_' in x else x))
    
    for key in sorted_keys:
        comparison = analysis_results['dataset_comparisons'][key]
        metrics = comparison['aggregate_metrics']
        summary_results['dataset_breakdown'][key] = {
            'comparisons': metrics['total_comparisons'],
            'identical': metrics['identical_after_stripping'],
            'identical_percentage': round(metrics['identical_percentage'], 1),
            'avg_similarity': round(metrics['avg_similarity'], 3),
            'avg_preservation_levenshtein': round(metrics['avg_preservation_levenshtein'], 3),
            'avg_length_change_percent': round(metrics['avg_length_change_percent'], 1)
        }
        
        # Include only aggregate metrics for each dataset, not individual comparisons
        summary_results['dataset_comparisons'][key] = {
            'aggregate_metrics': comparison['aggregate_metrics']
        }
    
    # Add timestamp
    summary_results['analysis_timestamp'] = analysis_results['analysis_timestamp']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"\nComparison summary saved to: {output_path}")
    
    # Save detailed results only if verbose flag is used
    if args.verbose:
        detailed_path = os.path.join(args.results_dir, "initial_final_comparison_detailed.json")
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"Detailed comparison results saved to: {detailed_path}")
    
    # Print verbose details if requested
    if args.verbose:
        print(f"\nDETAILED INDIVIDUAL COMPARISONS:")
        for key, comparison in analysis_results['dataset_comparisons'].items():
            print(f"\n{key}:")
            for comp in comparison.get('individual_comparisons', [])[:5]:  # Show first 5
                print(f"  ID: {comp['id']}")
                print(f"  Similarity: {comp['text_similarity']:.3f}")
                print(f"  Preservation (PresLev): {comp['preservation_levenshtein']:.3f}")
                print(f"  Length change: {comp['length_metrics']['length_change_percent']:.1f}%")
                print(f"  Identical: {comp['are_identical_after_stripping']}")
                if not comp['are_identical_after_stripping']:
                    print(f"    Initial: {comp['initial_stripped'][:100]}...")
                    print(f"    Final:   {comp['final_stripped'][:100]}...")
                print()

if __name__ == "__main__":
    main()
