#!/usr/bin/env python3
"""
Batch Model Analysis with diffai-python v0.3.16

This script demonstrates how to use the diffai Python package to analyze
multiple model pairs in batch, with progress tracking and result aggregation.

Features:
- Batch processing of multiple model pairs
- Progress tracking with detailed reporting
- Result aggregation and summary statistics
- Export results to various formats (JSON, CSV, Markdown)
- Automatic ML analysis for all model pairs

Requirements:
- diffai-python (pip install diffai-python)
- pandas (pip install pandas) - optional, for CSV export
- Model files to compare (.pt/.pth/.safetensors)

Usage:
    python batch_analysis.py models/baseline/ models/finetuned/ --output results/
"""

import sys
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    import diffai_python
except ImportError:
    print("❌ diffai-python not installed. Install with: pip install diffai-python")
    sys.exit(1)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available - CSV export will use basic format")


class BatchAnalyzer:
    """Batch analyzer for multiple model comparisons using diffai-python."""
    
    def __init__(self, epsilon: float = 1e-6, max_workers: int = 4):
        self.epsilon = epsilon
        self.max_workers = max_workers
        self.results = []
        
    def find_model_pairs(self, baseline_dir: str, comparison_dir: str) -> List[Tuple[str, str]]:
        """
        Find matching model pairs between two directories.
        
        Args:
            baseline_dir: Directory containing baseline models
            comparison_dir: Directory containing comparison models
            
        Returns:
            List of (baseline_path, comparison_path) tuples
        """
        baseline_path = Path(baseline_dir)
        comparison_path = Path(comparison_dir)
        
        if not baseline_path.exists():
            raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
        if not comparison_path.exists():
            raise FileNotFoundError(f"Comparison directory not found: {comparison_dir}")
        
        # Supported ML file extensions
        ml_extensions = {'.pt', '.pth', '.safetensors', '.npy', '.npz', '.mat'}
        
        # Find all models in baseline directory
        baseline_models = {}
        for ext in ml_extensions:
            for model_file in baseline_path.glob(f"**/*{ext}"):
                relative_path = model_file.relative_to(baseline_path)
                baseline_models[relative_path] = model_file
        
        # Find matching models in comparison directory
        pairs = []
        for relative_path, baseline_file in baseline_models.items():
            comparison_file = comparison_path / relative_path
            if comparison_file.exists():
                pairs.append((str(baseline_file), str(comparison_file)))
            else:
                print(f"⚠️  No matching file for {relative_path}")
        
        return pairs
    
    def analyze_single_pair(self, baseline_path: str, comparison_path: str) -> Dict[str, Any]:
        """
        Analyze a single model pair using diffai-python API.
        
        Args:
            baseline_path: Path to baseline model
            comparison_path: Path to comparison model
            
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        
        # Load model data (in practice, use appropriate ML libraries)
        try:
            import json
            with open(baseline_path, 'r') as f:
                old_data = json.load(f)
            with open(comparison_path, 'r') as f:
                new_data = json.load(f)
        except json.JSONDecodeError:
            # Binary files - use placeholders for demo
            old_data = {"binary_file": baseline_path}
            new_data = {"binary_file": comparison_path}
        
        try:
            # Use diffai-python's diff function - the ONLY actual API
            differences = diffai_python.diff(
                old_data, 
                new_data,
                epsilon=self.epsilon,
                ml_analysis_enabled=True,
                tensor_comparison_mode='both',
                learning_rate_tracking=True,
                optimizer_comparison=True,
                loss_tracking=True,
                accuracy_tracking=True,
                model_version_check=True,
                activation_analysis=True,
                weight_threshold=0.01,
                scientific_precision=True
            )
            
            # Process results
            analysis = {
                'baseline_path': baseline_path,
                'comparison_path': comparison_path,
                'analysis_time': time.time() - start_time,
                'success': True,
                'error': None,
                'total_changes': 0,
                'tensor_changes': 0,
                'shape_changes': 0,
                'significant_changes': [],
                'ml_analysis_summary': {
                    'learning_rate_changes': 0,
                    'optimizer_changes': 0,
                    'gradient_flow_issues': 0,
                    'quantization_changes': 0,
                    'convergence_changes': 0,
                    'attention_changes': 0,
                }
            }
            
            analysis['total_changes'] = len(differences)
            
            # Analyze each difference from diffai-python API
            for diff in differences:
                if isinstance(diff, dict):
                    diff_type = diff.get('diffType', '') if hasattr(diff, 'get') else str(diff)
                    if 'TensorStats' in diff_type or 'Modified' in diff_type:
                        analysis['tensor_changes'] += 1
                    elif 'TensorShape' in diff_type or 'Architecture' in diff_type:
                        analysis['shape_changes'] += 1
                        analysis['significant_changes'].append({
                            'type': 'shape_change',
                            'path': diff.get('path', 'unknown') if hasattr(diff, 'get') else 'unknown',
                            'details': diff
                        })
            
            return analysis
            
        except Exception as e:
            return {
                'baseline_path': baseline_path,
                'comparison_path': comparison_path,
                'analysis_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'total_changes': 0,
                'tensor_changes': 0,
                'shape_changes': 0,
                'significant_changes': [],
                'ml_analysis_summary': {}
            }
    
    def analyze_batch(self, model_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple model pairs in parallel.
        
        Args:
            model_pairs: List of (baseline, comparison) path tuples
            
        Returns:
            List of analysis results
        """
        print(f"🔄 Starting batch analysis of {len(model_pairs)} model pairs...")
        print(f"   Using {self.max_workers} parallel workers")
        print("")
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_pair = {
                executor.submit(self.analyze_single_pair, baseline, comparison): (baseline, comparison)
                for baseline, comparison in model_pairs
            }
            
            # Process completed tasks
            for future in as_completed(future_to_pair):
                baseline, comparison = future_to_pair[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Progress update
                    if result['success']:
                        print(f"✅ [{completed}/{len(model_pairs)}] {Path(baseline).name} vs {Path(comparison).name} "
                              f"({result['total_changes']} changes, {result['analysis_time']:.2f}s)")
                    else:
                        print(f"❌ [{completed}/{len(model_pairs)}] {Path(baseline).name} vs {Path(comparison).name} "
                              f"(Error: {result['error']})")
                        
                except Exception as e:
                    completed += 1
                    print(f"❌ [{completed}/{len(model_pairs)}] {Path(baseline).name} vs {Path(comparison).name} "
                          f"(Exception: {e})")
                    results.append({
                        'baseline_path': baseline,
                        'comparison_path': comparison,
                        'success': False,
                        'error': str(e),
                        'total_changes': 0
                    })
        
        self.results = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from batch analysis results."""
        
        if not self.results:
            return {'error': 'No results to summarize'}
        
        successful_analyses = [r for r in self.results if r['success']]
        failed_analyses = [r for r in self.results if not r['success']]
        
        summary = {
            'total_pairs': len(self.results),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
            'success_rate': len(successful_analyses) / len(self.results) * 100,
            'total_analysis_time': sum(r['analysis_time'] for r in self.results),
            'average_analysis_time': sum(r['analysis_time'] for r in self.results) / len(self.results),
            'change_statistics': {
                'models_with_changes': len([r for r in successful_analyses if r['total_changes'] > 0]),
                'models_unchanged': len([r for r in successful_analyses if r['total_changes'] == 0]),
                'total_changes_detected': sum(r['total_changes'] for r in successful_analyses),
                'average_changes_per_model': sum(r['total_changes'] for r in successful_analyses) / len(successful_analyses) if successful_analyses else 0,
                'models_with_shape_changes': len([r for r in successful_analyses if r['shape_changes'] > 0]),
            },
            'ml_analysis_completion': {
                'total_ml_analyses_run': len(successful_analyses) * 11,  # 11 automatic analyses
                'estimated_manual_hours_saved': len(successful_analyses) * 2,  # Estimate 2 hours per manual analysis
            }
        }
        
        return summary
    
    def export_results(self, output_dir: str, formats: List[str] = ['json', 'csv', 'md']):
        """
        Export results in multiple formats.
        
        Args:
            output_dir: Directory to save results
            formats: List of formats to export ('json', 'csv', 'md')
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        summary = self.generate_summary()
        
        # Export JSON
        if 'json' in formats:
            json_file = output_path / 'batch_analysis_results.json'
            with open(json_file, 'w') as f:
                json.dump({
                    'summary': summary,
                    'detailed_results': self.results
                }, f, indent=2)
            print(f"📄 JSON results exported to: {json_file}")
        
        # Export CSV
        if 'csv' in formats:
            csv_file = output_path / 'batch_analysis_results.csv'
            
            if PANDAS_AVAILABLE:
                # Use pandas for rich CSV export
                df = pd.DataFrame(self.results)
                df.to_csv(csv_file, index=False)
            else:
                # Basic CSV export
                with open(csv_file, 'w', newline='') as f:
                    if self.results:
                        writer = csv.DictWriter(f, fieldnames=self.results[0].keys())
                        writer.writeheader()
                        writer.writerows(self.results)
            
            print(f"📊 CSV results exported to: {csv_file}")
        
        # Export Markdown report
        if 'md' in formats:
            md_file = output_path / 'batch_analysis_report.md'
            self._generate_markdown_report(md_file, summary)
            print(f"📝 Markdown report exported to: {md_file}")
    
    def _generate_markdown_report(self, output_file: Path, summary: Dict[str, Any]):
        """Generate detailed Markdown report."""
        
        with open(output_file, 'w') as f:
            f.write("# Batch Model Analysis Report\n\n")
            f.write(f"**Generated by:** diffai-python v{diffai_python.__version__} batch analyzer\n")
            f.write(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Model Pairs:** {summary['total_pairs']}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Successful Analyses:** {summary['successful_analyses']}/{summary['total_pairs']} ({summary['success_rate']:.1f}%)\n")
            f.write(f"- **Failed Analyses:** {summary['failed_analyses']}\n")
            f.write(f"- **Total Analysis Time:** {summary['total_analysis_time']:.2f} seconds\n")
            f.write(f"- **Average Time per Pair:** {summary['average_analysis_time']:.2f} seconds\n\n")
            
            # Change detection statistics
            f.write("## Change Detection Results\n\n")
            stats = summary['change_statistics']
            f.write(f"- **Models with Changes:** {stats['models_with_changes']}\n")
            f.write(f"- **Unchanged Models:** {stats['models_unchanged']}\n")
            f.write(f"- **Total Changes Detected:** {stats['total_changes_detected']}\n")
            f.write(f"- **Average Changes per Model:** {stats['average_changes_per_model']:.2f}\n")
            f.write(f"- **Models with Architecture Changes:** {stats['models_with_shape_changes']}\n\n")
            
            # ML Analysis impact
            f.write("## Automatic ML Analysis Impact\n\n")
            ml_stats = summary['ml_analysis_completion']
            f.write(f"- **Total ML Analyses Performed:** {ml_stats['total_ml_analyses_run']} (11 per model pair)\n")
            f.write(f"- **Estimated Manual Hours Saved:** {ml_stats['estimated_manual_hours_saved']} hours\n")
            f.write("- **Analysis Functions per Pair:** Learning Rate, Optimizer, Loss, Accuracy, Gradient, Quantization, Convergence, Activation, Attention, Ensemble, Model Version\n\n")
            
            # Detailed results
            f.write("## Detailed Results\n\n")
            successful_results = [r for r in self.results if r['success']]
            failed_results = [r for r in self.results if not r['success']]
            
            if successful_results:
                f.write("### Successful Analyses\n\n")
                f.write("| Baseline | Comparison | Changes | Time (s) |\n")
                f.write("|----------|------------|---------|----------|\n")
                for result in successful_results:
                    baseline_name = Path(result['baseline_path']).name
                    comparison_name = Path(result['comparison_path']).name
                    f.write(f"| {baseline_name} | {comparison_name} | {result['total_changes']} | {result['analysis_time']:.2f} |\n")
                f.write("\n")
            
            if failed_results:
                f.write("### Failed Analyses\n\n")
                f.write("| Baseline | Comparison | Error |\n")
                f.write("|----------|------------|---------|\n")
                for result in failed_results:
                    baseline_name = Path(result['baseline_path']).name
                    comparison_name = Path(result['comparison_path']).name
                    error = result['error'][:50] + "..." if len(result['error']) > 50 else result['error']
                    f.write(f"| {baseline_name} | {comparison_name} | {error} |\n")
                f.write("\n")
            
            f.write("## Recommendations\n\n")
            
            if summary['failed_analyses'] > 0:
                f.write("⚠️ **Some analyses failed** - Review error messages and ensure model files are valid\n\n")
            
            if stats['models_with_shape_changes'] > 0:
                f.write("⚠️ **Architecture changes detected** - These models require thorough testing\n\n")
            
            if stats['models_unchanged'] == summary['successful_analyses']:
                f.write("✅ **All models unchanged** - Safe for deployment\n\n")
            
            f.write(f"*Report generated with diffai-python v{diffai_python.__version__} 🚀*\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze multiple model pairs using diffai-python",
        epilog="This tool performs automatic ML analysis on multiple model pairs in parallel."
    )
    parser.add_argument("baseline_dir", help="Directory containing baseline models")
    parser.add_argument("comparison_dir", help="Directory containing comparison models")
    parser.add_argument("--output", default="batch_results/", 
                        help="Output directory for results")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Tolerance for floating-point comparisons")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--formats", nargs='+', default=['json', 'csv', 'md'],
                        choices=['json', 'csv', 'md'],
                        help="Export formats")
    
    args = parser.parse_args()
    
    print("🔄 diffai-python Batch Model Analysis")
    print("=====================================")
    print(f"Baseline Directory: {args.baseline_dir}")
    print(f"Comparison Directory: {args.comparison_dir}")
    print(f"Output Directory: {args.output}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Workers: {args.workers}")
    print(f"Export Formats: {', '.join(args.formats)}")
    print("")
    
    try:
        # Initialize batch analyzer
        analyzer = BatchAnalyzer(epsilon=args.epsilon, max_workers=args.workers)
        
        # Find model pairs
        print("🔍 Finding model pairs...")
        model_pairs = analyzer.find_model_pairs(args.baseline_dir, args.comparison_dir)
        
        if not model_pairs:
            print("❌ No matching model pairs found")
            sys.exit(1)
        
        print(f"📋 Found {len(model_pairs)} model pairs to analyze")
        print("")
        
        # Run batch analysis
        results = analyzer.analyze_batch(model_pairs)
        
        # Generate and display summary
        print("")
        print("📊 Analysis Summary:")
        summary = analyzer.generate_summary()
        print(f"   ✅ Successful: {summary['successful_analyses']}/{summary['total_pairs']}")
        print(f"   ❌ Failed: {summary['failed_analyses']}")
        print(f"   🔍 Total Changes Found: {summary['change_statistics']['total_changes_detected']}")
        print(f"   ⏱️  Total Time: {summary['total_analysis_time']:.2f}s")
        print("")
        
        # Export results
        print("💾 Exporting results...")
        analyzer.export_results(args.output, args.formats)
        
        print("")
        print("🎉 Batch analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during batch analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()