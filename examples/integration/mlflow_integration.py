#!/usr/bin/env python3
"""
MLflow Integration Example for diffai

This script demonstrates how to integrate diffai model comparison
with MLflow experiment tracking. It compares two models and logs
the differences as MLflow artifacts and metrics.

Requirements:
- mlflow
- diffai CLI tool installed
- Model files to compare

Usage:
    python mlflow_integration.py model1.safetensors model2.safetensors
"""

import subprocess
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import os

try:
    import mlflow
    import mlflow.artifacts
except ImportError:
    print("❌ MLflow not installed. Install with: pip install mlflow")
    sys.exit(1)


def run_diffai_comparison(model1_path: str, model2_path: str, 
                         epsilon: Optional[float] = None) -> Dict[str, Any]:
    """
    Run diffai comparison and return results as JSON.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        epsilon: Tolerance for floating-point comparisons
        
    Returns:
        Dictionary containing comparison results
    """
    cmd = ['diffai', model1_path, model2_path, '--output', 'json']
    
    if epsilon is not None:
        cmd.extend(['--epsilon', str(epsilon)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            return json.loads(result.stdout)
        else:
            return []  # No differences found
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running diffai: {e}")
        print(f"   stderr: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing diffai output: {e}")
        print(f"   stdout: {result.stdout}")
        raise


def analyze_differences(differences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze diffai results and extract key metrics.
    
    Args:
        differences: List of difference objects from diffai
        
    Returns:
        Dictionary of analyzed metrics
    """
    analysis = {
        'total_changes': len(differences),
        'tensor_stats_changes': 0,
        'tensor_shape_changes': 0,
        'tensors_added': 0,
        'tensors_removed': 0,
        'architecture_changes': 0,
        'significant_changes': [],
        'parameter_count_change': 0,
        'max_mean_change': 0.0,
        'max_std_change': 0.0,
    }
    
    for diff in differences:
        if 'TensorStatsChanged' in diff:
            analysis['tensor_stats_changes'] += 1
            name, old_stats, new_stats = diff['TensorStatsChanged']
            
            mean_change = abs(new_stats['mean'] - old_stats['mean'])
            std_change = abs(new_stats['std'] - old_stats['std'])
            
            analysis['max_mean_change'] = max(analysis['max_mean_change'], mean_change)
            analysis['max_std_change'] = max(analysis['max_std_change'], std_change)
            
            # Consider significant if change > 1% of the original value
            if mean_change > abs(old_stats['mean']) * 0.01:
                analysis['significant_changes'].append({
                    'type': 'TensorStatsChanged',
                    'name': name,
                    'mean_change': new_stats['mean'] - old_stats['mean'],
                    'std_change': new_stats['std'] - old_stats['std']
                })
                
        elif 'TensorShapeChanged' in diff:
            analysis['tensor_shape_changes'] += 1
            name, old_shape, new_shape = diff['TensorShapeChanged']
            
            old_params = 1
            for dim in old_shape:
                old_params *= dim
                
            new_params = 1
            for dim in new_shape:
                new_params *= dim
                
            param_change = new_params - old_params
            analysis['parameter_count_change'] += param_change
            
            analysis['significant_changes'].append({
                'type': 'TensorShapeChanged',
                'name': name,
                'old_shape': old_shape,
                'new_shape': new_shape,
                'parameter_change': param_change
            })
            
        elif 'Added' in diff:
            analysis['tensors_added'] += 1
            
        elif 'Removed' in diff:
            analysis['tensors_removed'] += 1
            
        elif 'ModelArchitectureChanged' in diff:
            analysis['architecture_changes'] += 1
    
    return analysis


def log_to_mlflow(model1_path: str, model2_path: str, 
                 differences: List[Dict[str, Any]], 
                 analysis: Dict[str, Any],
                 experiment_name: str = "model_comparison"):
    """
    Log comparison results to MLflow.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model
        differences: Raw diffai results
        analysis: Analyzed metrics
        experiment_name: MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log basic info
        mlflow.log_param("model1_path", model1_path)
        mlflow.log_param("model2_path", model2_path)
        mlflow.log_param("model1_name", Path(model1_path).name)
        mlflow.log_param("model2_name", Path(model2_path).name)
        
        # Log file sizes
        model1_size = Path(model1_path).stat().st_size
        model2_size = Path(model2_path).stat().st_size
        mlflow.log_metric("model1_size_mb", model1_size / (1024 * 1024))
        mlflow.log_metric("model2_size_mb", model2_size / (1024 * 1024))
        mlflow.log_metric("size_change_mb", (model2_size - model1_size) / (1024 * 1024))
        
        # Log analysis metrics
        mlflow.log_metric("total_changes", analysis['total_changes'])
        mlflow.log_metric("tensor_stats_changes", analysis['tensor_stats_changes'])
        mlflow.log_metric("tensor_shape_changes", analysis['tensor_shape_changes'])
        mlflow.log_metric("tensors_added", analysis['tensors_added'])
        mlflow.log_metric("tensors_removed", analysis['tensors_removed'])
        mlflow.log_metric("parameter_count_change", analysis['parameter_count_change'])
        mlflow.log_metric("max_mean_change", analysis['max_mean_change'])
        mlflow.log_metric("max_std_change", analysis['max_std_change'])
        
        # Determine change severity
        severity = "minimal"
        if analysis['tensor_shape_changes'] > 0 or analysis['architecture_changes'] > 0:
            severity = "major"
        elif len(analysis['significant_changes']) > 0:
            severity = "moderate"
        elif analysis['total_changes'] > 0:
            severity = "minor"
            
        mlflow.log_param("change_severity", severity)
        
        # Log detailed results as artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save raw diffai output
            diffai_output_path = os.path.join(tmp_dir, "diffai_output.json")
            with open(diffai_output_path, 'w') as f:
                json.dump(differences, f, indent=2)
            mlflow.log_artifact(diffai_output_path)
            
            # Save analysis summary
            analysis_path = os.path.join(tmp_dir, "analysis_summary.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            mlflow.log_artifact(analysis_path)
            
            # Generate human-readable report
            report_path = os.path.join(tmp_dir, "comparison_report.md")
            generate_report(model1_path, model2_path, analysis, report_path)
            mlflow.log_artifact(report_path)
        
        # Log tags for easy filtering
        mlflow.set_tag("tool", "diffai")
        mlflow.set_tag("comparison_type", "model_diff")
        mlflow.set_tag("severity", severity)
        
        if analysis['tensor_shape_changes'] > 0:
            mlflow.set_tag("has_shape_changes", "true")
        if analysis['tensors_added'] > 0 or analysis['tensors_removed'] > 0:
            mlflow.set_tag("has_topology_changes", "true")
        
        print(f"✅ Results logged to MLflow experiment: {experiment_name}")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")


def generate_report(model1_path: str, model2_path: str, 
                   analysis: Dict[str, Any], output_path: str):
    """Generate a human-readable comparison report."""
    
    with open(output_path, 'w') as f:
        f.write("# Model Comparison Report\n\n")
        f.write(f"**Generated by:** diffai + MLflow integration\n")
        f.write(f"**Date:** {mlflow.utils.time.get_current_time_millis()}\n\n")
        
        f.write("## Models Compared\n\n")
        f.write(f"- **Model 1:** `{model1_path}`\n")
        f.write(f"- **Model 2:** `{model2_path}`\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Changes:** {analysis['total_changes']}\n")
        f.write(f"- **Tensor Statistics Changes:** {analysis['tensor_stats_changes']}\n")
        f.write(f"- **Tensor Shape Changes:** {analysis['tensor_shape_changes']}\n")
        f.write(f"- **Tensors Added:** {analysis['tensors_added']}\n")
        f.write(f"- **Tensors Removed:** {analysis['tensors_removed']}\n")
        f.write(f"- **Parameter Count Change:** {analysis['parameter_count_change']:,}\n\n")
        
        if analysis['significant_changes']:
            f.write("## Significant Changes\n\n")
            for change in analysis['significant_changes']:
                f.write(f"### {change['type']}: {change['name']}\n\n")
                if change['type'] == 'TensorStatsChanged':
                    f.write(f"- **Mean Change:** {change['mean_change']:+.6f}\n")
                    f.write(f"- **Std Change:** {change['std_change']:+.6f}\n")
                elif change['type'] == 'TensorShapeChanged':
                    f.write(f"- **Old Shape:** {change['old_shape']}\n")
                    f.write(f"- **New Shape:** {change['new_shape']}\n")
                    f.write(f"- **Parameter Change:** {change['parameter_change']:+,}\n")
                f.write("\n")
        
        f.write("## Recommendations\n\n")
        
        if analysis['tensor_shape_changes'] > 0:
            f.write("⚠️ **Architecture Changes Detected**\n")
            f.write("- Model architecture has been modified\n")
            f.write("- Thorough testing recommended before deployment\n")
            f.write("- Check compatibility with existing inference pipelines\n\n")
        elif analysis['total_changes'] == 0:
            f.write("✅ **No Significant Changes**\n")
            f.write("- Models are functionally identical\n")
            f.write("- Safe for deployment\n\n")
        else:
            f.write("ℹ️ **Parameter Updates Detected**\n")
            f.write("- Likely fine-tuning or continued training\n")
            f.write("- Validate performance on test set\n")
            f.write("- Monitor for regression\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML models using diffai and log results to MLflow"
    )
    parser.add_argument("model1", help="Path to first model")
    parser.add_argument("model2", help="Path to second model")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Tolerance for floating-point comparisons")
    parser.add_argument("--experiment", default="model_comparison",
                        help="MLflow experiment name")
    parser.add_argument("--tracking-uri", 
                        help="MLflow tracking server URI")
    
    args = parser.parse_args()
    
    # Validate model files exist
    if not Path(args.model1).exists():
        print(f"❌ Model file not found: {args.model1}")
        sys.exit(1)
        
    if not Path(args.model2).exists():
        print(f"❌ Model file not found: {args.model2}")
        sys.exit(1)
    
    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    
    print("🤖 diffai + MLflow Model Comparison")
    print("====================================")
    print(f"Model 1: {args.model1}")
    print(f"Model 2: {args.model2}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Experiment: {args.experiment}")
    print("")
    
    try:
        # Run diffai comparison
        print("🔍 Running diffai comparison...")
        differences = run_diffai_comparison(args.model1, args.model2, args.epsilon)
        
        # Analyze results
        print("📊 Analyzing results...")
        analysis = analyze_differences(differences)
        
        # Log to MLflow
        print("📝 Logging to MLflow...")
        log_to_mlflow(args.model1, args.model2, differences, analysis, args.experiment)
        
        print("")
        print("🎉 Comparison completed successfully!")
        print(f"Found {analysis['total_changes']} differences")
        
        if analysis['significant_changes']:
            print(f"⚠️  {len(analysis['significant_changes'])} significant changes detected")
        else:
            print("✅ No significant changes detected")
            
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()