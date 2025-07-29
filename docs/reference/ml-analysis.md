# ML Analysis Features - Reference Guide

Complete reference for diffai's 11 automatic ML analysis capabilities.

## Overview

diffai automatically provides comprehensive AI/ML analysis without requiring any configuration. All analysis features are **standard capabilities** that activate based on file format, not optional flags.

**Design Philosophy**: ML analysis features are not "options" but standard functionality that diffai provides automatically for each file format based on technical feasibility.

## Core Analysis Features

diffai automatically analyzes all applicable aspects of AI/ML models:

**Usage**:
```bash
diffai model1.pt model2.pt
```

**Comprehensive Output Example**:
```
TensorStatsChanged: fc1.weight
  Old: mean=-0.0002, std=0.0514, shape=[128, 256], dtype=float32
  New: mean=-0.0001, std=0.0716, shape=[128, 256], dtype=float32

ModelArchitectureChanged: model
  Old: {layers: 12, parameters: 124439808, types: [conv, linear, norm]}
  New: {layers: 12, parameters: 124440064, types: [conv, linear, norm, attention]}

WeightSignificantChange: transformer.attention.query.weight
  Change Magnitude: 0.0234 (above threshold: 0.01)

MemoryAnalysis: memory_change
  Old: 487.2MB (tensors: 485.1MB, metadata: 2.1MB)
  New: memory_change: +12.5MB, breakdown: tensors: +12.3MB, metadata: +0.2MB

LearningRateChanged: optimizer.learning_rate
  Old: 0.001, New: 0.0005 (scheduler: step_decay, epoch: 10)

ConvergenceAnalysis: convergence_patterns
  Old: evaluating
  New: loss: improving (trend: decreasing), stability: gradient_norm: stable, epoch: 10 → 11

GradientAnalysis: gradient_magnitudes
  Old: norm: 0.018456, max: 0.145234, var: 0.000234
  New: total_norm: 0.021234 (+14.8%, increasing), max_gradient: 0.156789 (+8.0%)
```

**All Analysis Features Included**:
- **Tensor Statistics**: Mean, std, min/max, shape, dtype analysis
- **Model Architecture**: Layer detection, parameter counting, structural changes
- **Weight Changes**: Significant parameter change detection with configurable thresholds
- **Memory Analysis**: Memory usage and optimization analysis
- **Learning Rate**: Learning rate tracking from optimizer state
- **Convergence**: Training convergence pattern analysis
- **Gradients**: Gradient flow and magnitude analysis
- **Attention**: Transformer attention mechanism analysis
- **Ensemble**: Multi-model ensemble analysis
- **Quantization**: Model quantization and precision analysis

**Use Cases**:
- Monitor comprehensive model changes during training
- Detect all types of statistical and architectural shifts
- Validate model consistency across all dimensions
- Analyze training dynamics and optimization patterns

## Implementation Status

All 11 ML analysis features are **fully implemented and automatically available**:

✅ **High Priority Features** (automatically active for all compatible formats):
1. **Tensor Statistics**: Complete statistical analysis of all model parameters
2. **Model Architecture**: Comprehensive structural analysis and comparison
3. **Weight Changes**: Significant parameter change detection with thresholds
4. **Memory Analysis**: Detailed memory usage and optimization analysis

✅ **Medium Priority Features** (automatically active when applicable):
5. **Learning Rate**: Learning rate detection from optimizer state and metadata
6. **Convergence**: Training convergence pattern analysis from model changes
7. **Gradients**: Gradient flow analysis estimated from parameter updates

✅ **Advanced Features** (automatically active for supported formats):
8. **Attention**: Transformer attention mechanism analysis
9. **Ensemble**: Multi-model ensemble composition analysis  
10. **Quantization**: Model quantization and precision analysis

## Format Compatibility

| Feature | PyTorch (.pt/.pth) | Safetensors (.safetensors) | NumPy (.npy/.npz) | MATLAB (.mat) |
|---------|-------------------|---------------------------|-------------------|---------------|
| Tensor Statistics | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Model Architecture | ✅ Full | ✅ Full | ✅ Basic | ✅ Basic |
| Weight Changes | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Memory Analysis | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| Learning Rate | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Convergence | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Gradients | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Attention | ✅ Full | ✅ Partial | ❌ N/A | ❌ N/A |
| Ensemble | ✅ Full | ❌ Limited | ❌ N/A | ❌ N/A |
| Quantization | ✅ Full | ✅ Full | ✅ Limited | ✅ Limited |

## Automatic Analysis Examples

**Basic Model Comparison**:
```bash
# All 11 features activate automatically based on file content
diffai model1.pt model2.pt
```

**JSON Output for Automation**:
```bash
# Machine-readable comprehensive analysis
diffai baseline.safetensors candidate.safetensors --output json
```

**Cross-Format Analysis**:
```bash
# Automatic format-aware feature selection
diffai pytorch_model.pt numpy_weights.npy
```

## Design Philosophy

**Standard Feature Principle**: All ML analysis capabilities are standard features that activate automatically. diffai determines the appropriate level of analysis based on:

1. **File Format Capabilities**: What each format technically supports
2. **Data Availability**: What information is extractable from the model
3. **Analysis Feasibility**: What can be reliably computed

**No Configuration Required**: Users never need to specify which analyses to run. diffai automatically provides the maximum possible analysis for each file format.

## Integration Examples

### MLflow Integration
```python
import subprocess
import json
import mlflow

def log_comprehensive_model_diff(model1_path, model2_path):
    # All 11 ML features analyzed automatically
    result = subprocess.run([
        "diffai", model1_path, model2_path, "--output", "json"
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "comprehensive_analysis.json")
        # Log specific metrics from automatic analysis
        mlflow.log_metric("tensor_changes", len([d for d in diff_data if d["type"] == "TensorStatsChanged"]))
        mlflow.log_metric("architecture_changes", len([d for d in diff_data if d["type"] == "ModelArchitectureChanged"]))
```

### CI/CD Pipeline
```yaml
name: Comprehensive Model Validation
on: [push, pull_request]

jobs:
  model-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install diffai
        run: cargo install diffai
        
      - name: Complete Model Analysis
        run: |
          # Automatic comprehensive analysis - all 11 features
          diffai models/baseline.pt models/candidate.pt --output json > complete_analysis.json
          
          # Analysis includes: tensor stats, architecture, weights, memory,
          # learning rate, convergence, gradients, attention, ensemble, quantization
```

### Training Pipeline Integration
```python
# Automatic training monitoring with full ML analysis
def monitor_training_checkpoint(prev_checkpoint, current_checkpoint):
    result = subprocess.run([
        "diffai", prev_checkpoint, current_checkpoint, "--output", "json"
    ], capture_output=True, text=True)
    
    analysis = json.loads(result.stdout)
    
    # All analysis results available automatically:
    # - TensorStatsChanged: parameter drift detection
    # - WeightSignificantChange: significant updates
    # - LearningRateChanged: optimizer state changes
    # - ConvergenceAnalysis: training progress assessment
    # - GradientAnalysis: gradient flow health
    # - MemoryAnalysis: resource usage optimization
    
    return analysis
```

## See Also

- [Implementation Details](ml-analysis-implemented.md) - Complete technical reference
- [CLI Reference](cli-reference.md) - Command-line interface documentation
- [Format Support](formats.md) - Detailed format compatibility information
- [Basic Usage Guide](../user-guide/basic-usage.md) - Getting started with comprehensive analysis