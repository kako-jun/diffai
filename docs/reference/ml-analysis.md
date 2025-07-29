# ML Analysis Functions

Guide to diffai's automatic machine learning analysis capabilities.

## Overview

diffai automatically analyzes AI/ML models without requiring any configuration or options.

## Automatic Analysis

### Current Capabilities
Diffai automatically provides analysis for PyTorch and Safetensors files:

**Usage**:
```bash
diffai model1.safetensors model2.safetensors
```

**Output**:
```
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

**Analysis Fields**:
- **mean**: Average parameter values
- **std**: Standard deviation of parameters
- **min/max**: Parameter value ranges
- **shape**: Tensor dimensions
- **dtype**: Data type precision

**Use Cases**:
- Monitor parameter changes during training
- Detect statistical shifts in model weights
- Validate model consistency

## Planned Features

The following analysis capabilities are under development and will be automatically included in future releases:

**Output**:
```
quantization_analysis: compression=0.25, precision_loss=minimal
```

**Analysis Fields**:
- **compression**: Model size reduction ratio
- **precision_loss**: Accuracy impact assessment
- **efficiency**: Performance vs quality trade-offs

**Use Cases**:
- Validate quantization quality
- Optimize deployment size
- Compare compression techniques

### 3. `--sort-by-change-magnitude` Change Magnitude Sorting
Sorts differences by magnitude for prioritization.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

**Output**: Results are sorted with largest changes first

**Use Cases**:
- Focus on most significant changes
- Prioritize debugging efforts
- Identify critical parameter shifts

### 4. `--show-layer-impact` Layer Impact Analysis
Analyzes layer-by-layer impact of changes.

**Usage**:
```bash
diffai baseline.safetensors modified.safetensors --show-layer-impact
```

**Output**: Per-layer change analysis

**Use Cases**:
- Understand which layers changed most
- Guide fine-tuning strategies
- Analyze architectural modifications

## Combined Analysis

Combine multiple features for comprehensive analysis:

```bash
# Comprehensive model analysis
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  \
  --quantization-analysis \
  --sort-by-change-magnitude \
  --show-layer-impact

# JSON output for automation
diffai model1.safetensors model2.safetensors \
  --output json
```

## Feature Selection Guide

**For Training Monitoring**:
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --sort-by-change-magnitude
```

**For Production Deployment**:
```bash
diffai current_prod.safetensors candidate.safetensors \
  --quantization-analysis
```

**For Research Analysis**:
```bash
diffai baseline.safetensors experiment.safetensors \
  --show-layer-impact
```

**For Quantization Validation**:
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis
```

### 5. `--architecture-comparison` Architecture Comparison
Compare model architectures and detect structural changes.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --architecture-comparison
```

**Output**:
```
architecture_comparison: transformer->transformer, complexity=similar_complexity, migration=easy
```

**Analysis Fields**:
- **Architecture type detection**: Transformer, CNN, RNN, or feedforward
- **Layer depth comparison**: Number of layers and structural changes
- **Parameter count analysis**: Size ratios and complexity assessment
- **Migration difficulty**: Assessment of upgrade complexity
- **Compatibility evaluation**: Cross-architecture compatibility

**Use Cases**:
- Compare different model architectures
- Assess architectural upgrade complexity
- Analyze structural model changes

### 6. `--memory-analysis` Memory Analysis
Analyze memory usage and optimization opportunities.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --memory-analysis
```

**Output**:
```
memory_analysis: delta=+12.5MB, peak=156.3MB, efficiency=0.85, recommendation=optimal
```

**Analysis Fields**:
- **Memory delta**: Exact memory change between models
- **Peak usage estimation**: Including gradients and activations
- **GPU utilization**: Estimated GPU memory usage
- **Optimization opportunities**: Gradient checkpointing, mixed precision
- **Memory leak detection**: Unusually large tensors identification

**Use Cases**:
- Optimize memory usage for deployment
- Detect memory inefficiencies
- Plan GPU resource allocation

### 7. `--anomaly-detection` Anomaly Detection
Detect numerical anomalies in model parameters.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --anomaly-detection
```

**Output**:
```
anomaly_detection: type=none, severity=none, affected_layers=[], confidence=0.95
```

**Analysis Fields**:
- **NaN/Inf detection**: Numerical instability identification
- **Gradient explosion/vanishing**: Parameter change magnitude analysis
- **Dead neurons**: Zero variance detection
- **Root cause analysis**: Suggested causes and solutions
- **Recovery probability**: Likelihood of training recovery

**Use Cases**:
- Debug training instabilities
- Detect numerical issues early
- Validate model health

### 8. `--change-summary` Change Summary
Generate detailed change summaries.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --change-summary
```

**Output**:
```
change_summary: layers_changed=6, magnitude=0.15, patterns=[weight_updates, bias_adjustments]
```

**Analysis Fields**:
- **Change magnitude**: Overall parameter change intensity
- **Change patterns**: Types of modifications detected
- **Most changed layers**: Ranking by modification intensity
- **Structural vs parameter changes**: Classification of change types
- **Change distribution**: By layer type and function

**Use Cases**:
- Summarize model evolution
- Track training progress
- Generate reports for stakeholders

### 9. `--convergence-analysis` Convergence Analysis
Analyze convergence patterns in model parameters.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --convergence-analysis
```

**Output**:
```
convergence_analysis: status=converging, stability=0.92, early_stopping=continue
```

**Analysis Fields**:
- **Convergence status**: Converged, converging, plateaued, or diverging
- **Parameter stability**: How stable parameters are between iterations
- **Plateau detection**: Identification of training plateaus
- **Early stopping recommendation**: When to stop training
- **Remaining iterations**: Estimated iterations to convergence

**Use Cases**:
- Optimize training duration
- Detect convergence issues
- Make early stopping decisions

### 10. `--gradient-analysis` Gradient Analysis
Analyze gradient information estimated from parameter changes.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --gradient-analysis
```

**Output**:
```
gradient_analysis: flow_health=healthy, norm=0.021, ratio=2.11, clipping=none
```

**Analysis Fields**:
- **Gradient flow health**: Overall gradient quality assessment
- **Gradient norm estimation**: Magnitude of parameter updates
- **Problematic layers**: Layers with gradient issues
- **Clipping recommendation**: Suggested gradient clipping values
- **Learning rate suggestions**: Adaptive LR recommendations

**Use Cases**:
- Debug gradient flow problems
- Optimize learning rates
- Detect vanishing/exploding gradients

### 11. `--similarity-matrix` Similarity Matrix
Generate similarity matrix for model comparison.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --similarity-matrix
```

**Output**:
```
similarity_matrix: dimensions=(6,6), mean_similarity=0.65, clustering=0.73
```

**Analysis Fields**:
- **Layer-to-layer similarities**: Cosine similarity matrix
- **Clustering coefficient**: How clustered the similarities are
- **Outlier detection**: Layers with unusual similarity patterns
- **Matrix quality score**: Overall similarity matrix quality
- **Correlation patterns**: Block diagonal, hierarchical structures

**Use Cases**:
- Analyze model relationships
- Detect redundant layers
- Compare model families

## Phase 3 Features (Now Available)

The above 7 new functions (5-11) represent Phase 3 features that are now fully implemented and available for use.

## Design Philosophy

diffai follows UNIX philosophy: simple, composable tools that do one thing well. Features are orthogonal and can be combined for powerful analysis workflows.

## Integration Examples

### MLflow Integration
```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    result = subprocess.run([
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
```

### CI/CD Pipeline
```yaml
name: Model Validation
on: [push, pull_request]

jobs:
  model-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install diffai
        run: cargo install diffai
        
      - name: Compare models
        run: |
          diffai models/baseline.safetensors models/candidate.safetensors \
            --output json > model_diff.json
```

## See Also

- [CLI Reference](cli-reference.md) - Complete command reference
- [Basic Usage Guide](../user-guide/basic-usage.md) - Get started with diffai
- [ML Model Comparison Guide](../user-guide/ml-model-comparison.md) - Advanced model comparison techniques