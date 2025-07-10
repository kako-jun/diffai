# ML Analysis Functions

Comprehensive guide to diffai's machine learning analysis functions for model comparison and analysis.

## Overview

diffai provides specialized analysis functions designed specifically for machine learning model comparison and analysis. These functions help with research and development, MLOps, and deployment workflows.

## Currently Available Functions (v0.2.0)

### 1. `--stats` Statistical Analysis
Provides detailed tensor statistics for model comparison.

**Usage**:
```bash
diffai model1.safetensors model2.safetensors --stats
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

### 2. `--quantization-analysis` Quantization Analysis
Analyzes quantization effects and efficiency.

**Usage**:
```bash
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis
```

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
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude --stats
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
  --stats \
  --quantization-analysis \
  --sort-by-change-magnitude \
  --show-layer-impact

# JSON output for automation
diffai model1.safetensors model2.safetensors \
  --stats --output json
```

## Feature Selection Guide

**For Training Monitoring**:
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --stats --sort-by-change-magnitude
```

**For Production Deployment**:
```bash
diffai current_prod.safetensors candidate.safetensors \
  --stats --quantization-analysis
```

**For Research Analysis**:
```bash
diffai baseline.safetensors experiment.safetensors \
  --stats --show-layer-impact
```

**For Quantization Validation**:
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --stats
```

## Future Features (Phase 3)

### Coming in Phase 3A (Core Features)
- `--architecture-comparison` - Compare model architectures and structural changes
- `--memory-analysis` - Analyze memory usage and optimization opportunities
- `--anomaly-detection` - Detect numerical anomalies in model parameters
- `--change-summary` - Generate detailed change summaries

### Coming in Phase 3B (Advanced Analysis)
- `--convergence-analysis` - Analyze convergence patterns in model parameters
- `--gradient-analysis` - Analyze gradient information when available
- `--similarity-matrix` - Generate similarity matrix for model comparison

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
        'diffai', model1_path, model2_path, '--stats', '--output', 'json'
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
            --stats --output json > model_diff.json
```

## See Also

- [CLI Reference](cli-reference.md) - Complete command reference
- [Basic Usage Guide](../user-guide/basic-usage.md) - Get started with diffai
- [ML Model Comparison Guide](../user-guide/ml-model-comparison.md) - Advanced model comparison techniques