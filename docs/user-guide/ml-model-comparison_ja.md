# ML Model Comparison Guide

This guide covers diffai's specialized features for comparing machine learning models, including PyTorch and Safetensors files.

## Overview

diffai provides native support for AI/ML model formats, allowing you to compare models at the tensor level rather than just as binary files. This enables meaningful analysis of model changes during training, fine-tuning, quantization, and deployment.

## Supported ML Formats

### PyTorch Models
- **`.pt` files**: PyTorch model files (pickle format with Candle integration)
- **`.pth` files**: PyTorch model files (alternative extension)

### Safetensors Models
- **`.safetensors` files**: HuggingFace Safetensors format (recommended)

### Future Support (Phase 3)
- **`.onnx` files**: ONNX format
- **`.h5` files**: Keras/TensorFlow HDF5 format
- **`.pb` files**: TensorFlow Protocol Buffer format

## What diffai Analyzes

### Tensor Statistics
For each tensor in the model, diffai calculates and compares:

- **Mean**: Average value of all parameters
- **Standard Deviation**: Measure of parameter variance
- **Minimum**: Smallest parameter value
- **Maximum**: Largest parameter value
- **Shape**: Tensor dimensions
- **Data Type**: Parameter precision (f32, f64, etc.)
- **Total Parameters**: Number of parameters in the tensor

### Model Architecture
- **Parameter count changes**: Total model parameters
- **Layer additions/removals**: New or deleted layers
- **Shape changes**: Modified layer dimensions

## Basic Model Comparison

### Simple Comparison

```bash
# Compare two PyTorch models (comprehensive analysis automatic)
diffai model1.pt model2.pt

# Compare Safetensors models (recommended, comprehensive analysis automatic)
diffai model1.safetensors model2.safetensors

# Automatic format detection with full analysis
diffai pretrained.safetensors finetuned.safetensors
```

### Example Output

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### Output Symbols

| Symbol | Meaning | Description |
|--------|---------|-------------|
| `~` | Statistics Changed | Tensor values changed but shape remained same |
| `+` | Added | New tensor/layer added |
| `-` | Removed | Tensor/layer removed |

## Advanced Options

### Epsilon Tolerance

Use epsilon to ignore minor floating-point differences:

```bash
# Ignore differences smaller than 1e-6
diffai model1.safetensors model2.safetensors --epsilon 1e-6

# Useful for quantization analysis
diffai fp32_model.safetensors int8_model.safetensors --epsilon 0.1
```

### Output Formats

```bash
# JSON output for automation
diffai model1.pt model2.pt --output json

# YAML output for readability
diffai model1.pt model2.pt --output yaml

# Pipe to file for processing
diffai model1.pt model2.pt --output json > changes.json
```

### Filtering Results

```bash
# Focus on specific layers
diffai model1.safetensors model2.safetensors --path "classifier"

# Ignore timestamp or metadata
diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(timestamp|_metadata)"
```

## Common Use Cases

### 1. Fine-tuning Analysis

Compare a pre-trained model with its fine-tuned version:

```bash
diffai pretrained_bert.safetensors finetuned_bert.safetensors

# Expected output: Comprehensive analysis with attention layer changes
# anomaly_detection: type=none, severity=none
# architecture_comparison: type=transformer, deployment_readiness=ready
# convergence_analysis: status=converged, stability=0.95
# ~ bert.encoder.layer.11.attention.self.query.weight: mean=-0.0001→0.0023
# ~ classifier.weight: mean=0.0000→0.0145, std=0.0200→0.0890
```

**Analysis**: 
- Small changes in early layers (feature extraction remains similar)
- Larger changes in final layers (task-specific adaptation)

### 2. Quantization Impact Assessment

Compare FP32 and quantized models:

```bash
diffai model_fp32.safetensors model_int8.safetensors --epsilon 0.1

# Expected output: Controlled precision loss
# ~ conv1.weight: mean=0.0045→0.0043, std=0.2341→0.2298
# No differences found (within epsilon tolerance)
```

**Analysis**:
- Small statistical changes indicate successful quantization
- Large changes may suggest quality loss

### 3. Training Progress Tracking

Compare checkpoints during training:

```bash
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt

# Expected output: Convergence patterns
# ~ layers.0.weight: mean=-0.0012→0.0034, std=1.2341→0.8907
# ~ layers.1.bias: mean=0.1234→0.0567, std=0.4567→0.3210
```

**Analysis**:
- Decreasing standard deviation suggests convergence
- Mean shifts show learning direction

### 4. Architecture Comparison

Compare different model architectures:

```bash
diffai resnet50.safetensors efficientnet_b0.safetensors

# Expected output: Structural differences
# ~ features.conv1.weight: shape=[64, 3, 7, 7] -> [32, 3, 3, 3]
# + features.mbconv.expand_conv.weight: shape=[96, 32, 1, 1]
# - features.layer4.2.downsample.0.weight: shape=[2048, 1024, 1, 1]
```

**Analysis**:
- Shape changes indicate different layer sizes
- Added/removed tensors show architectural innovations

## Performance Optimization

### Memory Considerations

For large models (>1GB), consider:

```bash
# Use recursive mode for directory comparison
diffai --recursive model_dir1/ model_dir2/

# Focus analysis on specific parts
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# Use higher epsilon for faster comparison
diffai model1.safetensors model2.safetensors --epsilon 1e-3
```

### Speed Optimization

```bash
# Use verbose mode for detailed processing info
diffai --verbose model1.safetensors model2.safetensors

# Focus on architecture differences only
diffai --architecture-comparison model1.safetensors model2.safetensors
```

## Integration Examples

### MLflow Integration

```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    # Run diffai comparison
    result = subprocess.run([
        'diffai', model1_path, model2_path, '--output', 'json'
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
        
        # Count change types
        stats_changes = len([d for d in diff_data if 'TensorStatsChanged' in d])
        shape_changes = len([d for d in diff_data if 'TensorShapeChanged' in d])
        
        mlflow.log_metric("stats_changes", stats_changes)
        mlflow.log_metric("shape_changes", shape_changes)
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
            
      - name: Analyze changes
        run: |
          # Fail if critical layers changed
          if jq -e '.[] | select(.TensorShapeChanged and (.TensorShapeChanged[0] | contains("classifier")))' model_diff.json; then
            echo "CRITICAL: Critical layer shape changes detected"
            exit 1
          fi
          
          # Warn if many parameters changed
          changes=$(jq length model_diff.json)
          if [ "$changes" -gt 10 ]; then
            echo "WARNING: Many parameter changes detected: $changes"
          fi
```

### Git Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

model_files=$(git diff --cached --name-only | grep -E '\.(pt|pth|safetensors)$')

for file in $model_files; do
    if [ -f "$file" ]; then
        echo "Analyzing model changes in $file"
        
        # Compare with previous version
        git show HEAD:"$file" > /tmp/old_model
        
        diffai /tmp/old_model "$file" --output json > /tmp/model_diff.json
        
        # Check for significant changes
        shape_changes=$(jq '[.[] | select(.TensorShapeChanged)] | length' /tmp/model_diff.json)
        
        if [ "$shape_changes" -gt 0 ]; then
            echo "WARNING: Architecture changes detected in $file"
            diffai /tmp/old_model "$file"
            
            read -p "Continue with commit? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        rm -f /tmp/old_model /tmp/model_diff.json
    fi
done
```

## Troubleshooting

### Common Issues

#### 1. "Failed to parse" Errors

```bash
# Check file format
file model.safetensors

# Show detailed statistics for single model analysis
diffai model.safetensors model.safetensors

# Try with explicit format
diffai --format safetensors model1.safetensors model2.safetensors
```

#### 2. Memory Issues with Large Models

```bash
# Use higher epsilon to reduce precision
diffai --epsilon 1e-3 large1.safetensors large2.safetensors

# Focus on specific layers
diffai --path "tensor.classifier" large1.safetensors large2.safetensors
```

#### 3. Binary File Errors

```bash
# Ensure files are actual model files, not corrupted
ls -la model*.safetensors

# Check if files are compressed
file model.safetensors

# Try extracting if compressed
gunzip model.safetensors.gz
```

## Best Practices

### 1. Choosing Epsilon Values

| Use Case | Recommended Epsilon | Reason |
|----------|-------------------|---------|
| Exact comparison | No epsilon | Detect all changes |
| Training progress | 1e-6 | Ignore numerical noise |
| Quantization analysis | 0.01-0.1 | Account for precision loss |
| Architecture check | 1e-3 | Focus on structural changes |

### 2. Output Format Selection

- **CLI**: Human review and debugging
- **JSON**: Automation and scripting
- **YAML**: Configuration files and documentation

### 3. Performance Tips

- Use `--path` to focus analysis on relevant layers
- Set appropriate epsilon values to avoid noise
- Consider model size when choosing comparison strategy

## ML Analysis Features

### Comprehensive Analysis (Automatic)

diffai now provides automatic comprehensive analysis with 30+ features for ML models:

### Automatic Analysis Features

All analysis is performed automatically when comparing PyTorch/Safetensors files:

```bash
# Single command triggers comprehensive analysis
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors

# Output includes:
# - Anomaly detection
# - Architecture comparison
# - Convergence analysis
# - Gradient analysis
# - Memory analysis
# - Quantization analysis
# - Regression testing
# - Deployment readiness
# - Statistical analysis
# - And 20+ more features
```

**Analysis Information (Automatic):**
- **Statistical metrics**: mean, std, min/max, shape, dtype
- **Architecture analysis**: model type, complexity, migration difficulty
- **Performance metrics**: memory usage, quantization impact, speedup
- **Training insights**: convergence status, gradient health, stability
- **Deployment readiness**: risk assessment, compatibility, optimization

**Use Cases:**
- Monitor complete training progress
- Validate production deployment readiness
- Comprehensive model comparison
- Automated quality assurance

## Simplified Usage Guide

**For Training Monitoring:**
```bash
# Comprehensive analysis automatic
diffai checkpoint_old.safetensors checkpoint_new.safetensors
```

**For Production Deployment:**
```bash
# Full deployment readiness analysis automatic
diffai current_prod.safetensors candidate.safetensors
```

**For Research Analysis:**
```bash
# Complete experimental comparison automatic
diffai baseline.safetensors experiment.safetensors
```

**For Quantization Validation:**
```bash
# Automatic quantization impact assessment
diffai fp32.safetensors quantized.safetensors
```

## Advanced Features (Integrated)

### Core Analysis Features (Automatic)

#### Architecture Comparison (`--architecture-comparison`)
Compare model architectures and detect structural changes:

```bash
diffai model1.safetensors model2.safetensors --architecture-comparison

# Output example:
# architecture_comparison: transformer->transformer, complexity=similar_complexity, migration=easy
```

**Analysis Information:**
- **Architecture type detection**: Transformer, CNN, RNN, or feedforward
- **Layer depth comparison**: Number of layers and structural changes
- **Parameter count analysis**: Size ratios and complexity assessment
- **Migration difficulty**: Assessment of upgrade complexity
- **Compatibility evaluation**: Cross-architecture compatibility

#### Memory Analysis (`--memory-analysis`) 
Analyze memory usage and optimization opportunities:

```bash
diffai model1.safetensors model2.safetensors --memory-analysis

# Output example:
# memory_analysis: delta=+12.5MB, peak=156.3MB, efficiency=0.85, recommendation=optimal
```

**Analysis Information:**
- **Memory delta**: Exact memory change between models
- **Peak usage estimation**: Including gradients and activations
- **GPU utilization**: Estimated GPU memory usage
- **Optimization opportunities**: Gradient checkpointing, mixed precision
- **Memory leak detection**: Unusually large tensors identification

#### Anomaly Detection (`--anomaly-detection`)
Detect numerical anomalies in model parameters:

```bash
diffai model1.safetensors model2.safetensors --anomaly-detection

# Output example:
# anomaly_detection: type=none, severity=none, affected_layers=[], confidence=0.95
```

**Analysis Information:**
- **NaN/Inf detection**: Numerical instability identification
- **Gradient explosion/vanishing**: Parameter change magnitude analysis
- **Dead neurons**: Zero variance detection
- **Root cause analysis**: Suggested causes and solutions
- **Recovery probability**: Likelihood of training recovery

#### Change Summary (`--change-summary`)
Generate detailed change summaries:

```bash
diffai model1.safetensors model2.safetensors --change-summary

# Output example:
# change_summary: layers_changed=6, magnitude=0.15, patterns=[weight_updates, bias_adjustments]
```

**Analysis Information:**
- **Change magnitude**: Overall parameter change intensity
- **Change patterns**: Types of modifications detected
- **Most changed layers**: Ranking by modification intensity
- **Structural vs parameter changes**: Classification of change types
- **Change distribution**: By layer type and function

### Phase 3B: Advanced Analysis Features

#### Convergence Analysis (`--convergence-analysis`)
Analyze convergence patterns in model parameters:

```bash
diffai model1.safetensors model2.safetensors --convergence-analysis

# Output example:
# convergence_analysis: status=converging, stability=0.92, early_stopping=continue
```

**Analysis Information:**
- **Convergence status**: Converged, converging, plateaued, or diverging
- **Parameter stability**: How stable parameters are between iterations
- **Plateau detection**: Identification of training plateaus
- **Early stopping recommendation**: When to stop training
- **Remaining iterations**: Estimated iterations to convergence

#### Gradient Analysis (`--gradient-analysis`)
Analyze gradient information estimated from parameter changes:

```bash
diffai model1.safetensors model2.safetensors --gradient-analysis

# Output example:
# gradient_analysis: flow_health=healthy, norm=0.021, ratio=2.11, clipping=none
```

**Analysis Information:**
- **Gradient flow health**: Overall gradient quality assessment
- **Gradient norm estimation**: Magnitude of parameter updates
- **Problematic layers**: Layers with gradient issues
- **Clipping recommendation**: Suggested gradient clipping values
- **Learning rate suggestions**: Adaptive LR recommendations

#### Similarity Matrix (`--similarity-matrix`)
Generate similarity matrix for model comparison:

```bash
diffai model1.safetensors model2.safetensors --similarity-matrix

# Output example:
# similarity_matrix: dimensions=(6,6), mean_similarity=0.65, clustering=0.73
```

**Analysis Information:**
- **Layer-to-layer similarities**: Cosine similarity matrix
- **Clustering coefficient**: How clustered the similarities are
- **Outlier detection**: Layers with unusual similarity patterns
- **Matrix quality score**: Overall similarity matrix quality
- **Correlation patterns**: Block diagonal, hierarchical structures

### Combined Analysis Examples

```bash
# Comprehensive Phase 3 analysis
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison \
  --memory-analysis \
  --anomaly-detection \
  --change-summary \
  --convergence-analysis \
  --gradient-analysis \
  --similarity-matrix

# JSON output for MLOps integration
diffai model1.safetensors model2.safetensors \
  --architecture-comparison \
  --memory-analysis \
  --output json
```

### Design Philosophy
diffai follows UNIX philosophy: simple, composable tools that do one thing well. Phase 3 features are orthogonal and can be combined for powerful analysis workflows.

## Next Steps

- [Basic Usage](basic-usage_ja.md) - Learn fundamental operations
- [Scientific Data Analysis](scientific-data_ja.md) - NumPy and MATLAB file comparison
- [CLIリファレンス](../reference/cli-reference_ja.md) - Complete command reference