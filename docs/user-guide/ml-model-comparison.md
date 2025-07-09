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
# Compare two PyTorch models
diffai model1.pt model2.pt --stats

# Compare Safetensors models (recommended)
diffai model1.safetensors model2.safetensors --stats

# Automatic format detection
diffai pretrained.safetensors finetuned.safetensors --stats
```

### Example Output

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
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
diffai pretrained_bert.safetensors finetuned_bert.safetensors --stats

# Expected output: Statistical changes in attention layers
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
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt --stats

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
diffai resnet50.safetensors efficientnet_b0.safetensors --stats

# Expected output: Structural differences
# ~ features.conv1.weight: shape=[64, 3, 7, 7] -> [32, 3, 3, 3]
# + features.mbconv.expand_conv.weight: shape=[96, 32, 1, 1]
# - features.layer4.2.downsample.0.weight: shape=[2048, 1024, 1, 1]
```

**Analysis**:
- Shape changes indicate different layer sizes
- Added/removed tensors show architectural innovations

## 📈 Performance Optimization

### Memory Considerations

For large models (>1GB), consider:

```bash
# Use streaming mode (future feature)
diffai --stream huge_model1.safetensors huge_model2.safetensors

# Focus analysis on specific parts
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# Use higher epsilon for faster comparison
diffai model1.safetensors model2.safetensors --epsilon 1e-3
```

### Speed Optimization

```bash
# Parallel processing (future feature)
diffai --threads 8 model1.safetensors model2.safetensors

# Skip statistical calculation for shape-only analysis
diffai --shape-only model1.safetensors model2.safetensors
```

## 🔧 Integration Examples

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
            echo "❌ Critical layer shape changes detected"
            exit 1
          fi
          
          # Warn if many parameters changed
          changes=$(jq length model_diff.json)
          if [ "$changes" -gt 10 ]; then
            echo "⚠️ Many parameter changes detected: $changes"
          fi
```

### Git Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

model_files=$(git diff --cached --name-only | grep -E '\.(pt|pth|safetensors)$')

for file in $model_files; do
    if [ -f "$file" ]; then
        echo "🔍 Analyzing model changes in $file"
        
        # Compare with previous version
        git show HEAD:"$file" > /tmp/old_model
        
        diffai /tmp/old_model "$file" --output json > /tmp/model_diff.json
        
        # Check for significant changes
        shape_changes=$(jq '[.[] | select(.TensorShapeChanged)] | length' /tmp/model_diff.json)
        
        if [ "$shape_changes" -gt 0 ]; then
            echo "⚠️ Architecture changes detected in $file"
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

## 🚨 Troubleshooting

### Common Issues

#### 1. "Failed to parse" Errors

```bash
# Check file format
file model.safetensors

# Verify file integrity
diffai --check model.safetensors

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

## 📊 Best Practices

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

## Advanced ML Analysis Features

diffai provides 28 advanced machine learning analysis features designed for comprehensive model evaluation:

### 1. Learning Progress Analysis (`--learning-progress`)

Analyzes training progression between model checkpoints:

```bash
# Compare training checkpoints
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress

# Output example:
# + learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

**Analysis Information:**
- **trend**: `improving`, `degrading`, or `stable`
- **magnitude**: Amount of change (0.0-1.0)
- **speed**: Rate of convergence (0.0-1.0)

**Use Cases:**
- Monitor training progress
- Detect learning plateaus
- Optimize training schedules

### 2. Convergence Analysis (`--convergence-analysis`)

Evaluates model stability and convergence status:

```bash
# Analyze convergence between checkpoints
diffai model_before.safetensors model_after.safetensors --convergence-analysis

# Output example:
# + convergence_analysis: status=stable, stability=0.0234, action="Continue training"
```

**Analysis Information:**
- **status**: `converged`, `diverging`, `oscillating`, `stable`
- **stability**: Variance in parameter changes (lower = more stable)
- **action**: Recommended next steps

**Use Cases:**
- Determine when to stop training
- Detect training instability
- Optimize hyperparameters

### 3. Anomaly Detection (`--anomaly-detection`)

Detects abnormal patterns in model weights:

```bash
# Detect training anomalies
diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection

# Output example:
# 🚨 anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

**Detected Anomalies:**
- **Gradient explosion**: Extremely large weight values
- **Gradient vanishing**: Near-zero gradients
- **Weight distribution shift**: Unusual statistical patterns
- **NaN/Inf values**: Invalid numerical values

**Use Cases:**
- Debug training problems
- Validate model health
- Prevent deployment of corrupted models

### 4. Memory Analysis (`--memory-analysis`)

Analyzes memory usage and model efficiency:

```bash
# Compare model memory footprints
diffai small_model.safetensors large_model.safetensors --memory-analysis

# Output example:
# 🧠 memory_analysis: delta=+2.7MB, gpu_est=4.5MB, efficiency=0.25
```

**Analysis Information:**
- **delta**: Memory difference between models
- **gpu_est**: Estimated GPU memory requirements
- **efficiency**: Parameters per MB ratio

**Use Cases:**
- Optimize for deployment constraints
- Compare architecture efficiency
- Plan hardware requirements

### 5. Architecture Comparison (`--architecture-comparison`)

Compares model architectures and structures:

```bash
# Compare different architectures
diffai resnet.safetensors transformer.safetensors --architecture-comparison

# Output example:
# 🏗️ architecture_comparison: type1=cnn, type2=transformer, differences=15
```

**Analysis Information:**
- **Architecture types**: CNN, RNN, Transformer, MLP, etc.
- **Layer differences**: Added, removed, or modified layers
- **Parameter distribution**: How parameters are allocated

**Use Cases:**
- Evaluate architecture changes
- Compare model families
- Design decision validation

### 6. Multi-Feature Analysis

Combine multiple features for comprehensive analysis:

```bash
# Comprehensive training analysis
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection \
  --memory-analysis \
  --stats

# Output example:
# + learning_progress: trend=improving, magnitude=0.0432, speed=0.75
# + convergence_analysis: status=stable, stability=0.0156
# 🧠 memory_analysis: delta=+0.1MB, efficiency=0.89
# 📊 Tensor statistics and detailed analysis...
```

### 7. Production Deployment Features

Essential features for production environments:

```bash
# Production readiness check
diffai production.safetensors candidate.safetensors \
  --anomaly-detection \
  --memory-analysis \
  --deployment-readiness

# Regression testing
diffai baseline.safetensors new_version.safetensors \
  --regression-test \
  --alert-on-degradation
```

### 8. Research and Development Features

Advanced analysis for research workflows:

```bash
# Hyperparameter impact analysis
diffai model_lr_001.safetensors model_lr_0001.safetensors \
  --hyperparameter-impact \
  --learning-rate-analysis

# Architecture efficiency analysis
diffai efficient_model.safetensors baseline_model.safetensors \
  --param-efficiency-analysis \
  --architecture-comparison
```

## 🎯 Feature Selection Guide

**For Training Monitoring:**
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress --convergence-analysis --anomaly-detection
```

**For Production Deployment:**
```bash
diffai current_prod.safetensors candidate.safetensors \
  --anomaly-detection --memory-analysis --deployment-readiness
```

**For Research Analysis:**
```bash
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison --hyperparameter-impact --stats
```

**For Quantization Validation:**
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --memory-analysis --performance-impact-estimate
```

## All 28 Advanced Features

### Learning & Convergence Analysis (4 features)
- `--learning-progress` - Track learning progress between checkpoints
- `--convergence-analysis` - Analyze convergence stability and patterns
- `--anomaly-detection` - Detect training anomalies (gradient explosion, vanishing gradients)
- `--gradient-analysis` - Analyze gradient characteristics and flow

### Architecture & Performance Analysis (4 features)
- `--architecture-comparison` - Compare model architectures and structural changes
- `--param-efficiency-analysis` - Analyze parameter efficiency between models
- `--memory-analysis` - Analyze memory usage and optimization opportunities
- `--inference-speed-estimate` - Estimate inference speed and performance characteristics

### MLOps & Deployment Support (7 features)
- `--deployment-readiness` - Assess deployment readiness and compatibility
- `--regression-test` - Perform automated regression testing
- `--risk-assessment` - Evaluate deployment risks and stability
- `--hyperparameter-impact` - Analyze hyperparameter impact on model changes
- `--learning-rate-analysis` - Analyze learning rate effects and optimization
- `--alert-on-degradation` - Alert on performance degradation beyond thresholds
- `--performance-impact-estimate` - Estimate performance impact of changes

### Experiment & Documentation Support (4 features)
- `--generate-report` - Generate comprehensive analysis reports
- `--markdown-output` - Output results in markdown format for documentation
- `--include-charts` - Include charts and visualizations in output
- `--review-friendly` - Generate review-friendly output for human reviewers

### Advanced Analysis Functions (6 features)
- `--embedding-analysis` - Analyze embedding layer changes and semantic drift
- `--similarity-matrix` - Generate similarity matrix for model comparison
- `--clustering-change` - Analyze clustering changes in model representations
- `--attention-analysis` - Analyze attention mechanism patterns (Transformer models)
- `--head-importance` - Analyze attention head importance and specialization
- `--attention-pattern-diff` - Compare attention patterns between models

### Additional Analysis Functions (3 features)
- `--quantization-analysis` - Analyze quantization effects and efficiency
- `--sort-by-change-magnitude` - Sort differences by magnitude for prioritization
- `--change-summary` - Generate detailed change summaries

## Next Steps

- [Basic Usage](basic-usage.md) - Learn fundamental operations
- [Scientific Data Analysis](scientific-data.md) - NumPy and MATLAB file comparison
- [CLI Reference](../reference/cli-reference.md) - Complete command reference