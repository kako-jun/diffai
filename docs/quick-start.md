# Quick Start - diffai

Get up and running with diffai in 5 minutes. diffai is a specialized diff tool for AI/ML models that automatically provides 11 comprehensive analysis functions when comparing PyTorch or Safetensors files.

## Installation

```bash
# Install from crates.io (recommended)
cargo install diffai

# Or from source
git clone https://github.com/kako-jun/diffai.git
cd diffai && cargo build --release
```

## Basic Usage

### Compare ML Models (Automatic Analysis)

```bash
# PyTorch models - 11 ML analyses run automatically
diffai model_old.pt model_new.pt

# Safetensors - 11 ML analyses run automatically  
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# Output example:
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
# ... + 6 more analyses
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### Scientific Data (Basic Analysis)

```bash
# NumPy arrays - tensor statistics only
diffai experiment_v1.npy experiment_v2.npy

# MATLAB files - tensor statistics only
diffai simulation_v1.mat simulation_v2.mat
```

## Output Formats

### JSON (MLOps Integration)
```bash
diffai model1.safetensors model2.safetensors --output json
```

### YAML (Human-Readable Reports)
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

### Verbose Mode (Diagnostics)
```bash
diffai model1.safetensors model2.safetensors --verbose
```

## What Makes diffai Different

### Automatic ML Analysis
- **No Configuration Required**: 11 ML analysis functions run automatically for PyTorch/Safetensors
- **Convention over Configuration**: Following lawkit patterns for zero-setup experience
- **Built on diffx-core**: Proven reliability for diff operations

### Specialized for AI/ML
- **Native Tensor Support**: Understands PyTorch, Safetensors, NumPy, MATLAB formats
- **Statistical Analysis**: Automatic tensor statistics (mean, std, shape, memory)
- **ML-Specific Insights**: Gradient analysis, quantization detection, convergence patterns

### Traditional Tools vs diffai
```bash
# Traditional diff
$ diff model_v1.pt model_v2.pt
Binary files model_v1.pt and model_v2.pt differ

# diffai
$ diffai model_v1.pt model_v2.pt
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
gradient_analysis: flow_health=healthy, norm=0.021069
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%
# ... comprehensive ML analysis automatically
```

## Common Use Cases

### Research & Development
```bash
# Compare before/after fine-tuning (automatic comprehensive analysis)
diffai pretrained_model.safetensors finetuned_model.safetensors

# Outputs: learning progress, convergence analysis, parameter evolution, etc.
```

### MLOps & CI/CD
```bash
# Automated model validation in CI/CD pipelines
diffai production_model.safetensors candidate_model.safetensors --output json

# Pipe to jq or other tools for automated processing
diffai baseline.pt improved.pt --output json | jq '.gradient_analysis'
```

### Model Optimization
```bash
# Analyze quantization effects
diffai full_precision.pt quantized.pt
# Auto-detects: mixed precision, compression ratios, precision loss

# Memory usage analysis
diffai large_model.safetensors optimized_model.safetensors --verbose
```

## Next Steps

- **[Examples](examples/)** - See real diffai outputs and use cases
- **[ML Analysis](ml-analysis.md)** - Understand the 11 automatic analysis functions  
- **[API Reference](reference/api-reference.md)** - Use diffai in your Rust/Python/JavaScript code
- **[File Formats](formats.md)** - Supported AI/ML file format details

## Key Options

```bash
# Basic comparison options
--epsilon <FLOAT>           # Tolerance for float comparisons
--output <FORMAT>           # cli (default), json, yaml
--verbose                   # Detailed diagnostic information
--no-color                  # Disable colored output

# Path filtering
--path <PATH>               # Filter differences by specific path
--ignore-keys-regex <REGEX> # Ignore keys matching regex pattern

# Memory optimization (for large models)
# Memory optimization is automatic - no configuration needed
```

diffai follows **Convention over Configuration**: ML analysis runs automatically when it detects AI/ML files, giving you comprehensive insights without any setup.