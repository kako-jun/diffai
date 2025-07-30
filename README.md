# diffai

> **AI/ML specialized diff tool for PyTorch, Safetensors, NumPy, and MATLAB files**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io CLI](https://img.shields.io/crates/v/diffai.svg?label=diffai-cli)](https://crates.io/crates/diffai)
[![Docs.rs Core](https://docs.rs/diffai-core/badge.svg)](https://docs.rs/diffai-core)
[![npm](https://img.shields.io/npm/v/diffai-js.svg?label=diffai-js)](https://www.npmjs.com/package/diffai-js)
[![PyPI](https://img.shields.io/pypi/v/diffai-python.svg?label=diffai-python)](https://pypi.org/project/diffai-python/)
[![Documentation](https://img.shields.io/badge/📚%20User%20Guide-Documentation-green)](https://github.com/kako-jun/diffai/tree/main/docs/index.md)
[![API Reference](https://img.shields.io/badge/🔧%20API%20Reference-docs.rs-blue)](https://docs.rs/diffai-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A next-generation diff tool specialized for **AI/ML and scientific computing workflows** that understands model structures, tensor statistics, and numerical data - not just text changes. Native support for PyTorch, Safetensors, NumPy arrays, MATLAB files, and structured data.

```bash
# Traditional diff fails with binary model files
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffai shows meaningful model changes with automatic ML analysis
$ diffai model_v1.safetensors model_v2.safetensors
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
ensemble_analysis: ensemble_detected=false, model_type=single
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

## Key Features

- **AI/ML Native**: Direct support for PyTorch (.pt/.pth), Safetensors (.safetensors), NumPy (.npy/.npz), and MATLAB (.mat) files
- **Tensor Analysis**: Automatic calculation of tensor statistics (mean, std, min, max, shape, memory usage)
- **Comprehensive ML Analysis**: 11 specialized ML analysis functions including gradient analysis, quantization analysis, convergence patterns, learning rate tracking, and deployment readiness - all automatically enabled for AI/ML files
- **Built on diffx-core**: Leverages proven diff engine from diffx project for reliable core functionality
- **Scientific Data Support**: NumPy arrays and MATLAB matrices with complex number support
- **Convention over Configuration**: Zero-configuration ML analysis - automatically detects and analyzes AI/ML content without manual setup
- **Pure Rust Implementation**: No system dependencies, works on Windows/Linux/macOS without additional installations
- **Multiple Output Formats**: Colored CLI, JSON for MLOps integration, YAML for human-readable reports
- **Fast and Memory Efficient**: Built in Rust for handling large model files efficiently

## Why diffai?

Traditional diff tools are inadequate for AI/ML workflows:

| Challenge | Traditional Tools | diffai |
|-----------|------------------|---------|
| **Binary model files** | "Binary files differ" | Tensor-level analysis with statistics |
| **Large files (GB+)** | Memory issues or failures | Efficient streaming and chunked processing |
| **Statistical changes** | No semantic understanding | Mean/std/shape comparison with significance |
| **ML-specific formats** | No support | Native PyTorch/Safetensors/NumPy/MATLAB |
| **Scientific workflows** | Text-only comparison | Numerical array analysis and visualization |

### diffai vs MLOps Tools

diffai complements existing MLOps tools by focusing on **structural comparison** rather than experiment management:

| Aspect | diffai | MLflow / DVC / ModelDB |
|--------|--------|------------------------|
| **Focus** | "Making incomparable things comparable" | Systematization, reproducibility, CI/CD integration |
| **Data Assumption** | Unknown origin files / black-box generated artifacts | Well-documented and tracked data |
| **Operation** | Structural and visual comparison optimization | Version control and experiment tracking specialization |
| **Scope** | Visualization of "ambiguous structures" including JSON/YAML/model files | Experiment metadata, version management, reproducibility |

## Installation

### From crates.io (Recommended)

```bash
cargo install diffai
```

### From Source

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build --release
```

## Quick Start

### Basic Model Comparison

```bash
# Compare PyTorch models (11 ML analyses run automatically)
diffai model_old.pt model_new.pt

# Compare Safetensors (11 ML analyses run automatically)
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# Compare NumPy arrays (basic tensor statistics)
diffai data_v1.npy data_v2.npy

# Compare MATLAB files (basic tensor statistics)
diffai experiment_v1.mat experiment_v2.mat
```

### ML Analysis Features

```bash
# All 11 ML analysis functions run automatically for PyTorch/Safetensors
diffai baseline.safetensors finetuned.safetensors
# Auto-outputs: learning_rate_analysis, gradient_analysis, convergence_analysis, etc.

# JSON output for MLOps integration
diffai model_v1.safetensors model_v2.safetensors --output json

# Detailed diagnostic information with verbose mode
diffai model_v1.safetensors model_v2.safetensors --verbose

# YAML output for human-readable reports
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 📚 Documentation

- **[Quick Start](docs/quick-start.md)** - Get up and running in 5 minutes
- **[ML Analysis](docs/ml-analysis.md)** - Understand the 11 automatic ML analysis functions
- **[File Formats](docs/formats.md)** - Supported formats and output options
- **[Examples](docs/examples/)** - Real usage examples and outputs
- **[API Reference](docs/reference/api-reference.md)** - Programming interfaces (Rust/Python/JavaScript)
- **[CLI Reference](docs/reference/cli-reference.md)** - Command-line options and usage

## Supported AI/ML File Formats

diffai is specialized for AI/ML and scientific computing files only:

### ML Model Formats
- **Safetensors** (.safetensors) - HuggingFace standard format
- **PyTorch** (.pt, .pth) - PyTorch model files with Candle integration

### Scientific Data Formats  
- **NumPy** (.npy, .npz) - NumPy arrays with full statistical analysis
- **MATLAB** (.mat) - MATLAB matrices with complex number support

**Note**: For general-purpose structured data formats (JSON, YAML, CSV, XML, etc.), please use our sibling project [diffx](https://github.com/kako-jun/diffx) which is specifically designed for those formats.

## ML Analysis Functions

### Automatic Comprehensive Analysis (v0.3.16)
When comparing PyTorch or Safetensors files, diffai automatically runs **11 specialized ML analysis functions** without requiring any configuration:

**Auto-Enabled ML Analysis Functions:**
1. **Learning Rate Analysis**: Track learning rate changes and training dynamics
2. **Optimizer Comparison**: Compare optimizer states and momentum information
3. **Loss Tracking**: Analyze loss function evolution and convergence patterns
4. **Accuracy Tracking**: Monitor accuracy changes and performance metrics
5. **Model Version Detection**: Identify model versioning and checkpoint information
6. **Gradient Analysis**: Analyze gradient flow, vanishing/exploding gradients, and stability
7. **Quantization Analysis**: Detect mixed precision (FP32/FP16/INT8/INT4) and compression effects
8. **Convergence Analysis**: Learning curve analysis, plateau detection, and optimization trajectory
9. **Activation Analysis**: Analyze activation function usage and distribution
10. **Attention Analysis**: Analyze attention mechanisms and transformer components (when present)
11. **Ensemble Analysis**: Detect and analyze ensemble model structures

**Automatic Trigger Conditions:**
- **PyTorch files (.pt/.pth)**: All 11 analyses run automatically
- **Safetensors files (.safetensors)**: All 11 analyses run automatically  
- **NumPy/MATLAB files**: Basic tensor statistics only (non-ML analysis)
- **Other formats**: Standard structural comparison

### Future Enhancements
- TensorFlow format support (.pb, .h5, SavedModel)
- ONNX format support
- Advanced visualization and charting features

### Design Philosophy
diffai follows the **Convention over Configuration** principle (inspired by lawkit patterns): comprehensive ML analysis runs automatically for AI/ML files, eliminating choice paralysis. Users get all relevant insights without needing to remember or specify analysis flags. Built on the proven diffx-core engine for reliability.

## Debugging and Diagnostics

### Verbose Mode (`--verbose` / `-v`)
Get comprehensive diagnostic information for debugging and performance analysis:

```bash
# Basic verbose output
diffai model1.safetensors model2.safetensors --verbose

# Verbose with structured data filtering
diffai data1.json data2.json --verbose --epsilon 0.001 --ignore-keys-regex "^id$"
```

**Verbose output includes:**
- **Configuration diagnostics**: Format settings, filters, analysis modes
- **File analysis**: Paths, sizes, detected formats, processing context
- **Performance metrics**: Processing time, difference counts, optimization status
- **Directory statistics**: File counts, comparison summaries (automatic for directories)

**Example verbose output:**
```
=== diffai verbose mode enabled ===
Configuration:
  Input format: Safetensors
  Output format: Cli
  ML analysis: Full analysis enabled (all 30 features)
  Epsilon tolerance: 0.001

File analysis:
  Input 1: model1.safetensors
  Input 2: model2.safetensors
  Detected format: Safetensors
  File 1 size: 1048576 bytes
  File 2 size: 1048576 bytes

Processing results:
  Total processing time: 1.234ms
  Differences found: 15
  ML/Scientific data analysis completed
```

📚 **See [CLI Reference](docs/reference/cli-reference.md) for detailed verbose mode usage**

## Output Formats

### CLI Output (Default)
Colored, human-readable output with intuitive symbols:
- `~` Changed tensors/arrays with statistical comparison
- `+` Added tensors/arrays with metadata
- `-` Removed tensors/arrays with metadata

### JSON Output
Structured output for MLOps integration and automation:
```bash
diffai model1.safetensors model2.safetensors --output json | jq .
```

### YAML Output  
Human-readable structured output for documentation:
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## Real-World Use Cases

### Research & Development
```bash
# Compare model before and after fine-tuning (full analysis automatic)
diffai pretrained_model.safetensors finetuned_model.safetensors
# Outputs: learning_progress, convergence_analysis, parameter stats, and 27 more analyses

# Analyze architectural changes during development
diffai baseline_architecture.pt improved_architecture.pt
# Outputs: architecture_comparison, param_efficiency_analysis, and full ML analysis
```

### MLOps & CI/CD
```bash
# Automated model validation in CI/CD (comprehensive analysis)
diffai production_model.safetensors candidate_model.safetensors
# Outputs: deployment_readiness, regression_test, risk_assessment, and 27 more analyses

# Performance impact assessment with JSON output for automation
diffai original_model.pt optimized_model.pt --output json
# Outputs: quantization_analysis, memory_analysis, performance_impact_estimate, etc.
```

### Scientific Computing
```bash
# Compare NumPy experiment results
diffai baseline_results.npy new_results.npy

# Analyze MATLAB simulation data
diffai simulation_v1.mat simulation_v2.mat

# Compare compressed NumPy archives
diffai dataset_v1.npz dataset_v2.npz
```

### Experiment Tracking
```bash
# Generate comprehensive reports
diffai experiment_baseline.safetensors experiment_improved.safetensors \
  --generate-report --markdown-output --review-friendly

# A/B test analysis
diffai model_a.safetensors model_b.safetensors \
  --statistical-significance --hyperparameter-comparison
```

## Command-Line Options

### Basic Options
- `-f, --format <FORMAT>` - Specify input file format
- `-o, --output <OUTPUT>` - Choose output format (cli, json, yaml)
- **Directory comparison** - Automatically recursive when directories are provided

**Note:** For ML models (PyTorch/Safetensors), all 11 specialized ML analysis functions run automatically without configuration

### Advanced Options
- `--path <PATH>` - Filter differences by specific path
- `--ignore-keys-regex <REGEX>` - Ignore keys matching regex pattern
- `--epsilon <FLOAT>` - Set tolerance for float comparisons
- `--array-id-key <KEY>` - Specify key for array element identification
- `--sort-by-change-magnitude` - Sort by change magnitude

## Examples

### Basic Tensor Comparison (All 11 ML Analyses Automatic)
```bash
$ diffai simple_model_v1.safetensors simple_model_v2.safetensors
learning_rate_analysis: old=0.001, new=0.001, change=0.0%, trend=stable
optimizer_comparison: type=Adam, momentum_change=+1.2%, state_evolution=stable
loss_tracking: loss_trend=decreasing, improvement_rate=8.5%, convergence_score=0.85
accuracy_tracking: accuracy_delta=+1.8%, performance_trend=improving
model_version_analysis: version_change=detected, checkpoint_evolution=minor
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+5.2%
quantization_analysis: mixed_precision=none, compression=0.0%, precision_loss=0.0%
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
activation_analysis: relu_usage=100%, activation_distribution=healthy
attention_analysis: transformer_components=none, attention_detected=false
ensemble_analysis: ensemble_detected=false, model_type=feedforward
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### JSON Output for Automation
```bash
$ diffai baseline.safetensors improved.safetensors --output json
{
  "learning_rate_analysis": {"old": 0.001, "new": 0.0015, "change": "+50.0%"},
  "optimizer_comparison": {"type": "Adam", "momentum_change": "+2.1%"},
  "loss_tracking": {"loss_trend": "decreasing", "improvement_rate": "15.2%"},
  "accuracy_tracking": {"accuracy_delta": "+3.2%", "performance_trend": "improving"},
  "model_version_analysis": {"version_change": "1.0->1.1", "evolution": "incremental"},
  "gradient_analysis": {"flow_health": "healthy", "norm": 0.021069, "variance_change": "+15.3%"},
  "quantization_analysis": {"mixed_precision": "FP16+FP32", "compression": "12.5%"},
  "convergence_analysis": {"status": "converging", "stability": 0.92},
  "activation_analysis": {"relu_usage": "45%", "gelu_usage": "55%"},
  "attention_analysis": {"head_count": 12, "attention_patterns": "stable"},
  "ensemble_analysis": {"ensemble_detected": false, "model_type": "single"}
  // All 11 ML analysis functions included automatically
}
```

### Scientific Data Analysis
```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB File Comparison
```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

## Performance

diffai is optimized for large files and scientific workflows:

- **Memory Efficient**: Streaming processing for GB+ files
- **Fast**: Rust implementation with optimized tensor operations
- **Scalable**: Handles models with millions/billions of parameters
- **Cross-Platform**: Works on Windows, Linux, and macOS without dependencies

## Contributing

We welcome contributions! Please see [CONTRIBUTING](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build
cargo test
```

### Running Tests

```bash
# Run all tests
cargo test

# Run specific test categories
cargo test --test integration
cargo test --test ml_analysis
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- **[diffx](https://github.com/kako-jun/diffx)** - General-purpose structured data diff tool (diffai's sibling project)
- **[safetensors](https://github.com/huggingface/safetensors)** - Simple, safe way to store and distribute tensors
- **[PyTorch](https://pytorch.org/)** - Machine learning framework
- **[NumPy](https://numpy.org/)** - Fundamental package for scientific computing with Python

