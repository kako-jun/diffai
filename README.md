# diffai

> **AI/ML specialized diff tool for PyTorch, Safetensors, NumPy, and MATLAB files**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/diffai.svg)](https://crates.io/crates/diffai)
[![Documentation](https://img.shields.io/badge/docs-GitHub-blue)](https://github.com/kako-jun/diffai/tree/main/docs/index.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A next-generation diff tool specialized for **AI/ML and scientific computing workflows** that understands model structures, tensor statistics, and numerical data - not just text changes. Native support for PyTorch, Safetensors, NumPy arrays, MATLAB files, and structured data.

```bash
# Traditional diff fails with binary model files
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffai shows meaningful model changes with full analysis
$ diffai model_v1.safetensors model_v2.safetensors
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  gradient_analysis: flow_health=healthy, norm=0.015000, ratio=1.0500
  deployment_readiness: readiness=0.92, strategy=blue_green, risk=low
  quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%

[WARNING]
• Memory usage increased moderately (+250MB). Monitor resource consumption.
• Inference speed moderately affected (1.3x slower). Consider optimization opportunities.
```

## Key Features

- **AI/ML Native**: Direct support for PyTorch (.pt/.pth), Safetensors (.safetensors), NumPy (.npy/.npz), and MATLAB (.mat) files
- **Tensor Analysis**: Automatic calculation of tensor statistics (mean, std, min, max, shape, memory usage)
- **Comprehensive ML Analysis**: 30+ analysis functions including quantization, architecture, memory, convergence, anomaly detection, and deployment readiness - all enabled by default
- **Scientific Data Support**: NumPy arrays and MATLAB matrices with complex number support
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
# Compare PyTorch models with full analysis (default)
diffai model_old.pt model_new.pt

# Compare Safetensors with complete ML analysis
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# Compare NumPy arrays
diffai data_v1.npy data_v2.npy

# Compare MATLAB files
diffai experiment_v1.mat experiment_v2.mat
```

### ML Analysis Features

```bash
# Full ML analysis runs automatically for PyTorch/Safetensors
diffai baseline.safetensors finetuned.safetensors
# Outputs: 30+ analysis types including quantization, architecture, memory, etc.

# JSON output for automation
diffai model_v1.safetensors model_v2.safetensors --output json

# Detailed diagnostic information with verbose mode
diffai model_v1.safetensors model_v2.safetensors --verbose

# YAML output for human-readable reports
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 📚 Documentation

- **[Working Examples & Demonstrations](docs/examples/)** - See diffai in action with real outputs
- **[API Documentation](https://docs.rs/diffai-core)** - Rust library documentation
- **[User Guide](docs/user-guide.md)** - Comprehensive usage guide
- **[ML Analysis Guide](docs/ml-analysis-guide.md)** - Deep dive into ML-specific features

## Supported File Formats

### ML Model Formats
- **Safetensors** (.safetensors) - HuggingFace standard format
- **PyTorch** (.pt, .pth) - PyTorch model files with Candle integration

### Scientific Data Formats  
- **NumPy** (.npy, .npz) - NumPy arrays with full statistical analysis
- **MATLAB** (.mat) - MATLAB matrices with complex number support

### Structured Data Formats
- **JSON** (.json) - JavaScript Object Notation
- **YAML** (.yaml, .yml) - YAML Ain't Markup Language
- **TOML** (.toml) - Tom's Obvious Minimal Language  
- **XML** (.xml) - Extensible Markup Language
- **INI** (.ini) - Configuration files
- **CSV** (.csv) - Comma-separated values

## ML Analysis Functions

### Automatic Comprehensive Analysis (v0.3.4)
When comparing PyTorch or Safetensors files, diffai automatically runs 30+ ML analysis features:

**Automatic Features Include:**
- **Statistical Analysis**: Detailed tensor statistics (mean, std, min, max, shape, memory)
- **Quantization Analysis**: Analyze quantization effects and efficiency
- **Architecture Comparison**: Compare model architectures and structural changes
- **Memory Analysis**: Analyze memory usage and optimization opportunities
- **Anomaly Detection**: Detect numerical anomalies in model parameters
- **Convergence Analysis**: Analyze convergence patterns in model parameters
- **Gradient Analysis**: Analyze gradient information when available
- **Deployment Readiness**: Assess production deployment readiness
- **Regression Testing**: Automatic performance degradation detection
- **Plus 20+ additional specialized features**

### Future Enhancements
- TensorFlow format support (.pb, .h5, SavedModel)
- ONNX format support
- Advanced visualization and charting features

### Design Philosophy
diffai provides comprehensive analysis by default for ML models, eliminating choice paralysis. Users get all relevant insights without needing to remember or specify dozens of analysis flags.

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
- **Directory statistics**: File counts, comparison summaries (with `--recursive`)

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

📚 **See [Verbose Output Guide](docs/user-guide/verbose-output.md) for detailed usage**

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
- `-r, --recursive` - Compare directories recursively

**Note:** For ML models (PyTorch/Safetensors), comprehensive analysis including statistics runs automatically

### Advanced Options
- `--path <PATH>` - Filter differences by specific path
- `--ignore-keys-regex <REGEX>` - Ignore keys matching regex pattern
- `--epsilon <FLOAT>` - Set tolerance for float comparisons
- `--array-id-key <KEY>` - Specify key for array element identification
- `--sort-by-change-magnitude` - Sort by change magnitude

## Examples

### Basic Tensor Comparison (Automatic)
```bash
$ diffai simple_model_v1.safetensors simple_model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
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
  "anomaly_detection": {"type": "none", "severity": "none"},
  "architecture_comparison": {"type1": "feedforward", "type2": "feedforward"},
  "deployment_readiness": {"readiness": 0.92, "strategy": "blue_green"},
  "quantization_analysis": {"compression": "0.0%", "speedup": "1.8x"},
  "regression_test": {"passed": true, "degradation": "-2.5%"}
  // ... plus 25+ additional analysis features
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

