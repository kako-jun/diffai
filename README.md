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

# diffai shows meaningful model changes
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  gradient_analysis: flow_health=healthy, norm=0.015000, ratio=1.0500
```

## Key Features

- **AI/ML Native**: Direct support for PyTorch (.pt/.pth), Safetensors (.safetensors), NumPy (.npy/.npz), and MATLAB (.mat) files
- **Tensor Analysis**: Automatic calculation of tensor statistics (mean, std, min, max, shape, memory usage)
- **ML Analysis Functions**: Statistical analysis, quantization analysis, architecture comparison, and more
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
# Compare PyTorch models
diffai model_old.pt model_new.pt --stats

# Compare Safetensors with statistical analysis
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors --stats

# Compare NumPy arrays
diffai data_v1.npy data_v2.npy --stats

# Compare MATLAB files
diffai experiment_v1.mat experiment_v2.mat --stats
```

### Advanced ML Analysis

```bash
# Current available analysis
diffai baseline.safetensors finetuned.safetensors --stats --quantization-analysis

# Combined analysis with sorting
diffai original.pt optimized.pt --stats --quantization-analysis --sort-by-change-magnitude

# JSON output for automation
diffai model_v1.safetensors model_v2.safetensors --stats --output json

# Detailed diagnostic information with verbose mode
diffai model_v1.safetensors model_v2.safetensors --verbose --stats --architecture-comparison

# Future Phase 3 features (coming soon)
diffai model_v1.safetensors model_v2.safetensors --architecture-comparison --memory-analysis
```

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

### Currently Available (v0.2.7)
- `--stats` - Detailed tensor statistics (mean, std, min, max, shape, memory)
- `--quantization-analysis` - Analyze quantization effects and efficiency
- `--sort-by-change-magnitude` - Sort differences by magnitude for prioritization
- `--show-layer-impact` - Layer-by-layer impact analysis
- `--architecture-comparison` - Compare model architectures and structural changes
- `--memory-analysis` - Analyze memory usage and optimization opportunities
- `--anomaly-detection` - Detect numerical anomalies in model parameters
- `--change-summary` - Generate detailed change summaries
- `--convergence-analysis` - Analyze convergence patterns in model parameters
- `--gradient-analysis` - Analyze gradient information when available
- `--similarity-matrix` - Generate similarity matrix for model comparison

### Coming in Phase 4 (ML Framework Expansion)
- TensorFlow format support (.pb, .h5, SavedModel)
- ONNX format support
- Advanced visualization and charting features

### Design Philosophy
diffai follows UNIX philosophy: simple, composable tools that do one thing well. Features are orthogonal and can be combined for powerful analysis workflows.

## Debugging and Diagnostics

### Verbose Mode (`--verbose` / `-v`)
Get comprehensive diagnostic information for debugging and performance analysis:

```bash
# Basic verbose output
diffai model1.safetensors model2.safetensors --verbose

# Verbose with ML analysis features
diffai data1.json data2.json --verbose --stats --epsilon 0.001 --ignore-keys-regex "^id$"
```

**Verbose output includes:**
- **Configuration diagnostics**: Active ML features, format settings, filters
- **File analysis**: Paths, sizes, detected formats, processing context
- **Performance metrics**: Processing time, difference counts, optimization status
- **Directory statistics**: File counts, comparison summaries (with `--recursive`)

**Example verbose output:**
```
=== diffai verbose mode enabled ===
Configuration:
  Input format: None
  Output format: Cli
  ML analysis features: statistics, architecture_comparison
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
# Compare model before and after fine-tuning
diffai pretrained_model.safetensors finetuned_model.safetensors \
  --learning-progress --convergence-analysis --stats

# Analyze architectural changes during development
diffai baseline_architecture.pt improved_architecture.pt \
  --architecture-comparison --param-efficiency-analysis
```

### MLOps & CI/CD
```bash
# Automated model validation in CI/CD
diffai production_model.safetensors candidate_model.safetensors \
  --deployment-readiness --regression-test --risk-assessment

# Performance impact assessment
diffai original_model.pt optimized_model.pt \
  --quantization-analysis --memory-analysis --performance-impact-estimate
```

### Scientific Computing
```bash
# Compare NumPy experiment results
diffai baseline_results.npy new_results.npy --stats

# Analyze MATLAB simulation data
diffai simulation_v1.mat simulation_v2.mat --stats

# Compare compressed NumPy archives
diffai dataset_v1.npz dataset_v2.npz --stats
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
- `--stats` - Show detailed statistics for ML models

### Advanced Options
- `--path <PATH>` - Filter differences by specific path
- `--ignore-keys-regex <REGEX>` - Ignore keys matching regex pattern
- `--epsilon <FLOAT>` - Set tolerance for float comparisons
- `--array-id-key <KEY>` - Specify key for array element identification
- `--sort-by-change-magnitude` - Sort by change magnitude

## Examples

### Basic Tensor Comparison
```bash
$ diffai simple_model_v1.safetensors simple_model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### Advanced Analysis
```bash
$ diffai baseline.safetensors improved.safetensors --deployment-readiness --architecture-comparison
deployment_readiness: readiness=0.92, strategy=blue_green, risk=low, timeline=ready_for_immediate_deployment
architecture_comparison: type1=feedforward, type2=feedforward, depth=3->3, differences=0
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

### Scientific Data Analysis
```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy --stats
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB File Comparison
```bash
$ diffai simulation_v1.mat simulation_v2.mat --stats
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

