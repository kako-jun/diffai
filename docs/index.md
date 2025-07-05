# diffai Documentation

Comprehensive documentation for diffai - AI/ML specialized diff tool

## 📖 Table of Contents

### 🚀 User Guide
- [**Installation**](user-guide/installation.md) - Setup guide for various environments
- [**Basic Usage**](user-guide/basic-usage.md) - Basic commands and operations
- [**ML/AI Workflows**](user-guide/ml-workflows.md) - Integration with ML development
- [**Configuration**](user-guide/configuration.md) - Configuration files and customization

### 🤖 AI/ML Specialized Features
- [**PyTorch Model Comparison**](examples/pytorch-models.md) - Model structure diff analysis
- [**Safetensors Support**](examples/safetensors.md) - Safe tensor format support
- [**Dataset Comparison**](examples/datasets.md) - Dataset format diff analysis
- [**Experiment Management**](examples/experiments.md) - MLflow integration examples

### 🏗️ Architecture
- [**Design Principles**](architecture/design-principles.md) - diffai design philosophy
- [**Core Features**](architecture/core-features.md) - Main functionality details
- [**Extensibility**](architecture/extensibility.md) - Plugin system and customization

### 📚 API Reference
- [**CLI API**](api/cli.md) - Command-line interface
- [**Rust API**](api/rust.md) - Using as Rust library
- [**Configuration Options**](api/config.md) - Complete configuration reference

## 🎯 Quick Start

```bash
# Compare PyTorch models
diffai model1.pth model2.pth

# Compare Safetensors files
diffai model1.safetensors model2.safetensors

# Compare datasets
diffai dataset1.csv dataset2.csv --format csv

# Compare experiment results
diffai experiment1/results experiment2/results --recursive
```

## 🌐 Language Support

- **English**: Current documentation
- **日本語**: [Japanese documentation](index_ja.md)

## 🔗 Related Links

- [GitHub Repository](https://github.com/kako-jun/diffai)
- [crates.io](https://crates.io/crates/diffai)
- [Issues & Support](https://github.com/kako-jun/diffai/issues)
- [Contributing Guide](../CONTRIBUTING.md)