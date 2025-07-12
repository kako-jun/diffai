# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.7] - 2025-01-11
### 🚀 Features
- Phase 3 TDD test suite completion: 16 test functions covering 7 ML analysis features
- 7 new ML analysis functions: architecture comparison, memory analysis, anomaly detection, change summary, convergence analysis, gradient analysis, similarity matrix
- CLI integration and bug fixes: argument order fixes, missing parameter additions
- Comprehensive implementation logic for all 7 new features
- Multi-language documentation updates: complete consistency across English, Japanese, and Chinese
- Documentation example testing: 20 test cases ensuring all usage examples work
- Complete diffx improvements sync: GitHub Actions modernization, act1/act2 release workflow
- CI optimization: Added --release flag to prevent timeouts in development

### 🔧 Improvements
- Code quality improvements: Clippy warning fixes, format compliance
- Total ML analysis functions: 31 features implemented
- TDD + documentation example test coverage
- Cross-platform pure Rust implementation without system dependencies
- Package naming aligned with diffx conventions: diffai-js, diffai-python
- Removed legacy API implementations following diffx modernization

## [0.2.6] - 2025-01-09
### 🚀 Features - Phase 2: Scientific Data Support Complete
- **NumPy Array Support**: Full support for .npy single files and .npz archives
- **MATLAB Array Support**: Complete .mat file support with matfile crate integration
- **Complex Number Support**: Full support for complex numbers in both NumPy and MATLAB
- **Statistical Analysis**: Comprehensive statistical analysis for scientific data
- **Colored CLI Output**: Visual representation of scientific data changes

### 🔧 Improvements
- **Header Analysis**: Detailed header parsing for NumPy and MATLAB files
- **All Numeric Types**: Support for all numeric data types in scientific formats
- **Scientific Computing Integration**: ML + scientific data comprehensive diff analysis system

## [0.2.5] - 2025-01-08
### 🚀 Features - Phase 1-2 Foundation
- **PyTorch Multi-dimensional Tensor Fix**: Fixed rank error with flatten_all() function
- **External CLI Dependency Removal**: Direct diffx-core integration for complete independence
- **Basic ML Analysis Functions**: Quantization analysis, statistical display, layer impact analysis
- **Self-contained Operation**: Eliminated legacy code dependencies

### 🔧 Improvements
- **Pure Rust Implementation**: Complete system independence
- **Cross-platform Support**: Stable operation across all environments

## [0.2.4] - 2025-01-08
### 🚀 Features
- **Safetensors Alignment Fix**: Manual byte conversion for HuggingFace compatibility
- **Real Model Validation**: Comprehensive testing with actual models
- **Binary Analysis Enhancement**: Improved safetensors binary parsing

## [0.2.3] - 2025-01-07
### 🚀 Features
- **PyTorch Integration**: Complete Candle framework integration
- **Real Model Testing Environment**: HuggingFace model download and validation system
- **Multi-dimensional Tensor Support**: Full support for complex tensor structures

## [0.2.2] - 2025-01-07
### 🚀 Features
- **Safetensors Complete Support**: Binary analysis, statistical computation, basic ML analysis
- **Statistical Analysis**: Mean, standard deviation, min/max, shape, memory size calculations
- **Colored CLI Output**: Visual distinction by change type, risk level, and data type

## [0.2.1] - 2025-01-06
### 🚀 Features
- **diffx-core Integration**: Direct integration eliminating external CLI dependencies
- **Legacy Code Cleanup**: Removed outdated diffx expansion analysis documents
- **Self-contained Architecture**: Complete independence from external tools

## [0.2.0] - 2025-01-06
### 🚀 Features - Phase 1: ML Model Support Complete
- **crates.io Publication**: https://crates.io/crates/diffai
- **Safetensors Format**: Complete binary analysis and statistical computation
- **PyTorch Format**: Full support with Candle integration
- **Basic ML Analysis**: 4 core functions implemented
  - `--stats`: Detailed tensor statistics
  - `--quantization-analysis`: Quantization analysis
  - `--sort-by-change-magnitude`: Change magnitude sorting
  - `--show-layer-impact`: Layer impact analysis
- **JSON/YAML Output**: MLOps tool integration and API compatibility
- **Pure Rust Implementation**: System-independent, cross-platform operation

### 🏗️ Architecture
- **Format Support**: ML (PyTorch/Safetensors) + structured data (JSON/YAML/TOML/XML/INI/CSV)
- **Statistical Engine**: Comprehensive tensor analysis with memory efficiency
- **Cross-platform**: Windows, Linux, macOS support without additional dependencies

## [0.1.0] - 2025-01-05
### 🚀 Features - Initial Release
- **Project Setup**: Initial diffai project structure
- **Basic Architecture**: Core diff engine implementation
- **Structured Data Support**: JSON, YAML, TOML, XML, INI, CSV formats
- **CLI Interface**: Basic command-line interface
- **CI/CD Pipeline**: GitHub Actions integration for testing and building
