# diffai Comprehensive Test Report

**Generated:** Mon Jul 14 02:00:34 PM JST 2025
**Version:** v0.3.4

This report demonstrates that diffai is working correctly across all major features and use cases.

## Test Results Summary

- **Total Tests:** 12
- **Successful:** 11 ✅
- **Failed:** 1 ❌

⚠️ **Some tests failed.** Please review the individual test files.

## Test Categories

### 1. ML Model Analysis
- SafeTensors and PyTorch model comparison
- Statistical analysis
- Architecture comparison
- Memory analysis
- Anomaly detection
- Convergence analysis

### 2. AI/ML Format Support
- PyTorch models (.pt, .pth)
- Safetensors models (.safetensors)
- NumPy arrays (.npy, .npz)
- MATLAB files (.mat)

### 3. CLI Features
- Help system
- Version information
- Advanced options
- Verbose mode
- Directory comparison

## Individual Test Files

### CLI and System Tests
- [advanced_directory_comparison.md](./advanced_directory_comparison.md) ✅
- [basic_verbose_mode.md](./basic_verbose_mode.md) ✅
- [cli_help_output.md](./cli_help_output.md) ✅
- [cli_version_info.md](./cli_version_info.md) ✅

### ML Model Analysis Tests
- [ml_anomaly_detection.md](./ml_anomaly_detection.md) ✅
- [ml_architecture_comparison.md](./ml_architecture_comparison.md) ✅
- [ml_basic_comparison.md](./ml_basic_comparison.md) ✅
- [ml_combined_features.md](./ml_combined_features.md) ✅
- [ml_convergence_analysis.md](./ml_convergence_analysis.md) ✅
- [ml_memory_analysis.md](./ml_memory_analysis.md) ✅
- [ml_stats_analysis.md](./ml_stats_analysis.md) ✅

### Status
- [README.md](./README.md) ❌

## Verification

Each test file contains:
- The exact command used
- Complete output produced
- Exit code
- Success/failure status

This demonstrates that diffai is not just passing unit tests, but actually producing
meaningful output for real-world use cases documented in the README and CLI help.
