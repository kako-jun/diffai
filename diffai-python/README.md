# diffai-python

AI/ML specialized diff tool for deep tensor comparison and analysis - Python Package

[![PyPI version](https://badge.fury.io/py/diffai-python.svg)](https://badge.fury.io/py/diffai-python)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**diffai-python** provides Python bindings for [diffai](https://github.com/kako-jun/diffai), an AI/ML specialized diff tool. This package bundles the high-performance Rust binary and provides a clean Python API for integration into ML workflows, notebooks, and automation scripts.

Following the same distribution pattern as [ruff](https://github.com/astral-sh/ruff), this package distributes a pre-compiled binary for maximum performance while providing a convenient Python interface.

## Features

- **High Performance**: Uses the native diffai Rust binary for maximum speed
- **Zero Dependencies**: Self-contained package with bundled binary
- **ML-Focused**: Specialized analysis for PyTorch, Safetensors, NumPy, and MATLAB files
- **Multiple Output Formats**: CLI, JSON, and YAML outputs for different use cases
- **Python Integration**: Clean API for programmatic use in ML pipelines

## Installation

```bash
pip install diffai-python
```

## Quick Start

### Command Line Usage

After installation, the `diffai` command is available:

```bash
# Compare ML models
diffai model_v1.safetensors model_v2.safetensors --stats

# Compare NumPy arrays
diffai data_v1.npy data_v2.npy --stats

# JSON output for automation
diffai model_v1.pt model_v2.pt --output json
```

### Python API Usage

```python
import diffai

# Basic comparison
result = diffai.diff("model_v1.safetensors", "model_v2.safetensors")
print(result.raw_output)

# With options
options = diffai.DiffOptions(
    stats=True,
    architecture_comparison=True,
    output_format=diffai.OutputFormat.JSON
)
result = diffai.diff("model_v1.pt", "model_v2.pt", options)

# Access structured data
if result.is_json:
    data = result.data
    print(f"Found {len(data)} differences")
```

### Advanced ML Analysis

```python
# Comprehensive ML model analysis
result = diffai.diff(
    "baseline.safetensors", 
    "improved.safetensors",
    stats=True,
    architecture_comparison=True,
    memory_analysis=True,
    anomaly_detection=True,
    convergence_analysis=True
)

print(result.raw_output)
```

## Supported Formats

### Input Formats
- **ML Models**: `.safetensors`, `.pt`, `.pth`, `.bin` (PyTorch)
- **Scientific Data**: `.npy`, `.npz` (NumPy), `.mat` (MATLAB)  
- **Structured Data**: `.json`, `.yaml`, `.toml`, `.xml`, `.ini`, `.csv`

### Output Formats
- **CLI**: Colored terminal output (default)
- **JSON**: Machine-readable format for automation
- **YAML**: Human-readable structured format

## ML Analysis Features

The package provides 11 specialized ML analysis features:

- `--stats`: Detailed tensor statistics
- `--architecture-comparison`: Model structure comparison
- `--memory-analysis`: Memory usage analysis  
- `--anomaly-detection`: Numerical anomaly detection
- `--convergence-analysis`: Training convergence analysis
- `--gradient-analysis`: Gradient information analysis
- `--similarity-matrix`: Layer similarity comparison
- `--change-summary`: Detailed change summary
- `--quantization-analysis`: Quantization impact analysis
- `--sort-by-change-magnitude`: Sort by change magnitude
- `--show-layer-impact`: Layer-specific impact analysis

## API Reference

### Main Functions

```python
# Compare two files
def diff(input1: str, input2: str, options: Optional[DiffOptions] = None, **kwargs) -> DiffResult

# Main CLI entry point
def main() -> None
```

### Configuration

```python
@dataclass
class DiffOptions:
    # Basic options
    input_format: Optional[str] = None
    output_format: Optional[OutputFormat] = None
    recursive: bool = False
    verbose: bool = False
    
    # ML analysis options  
    stats: bool = False
    architecture_comparison: bool = False
    memory_analysis: bool = False
    anomaly_detection: bool = False
    # ... and more
```

### Results

```python
class DiffResult:
    raw_output: str           # Raw output from diffai
    format_type: str          # Output format used
    return_code: int          # Process return code
    
    @property
    def data(self) -> Any     # Parsed data (JSON when applicable)
    
    @property  
    def is_json(self) -> bool # True if JSON format
```

## Use Cases

### Research & Development
```python
# Compare fine-tuning results
before = "model_baseline.safetensors"
after = "model_finetuned.safetensors"

result = diffai.diff(before, after, 
                    stats=True, 
                    convergence_analysis=True)
```

### MLOps Integration
```python
# Automated model validation in CI/CD
def validate_model_changes(old_model, new_model):
    result = diffai.diff(old_model, new_model,
                        output_format=diffai.OutputFormat.JSON,
                        anomaly_detection=True,
                        architecture_comparison=True)
    
    if result.return_code != 0:
        raise ValueError("Model validation failed")
    
    return result.data
```

### Jupyter Notebooks
```python
# Interactive analysis in notebooks
result = diffai.diff("checkpoint_100.pt", "checkpoint_200.pt", 
                    stats=True, memory_analysis=True)

# Display results
if result.is_json:
    from IPython.display import display, JSON
    display(JSON(result.data))
else:
    print(result.raw_output)
```

## Binary Distribution

This package follows the same pattern as [ruff](https://github.com/astral-sh/ruff):

- Pre-compiled `diffai` binary is bundled with the Python package
- No external dependencies or system requirements
- Cross-platform compatibility (Windows, macOS, Linux)
- Maximum performance through native Rust implementation

## Testing

Run the integration tests:

```bash
cd diffai-python
python test_integration.py
```

The test suite includes:
- Binary availability verification
- Basic diff functionality
- JSON output parsing
- ML analysis options
- Error handling

## Contributing

This package is part of the [diffai](https://github.com/kako-jun/diffai) project. Please see the main repository for contribution guidelines.

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Related Projects

- **[diffai](https://github.com/kako-jun/diffai)**: Main Rust CLI tool
- **[diffx](https://github.com/kako-jun/diffx)**: Generic structured data diff tool
- **[ruff](https://github.com/astral-sh/ruff)**: Inspiration for Python packaging approach