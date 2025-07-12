# diffai - AI/ML Specialized Diff Tool (Python Package)

[![PyPI version](https://badge.fury.io/py/diffai-python.svg)](https://badge.fury.io/py/diffai-python)
[![Downloads](https://img.shields.io/pypi/dm/diffai-python.svg)](https://pypi.org/project/diffai-python/)
[![Python Versions](https://img.shields.io/pypi/pyversions/diffai-python.svg)](https://pypi.org/project/diffai-python/)

AI/ML specialized data diff tool for deep tensor comparison and analysis. This Python package provides a convenient and type-safe interface to diffai through Python.

## 🚀 Quick Start

### Installation

```bash
# Install via pip
pip install diffai-python

# Development installation
pip install diffai-python[dev]
```

### Basic Usage

```python
import diffai

# Simple model comparison
result = diffai.diff("model_v1.safetensors", "model_v2.safetensors", stats=True)
print(result)

# Advanced ML analysis with type-safe configuration
options = diffai.DiffOptions(
    stats=True,
    architecture_comparison=True,
    memory_analysis=True,
    output_format=diffai.OutputFormat.JSON
)

result = diffai.diff("baseline.safetensors", "improved.safetensors", options)
if result.is_json:
    for change in result.changes:
        print(f"Changed: {change}")
```

### Command Line Usage

```bash
# The package also installs the diffai binary
diffai model1.safetensors model2.safetensors --stats

# Download binary manually if needed
diffai-download-binary
```

## 📦 Supported File Formats

### AI/ML Formats (Specialized Analysis)
- **Safetensors** (.safetensors) - PyTorch model format with ML analysis
- **PyTorch** (.pt, .pth) - Native PyTorch models with tensor statistics
- **NumPy** (.npy, .npz) - Scientific computing arrays with statistical analysis
- **MATLAB** (.mat) - Engineering/scientific data with numerical analysis

### Structured Data Formats (Universal)
- **JSON** (.json) - API configurations, model metadata
- **YAML** (.yaml, .yml) - Configuration files, CI/CD pipelines
- **TOML** (.toml) - Rust configs, Python pyproject.toml
- **XML** (.xml) - Legacy configurations, model definitions
- **CSV** (.csv) - Datasets, experiment results
- **INI** (.ini) - Legacy configuration files

## 🔬 35 ML Analysis Functions

### Core Analysis Functions
```python
# Statistical analysis
result = diffai.diff("model1.safetensors", "model2.safetensors", stats=True)

# Quantization analysis
result = diffai.diff("fp32.safetensors", "quantized.safetensors", 
                    quantization_analysis=True)

# Change magnitude sorting
result = diffai.diff("model1.safetensors", "model2.safetensors", 
                    sort_by_change_magnitude=True, stats=True)
```

### Phase 3 Advanced Analysis (v0.2.7+)
```python
# Architecture comparison
result = diffai.diff("model1.safetensors", "model2.safetensors", 
                    architecture_comparison=True)

# Memory analysis for deployment
result = diffai.diff("model1.safetensors", "model2.safetensors", 
                    memory_analysis=True)

# Anomaly detection for debugging
result = diffai.diff("stable.safetensors", "problematic.safetensors", 
                    anomaly_detection=True)

# Comprehensive analysis
options = diffai.DiffOptions(
    stats=True,
    architecture_comparison=True,
    memory_analysis=True,
    anomaly_detection=True,
    convergence_analysis=True,
    gradient_analysis=True,
    similarity_matrix=True,
    change_summary=True
)
result = diffai.diff("baseline.safetensors", "improved.safetensors", options)
```

## 💡 Python API Examples

### Type-Safe Configuration
```python
from diffai import DiffOptions, OutputFormat

# Create type-safe configuration
options = DiffOptions(
    stats=True,
    architecture_comparison=True,
    memory_analysis=True,
    output_format=OutputFormat.JSON
)

# Compare models
result = diffai.diff("model1.safetensors", "model2.safetensors", options)

# Access structured results
if result.is_json:
    print(f"Found {len(result.changes)} changes")
    for change in result.changes:
        print(f"  {change.get('path')}: {change.get('type')}")
```

### Scientific Data Analysis
```python
# NumPy array comparison
result = diffai.diff("experiment_v1.npy", "experiment_v2.npy", stats=True)
print(f"Statistical changes: {result}")

# MATLAB data comparison
result = diffai.diff("simulation_v1.mat", "simulation_v2.mat", 
                    stats=True, sort_by_change_magnitude=True)
```

### JSON Output for Automation
```python
# Get JSON results for MLOps integration
result = diffai.diff("model1.safetensors", "model2.safetensors", 
                    stats=True, output_format=diffai.OutputFormat.JSON)

if result.is_json:
    # Process structured data
    changes = result.changes
    summary = result.summary
    
    # Integration with MLflow, Weights & Biases, etc.
    log_model_comparison(changes, summary)
```

### Error Handling
```python
try:
    result = diffai.diff("model1.safetensors", "model2.safetensors", stats=True)
    print(result)
except diffai.BinaryNotFoundError:
    print("diffai binary not found. Please install: pip install diffai-python")
except diffai.InvalidInputError as e:
    print(f"Invalid input: {e}")
except diffai.DiffaiError as e:
    print(f"diffai error: {e}")
```

### String Comparison (Temporary Files)
```python
# Compare JSON strings directly
json1 = '{"model": "gpt-2", "layers": 12}'
json2 = '{"model": "gpt-2", "layers": 24}'

result = diffai.diff_string(json1, json2, output_format=diffai.OutputFormat.JSON)
print(result)
```

## 🔧 Advanced Usage

### Installation Verification
```python
# Check if diffai is properly installed
try:
    info = diffai.verify_installation()
    print(f"diffai version: {info['version']}")
    print(f"Binary path: {info['binary_path']}")
except diffai.BinaryNotFoundError as e:
    print(f"Installation issue: {e}")
```

### Manual Binary Management
```python
# Download binary programmatically
from diffai.installer import install_binary

success = install_binary(force=True)  # Force reinstall
if success:
    print("Binary installed successfully")
```

### Low-Level API Access
```python
# Direct command execution
result = diffai.run_diffai([
    "model1.safetensors", 
    "model2.safetensors", 
    "--stats", 
    "--architecture-comparison",
    "--output", "json"
])

print(f"Exit code: {result.exit_code}")
print(f"Output: {result.raw_output}")
```

## 🔗 Integration Examples

### MLflow Integration
```python
import mlflow
import diffai

def log_model_comparison(model1_path, model2_path, run_id=None):
    with mlflow.start_run(run_id=run_id):
        # Compare models with comprehensive analysis
        result = diffai.diff(
            model1_path, model2_path,
            stats=True,
            architecture_comparison=True,
            memory_analysis=True,
            output_format=diffai.OutputFormat.JSON
        )
        
        if result.is_json:
            # Log structured comparison data
            mlflow.log_dict(result.data, "model_comparison.json")
            
            # Log metrics
            if result.changes:
                mlflow.log_metric("total_changes", len(result.changes))
                mlflow.log_metric("significant_changes", 
                                sum(1 for c in result.changes 
                                    if c.get('magnitude', 0) > 0.1))

# Usage
log_model_comparison("baseline.safetensors", "candidate.safetensors")
```

### Weights & Biases Integration
```python
import wandb
import diffai

def wandb_log_model_diff(model1, model2, **kwargs):
    result = diffai.diff(model1, model2, 
                        stats=True, 
                        output_format=diffai.OutputFormat.JSON,
                        **kwargs)
    
    if result.is_json and result.changes:
        # Log to wandb
        wandb.log({
            "model_comparison": wandb.Table(
                columns=["parameter", "change_type", "magnitude"],
                data=[[c.get("path"), c.get("type"), c.get("magnitude")] 
                      for c in result.changes[:100]]  # Limit rows
            )
        })

# Initialize wandb run
wandb.init(project="model-comparison")
wandb_log_model_diff("model_v1.safetensors", "model_v2.safetensors")
```

### Flask API Endpoint
```python
from flask import Flask, request, jsonify
import diffai

app = Flask(__name__)

@app.route('/compare', methods=['POST'])
def compare_models():
    try:
        files = request.files
        model1 = files['model1']
        model2 = files['model2']
        
        # Save temporary files
        model1.save('/tmp/model1.safetensors')
        model2.save('/tmp/model2.safetensors')
        
        # Compare models
        result = diffai.diff('/tmp/model1.safetensors', '/tmp/model2.safetensors',
                           stats=True, 
                           architecture_comparison=True,
                           output_format=diffai.OutputFormat.JSON)
        
        return jsonify({
            "status": "success",
            "comparison": result.data if result.is_json else result.raw_output
        })
        
    except diffai.DiffaiError as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

## 🏗️ Platform Support

This package automatically downloads platform-specific binaries:

- **Linux** (x86_64, ARM64)
- **macOS** (Intel x86_64, Apple Silicon ARM64)
- **Windows** (x86_64)

The binary is downloaded during installation and cached. If download fails, the package falls back to system PATH.

## 🔗 Related Projects

- **[diffx-python](https://pypi.org/project/diffx-python/)** - General-purpose structured data diff tool
- **[diffai (npm)](https://www.npmjs.com/package/diffai)** - Node.js package for diffai
- **[diffai (GitHub)](https://github.com/diffai-team/diffai)** - Main repository

## 📚 Documentation

- [CLI Reference](https://github.com/diffai-team/diffai/blob/main/docs/reference/cli-reference.md)
- [ML Analysis Guide](https://github.com/diffai-team/diffai/blob/main/docs/reference/ml-analysis.md)
- [User Guide](https://github.com/diffai-team/diffai/blob/main/docs/user-guide/)
- [API Documentation](https://github.com/diffai-team/diffai/blob/main/docs/reference/api-reference.md)

## 📄 License

MIT License - see [LICENSE](https://github.com/diffai-team/diffai/blob/main/LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/diffai-team/diffai/blob/main/CONTRIBUTING.md) for guidelines.

---

**diffai** - Making AI/ML data differences visible, measurable, and actionable through Python. 🐍🚀