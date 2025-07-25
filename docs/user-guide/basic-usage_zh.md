# Basic Usage

Learn the fundamental operations of diffai - AI/ML specialized diff tool.

## Quick Start

### Basic File Comparison

```bash
# Compare two model files (comprehensive analysis automatic)
diffai model1.safetensors model2.safetensors

# Output in JSON format
diffai model1.safetensors model2.safetensors --output json

# Output in YAML format  
diffai model1.safetensors model2.safetensors --output yaml
```

### Directory Comparison

```bash
# Compare entire directories with automatic detection
diffai dir1/ dir2/

# Compare with specific file format
diffai models_v1/ models_v2/ --format safetensors
```

**注意**: 当路径以 `/` 结尾时，会自动检测目录比较。该工具智能分析目录中的所有ML/AI文件，并提供全面的比较报告。

## AI/ML Specialized Features

### PyTorch Model Comparison

```bash
# Compare PyTorch model files (full analysis automatic)
diffai model1.pt model2.pt

# Compare training checkpoints  
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt

# Compare baseline vs improved model
diffai baseline_model.pt improved_model.pt
```

**Example Output (Full Analysis):**
```
anomaldy_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.25, speedup=1.8x, precision_loss=minimal
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### Safetensors File Comparison

```bash
# Compare Safetensors files (comprehensive analysis automatic)
diffai model1.safetensors model2.safetensors

# For production deployment validation
diffai baseline.safetensors candidate.safetensors
```

### Scientific Data Comparison

```bash
# Compare NumPy arrays (automatic statistics)
diffai data_v1.npy data_v2.npy

# Compare MATLAB files (automatic statistics)
diffai simulation_v1.mat simulation_v2.mat

# Compare compressed NumPy archives (automatic statistics)
diffai dataset_v1.npz dataset_v2.npz
```

## Command Options

### Basic Options

| Option | Description | Example |
|--------|-------------|---------|
| `-f, --format` | Specify input file format | `--format safetensors` |
| `-o, --output` | Choose output format | `--output json` |
| `-v, --verbose` | Show verbose processing info | `--verbose` |

### Advanced Options

| Option | Description | Example |
|--------|-------------|---------|
| `--path` | Filter by specific path | `--path "config.model"` |
| `--ignore-keys-regex` | Ignore keys matching regex | `--ignore-keys-regex "^id$"` |
| `--epsilon` | Float comparison tolerance | `--epsilon 0.001` |
| `--array-id-key` | Array element identification | `--array-id-key "id"` |

## Output Formats

### CLI Output (Default - Full Analysis)

Human-readable colored output with comprehensive analysis:

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaldy_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### JSON Output

```bash
diffai model1.safetensors model2.safetensors --output json
```

```json
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518, "shape": [64], "dtype": "f32"},
      {"mean": 0.0017, "std": 0.0647, "shape": [64], "dtype": "f32"}
    ]
  }
]
```

### YAML Output

```bash
diffai model1.safetensors model2.safetensors --output yaml
```

```yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
    shape: [64]
    dtype: f32
  - mean: 0.0017
    std: 0.0647
    shape: [64]
    dtype: f32
```


## Practical Examples

### Experiment Comparison

```bash
# 通过自动ML分析比较两个实验结果
diffai experiment_v1/ experiment_v2/

# Compare model checkpoints (automatic learning analysis)
diffai checkpoints/epoch_10.safetensors checkpoints/epoch_20.safetensors
```

### CI/CD Usage

```yaml
- name: Compare models
  run: |
    diffai baseline/model.safetensors new/model.safetensors --output json > model_diff.json
    
- name: Check deployment readiness (included in analysis)
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors
```

### Scientific Data Analysis

```bash
# Compare NumPy experiment results (automatic statistics)
diffai baseline_results.npy new_results.npy

# Compare MATLAB simulation data
diffai simulation_v1.mat simulation_v2.mat
```

## Supported File Formats

### ML Model Formats
- **Safetensors** (.safetensors) - HuggingFace standard format
- **PyTorch** (.pt, .pth) - PyTorch model files

### Scientific Data Formats
- **NumPy** (.npy, .npz) - NumPy arrays with statistical analysis
- **MATLAB** (.mat) - MATLAB matrices with complex number support

### Structured Data Formats
- **JSON** (.json), **YAML** (.yaml, .yml), **TOML** (.toml)
- **XML** (.xml), **INI** (.ini), **CSV** (.csv)

## Next Steps

- [ML Model Comparison](ml-model-comparison_zh.md) - Advanced ML model analysis
- [Scientific Data Analysis](scientific-data_zh.md) - NumPy and MATLAB file comparison
- [CLI参考](../reference/cli-reference_zh.md) - Complete command reference

