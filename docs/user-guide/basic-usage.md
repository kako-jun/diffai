# Basic Usage

Learn the fundamental operations of diffai - AI/ML specialized diff tool.

## Quick Start

### Basic File Comparison

```bash
# Compare two model files
diffai model1.safetensors model2.safetensors

# Show detailed tensor statistics
diffai model1.safetensors model2.safetensors --stats

# Output in JSON format
diffai model1.safetensors model2.safetensors --output json
```

### Directory Comparison

```bash
# Compare entire directories recursively
diffai dir1/ dir2/ --recursive

# Compare with specific file format
diffai models_v1/ models_v2/ --format safetensors --recursive
```

## AI/ML Specialized Features

### PyTorch Model Comparison

```bash
# Compare PyTorch model files
diffai model1.pt model2.pt --stats

# With learning progress analysis
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt --learning-progress

# With architecture comparison
diffai baseline_model.pt improved_model.pt --architecture-comparison
```

**Example Output:**
```
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  learning_progress: trend=improving, magnitude=0.0234, speed=0.0156
```

### Safetensors File Comparison

```bash
# Compare Safetensors files with statistics
diffai model1.safetensors model2.safetensors --stats

# With deployment readiness analysis
diffai baseline.safetensors candidate.safetensors --deployment-readiness
```

### Scientific Data Comparison

```bash
# Compare NumPy arrays
diffai data_v1.npy data_v2.npy --stats

# Compare MATLAB files
diffai simulation_v1.mat simulation_v2.mat --stats

# Compare compressed NumPy archives
diffai dataset_v1.npz dataset_v2.npz --stats
```

## Command Options

### Basic Options

| Option | Description | Example |
|--------|-------------|---------|
| `-f, --format` | Specify input file format | `--format safetensors` |
| `-o, --output` | Choose output format | `--output json` |
| `-r, --recursive` | Compare directories recursively | `--recursive` |
| `--stats` | Show detailed statistics | `--stats` |

### Advanced Options

| Option | Description | Example |
|--------|-------------|---------|
| `--path` | Filter by specific path | `--path "config.model"` |
| `--ignore-keys-regex` | Ignore keys matching regex | `--ignore-keys-regex "^id$"` |
| `--epsilon` | Float comparison tolerance | `--epsilon 0.001` |
| `--array-id-key` | Array element identification | `--array-id-key "id"` |
| `--sort-by-change-magnitude` | Sort by change magnitude | `--sort-by-change-magnitude` |

### ML Analysis Options

| Option | Description | Example |
|--------|-------------|---------|
| `--learning-progress` | Track learning progress | `--learning-progress` |
| `--convergence-analysis` | Analyze convergence | `--convergence-analysis` |
| `--architecture-comparison` | Compare architectures | `--architecture-comparison` |
| `--deployment-readiness` | Assess deployment readiness | `--deployment-readiness` |
| `--quantization-analysis` | Analyze quantization effects | `--quantization-analysis` |

## Output Formats

### CLI Output (Default)

Human-readable colored output with symbols:

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
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

## Configuration

### Global Configuration

`~/.config/diffx/config.toml`:

```toml
[diffai]
default_output = "cli"
default_format = "auto"
epsilon = 0.001
sort_by_magnitude = false

[ml_analysis]
enable_all = false
learning_progress = true
convergence_analysis = true
memory_analysis = true
```

### Environment Variables

```bash
# Set configuration file path
export DIFFAI_CONFIG="/path/to/config.toml"

# Set log level
export DIFFAI_LOG_LEVEL="info"

# Set maximum memory usage
export DIFFAI_MAX_MEMORY="1024"
```

## Practical Examples

### Experiment Comparison

```bash
# Compare two experiment results
diffai experiment_v1/ experiment_v2/ --recursive

# Compare model checkpoints with learning analysis
diffai checkpoints/epoch_10.safetensors checkpoints/epoch_20.safetensors --learning-progress
```

### CI/CD Usage

```yaml
- name: Compare models
  run: |
    diffai baseline/model.safetensors new/model.safetensors --output json > model_diff.json
    
- name: Check deployment readiness
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors --deployment-readiness
```

### Scientific Data Analysis

```bash
# Compare NumPy experiment results
diffai baseline_results.npy new_results.npy --stats

# Compare MATLAB simulation data
diffai simulation_v1.mat simulation_v2.mat --stats
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

- [ML Model Comparison](ml-model-comparison.md) - Advanced ML model analysis
- [Scientific Data Analysis](scientific-data.md) - NumPy and MATLAB file comparison
- [CLI Reference](../reference/cli-reference.md) - Complete command reference

