# CLI Reference

Complete command-line reference for diffai v0.2.0 - AI/ML specialized diff tool.

## Synopsis

```
diffai [OPTIONS] <INPUT1> <INPUT2>
```

## Description

diffai is a specialized diff tool for AI/ML workflows that understands model structures, tensor statistics, and scientific data. It compares PyTorch models, Safetensors files, NumPy arrays, MATLAB matrices, and structured data files, focusing on semantic changes rather than formatting differences.

## Arguments

### Required Arguments

#### `<INPUT1>`
First input file or directory to compare.

- **Type**: File path or directory path
- **Formats**: PyTorch (.pt/.pth), Safetensors (.safetensors), NumPy (.npy/.npz), MATLAB (.mat), JSON, YAML, TOML, XML, INI, CSV
- **Special**: Use `-` for stdin

#### `<INPUT2>`
Second input file or directory to compare.

- **Type**: File path or directory path
- **Formats**: Same as INPUT1
- **Special**: Use `-` for stdin

**Examples**:
```bash
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
diffai config.json config_new.json
diffai - config.json < input.json
```

## Options

### Basic Options

#### `-f, --format <FORMAT>`
Specify input file format explicitly.

- **Possible values**: `json`, `yaml`, `toml`, `ini`, `xml`, `csv`, `safetensors`, `pytorch`, `numpy`, `npz`, `matlab`
- **Default**: Auto-detected from file extension
- **Example**: `--format safetensors`

#### `-o, --output <OUTPUT>`
Choose output format.

- **Possible values**: `cli`, `json`, `yaml`, `unified`
- **Default**: `cli`
- **Example**: `--output json`

#### `-r, --recursive`
Compare directories recursively.

- **Example**: `diffai dir1/ dir2/ --recursive`

#### `--stats`
Show detailed statistics for ML models and scientific data.

- **Example**: `diffai model.safetensors model2.safetensors --stats`

### Advanced Options

#### `--path <PATH>`
Filter differences by a specific path.

- **Example**: `--path "config.users[0].name"`
- **Format**: JSONPath-like syntax

#### `--ignore-keys-regex <REGEX>`
Ignore keys matching a regular expression.

- **Example**: `--ignore-keys-regex "^id$"`
- **Format**: Standard regex pattern

#### `--epsilon <FLOAT>`
Set tolerance for float comparisons.

- **Example**: `--epsilon 0.001`
- **Default**: Machine epsilon

#### `--array-id-key <KEY>`
Specify key for identifying array elements.

- **Example**: `--array-id-key "id"`
- **Usage**: For structured array comparison

#### `--sort-by-change-magnitude`
Sort differences by change magnitude (ML models only).

- **Example**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

## ML Analysis Functions

### Currently Available (v0.2.0)

The following ML analysis functions are currently implemented:

#### `--stats`
Show detailed statistics for ML models and scientific data.

- **Output**: Mean, standard deviation, min/max, shape, dtype for each tensor
- **Example**: `diffai model.safetensors model2.safetensors --stats`

#### `--quantization-analysis`
Analyze quantization effects and efficiency.

- **Output**: Compression ratio, precision loss analysis
- **Example**: `diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis`

#### `--sort-by-change-magnitude`
Sort differences by magnitude for prioritization.

- **Output**: Magnitude-sorted difference list
- **Example**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

#### `--show-layer-impact`
Analyze layer-by-layer impact of changes.

- **Output**: Per-layer change analysis
- **Example**: `diffai baseline.safetensors modified.safetensors --show-layer-impact`

### Phase 3 Features (Now Available)

#### Architecture & Performance Analysis

##### `--architecture-comparison`
Compare model architectures and detect structural changes.

- **Output**: Architecture type detection, layer depth comparison, migration difficulty assessment
- **Example**: `diffai model1.safetensors model2.safetensors --architecture-comparison`

##### `--memory-analysis`
Analyze memory usage and optimization opportunities.

- **Output**: Memory delta, peak usage estimation, GPU utilization, optimization recommendations
- **Example**: `diffai model1.safetensors model2.safetensors --memory-analysis`

##### `--anomaly-detection`
Detect numerical anomalies in model parameters.

- **Output**: NaN/Inf detection, gradient explosion/vanishing analysis, dead neuron detection
- **Example**: `diffai model1.safetensors model2.safetensors --anomaly-detection`

##### `--change-summary`
Generate detailed change summaries.

- **Output**: Change magnitude, patterns, layer rankings, structural vs parameter changes
- **Example**: `diffai model1.safetensors model2.safetensors --change-summary`

#### Advanced Analysis

##### `--convergence-analysis`
Analyze convergence patterns in model parameters.

- **Output**: Convergence status, parameter stability, early stopping recommendations
- **Example**: `diffai model1.safetensors model2.safetensors --convergence-analysis`

##### `--gradient-analysis`
Analyze gradient information estimated from parameter changes.

- **Output**: Gradient flow health, norm estimation, problematic layers, clipping recommendations
- **Example**: `diffai model1.safetensors model2.safetensors --gradient-analysis`

##### `--similarity-matrix`
Generate similarity matrix for model comparison.

- **Output**: Layer-to-layer similarities, clustering coefficient, outlier detection
- **Example**: `diffai model1.safetensors model2.safetensors --similarity-matrix`

## Output Examples

### CLI Output (Default)

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### Combined Analysis Output

```bash
$ diffai baseline.safetensors improved.safetensors --stats --quantization-analysis --sort-by-change-magnitude
quantization_analysis: compression=0.25, precision_loss=minimal
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
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

### JSON Output

```bash
$ diffai model_v1.safetensors model_v2.safetensors --output json
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
$ diffai model_v1.safetensors model_v2.safetensors --output yaml
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

## Exit Codes

- **0**: Success - differences found or no differences
- **1**: Error - invalid arguments or file access issues
- **2**: Fatal error - internal processing failure

## Environment Variables

- **DIFFAI_CONFIG**: Path to configuration file
- **DIFFAI_LOG_LEVEL**: Log level (error, warn, info, debug)
- **DIFFAI_MAX_MEMORY**: Maximum memory usage (in MB)

## Configuration File

diffai supports configuration files in TOML format. Place configuration at:

- Unix: `~/.config/diffx/config.toml`
- Windows: `%APPDATA%/diffx/config.toml`
- Current directory: `.diffx.toml`

Example configuration:
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

## Performance Considerations

- **Large Files**: diffai uses streaming processing for GB+ files
- **Memory Usage**: Configurable memory limits with `DIFFAI_MAX_MEMORY`
- **Parallel Processing**: Automatic parallelization for multi-file comparisons
- **Caching**: Intelligent caching for repeated comparisons

## Troubleshooting

### Common Issues

1. **"Binary files differ" message**: Use `--format` to specify file type
2. **Out of memory**: Set `DIFFAI_MAX_MEMORY` environment variable
3. **Slow processing**: Use `--stats` only when needed for large models
4. **Missing dependencies**: Ensure Rust toolchain is properly installed

### Debug Mode

Enable debug output with:
```bash
DIFFAI_LOG_LEVEL=debug diffai model1.safetensors model2.safetensors
```

## See Also

- [Basic Usage Guide](../user-guide/basic-usage.md)
- [ML Model Comparison Guide](../user-guide/ml-model-comparison.md)
- [Scientific Data Analysis Guide](../user-guide/scientific-data.md)
- [Output Format Reference](output-formats.md)
- [Supported Formats Reference](formats.md)