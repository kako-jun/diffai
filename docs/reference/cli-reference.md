# CLI Reference

Complete command-line reference for diffai v0.3.4 - AI/ML specialized diff tool with automatic comprehensive analysis.

## Synopsis

```
diffai <INPUT1> <INPUT2>
```

## Description

diffai is a specialized diff tool for AI/ML workflows that automatically provides comprehensive analysis of model structures, tensor statistics, and scientific data. It compares PyTorch models, Safetensors files, NumPy arrays, and MATLAB matrices with intelligent automatic analysis - no complex options required.

**Key Features:**
- **Automatic Analysis**: Comprehensive ML-specific analysis by default
- **Zero Configuration**: No options needed for detailed insights
- **AI/ML Focused**: Optimized for model comparison workflows

## Arguments

### Required Arguments

#### `<INPUT1>`
First input file or directory to compare.

- **Type**: File path or directory path
- **Formats**: PyTorch (.pt/.pth), Safetensors (.safetensors), NumPy (.npy/.npz), MATLAB (.mat)
- **Special**: Use `-` for stdin

#### `<INPUT2>`
Second input file or directory to compare.

- **Type**: File path or directory path
- **Formats**: Same as INPUT1
- **Special**: Use `-` for stdin

**Note**: stdin is not supported for AI/ML files as they are binary formats. Use file paths only.

**Examples**:
```bash
# Basic file comparison
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
# For general structured data, use diffx:
# diffx config.json config_new.json

# Directory comparison (automatic recursive)
diffai dir1/ dir2/

# Stdin not supported for binary AI/ML files
# For general data comparison, use diffx:
# cat config.json | diffx - config_new.json
# echo '{"old": "data"}
# {"new": "data"}' | diffx - -
```

## Options

### Basic Options

#### `-h, --help`
Show help information.

#### `-V, --version`
Show version information.

#### `--no-color`
Disable colored output for better compatibility with scripts and automated environments.

- **Example**: `diffai model1.safetensors model2.safetensors --no-color`
- **Usage**: Plain text output without color formatting


## Automatic Analysis

### Comprehensive AI/ML Analysis

**diffai automatically performs comprehensive analysis without requiring any options:**

#### Current Features

- **Tensor Comparison**: Shape, dtype, and basic statistical analysis
- **Model Structure**: Layer detection and parameter counting
- **File Format**: Automatic format detection and parsing
- **Change Detection**: Added, removed, and modified tensor identification

#### Planned Features (Under Development)

- **Advanced Statistics**: Mean, standard deviation, min/max analysis
- **Anomaly Detection**: NaN/Inf detection, gradient analysis
- **Architecture Analysis**: Layer structure comparison
- **Memory Analysis**: Usage estimation and optimization
- **Convergence Analysis**: Training progress tracking
- **Statistical Significance**: Change significance testing
- **Transfer Learning Analysis**: Fine-tuning effectiveness
- **Ensemble Analysis**: Multi-model comparison
- **Hyperparameter Impact**: Configuration change effects
- **Learning Rate Analysis**: Optimization schedule effectiveness
- **And more...**

**🎯 No flags required** - all analysis is performed automatically for optimal user experience.

**Example**: Simply run `diffai model1.safetensors model2.safetensors` to get comprehensive analysis.

## Output Examples

### CLI Output (Default - Full Analysis)

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low, timeline=ready_for_immediate_deployment
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### Comprehensive Analysis Benefits

- **30+ analysis functions** run automatically
- **No option selection needed** - get all insights by default
- **Same processing time** - no performance penalty
- **Production-ready insights** - deployment readiness, risk assessment, etc.

### Scientific Data Analysis (Automatic)

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB File Comparison (Automatic)

```bash
$ diffai simulation_v1.mat simulation_v2.mat
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

diffai does not use environment variables for configuration. All settings are controlled through command-line options.

## Performance Considerations

- **Large Files**: diffai uses streaming processing for GB+ files
- **Memory Usage**: Automatic memory optimization for large files
- **Parallel Processing**: Automatic parallelization for multi-file comparisons
- **Caching**: Intelligent caching for repeated comparisons

## Troubleshooting

### Common Issues

1. **"Binary files differ" message**: Use `--format` to specify file type
2. **Out of memory**: Memory optimization is automatic for large files
3. **Slow processing**: Analysis is optimized for large models automatically
4. **Missing dependencies**: Ensure Rust toolchain is properly installed

### Debug Mode

Enable debug output with the `--verbose` option:
```bash
diffai model1.safetensors model2.safetensors --verbose
```

## See Also

- [Basic Usage Guide](../user-guide/basic-usage.md)
- [ML Model Comparison Guide](../user-guide/ml-model-comparison.md)
- [Scientific Data Analysis Guide](../user-guide/scientific-data.md)
- [Output Format Reference](output-formats.md)
- [Supported Formats Reference](formats.md)