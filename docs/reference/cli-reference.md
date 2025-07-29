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

**diffai automatically performs all 11 ML analysis features without requiring any options:**

#### ✅ Fully Implemented Features (All Available Now)

**High Priority Features:**
1. **Tensor Statistics**: Complete statistical analysis (mean, std, min/max, shape, dtype)
2. **Model Architecture**: Layer detection, parameter counting, structural changes
3. **Weight Changes**: Significant parameter change detection with configurable thresholds
4. **Memory Analysis**: Memory usage analysis and optimization recommendations

**Medium Priority Features:**
5. **Learning Rate**: Learning rate detection from optimizer state and training metadata
6. **Convergence Analysis**: Training convergence pattern analysis from model changes
7. **Gradient Analysis**: Gradient flow analysis estimated from parameter updates

**Advanced Features:**
8. **Attention Analysis**: Transformer attention mechanism analysis and patterns
9. **Ensemble Analysis**: Multi-model ensemble composition and voting strategy analysis
10. **Quantization Analysis**: Model quantization detection and precision analysis

#### Format-Aware Automatic Feature Selection

- **PyTorch (.pt/.pth)**: All 11 features fully active
- **Safetensors (.safetensors)**: 10 features active (limited ensemble analysis)
- **NumPy (.npy/.npz)**: 4 core features active (tensor stats, architecture basics, weights, memory)
- **MATLAB (.mat)**: 4 core features active with basic quantization support

**🎯 Zero configuration required** - optimal analysis for each format automatically selected.

**Example**: Simply run `diffai model1.pt model2.pt` to get all applicable analysis features.

## Output Examples

### CLI Output (Default - Full Analysis)

```bash
$ diffai model_v1.pt model_v2.pt
TensorStatsChanged: fc1.weight
  Old: mean=-0.0002, std=0.0514, shape=[128, 256], dtype=float32
  New: mean=-0.0001, std=0.0716, shape=[128, 256], dtype=float32

ModelArchitectureChanged: model
  Old: {layers: 12, parameters: 124439808, types: [conv, linear, norm]}
  New: {layers: 12, parameters: 124440064, types: [conv, linear, norm, attention]}

WeightSignificantChange: transformer.attention.query.weight
  Change Magnitude: 0.0234 (above threshold: 0.01)

MemoryAnalysis: memory_change
  Old: 487.2MB (tensors: 485.1MB, metadata: 2.1MB)
  New: memory_change: +12.5MB, breakdown: tensors: +12.3MB, metadata: +0.2MB

LearningRateChanged: optimizer.learning_rate
  Old: 0.001, New: 0.0005 (scheduler: step_decay, epoch: 10)

ConvergenceAnalysis: convergence_patterns
  Old: evaluating
  New: loss: improving (trend: decreasing), stability: gradient_norm: stable, epoch: 10 → 11

GradientAnalysis: gradient_magnitudes
  Old: norm: 0.018456, max: 0.145234, var: 0.000234
  New: total_norm: 0.021234 (+14.8%, increasing), max_gradient: 0.156789 (+8.0%)

AttentionAnalysis: attention_heads
  Old: heads: 8, dim: 64, patterns: 4
  New: num_heads: 8 → 12, head_dim: 64 → 48, patterns: +query, +value

QuantizationAnalysis: quantization_precision
  Old: 32bit float32, layers: 0, mixed: false
  New: bit_width: 32 → 8, data_type: float32 → int8, quantized_layers: 8 (+8)
```

### Comprehensive Analysis Benefits

- **All 11 ML analysis features** run automatically
- **Format-aware feature selection** - optimal analysis for each file type
- **No configuration required** - maximum insights by default
- **Production-ready analysis** - comprehensive model assessment

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