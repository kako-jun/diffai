# CLI 参考

diffai v0.3.4 的完整命令行参考 - 具有简化界面的 AI/ML 专用 diff 工具。

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

**标准输入支持:**
- **一个标准输入，一个文件**: `diffai - file.json` 或 `diffai file.json -`
- **两个都来自标准输入**: `diffai - -` (从标准输入读取两个数据集)
  - **JSON**: 由换行符分隔或连接的两个 JSON 对象
  - **YAML**: 由 `---` 分隔的两个 YAML 文档

**示例**:
```bash
# 基本文件比较
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
diffai config.json config_new.json

# 目录比较（自动递归）
diffai dir1/ dir2/

# 标准输入和文件
cat config.json | diffai - config_new.json

# 两者都从标准输入（管道两者）
echo '{"old": "data"}
{"new": "data"}' | diffai - -

# 从标准输入读取两个 YAML 文档
echo 'name: Alice
age: 25
---
name: Bob
age: 30' | diffai - - --format yaml

# API 响应比较（通过标准输入）
(curl -s https://api.example.com/v1/model; echo; curl -s https://api.example.com/v2/model) | diffai - -
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

- **Possible values**: `diffai`, `json`, `yaml`
- **Default**: `diffai`
- **Example**: `--output json`

#### 目录比较（自动）
比较目录时，diffai 会自动检测并递归比较所有文件。

- **行为**: 自动递归目录遍历
- **示例**: `diffai dir1/ dir2/`（无需标志）
- **注意**: 当提供目录路径时，会自动启用目录比较


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

#### `-v, --verbose`
Show verbose processing information including performance metrics, configuration details, and diagnostic output.

- **Example**: `diffai model1.safetensors model2.safetensors --verbose`
- **Usage**: Debug analysis process and performance

#### `--no-color`
Disable colored output for better compatibility with scripts, pipelines, or terminals that don't support ANSI colors.

- **Example**: `diffai config.json config.new.json --no-color`
- **Usage**: Plain text output without color formatting
- **Note**: Particularly useful for CI/CD environments and automated scripts


## ML Analysis Functions

### ML Analysis (Automatic for PyTorch/Safetensors)

**For PyTorch (.pt/.pth) and Safetensors (.safetensors) files, diffai automatically performs comprehensive analysis including:**

#### Comprehensive Analysis Suite (30+ Features)

- **Basic Statistics**: Mean, standard deviation, min/max, shape, dtype for each tensor
- **Quantization Analysis**: Compression ratio, precision loss analysis  
- **Architecture Comparison**: Structure detection, layer depth comparison, migration assessment
- **Memory Analysis**: Memory delta, peak usage estimation, optimization recommendations
- **Anomaly Detection**: NaN/Inf detection, gradient explosion/vanishing analysis
- **Convergence Analysis**: Parameter stability, early stopping recommendations
- **Gradient Analysis**: Gradient flow health, norm estimation, problematic layers
- **Change Summary**: Magnitude analysis, patterns, layer rankings
- **Similarity Matrix**: Layer-to-layer similarities, clustering coefficient
- **Deployment Readiness**: Production deployment safety assessment
- **Risk Assessment**: Change impact evaluation
- **Performance Impact**: Speed and efficiency analysis
- **Parameter Efficiency**: Optimization opportunities
- **Regression Testing**: Quality assurance validation
- **Learning Progress**: Training progress tracking
- **Embedding Analysis**: Semantic drift detection
- **Attention Analysis**: Transformer attention pattern analysis
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

- [Basic Usage Guide](../user-guide/basic-usage_zh.md)
- [ML Model Comparison Guide](../user-guide/ml-model-comparison_zh.md)
- [Scientific Data Analysis Guide](../user-guide/scientific-data_zh.md)
- [Output Format Reference](output-formats_zh.md)
- [Supported Formats Reference](formats_zh.md)