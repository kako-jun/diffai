# CLI参考

diffai v0.3.4的完整命令行参考 - 具有自动全面分析功能的AI/ML专用差异工具。

## 语法

```
diffai <INPUT1> <INPUT2>
```

## 描述

diffai是专门为AI/ML工作流设计的差异工具，自动提供模型结构、张量统计和科学数据的全面分析。它通过智能自动分析比较PyTorch模型、Safetensors文件、NumPy数组和MATLAB矩阵，无需复杂选项。

**主要功能：**
- **自动分析**：默认提供ML特定的全面分析
- **零配置**：无需选项即可获得详细洞察
- **AI/ML专注**：针对模型比较工作流优化

## 参数

### 必需参数

#### `<INPUT1>`
要比较的第一个输入文件或目录。

- **类型**：文件路径或目录路径
- **格式**：PyTorch（.pt/.pth）、Safetensors（.safetensors）、NumPy（.npy/.npz）、MATLAB（.mat）
- **特殊**：使用`-`表示stdin

#### `<INPUT2>`
要比较的第二个输入文件或目录。

- **类型**：文件路径或目录路径
- **格式**：与INPUT1相同
- **特殊**：使用`-`表示stdin

**注意**：AI/ML文件是二进制格式，不支持stdin。请仅使用文件路径。

**示例**：
```bash
# 基本文件比较
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
# 对于一般结构化数据，使用diffx：
# diffx config.json config_new.json

# 目录比较（自动递归）
diffai dir1/ dir2/

# 二进制AI/ML文件不支持stdin
# 对于一般数据比较，使用diffx：
# cat config.json | diffx - config_new.json
# echo '{"old": "data"}
# {"new": "data"}' | diffx - -
```

## 选项

### 基本选项

#### `-h, --help`
显示帮助信息。

#### `-V, --version`
显示版本信息。

#### `--no-color`
禁用彩色输出，以便与脚本和自动化环境更好兼容。

- **示例**：`diffai model1.safetensors model2.safetensors --no-color`
- **用途**：无颜色格式的纯文本输出

## 自动分析

### 全面AI/ML分析

**diffai自动执行所有11项ML分析功能，无需任何选项：**

#### ✅ 完全实现的功能（现在全部可用）

**高优先级功能：**
1. **张量统计**：完整统计分析（均值、标准差、最小/最大值、形状、数据类型）
2. **模型架构**：层检测、参数计数、结构变化
3. **权重变化**：具有可配置阈值的重要参数变化检测
4. **内存分析**：内存使用分析和优化建议

**中等优先级功能：**
5. **学习率**：从优化器状态和训练元数据中检测学习率
6. **收敛分析**：从模型变化分析训练收敛模式
7. **梯度分析**：从参数更新估计的梯度流分析

**高级功能：**
8. **注意力分析**：Transformer注意力机制分析和模式
9. **集成分析**：多模型集成组合和投票策略分析
10. **量化分析**：模型量化检测和精度分析

#### 格式感知自动功能选择

- **PyTorch（.pt/.pth）**：全部11项功能完全激活
- **Safetensors（.safetensors）**：10项功能激活（集成分析有限）
- **NumPy（.npy/.npz）**：4项核心功能激活（张量统计、基础架构、权重、内存）
- **MATLAB（.mat）**：4项核心功能激活，具有基本量化支持

**🎯 无需配置** - 自动为每种格式选择最佳分析。

**示例**：只需运行`diffai model1.pt model2.pt`即可获得所有适用的分析功能。

## 输出示例

### CLI输出（默认 - 完整分析）

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

### 全面分析的优势

- **全部11项ML分析功能**自动运行
- **格式感知功能选择** - 每种文件类型的最佳分析
- **无需配置** - 默认提供最大洞察
- **生产就绪分析** - 全面的模型评估

### 科学数据分析（自动）

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB文件比较（自动）

```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### JSON输出

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

### YAML输出

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

## 退出代码

- **0**：成功 - 找到差异或无差异
- **1**：错误 - 无效参数或文件访问问题
- **2**：致命错误 - 内部处理失败

## 环境变量

diffai不使用环境变量进行配置。所有设置都通过命令行选项控制。

## 性能考虑

- **大文件**：diffai对GB+文件使用流处理
- **内存使用**：大文件自动内存优化
- **并行处理**：多文件比较自动并行化
- **缓存**：重复比较的智能缓存

## 故障排除

### 常见问题

1. **"Binary files differ"消息**：使用`--format`指定文件类型
2. **内存不足**：大文件的内存优化是自动的
3. **处理缓慢**：大模型分析自动优化
4. **缺少依赖**：确保Rust工具链正确安装

### 调试模式

使用`--verbose`选项启用调试输出：
```bash
diffai model1.safetensors model2.safetensors --verbose
```

## 参见

- [基本使用指南](../user-guide/basic-usage_zh.md)
- [ML模型比较指南](../user-guide/ml-model-comparison_zh.md)
- [科学数据分析指南](../user-guide/scientific-data_zh.md)
- [输出格式参考](output-formats_zh.md)
- [支持格式参考](formats_zh.md)