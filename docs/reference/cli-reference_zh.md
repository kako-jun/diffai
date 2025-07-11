# CLI参考手册

diffai v0.2.0 完整命令行参考手册 - AI/ML专用差分工具

## 命令概要

```
diffai [选项] <输入文件1> <输入文件2>
```

## 描述

diffai是专门为AI/ML工作流设计的差分工具，能够理解模型结构、张量统计和科学数据。它比较PyTorch模型、Safetensors文件、NumPy数组、MATLAB矩阵和结构化数据文件，专注于语义变化而非格式差异。

## 参数

### 必需参数

#### `<输入文件1>`
第一个要比较的输入文件或目录

- **类型**: 文件路径或目录路径
- **格式**: PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)、MATLAB (.mat)、JSON、YAML、TOML、XML、INI、CSV
- **特殊用法**: 使用 `-` 表示从标准输入读取

#### `<输入文件2>`
第二个要比较的输入文件或目录

- **类型**: 文件路径或目录路径
- **格式**: 与输入文件1相同
- **特殊用法**: 使用 `-` 表示从标准输入读取

**示例**:
```bash
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
diffai config.json config_new.json
diffai - config.json < input.json
```

## 选项

### 基本选项

#### `-f, --format <格式>`
显式指定输入文件格式

- **可能的值**: `json`、`yaml`、`toml`、`ini`、`xml`、`csv`、`safetensors`、`pytorch`、`numpy`、`npz`、`matlab`
- **默认值**: 从文件扩展名自动检测
- **示例**: `--format safetensors`

#### `-o, --output <输出格式>`
选择输出格式

- **可能的值**: `cli`、`json`、`yaml`、`unified`
- **默认值**: `cli`
- **示例**: `--output json`

#### `-r, --recursive`
递归比较目录

- **示例**: `diffai dir1/ dir2/ --recursive`

#### `--stats`
显示ML模型和科学数据的详细统计信息

- **示例**: `diffai model.safetensors model2.safetensors --stats`

### 高级选项

#### `--path <路径>`
按特定路径过滤差异

- **示例**: `--path "config.users[0].name"`
- **格式**: 类JSONPath语法

#### `--ignore-keys-regex <正则表达式>`
忽略匹配正则表达式的键

- **示例**: `--ignore-keys-regex "^id$"`
- **格式**: 标准正则表达式模式

#### `--epsilon <浮点数>`
设置浮点数比较的容差

- **示例**: `--epsilon 0.001`
- **默认值**: 机器精度

#### `--array-id-key <键>`
指定用于识别数组元素的键

- **示例**: `--array-id-key "id"`
- **用途**: 用于结构化数组比较

#### `--sort-by-change-magnitude`
按变化幅度排序差异（仅限ML模型）

- **示例**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

## ML分析功能

### 当前可用功能（v0.2.0）

以下ML分析功能当前已实现：

#### `--stats`
显示ML模型和科学数据的详细统计信息

- **输出**: 每个张量的平均值、标准差、最小/最大值、形状、数据类型
- **示例**: `diffai model.safetensors model2.safetensors --stats`

#### `--quantization-analysis`
分析量化效果和效率

- **输出**: 压缩比例、精度损失分析
- **示例**: `diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis`

#### `--sort-by-change-magnitude`
按幅度排序差异以便确定优先级

- **输出**: 按幅度排序的差异列表
- **示例**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

#### `--show-layer-impact`
分析逐层变化影响

- **输出**: 每层变化分析
- **示例**: `diffai baseline.safetensors modified.safetensors --show-layer-impact`

### 第3阶段功能（现已可用）

#### 架构与性能分析

##### `--architecture-comparison`
比较模型架构和检测结构变化

- **输出**: 架构类型检测、层深度比较、迁移难度评估
- **示例**: `diffai model1.safetensors model2.safetensors --architecture-comparison`

##### `--memory-analysis`
分析内存使用和优化机会

- **输出**: 内存增量、峰值使用估算、GPU利用率、优化建议
- **示例**: `diffai model1.safetensors model2.safetensors --memory-analysis`

##### `--anomaly-detection`
检测模型参数中的数值异常

- **输出**: NaN/Inf检测、梯度爆炸/消失分析、死神经元检测
- **示例**: `diffai model1.safetensors model2.safetensors --anomaly-detection`

##### `--change-summary`
生成详细的变化摘要

- **输出**: 变化幅度、模式、层排名、结构vs参数变化
- **示例**: `diffai model1.safetensors model2.safetensors --change-summary`

#### 高级分析

##### `--convergence-analysis`
分析模型参数中的收敛模式

- **输出**: 收敛状态、参数稳定性、早停建议
- **示例**: `diffai model1.safetensors model2.safetensors --convergence-analysis`

##### `--gradient-analysis`
分析从参数变化估算的梯度信息

- **输出**: 梯度流健康度、范数估算、问题层、裁剪建议
- **示例**: `diffai model1.safetensors model2.safetensors --gradient-analysis`

##### `--similarity-matrix`
生成模型比较的相似度矩阵

- **输出**: 层间相似度、聚类系数、异常值检测
- **示例**: `diffai model1.safetensors model2.safetensors --similarity-matrix`

## 输出示例

### CLI输出（默认）

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### 组合分析输出

```bash
$ diffai baseline.safetensors improved.safetensors --stats --quantization-analysis --sort-by-change-magnitude
quantization_analysis: compression=0.25, precision_loss=minimal
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
```

### 科学数据分析

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy --stats
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB文件比较

```bash
$ diffai simulation_v1.mat simulation_v2.mat --stats
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

- **0**: 成功 - 找到差异或无差异
- **1**: 错误 - 无效参数或文件访问问题
- **2**: 致命错误 - 内部处理失败

## 环境变量

- **DIFFAI_LOG_LEVEL**: 日志级别 (error, warn, info, debug)
- **DIFFAI_MAX_MEMORY**: 最大内存使用量 (MB为单位)

## 性能考虑

- **大文件**: diffai对GB+文件使用流处理
- **内存使用**: 可通过`DIFFAI_MAX_MEMORY`配置内存限制
- **并行处理**: 多文件比较的自动并行化
- **缓存**: 重复比较的智能缓存

## 故障排除

### 常见问题

1. **"Binary files differ"消息**: 使用`--format`指定文件类型
2. **内存不足**: 设置`DIFFAI_MAX_MEMORY`环境变量
3. **处理缓慢**: 对于大模型，仅在需要时使用`--stats`
4. **缺少依赖**: 确保Rust工具链已正确安装

### 调试模式

启用调试输出：
```bash
DIFFAI_LOG_LEVEL=debug diffai model1.safetensors model2.safetensors
```

## 相关文档

- [基本用法指南](../user-guide/basic-usage_zh.md)
- [ML模型比较指南](../user-guide/ml-model-comparison_zh.md)
- [科学数据分析指南](../user-guide/scientific-data_zh.md)
- [输出格式参考](output-formats_zh.md)
- [支持格式参考](formats_zh.md)