# 基本用法

diffai 的基本操作和常见用法模式。

## 简介

diffai 是专为 AI/ML 和科学计算工作流程设计的差异工具。与传统的文本差异工具不同，diffai 理解模型结构、张量统计和数值数据。

## 基本语法

```bash
diffai [OPTIONS] <INPUT1> <INPUT2>
```

### 最简单的用法

```bash
# 比较两个模型文件（自动综合分析）
diffai model1.safetensors model2.safetensors

# 比较 JSON 配置文件
diffai config_old.json config_new.json

# 比较 NumPy 数组（自动统计分析）
diffai data_v1.npy data_v2.npy
```

## 文件格式支持

diffai 自动检测文件格式并适用相应的比较策略：

### ML 模型格式
```bash
# PyTorch 模型（自动综合分析）
diffai model1.pt model2.pt

# Safetensors 模型（推荐，自动综合分析）
diffai model1.safetensors model2.safetensors

# NumPy 数组（自动统计分析）
diffai experiment1.npy experiment2.npy

# MATLAB 文件（自动统计分析）
diffai data1.mat data2.mat
```

### 结构化数据格式
```bash
# JSON 文件
diffai config1.json config2.json

# YAML 文件
diffai settings1.yaml settings2.yaml

# TOML 文件
diffai Cargo.toml Cargo_new.toml
```

## 基本选项

### 自动综合分析

```bash
# PyTorch/Safetensors文件的自动综合分析（30+项功能）
diffai model1.safetensors model2.safetensors

# 输出示例（完整分析）：
# anomaly_detection: type=none, severity=none, action="continue_training"
# architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
# convergence_analysis: status=converging, stability=0.92
# gradient_analysis: flow_health=healthy, norm=0.021069
# memory_analysis: delta=+0.0MB, efficiency=1.000000
# quantization_analysis: compression=0.25, speedup=1.8x, precision_loss=minimal
# deployment_readiness: readiness=0.92, risk=low
# ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
# ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

### 输出格式

```bash
# 默认彩色 CLI 输出
diffai model1.pt model2.pt

# JSON 格式（用于自动化）
diffai model1.pt model2.pt --output json

# YAML 格式（人类可读）
diffai model1.pt model2.pt --output yaml

# 传统 unified diff 格式
diffai config1.json config2.json --output unified
```

### 文件格式指定

```bash
# 自动检测（默认）
diffai model1.safetensors model2.safetensors

# 显式指定格式
diffai model1.file model2.file --format safetensors

# 用于不明确的扩展名
diffai data1.bin data2.bin --format pytorch
```

## 理解输出

### CLI 输出符号

| 符号 | 含义 | 描述 |
|------|------|------|
| `~` | 已修改 | 值已更改但结构相同 |
| `+` | 已添加 | 新元素已添加 |
| `-` | 已删除 | 元素已删除 |
| `□` | 形状已更改 | 张量维度已更改 |

### 示例输出解读

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  + new_layer.weight: shape=[64, 64], dtype=f32, params=4096
  - old_layer.bias: shape=[256], dtype=f32, params=256
```

**解读**:
- `fc1.bias` 和 `fc1.weight` 的统计信息发生了变化
- 添加了新的 `new_layer.weight` 层
- 删除了 `old_layer.bias` 层

## 常见使用场景

### 训练进度监控

```bash
# 比较训练检查点（自动学习分析）
diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt

# 查看学习进度（自动综合分析）
diffai checkpoint_old.pt checkpoint_new.pt
```

### 模型验证

```bash
# 验证模型保存和加载（自动综合分析）
diffai original_model.pt saved_loaded_model.pt

# 检查微调效果（自动综合分析）
diffai pretrained.safetensors finetuned.safetensors
```

### 配置文件管理

```bash
# 比较配置更改
diffai config_v1.json config_v2.json

# 检查 YAML 配置
diffai settings_old.yaml settings_new.yaml
```

### 数据验证

```bash
# 比较 NumPy 数组（自动统计分析）
diffai data_processed.npy data_corrected.npy

# 检查 MATLAB 计算结果（自动统计分析）
diffai results_v1.mat results_v2.mat
```

## 高级选项

### 数值容差

```bash
# 忽略小的浮点差异
diffai model1.pt model2.pt --epsilon 1e-6

# 用于量化模型比较
diffai fp32_model.pt int8_model.pt --epsilon 0.1
```

### 路径过滤

```bash
# 只比较特定路径
diffai model1.pt model2.pt --path "classifier"

# 使用正则表达式忽略键
diffai config1.json config2.json --ignore-keys-regex "^(timestamp|id)$"
```

### 递归目录比较

```bash
# 比较整个目录
diffai models_v1/ models_v2/ --recursive

# 与其他选项结合使用
diffai data_v1/ data_v2/ --recursive
```

## 输出重定向和处理

### 保存到文件

```bash
# 保存 CLI 输出
diffai model1.pt model2.pt > comparison.txt

# 保存 JSON 输出
diffai model1.pt model2.pt --output json > comparison.json

# 保存 YAML 输出
diffai model1.pt model2.pt --output yaml > comparison.yaml
```

### 管道处理

```bash
# 使用 jq 处理 JSON 输出
diffai model1.pt model2.pt --output json | jq '.'

# 计算变化数量
diffai model1.pt model2.pt --output json | jq 'length'

# 过滤特定类型的变化
diffai model1.pt model2.pt --output json | jq '.[] | select(.TensorStatsChanged)'
```

## 实际工作流程示例

### 开发工作流程

```bash
# 1. 开发前检查当前状态
diffai --version

# 2. 训练后比较检查点（自动综合分析）
diffai checkpoint_before.pt checkpoint_after.pt

# 3. 验证模型导出（自动综合分析）
diffai model.pt exported_model.pt

# 4. 部署前最终检查（自动综合分析）
diffai staging_model.pt production_model.pt
```

### 实验工作流程

```bash
# 1. 设置基准
cp baseline_model.pt reference_model.pt

# 2. 运行实验
python train.py --experiment-name exp1

# 3. 比较结果（自动综合分析）
diffai reference_model.pt experiment1_model.pt

# 4. 生成报告
diffai reference_model.pt experiment1_model.pt --output yaml > report.yaml
```

## 性能考虑

### 大文件处理

```bash
# 对于大型模型文件
diffai large_model1.pt large_model2.pt --epsilon 1e-3

# 限制到特定层
diffai large_model1.pt large_model2.pt --path "features"
```

### 内存优化

```bash
# 使用适当的容差
diffai model1.pt model2.pt --epsilon 1e-6

# 智能分析：自动决定需要哪些分析功能
diffai model1.pt model2.pt  # 自动综合分析
```

## 故障排除

### 常见错误

#### 文件未找到
```bash
# 检查文件是否存在
ls -la model1.pt model2.pt

# 使用绝对路径
diffai /path/to/model1.pt /path/to/model2.pt
```

#### 格式检测失败
```bash
# 显式指定格式
diffai file1.bin file2.bin --format pytorch

# 检查文件内容
file model.pt
```

#### 内存不足
```bash
# 使用更大的 epsilon
diffai large1.pt large2.pt --epsilon 1e-3

# 限制分析范围
diffai large1.pt large2.pt --path "classifier"
```

## 下一步

- 了解 [ML 模型比较](ml-model-comparison_zh.md) 的高级功能
- 探索 [科学数据分析](scientific-data_zh.md) 的 NumPy 和 MATLAB 支持
- 查看 [CLI 参考](../reference/cli-reference_zh.md) 获取完整的选项列表