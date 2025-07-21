# 基本用法

学习 diffai - AI/ML 专用差异工具的基本操作。

## 快速开始

### 基本文件比较

```bash
# 比较两个模型文件（自动进行综合分析）
diffai model1.safetensors model2.safetensors

# 以 JSON 格式输出
diffai model1.safetensors model2.safetensors --output json

# 以 YAML 格式输出
diffai model1.safetensors model2.safetensors --output yaml
```

### 目录比较

```bash
# 递归比较整个目录
diffai dir1/ dir2/ --recursive

# 指定特定文件格式进行比较
diffai models_v1/ models_v2/ --format safetensors --recursive
```

## AI/ML 专用功能

### PyTorch 模型比较

```bash
# 比较 PyTorch 模型文件（自动进行完整分析）
diffai model1.pt model2.pt

# 比较训练检查点
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt

# 比较基线与改进模型
diffai baseline_model.pt improved_model.pt
```

**示例输出（完整分析）：**
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

### Safetensors 文件比较

```bash
# 比较 Safetensors 文件（自动进行综合分析）
diffai model1.safetensors model2.safetensors

# 用于生产部署验证
diffai baseline.safetensors candidate.safetensors
```

### 科学数据比较

```bash
# 比较 NumPy 数组（自动统计）
diffai data_v1.npy data_v2.npy

# 比较 MATLAB 文件（自动统计）
diffai simulation_v1.mat simulation_v2.mat

# 比较压缩的 NumPy 归档（自动统计）
diffai dataset_v1.npz dataset_v2.npz
```

## 命令选项

### 基本选项

| 选项 | 描述 | 示例 |
|------|------|------|
| `-f, --format` | 指定输入文件格式 | `--format safetensors` |
| `-o, --output` | 选择输出格式 | `--output json` |
| `-r, --recursive` | 递归比较目录 | `--recursive` |
| `-v, --verbose` | 显示详细处理信息 | `--verbose` |

### 高级选项

| 选项 | 描述 | 示例 |
|------|------|------|
| `--path` | 按特定路径过滤 | `--path "config.model"` |
| `--ignore-keys-regex` | 忽略匹配正则表达式的键 | `--ignore-keys-regex "^id$"` |
| `--epsilon` | 浮点比较容差 | `--epsilon 0.001` |
| `--array-id-key` | 数组元素标识 | `--array-id-key "id"` |

## 输出格式

### CLI 输出（默认 - 完整分析）

带有综合分析的人类可读彩色输出：

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

### JSON 输出

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

### YAML 输出

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


## 实际示例

### 实验比较

```bash
# 比较两个实验结果
diffai experiment_v1/ experiment_v2/ --recursive

# 比较模型检查点（自动学习分析）
diffai checkpoints/epoch_10.safetensors checkpoints/epoch_20.safetensors
```

### CI/CD 用法

```yaml
- name: Compare models
  run: |
    diffai baseline/model.safetensors new/model.safetensors --output json > model_diff.json
    
- name: Check deployment readiness (included in analysis)
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors
```

### 科学数据分析

```bash
# 比较 NumPy 实验结果（自动统计）
diffai baseline_results.npy new_results.npy

# 比较 MATLAB 仿真数据
diffai simulation_v1.mat simulation_v2.mat
```

## 支持的文件格式

### ML 模型格式
- **Safetensors** (.safetensors) - HuggingFace 标准格式
- **PyTorch** (.pt, .pth) - PyTorch 模型文件

### 科学数据格式
- **NumPy** (.npy, .npz) - 带统计分析的 NumPy 数组
- **MATLAB** (.mat) - 支持复数的 MATLAB 矩阵

### 结构化数据格式
- **JSON** (.json), **YAML** (.yaml, .yml), **TOML** (.toml)
- **XML** (.xml), **INI** (.ini), **CSV** (.csv)

## 下一步

- [ML 模型比较](ml-model-comparison_zh.md) - 高级 ML 模型分析
- [科学数据分析](scientific-data_zh.md) - NumPy 和 MATLAB 文件比较
- [CLI 参考](../reference/cli-reference_zh.md) - 完整命令参考