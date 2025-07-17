# diffai

> **PyTorch、Safetensors、NumPy 和 MATLAB 文件的 AI/ML 专用差异工具**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/diffai.svg)](https://crates.io/crates/diffai)
[![Documentation](https://img.shields.io/badge/docs-GitHub-blue)](https://github.com/kako-jun/diffai/tree/main/docs/index.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

专为 **AI/ML 和科学计算工作流程** 设计的新一代差异工具，理解模型结构、张量统计和数值数据，而不仅仅是文本变化。原生支持 PyTorch、Safetensors、NumPy 数组、MATLAB 文件和结构化数据。

```bash
# 传统 diff 对二进制模型文件失效
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffai 显示有意义的模型变化及完整分析
$ diffai model_v1.safetensors model_v2.safetensors
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  gradient_analysis: flow_health=healthy, norm=0.015000, ratio=1.0500
  deployment_readiness: readiness=0.92, strategy=blue_green, risk=low
  quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%

[WARNING]
• Memory usage increased moderately (+250MB). Monitor resource consumption.
• Inference speed moderately affected (1.3x slower). Consider optimization opportunities.
```

## 核心特性

- **AI/ML 原生支持**: 直接支持 PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz) 和 MATLAB (.mat) 文件
- **张量分析**: 自动计算张量统计（均值、标准差、最小值、最大值、形状、内存使用）
- **全面ML分析**: 针对PyTorch/Safetensors文件自动执行30+分析功能（量子化、架构、内存、收敛、异常检测、部署就绪性等） - 全部默认启用
- **科学数据支持**: 支持复数的 NumPy 数组和 MATLAB 矩阵
- **纯 Rust 实现**: 无系统依赖，在 Windows/Linux/macOS 上无需额外安装即可运行
- **多种输出格式**: 彩色 CLI、用于 MLOps 集成的 JSON、人类可读的 YAML 报告
- **快速且内存高效**: 使用 Rust 构建，高效处理大型模型文件

## 为什么选择 diffai？

传统差异工具不适合 AI/ML 工作流程：

| 挑战 | 传统工具 | diffai |
|------|----------|--------|
| **二进制模型文件** | "Binary files differ" | 张量级分析和统计 |
| **大文件 (GB+)** | 内存问题或失败 | 高效流式处理和分块处理 |
| **统计变化** | 无语义理解 | 均值/标准差/形状比较和显著性分析 |
| **ML 特定格式** | 不支持 | 原生 PyTorch/Safetensors/NumPy/MATLAB |
| **科学工作流程** | 仅文本比较 | 数值数组分析和可视化 |

### diffai vs MLOps 工具

diffai 通过专注于**结构化比较**来补充现有的 MLOps 工具，而非实验管理：

| 方面 | diffai | MLflow / DVC / ModelDB |
|------|--------|------------------------|
| **焦点** | "让不可比较的东西变得可比较" | 系统化、可复现性、CI/CD 集成 |
| **数据假设** | 来源未知的文件／黑盒生成产物 | 有记录和跟踪的数据 |
| **操作性** | 结构化和可视化比较优化 | 版本控制和实验跟踪专业化 |
| **适用范围** | 包括 JSON/YAML/模型文件等"模糊结构"的可视化 | 实验元数据、版本管理、可复现性 |

## 安装

### 从 crates.io 安装（推荐）

```bash
cargo install diffai
```

### 从源码构建

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build --release
```

## 快速开始

### 基本模型比较

```bash
# 比较 PyTorch 模型（默认全面分析）
diffai model_old.pt model_new.pt

# 比较 Safetensors（包含30+ML分析功能的综合分析）
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 比较 NumPy 数组
diffai data_v1.npy data_v2.npy

# 比较 MATLAB 文件
diffai experiment_v1.mat experiment_v2.mat
```

### 自动ML分析

```bash
# PyTorch/Safetensors文件自动执行完整ML分析
diffai baseline.safetensors finetuned.safetensors
# 输出：量子化、架构、内存分析等30+类型分析

# 自动化用JSON输出
diffai model_v1.safetensors model_v2.safetensors --output json

# 带详细诊断信息的verbose模式
diffai model_v1.safetensors model_v2.safetensors --verbose

# 人类可读报告用YAML输出
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 支持的文件格式

### ML 模型格式
- **Safetensors** (.safetensors) - HuggingFace 标准格式，推荐用于高效安全的模型存储
- **PyTorch** (.pt/.pth) - PyTorch 原生格式，通过 Candle 库集成
- **NumPy** (.npy/.npz) - 科学计算数据格式，支持所有数据类型
- **MATLAB** (.mat) - MATLAB 矩阵格式，支持复数和变量名

### 结构化数据格式
- **JSON** (.json) - 用于配置和 API 响应
- **YAML** (.yaml/.yml) - 用于配置文件和文档
- **TOML** (.toml) - 用于 Rust 项目配置
- **XML** (.xml) - 用于数据交换
- **INI** (.ini) - 用于传统配置文件
- **CSV** (.csv) - 用于表格数据

## 调试和诊断

### 详细模式（`--verbose` / `-v`）
获取用于调试和性能分析的综合诊断信息：

```bash
# 基本详细输出（ML分析功能自动执行）
diffai model1.safetensors model2.safetensors --verbose

# 结构化数据的详细输出
diffai data1.json data2.json --verbose --epsilon 0.001 --ignore-keys-regex "^id$"
```

**详细输出包含信息：**
- **配置诊断**: 活动的ML功能、格式设置、过滤器
- **文件分析**: 路径、大小、检测的格式、处理上下文
- **性能指标**: 处理时间、差异计数、优化状态
- **目录统计**: 文件计数、比较摘要（使用`--recursive`时）

**详细输出示例：**
```
=== diffai verbose mode enabled ===
Configuration:
  Input format: None
  Output format: Cli
  ML analysis features: statistics, architecture_comparison
  Epsilon tolerance: 0.001

File analysis:
  Input 1: model1.safetensors
  Input 2: model2.safetensors
  Detected format: Safetensors
  File 1 size: 1048576 bytes
  File 2 size: 1048576 bytes

Processing results:
  Total processing time: 1.234ms
  Differences found: 15
  ML/Scientific data analysis completed
```

📚 **详细信息请参见[详细输出指南](docs/user-guide/verbose-output_zh.md)**

## ML分析功能

### 自动全面分析 (v0.3.4)
比较PyTorch或Safetensors文件时，diffai自动执行30+ML分析功能：

**自动功能包括：**
- **统计分析**: 详细的张量统计（均值、标准差、最小值、最大值、形状、内存）
- **量子化分析**: 分析量子化效果和效率
- **架构比较**: 比较模型架构和结构变化
- **内存分析**: 分析内存使用和优化机会
- **异常检测**: 检测模型参数中的数值异常
- **收敛分析**: 分析模型参数中的收敛模式
- **梯度分析**: 分析梯度信息（如可用）
- **部署就绪性**: 评估生产部署就绪状态
- **回归测试**: 自动性能退化检测
- **其他20+专业功能**

### 未来增强
- TensorFlow格式支持 (.pb, .h5, SavedModel)
- ONNX格式支持
- 高级可视化和图表功能

### 设计理念
diffai默认为ML模型提供全面分析，消除选择困难。用户无需记住或指定数十个分析标志，即可获得所有相关洞察。


## 使用示例

### 训练监控

```bash
# 监控学习进度和收敛（自动完整分析）
diffai checkpoint_old.safetensors checkpoint_new.safetensors
# 输出：学习进度、收敛分析、异常检测等30+分析
```

### 生产部署

```bash
# 部署前评估（自动完整分析）
diffai current_prod.safetensors candidate.safetensors
# 输出：部署就绪性、风险评估、回归测试等30+分析
```

### 研究分析

```bash
# 模型实验比较（自动完整分析）
diffai baseline.safetensors experiment.safetensors
# 输出：架构比较、嵌入分析、全面报告等30+分析
```

### 量化验证

```bash
# 量化效果评估（自动完整分析）
diffai fp32.safetensors quantized.safetensors
# 输出：量化分析、内存分析、性能影响估计等30+分析
```

## 命令行选项

### 基本选项
- `-f, --format <FORMAT>` - 指定输入文件格式
- `-o, --output <OUTPUT>` - 选择输出格式（cli, json, yaml）
- `-r, --recursive` - 递归比较目录

**注意：** 对于ML模型（PyTorch/Safetensors），包含统计的综合分析会自动运行

### 高级选项
- `--path <PATH>` - 按特定路径过滤差异
- `--ignore-keys-regex <REGEX>` - 忽略匹配正则表达式模式的键
- `--epsilon <FLOAT>` - 设置浮点数比较的容差
- `--array-id-key <KEY>` - 指定数组元素标识键
- `--sort-by-change-magnitude` - 按变化幅度排序

## 输出格式

### CLI 输出（默认）
带有直观符号的彩色人类可读输出：
- `~` 变更的张量/数组及统计比较
- `+` 添加的张量/数组及元数据
- `-` 删除的张量/数组及元数据

### JSON 输出
用于MLOps集成和自动化的结构化输出：
```bash
diffai model1.safetensors model2.safetensors --output json | jq .
```

### YAML 输出
用于文档的人类可读结构化输出：
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## 示例

### 基本张量比较（自动）
```bash
$ diffai simple_model_v1.safetensors simple_model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### 自动化用JSON输出
```bash
$ diffai baseline.safetensors improved.safetensors --output json
{
  "anomaly_detection": {"type": "none", "severity": "none"},
  "architecture_comparison": {"type1": "feedforward", "type2": "feedforward"},
  "deployment_readiness": {"readiness": 0.92, "strategy": "blue_green"},
  "quantization_analysis": {"compression": "0.0%", "speedup": "1.8x"},
  "regression_test": {"passed": true, "degradation": "-2.5%"}
  // ... 其他25+分析功能
}
```

### 科学数据分析
```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB文件比较
```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

## 实际应用场景

### 研究开发
```bash
# 微调前后的模型比较（自动完整分析）
diffai pretrained_model.safetensors finetuned_model.safetensors
# 输出：学习进度、收敛分析、参数统计，以及其他27+分析

# 开发中的架构变化分析
diffai baseline_architecture.pt improved_architecture.pt
# 输出：架构比较、参数效率分析和完整ML分析
```

### MLOps & CI/CD
```bash
# CI/CD中的自动模型验证（综合分析）
diffai production_model.safetensors candidate_model.safetensors
# 输出：部署就绪性、回归测试、风险评估以及其他27+分析

# 自动化性能影响评估（JSON输出）
diffai original_model.pt optimized_model.pt --output json
# 输出：量子化分析、内存分析、性能影响估计等
```

### 科学计算
```bash
# NumPy实验结果比较
diffai baseline_results.npy new_results.npy

# MATLAB仿真数据分析
diffai simulation_v1.mat simulation_v2.mat

# 压缩NumPy归档比较
diffai dataset_v1.npz dataset_v2.npz
```

### 实验跟踪
```bash
# 生成综合报告
diffai experiment_baseline.safetensors experiment_improved.safetensors \
  --generate-report --markdown-output --review-friendly

# A/B测试分析
diffai model_a.safetensors model_b.safetensors \
  --statistical-significance --hyperparameter-comparison
```

## 实际应用场景

### 训练进度监控
```bash
# 比较训练检查点
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors

# 输出分析学习趋势和收敛速度
+ learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

### 模型微调分析
```bash
# 分析微调前后的变化
diffai pretrained_bert.safetensors finetuned_bert.safetensors

# 显示统计变化
~ bert.encoder.layer.11.attention.self.query.weight: mean=-0.0001→0.0023
~ classifier.weight: mean=0.0000→0.0145, std=0.0200→0.0890
```

### 量化影响评估
```bash
# 评估量化对模型的影响
diffai fp32_model.safetensors int8_model.safetensors

# 分析压缩效果
quantization_analysis: compression=75.0%, speedup=2.5x, precision_loss=2.0%, suitability=good
```

### 部署准备度检查
```bash
# 检查模型是否准备好部署
diffai production.safetensors candidate.safetensors

# 评估部署风险
deployment_readiness: readiness=0.75, strategy=gradual, risk=medium
```

## 集成示例

### MLflow 集成
```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    # 运行 diffai 比较
    result = subprocess.run([
        'diffai', model1_path, model2_path, '--output', 'json'
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    # 记录到 MLflow
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
```

### CI/CD 管道
```yaml
name: Model Validation
on: [push, pull_request]

jobs:
  model-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install diffai
        run: cargo install diffai
        
      - name: Compare models
        run: |
          diffai models/baseline.safetensors models/candidate.safetensors \
            --output json > model_diff.json
            
      - name: Analyze changes
        run: |
          # 如果关键层发生变化则失败
          if jq -e '.[] | select(.TensorShapeChanged and (.TensorShapeChanged[0] | contains("classifier")))' model_diff.json; then
            echo "CRITICAL: Critical layer shape changes detected"
            exit 1
          fi
```

## 性能考量

### 内存使用优化
```bash
# 对于大型模型，使用 epsilon 减少内存使用
diffai large1.safetensors large2.safetensors --epsilon 1e-3

# 限制分析到特定路径
diffai model1.pt model2.pt --path "classifier"
```

### 速度优化提示
1. **使用 epsilon**: 忽略小的差异以加快处理速度
2. **路径过滤**: 只比较必要的部分
3. **适当的输出格式**: 根据用途选择最优格式

## 故障排除

### 常见问题

#### "Failed to parse" 错误
```bash
# 显式指定文件格式
diffai --format safetensors model1.safetensors model2.safetensors

# 检查文件完整性
file model.safetensors
```

#### 内存不足错误
```bash
# 使用更大的 epsilon
diffai --epsilon 1e-3 large1.pt large2.pt

# 只分析特定层
diffai --path "classifier" model1.pt model2.pt
```

#### 权限错误
```bash
# 检查读取权限
ls -la model.safetensors

# 必要时更改权限
chmod 644 model.safetensors
```


## 贡献

欢迎贡献！请查看 [CONTRIBUTING](CONTRIBUTING.md) 了解详细信息。

### 开发设置
```bash
# 克隆仓库
git clone https://github.com/kako-jun/diffai.git
cd diffai

# 运行测试
cargo test

# 构建发布版本
cargo build --release
```

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 相关项目

- [diffx](https://github.com/kako-jun/diffx) - 通用结构化数据差异工具（diffai 的兄弟项目）

## 支持

- 📖 [文档](https://github.com/kako-jun/diffai/tree/main/docs/index.md)
- 🐛 [问题报告](https://github.com/kako-jun/diffai/issues)
- 💬 [讨论](https://github.com/kako-jun/diffai/discussions)

---

**diffai** - 为 AI/ML 时代设计的智能差异工具 🚀