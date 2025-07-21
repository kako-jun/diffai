# diffai

> **PyTorch、Safetensors、NumPy和MATLAB文件专用的AI/ML特化diff工具**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/diffai.svg)](https://crates.io/crates/diffai)
[![Documentation](https://img.shields.io/badge/docs-GitHub-blue)](https://github.com/kako-jun/diffai/tree/main/docs/index.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

专为**AI/ML和科学计算工作流**设计的下一代diff工具，理解模型结构、张量统计和数值数据，而非仅仅是文本变化。原生支持PyTorch、Safetensors、NumPy数组、MATLAB文件和结构化数据。

```bash
# 传统diff在二进制模型文件上失效
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffai显示有意义的模型变化和完整分析
$ diffai model_v1.safetensors model_v2.safetensors
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  gradient_analysis: flow_health=healthy, norm=0.015000, ratio=1.0500
  deployment_readiness: readiness=0.92, strategy=blue_green, risk=low
  quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%

[WARNING]
• 内存使用量适度增加（+250MB）。监控资源消耗。
• 推理速度受到适度影响（1.3倍较慢）。考虑优化机会。
```

## 主要特性

- **AI/ML原生支持**: 直接支持PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)和MATLAB (.mat)文件
- **张量分析**: 自动计算张量统计（均值、标准差、最小值、最大值、形状、内存使用）
- **全面ML分析**: 30+分析功能包括量化、架构、内存、收敛、异常检测和部署就绪性 - 默认全部启用
- **科学数据支持**: NumPy数组和MATLAB矩阵，支持复数
- **纯Rust实现**: 无系统依赖，在Windows/Linux/macOS上无需额外安装即可工作
- **多种输出格式**: 彩色CLI、用于MLOps集成的JSON、用于人类可读报告的YAML
- **快速且内存高效**: 用Rust构建，高效处理大型模型文件

## 为什么选择diffai？

传统diff工具不适合AI/ML工作流：

| 挑战 | 传统工具 | diffai |
|------|----------|---------|
| **二进制模型文件** | "二进制文件不同" | 带统计的张量级分析 |
| **大文件(GB+)** | 内存问题或失败 | 高效流式和分块处理 |
| **统计变化** | 无语义理解 | 均值/标准差/形状比较及显著性 |
| **ML专用格式** | 不支持 | 原生PyTorch/Safetensors/NumPy/MATLAB |
| **科学工作流** | 仅文本比较 | 数值数组分析和可视化 |

### diffai vs MLOps工具

diffai通过专注于**结构比较**而非实验管理来补充现有MLOps工具：

| 方面 | diffai | MLflow / DVC / ModelDB |
|------|--------|------------------------|
| **焦点** | "使不可比较的事物变得可比较" | 系统化、可重现性、CI/CD集成 |
| **数据假设** | 未知来源文件/黑盒生成工件 | 有充分文档和跟踪的数据 |
| **操作** | 结构和视觉比较优化 | 版本控制和实验跟踪专业化 |
| **范围** | "模糊结构"可视化，包括JSON/YAML/模型文件 | 实验元数据、版本管理、可重现性 |

## 安装

### 从crates.io安装（推荐）

```bash
cargo install diffai
```

### 从源码安装

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build --release
```

## 快速开始

### 基本模型比较

```bash
# 比较PyTorch模型（默认完整分析）
diffai model_old.pt model_new.pt

# 比较Safetensors，完整ML分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 比较NumPy数组
diffai data_v1.npy data_v2.npy

# 比较MATLAB文件
diffai experiment_v1.mat experiment_v2.mat
```

### ML分析功能

```bash
# 对PyTorch/Safetensors自动运行完整ML分析
diffai baseline.safetensors finetuned.safetensors
# 输出：30+分析类型，包括量化、架构、内存等

# 用于自动化的JSON输出
diffai model_v1.safetensors model_v2.safetensors --output json

# 详细模式的详细诊断信息
diffai model_v1.safetensors model_v2.safetensors --verbose

# 用于人类可读报告的YAML输出
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 📚 文档

- **[工作示例和演示](docs/examples/)** - 查看diffai的实际输出
- **[API文档](https://docs.rs/diffai-core)** - Rust库文档
- **[用户指南](docs/user-guide.md)** - 综合使用指南
- **[ML分析指南](docs/ml-analysis-guide.md)** - ML特定功能深入指南

## 支持的文件格式

### ML模型格式
- **Safetensors** (.safetensors) - HuggingFace标准格式
- **PyTorch** (.pt, .pth) - 与Candle集成的PyTorch模型文件

### 科学数据格式
- **NumPy** (.npy, .npz) - 带完整统计分析的NumPy数组
- **MATLAB** (.mat) - 支持复数的MATLAB矩阵

### 结构化数据格式
- **JSON** (.json) - JavaScript对象表示法
- **YAML** (.yaml, .yml) - YAML不是标记语言
- **TOML** (.toml) - Tom的明显最小语言
- **XML** (.xml) - 可扩展标记语言
- **INI** (.ini) - 配置文件
- **CSV** (.csv) - 逗号分隔值

## ML分析功能

### 自动综合分析（v0.3.4）
比较PyTorch或Safetensors文件时，diffai自动运行30+ML分析功能：

**自动功能包括：**
- **统计分析**: 详细张量统计（均值、标准差、最小值、最大值、形状、内存）
- **量化分析**: 分析量化效果和效率
- **架构比较**: 比较模型架构和结构变化
- **内存分析**: 分析内存使用和优化机会
- **异常检测**: 检测模型参数中的数值异常
- **收敛分析**: 分析模型参数中的收敛模式
- **梯度分析**: 可用时分析梯度信息
- **部署就绪性**: 评估生产部署就绪性
- **回归测试**: 自动性能降级检测
- **加上20+其他专业功能**

### 未来增强
- TensorFlow格式支持（.pb, .h5, SavedModel）
- ONNX格式支持
- 高级可视化和图表功能

### 设计理念
diffai默认为ML模型提供全面分析，消除选择困难。用户无需记住或指定数十个分析标志即可获得所有相关洞察。

## 调试和诊断

### 详细模式（`--verbose` / `-v`）
获取用于调试和性能分析的综合诊断信息：

```bash
# 基本详细输出
diffai model1.safetensors model2.safetensors --verbose

# 带结构化数据过滤的详细输出
diffai data1.json data2.json --verbose --epsilon 0.001 --ignore-keys-regex "^id$"
```

**详细输出包括：**
- **配置诊断**: 格式设置、过滤器、分析模式
- **文件分析**: 路径、大小、检测到的格式、处理上下文
- **性能指标**: 处理时间、差异计数、优化状态
- **目录统计**: 文件计数、比较摘要（使用`--recursive`）

**详细输出示例：**
```
=== diffai详细模式已启用 ===
配置：
  输入格式：Safetensors
  输出格式：Cli
  ML分析：已启用完整分析（全部30个功能）
  Epsilon容差：0.001

文件分析：
  输入1：model1.safetensors
  输入2：model2.safetensors
  检测到的格式：Safetensors
  文件1大小：1048576字节
  文件2大小：1048576字节

处理结果：
  总处理时间：1.234ms
  发现差异：15
  ML/科学数据分析完成
```

📚 **详细用法请参见[详细输出指南](docs/user-guide/verbose-output.md)**

## 输出格式

### CLI输出（默认）
带有直观符号的彩色、人类可读输出：
- `~` 已更改的张量/数组，带统计比较
- `+` 已添加的张量/数组，带元数据
- `-` 已删除的张量/数组，带元数据

### JSON输出
用于MLOps集成和自动化的结构化输出：
```bash
diffai model1.safetensors model2.safetensors --output json | jq .
```

### YAML输出
用于文档的人类可读结构化输出：
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## 实际用例

### 研究与开发
```bash
# 比较微调前后的模型（自动完整分析）
diffai pretrained_model.safetensors finetuned_model.safetensors
# 输出：学习进度、收敛分析、参数统计和27项更多分析

# 开发过程中分析架构变化
diffai baseline_architecture.pt improved_architecture.pt
# 输出：架构比较、参数效率分析和完整ML分析
```

### MLOps和CI/CD
```bash
# CI/CD中的自动化模型验证（综合分析）
diffai production_model.safetensors candidate_model.safetensors
# 输出：部署就绪性、回归测试、风险评估和27项更多分析

# 带JSON输出的性能影响评估用于自动化
diffai original_model.pt optimized_model.pt --output json
# 输出：量化分析、内存分析、性能影响估计等
```

### 科学计算
```bash
# 比较NumPy实验结果
diffai baseline_results.npy new_results.npy

# 分析MATLAB仿真数据
diffai simulation_v1.mat simulation_v2.mat

# 比较压缩的NumPy存档
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

## 命令行选项

### 基本选项
- `-f, --format <FORMAT>` - 指定输入文件格式
- `-o, --output <OUTPUT>` - 选择输出格式（cli, json, yaml）
- `-r, --recursive` - 递归比较目录

**注意：** 对于ML模型（PyTorch/Safetensors），包括统计的综合分析自动运行

### 高级选项
- `--path <PATH>` - 按特定路径过滤差异
- `--ignore-keys-regex <REGEX>` - 忽略匹配正则表达式模式的键
- `--epsilon <FLOAT>` - 设置浮点比较的容差
- `--array-id-key <KEY>` - 指定数组元素标识的键
- `--sort-by-change-magnitude` - 按变化幅度排序

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

### 用于自动化的JSON输出
```bash
$ diffai baseline.safetensors improved.safetensors --output json
{
  "anomaly_detection": {"type": "none", "severity": "none"},
  "architecture_comparison": {"type1": "feedforward", "type2": "feedforward"},
  "deployment_readiness": {"readiness": 0.92, "strategy": "blue_green"},
  "quantization_analysis": {"compression": "0.0%", "speedup": "1.8x"},
  "regression_test": {"passed": true, "degradation": "-2.5%"}
  // ... 加上25+其他分析功能
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

## 性能

diffai针对大文件和科学工作流进行了优化：

- **内存高效**: 对GB+文件进行流式处理
- **快速**: Rust实现，优化的张量操作
- **可扩展**: 处理具有数百万/数十亿参数的模型
- **跨平台**: 在Windows、Linux和macOS上无依赖运行

## 贡献

我们欢迎贡献！请参见[CONTRIBUTING](CONTRIBUTING.md)获取指南。

### 开发设置

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build
cargo test
```

### 运行测试

```bash
# 运行所有测试
cargo test

# 运行特定测试类别
cargo test --test integration
cargo test --test ml_analysis
```

## 许可证

本项目在MIT许可证下授权 - 详情请参见[LICENSE](LICENSE)文件。

## 相关项目

- **[diffx](https://github.com/kako-jun/diffx)** - 通用结构化数据diff工具（diffai的兄弟项目）
- **[safetensors](https://github.com/huggingface/safetensors)** - 存储和分发张量的简单、安全方式
- **[PyTorch](https://pytorch.org/)** - 机器学习框架
- **[NumPy](https://numpy.org/)** - Python科学计算基础包

