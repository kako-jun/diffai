# diffai

> **PyTorch、Safetensors、NumPy、MATLAB文件专用的AI/ML特化diff工具**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io CLI](https://img.shields.io/crates/v/diffai.svg?label=diffai-cli)](https://crates.io/crates/diffai)
[![Docs.rs Core](https://docs.rs/diffai-core/badge.svg)](https://docs.rs/diffai-core)
[![npm](https://img.shields.io/npm/v/diffai-js.svg?label=diffai-js)](https://www.npmjs.com/package/diffai-js)
[![PyPI](https://img.shields.io/pypi/v/diffai-python.svg?label=diffai-python)](https://pypi.org/project/diffai-python/)
[![Documentation](https://img.shields.io/badge/📚%20User%20Guide-Documentation-green)](https://github.com/kako-jun/diffai/tree/main/docs/index_zh.md)
[![API Reference](https://img.shields.io/badge/🔧%20API%20Reference-docs.rs-blue)](https://docs.rs/diffai-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

专为**AI/ML和科学计算工作流**设计的下一代diff工具，能够理解模型结构、张量统计和数值数据，而不仅仅是文本变化。原生支持PyTorch、Safetensors、NumPy数组、MATLAB文件和结构化数据。

```bash
# Traditional diff fails with binary model files
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffai shows meaningful model changes with full analysis
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

## 核心功能

- **AI/ML原生支持**: 直接支持PyTorch（.pt/.pth）、Safetensors（.safetensors）、NumPy（.npy/.npz）和MATLAB（.mat）文件
- **张量分析**: 自动计算张量统计信息（均值、标准差、最小值、最大值、形状、内存使用量）
- **全面ML分析**: 包含量化、架构、内存、收敛、异常检测和部署就绪性在内的30+种分析功能 - 全部默认启用
- **科学数据支持**: 支持复数的NumPy数组和MATLAB矩阵
- **纯Rust实现**: 无系统依赖，在Windows/Linux/macOS上无需额外安装即可运行
- **多种输出格式**: 彩色CLI、用于MLOps集成的JSON、便于阅读的YAML报告
- **高速且内存高效**: 采用Rust构建，能够高效处理大型模型文件

## 为什么选择diffai？

传统的diff工具不适用于AI/ML工作流：

| 挑战 | 传统工具 | diffai |
|------|----------|--------|
| **二进制模型文件** | "Binary files differ" | 带有统计信息的张量级分析 |
| **大文件（GB+）** | 内存问题或处理失败 | 高效流式处理和分块处理 |
| **统计变化** | 无语义理解 | 带有统计显著性的均值/标准差/形状比较 |
| **ML专用格式** | 不支持 | 原生支持PyTorch/Safetensors/NumPy/MATLAB |
| **科学计算工作流** | 仅支持文本比较 | 数值数组分析和可视化 |

### diffai vs MLOps工具

diffai通过专注于**结构比较**而非实验管理来补充现有的MLOps工具：

| 方面 | diffai | MLflow / DVC / ModelDB |
|------|--------|------------------------|
| **焦点** | “让不可比较的东西变得可比较” | 系统化、可重现性、CI/CD集成 |
| **数据假设** | 未知源文件 / 黑盒生成的产物 | 充分文档化和跟踪的数据 |
| **操作** | 结构和视觉比较优化 | 版本控制和实验跟踪专业化 |
| **范围** | 包括JSON/YAML/模型文件在内的“模糊结构”可视化 | 实验元数据、版本管理、可重现性 |

## 安装

### 从 crates.io 安装（推荐）

```bash
cargo install diffai
```

### 从源码安装

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build --release
```

## 快速入门

### 基本模型比较

```bash
# 使用全面分析比较PyTorch模型（默认）
diffai model_old.pt model_new.pt

# 使用完整ML分析比较Safetensors
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 比较NumPy数组
diffai data_v1.npy data_v2.npy

# 比较MATLAB文件
diffai experiment_v1.mat experiment_v2.mat
```

### ML分析功能

```bash
# PyTorch/Safetensors文件自动运行全面ML分析
diffai baseline.safetensors finetuned.safetensors
# 输出：包括量化、架构、内存等在30+种分析类型

# 用于自动化的JSON输出
diffai model_v1.safetensors model_v2.safetensors --output json

# 使用详细模式显示详细诊断信息
diffai model_v1.safetensors model_v2.safetensors --verbose

# 用于可读报告的YAML输出
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 📚 文档

- **[实用示例和演示](docs/examples/)** - 查看带有真实输出的diffai实际操作
- **[API文档](https://docs.rs/diffai-core)** - Rust库文档
- **[用户指南](docs/user-guide/getting-started_zh.md)** - 全面的使用指南
- **[ML分析指南](docs/reference/ml-analysis_zh.md)** - ML专用功能的深入介绍

## 支持的文件格式

### ML模型格式
- **Safetensors** (.safetensors) - HuggingFace标准格式
- **PyTorch** (.pt, .pth) - 集成Candle的PyTorch模型文件

### 科学数据格式  
- **NumPy** (.npy, .npz) - 带有完整统计分析的NumPy数组
- **MATLAB** (.mat) - 支持复数的MATLAB矩阵

### 结构化数据格式
- **JSON** (.json) - JavaScript Object Notation
- **YAML** (.yaml, .yml) - YAML Ain't Markup Language
- **TOML** (.toml) - Tom's Obvious Minimal Language  
- **XML** (.xml) - Extensible Markup Language
- **INI** (.ini) - 配置文件
- **CSV** (.csv) - 逗号分隔值

## ML分析功能

### 自动全面分析（v0.3.4）
在比较PyTorch或Safetensors文件时，diffai会自动运行30+种ML分析功能：

**自动功能包括：**
- **统计分析**: 详细的张量统计信息（均值、标准差、最小值、最大值、形状、内存）
- **量化分析**: 分析量化效果和效率
- **架构比较**: 比较模型架构和结构变化
- **内存分析**: 分析内存使用情况和优化机会
- **异常检测**: 检测模型参数中的数值异常
- **收敛分析**: 分析模型参数中的收敛模式
- **梯度分析**: 分析可用的梯度信息
- **部署就绪性**: 评估生产部署的就绪性
- **回归测试**: 自动性能降级检测
- **另外还20+种专业功能**

### 未来增强功能
- TensorFlow格式支持（.pb, .h5, SavedModel）
- ONNX格式支持
- 高级可视化和图表功能

### 设计理念
diffai为ML模型默认提供全面分析，消除选择麻痹。用户无需记住或指定数十个分析标志，就能获得所有相关洞察。

## Debugging and Diagnostics

### Verbose Mode (`--verbose` / `-v`)
Get comprehensive diagnostic information for debugging and performance analysis:

```bash
# Basic verbose output
diffai model1.safetensors model2.safetensors --verbose

# Verbose with structured data filtering
diffai data1.json data2.json --verbose --epsilon 0.001 --ignore-keys-regex "^id$"
```

**Verbose output includes:**
- **Configuration diagnostics**: Format settings, filters, analysis modes
- **File analysis**: Paths, sizes, detected formats, processing context
- **Performance metrics**: Processing time, difference counts, optimization status
- **Directory statistics**: File counts, comparison summaries (目录自动处理)

**Example verbose output:**
```
=== diffai verbose mode enabled ===
Configuration:
  Input format: Safetensors
  Output format: Cli
  ML analysis: Full analysis enabled (all 30 features)
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

📚 **See [Verbose Output Guide](docs/user-guide/verbose-output_zh.md) for detailed usage**

## Output Formats

### CLI Output (Default)
Colored, human-readable output with intuitive symbols:
- `~` Changed tensors/arrays with statistical comparison
- `+` Added tensors/arrays with metadata
- `-` Removed tensors/arrays with metadata

### JSON Output
Structured output for MLOps integration and automation:
```bash
diffai model1.safetensors model2.safetensors --output json | jq .
```

### YAML Output  
Human-readable structured output for documentation:
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## Real-World Use Cases

### Research & Development
```bash
# Compare model before and after fine-tuning (full analysis automatic)
diffai pretrained_model.safetensors finetuned_model.safetensors
# Outputs: learning_progress, convergence_analysis, parameter stats, and 27 more analyses

# Analyze architectural changes during development
diffai baseline_architecture.pt improved_architecture.pt
# Outputs: architecture_comparison, param_efficiency_analysis, and full ML analysis
```

### MLOps & CI/CD
```bash
# Automated model validation in CI/CD (comprehensive analysis)
diffai production_model.safetensors candidate_model.safetensors
# Outputs: deployment_readiness, regression_test, risk_assessment, and 27 more analyses

# Performance impact assessment with JSON output for automation
diffai original_model.pt optimized_model.pt --output json
# Outputs: quantization_analysis, memory_analysis, performance_impact_estimate, etc.
```

### Scientific Computing
```bash
# Compare NumPy experiment results
diffai baseline_results.npy new_results.npy

# Analyze MATLAB simulation data
diffai simulation_v1.mat simulation_v2.mat

# Compare compressed NumPy archives
diffai dataset_v1.npz dataset_v2.npz
```

### Experiment Tracking
```bash
# Generate comprehensive reports
diffai experiment_baseline.safetensors experiment_improved.safetensors \
  --generate-report --markdown-output --review-friendly

# A/B test analysis
diffai model_a.safetensors model_b.safetensors \
  --statistical-significance --hyperparameter-comparison
```

## Command-Line Options

### Basic Options
- `-f, --format <FORMAT>` - 指定输入文件格式
- `-o, --output <OUTPUT>` - 选择输出格式 (cli, json, yaml)
- **目录比较** - 提供目录时自动递归处理

**Note:** For ML models (PyTorch/Safetensors), comprehensive analysis including statistics runs automatically

### Advanced Options
- `--path <PATH>` - Filter differences by specific path
- `--ignore-keys-regex <REGEX>` - Ignore keys matching regex pattern
- `--epsilon <FLOAT>` - Set tolerance for float comparisons
- `--array-id-key <KEY>` - Specify key for array element identification
- `--sort-by-change-magnitude` - Sort by change magnitude

## Examples

### Basic Tensor Comparison (Automatic)
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

### JSON Output for Automation
```bash
$ diffai baseline.safetensors improved.safetensors --output json
{
  "anomaly_detection": {"type": "none", "severity": "none"},
  "architecture_comparison": {"type1": "feedforward", "type2": "feedforward"},
  "deployment_readiness": {"readiness": 0.92, "strategy": "blue_green"},
  "quantization_analysis": {"compression": "0.0%", "speedup": "1.8x"},
  "regression_test": {"passed": true, "degradation": "-2.5%"}
  // ... plus 25+ additional analysis features
}
```

### Scientific Data Analysis
```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB File Comparison
```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

## 性能

diffai为大文件和科学计算工作流进行了优化：

- **内存高效**: 对GB+文件进行流式处理
- **高速**: 采用优化张量操作的Rust实现
- **可扩展**: 处理具有数百万/数十亿参数的模型
- **跨平台**: 在Windows、Linux和macOS上无依赖运行

## 贡献

欢迎贡献！请参阅[CONTRIBUTING](CONTRIBUTING.md)获取指导。

### 开发环境设置

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

该项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件。

## 相关项目

- **[diffx](https://github.com/kako-jun/diffx)** - 通用结构化数据diff工具（diffai的姊妹项目）
- **[safetensors](https://github.com/huggingface/safetensors)** - 存储和分发张量的简单安全方式
- **[PyTorch](https://pytorch.org/)** - 机器学习框架
- **[NumPy](https://numpy.org/)** - Python科学计算的基础包

