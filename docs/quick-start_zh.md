# 快速入门 - diffai

5分钟内开始使用diffai。diffai是专门为AI/ML模型设计的差异工具，在比较PyTorch或Safetensors文件时自动提供11种全面的分析功能。

## 安装

```bash
# 从crates.io安装（推荐）
cargo install diffai

# 或从源代码安装
git clone https://github.com/kako-jun/diffai.git
cd diffai && cargo build --release
```

## 基本用法

### 比较ML模型（自动分析）

```bash
# PyTorch模型 - 自动运行11项ML分析
diffai model_old.pt model_new.pt

# Safetensors - 自动运行11项ML分析  
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 输出示例：
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
# ... + 其他6项分析
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### 科学数据（基础分析）

```bash
# NumPy数组 - 仅张量统计
diffai experiment_v1.npy experiment_v2.npy

# MATLAB文件 - 仅张量统计
diffai simulation_v1.mat simulation_v2.mat
```

## 输出格式

### JSON（MLOps集成）
```bash
diffai model1.safetensors model2.safetensors --output json
```

### YAML（人类可读报告）
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

### 详细模式（诊断信息）
```bash
diffai model1.safetensors model2.safetensors --verbose
```

## diffai的独特之处

### 自动ML分析
- **无需配置**：对PyTorch/Safetensors自动运行11项ML分析功能
- **约定优于配置**：遵循lawkit模式，零配置体验
- **基于diffx-core**：经过验证的可靠差异操作

### AI/ML专门设计
- **原生张量支持**：理解PyTorch、Safetensors、NumPy、MATLAB格式
- **统计分析**：自动张量统计（均值、标准差、形状、内存）
- **ML特定洞察**：梯度分析、量化检测、收敛模式

### 传统工具 vs diffai
```bash
# 传统diff
$ diff model_v1.pt model_v2.pt
Binary files model_v1.pt and model_v2.pt differ

# diffai
$ diffai model_v1.pt model_v2.pt
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
gradient_analysis: flow_health=healthy, norm=0.021069
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%
# ... 自动进行全面的ML分析
```

## 常见用例

### 研究与开发
```bash
# 比较微调前后（自动全面分析）
diffai pretrained_model.safetensors finetuned_model.safetensors

# 输出：学习进度、收敛分析、参数演变等
```

### MLOps & CI/CD
```bash
# CI/CD管道中的自动模型验证
diffai production_model.safetensors candidate_model.safetensors --output json

# 通过管道传递给jq或其他工具进行自动处理
diffai baseline.pt improved.pt --output json | jq '.gradient_analysis'
```

### 模型优化
```bash
# 分析量化效果
diffai full_precision.pt quantized.pt
# 自动检测：混合精度、压缩比、精度损失

# 内存使用分析
diffai large_model.safetensors optimized_model.safetensors --verbose
```

## 下一步

- **[示例](examples/)** - 查看真实的diffai输出和用例
- **[ML分析](ml-analysis_zh.md)** - 了解11种自动分析功能  
- **[API参考](reference/api-reference_zh.md)** - 在Rust/Python/JavaScript代码中使用diffai
- **[文件格式](formats_zh.md)** - 支持的AI/ML文件格式详情

## 主要选项

```bash
# 基本比较选项
--epsilon <FLOAT>           # 浮点比较的容差
--output <FORMAT>           # cli（默认）、json、yaml
--verbose                   # 详细的诊断信息
--no-color                  # 禁用彩色输出

# 路径过滤
--path <PATH>               # 按特定路径过滤差异
--ignore-keys-regex <REGEX> # 忽略匹配正则表达式的键

# 内存优化（用于大型模型）
# 内存优化是自动的 - 无需配置
```

diffai遵循**约定优于配置**：当检测到AI/ML文件时，ML分析会自动运行，无需任何设置即可提供全面的洞察。