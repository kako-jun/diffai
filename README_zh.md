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

# diffai 显示有意义的模型变化
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  gradient_analysis: flow_health=healthy, norm=0.015000, ratio=1.0500
```

## 核心特性

- **AI/ML 原生支持**: 直接支持 PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz) 和 MATLAB (.mat) 文件
- **张量分析**: 自动计算张量统计（均值、标准差、最小值、最大值、形状、内存使用）
- **28 个 ML 分析功能**: 学习进度、收敛分析、架构比较、部署准备度等
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
# 比较 PyTorch 模型
diffai model_old.pt model_new.pt --stats

# 比较 Safetensors 并进行详细分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors --learning-progress

# 比较 NumPy 数组
diffai data_v1.npy data_v2.npy --stats

# 比较 MATLAB 文件
diffai experiment_v1.mat experiment_v2.mat --stats
```

### 高级 ML 分析

```bash
# 学习进度分析
diffai baseline.safetensors finetuned.safetensors --learning-progress --convergence-analysis

# 架构和部署分析
diffai model_v1.safetensors model_v2.safetensors --architecture-comparison --deployment-readiness

# 性能影响评估
diffai original.pt optimized.pt --quantization-analysis --memory-analysis

# 生成 JSON 格式的详细报告
diffai model_v1.safetensors model_v2.safetensors --generate-report --output json
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

## 28 个高级 ML 分析功能

### 学习和收敛分析（4 个功能）
- `--learning-progress` - 跟踪检查点间的学习进度
- `--convergence-analysis` - 分析收敛稳定性和模式
- `--anomaly-detection` - 检测训练异常（梯度爆炸、消失）
- `--gradient-analysis` - 分析梯度特征和流向

### 架构和性能分析（4 个功能）
- `--architecture-comparison` - 比较模型架构和结构变化
- `--param-efficiency-analysis` - 分析模型间参数效率
- `--memory-analysis` - 分析内存使用和优化机会
- `--inference-speed-estimate` - 估算推理速度和性能特征

### MLOps 和部署支持（7 个功能）
- `--deployment-readiness` - 评估部署准备度和兼容性
- `--regression-test` - 执行自动回归测试
- `--risk-assessment` - 评估部署风险和稳定性
- `--hyperparameter-impact` - 分析超参数对模型变化的影响
- `--learning-rate-analysis` - 分析学习率效果和优化
- `--alert-on-degradation` - 超出阈值时的性能降级警报
- `--performance-impact-estimate` - 估算变化的性能影响

### 实验和文档支持（4 个功能）
- `--generate-report` - 生成全面的分析报告
- `--markdown-output` - 以 Markdown 格式输出用于文档
- `--include-charts` - 在输出中包含图表和可视化
- `--review-friendly` - 生成适合人工审查的输出

### 高级分析功能（6 个功能）
- `--embedding-analysis` - 分析嵌入层变化和语义偏移
- `--similarity-matrix` - 生成模型比较的相似度矩阵
- `--clustering-change` - 分析模型表示中的聚类变化
- `--attention-analysis` - 分析注意力机制模式（Transformer 模型）
- `--head-importance` - 分析注意力头的重要性和专业化
- `--attention-pattern-diff` - 比较模型间的注意力模式

### 其他分析功能（3 个功能）
- `--quantization-analysis` - 分析量化效果和效率
- `--sort-by-change-magnitude` - 按变化幅度排序以便优先处理
- `--change-summary` - 生成详细的变化摘要

## 使用示例

### 训练监控

```bash
# 监控学习进度和收敛
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection
```

### 生产部署

```bash
# 部署前评估
diffai current_prod.safetensors candidate.safetensors \
  --deployment-readiness \
  --risk-assessment \
  --regression-test
```

### 研究分析

```bash
# 模型实验比较
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison \
  --embedding-analysis \
  --generate-report
```

### 量化验证

```bash
# 量化效果评估
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis \
  --memory-analysis \
  --performance-impact-estimate
```

## 输出格式

### CLI 输出（默认）
```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  + new_layer.weight: shape=[64, 64], dtype=f32, params=4096
  - old_layer.bias: shape=[256], dtype=f32, params=256
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### JSON 输出
```bash
$ diffai model1.pt model2.pt --output json --learning-progress
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518},
      {"mean": 0.0017, "std": 0.0647}
    ]
  },
  {
    "LearningProgress": [
      "learning_progress",
      {"trend": "improving", "magnitude": 0.0543, "speed": 0.80}
    ]
  }
]
```

### YAML 输出
```bash
$ diffai config1.yaml config2.yaml --output yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
  - mean: 0.0017
    std: 0.0647
```

## 实际应用场景

### 训练进度监控
```bash
# 比较训练检查点
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress

# 输出分析学习趋势和收敛速度
+ learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

### 模型微调分析
```bash
# 分析微调前后的变化
diffai pretrained_bert.safetensors finetuned_bert.safetensors --stats

# 显示统计变化
~ bert.encoder.layer.11.attention.self.query.weight: mean=-0.0001→0.0023
~ classifier.weight: mean=0.0000→0.0145, std=0.0200→0.0890
```

### 量化影响评估
```bash
# 评估量化对模型的影响
diffai fp32_model.safetensors int8_model.safetensors --quantization-analysis

# 分析压缩效果
quantization_analysis: compression=75.0%, speedup=2.5x, precision_loss=2.0%, suitability=good
```

### 部署准备度检查
```bash
# 检查模型是否准备好部署
diffai production.safetensors candidate.safetensors --deployment-readiness

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

## 开发路线图

### 当前状态（Phase 1-2 完成）
- ✅ **完整的 AI/ML 支持**: 28 个分析功能
- ✅ **多格式支持**: PyTorch、Safetensors、NumPy、MATLAB
- ✅ **科学数据分析**: 统计计算和数值比较
- ✅ **生产就绪**: 全面测试和文档

### 即将推出（Phase 3）
- 🔄 **TensorFlow 支持**: .pb、.h5、SavedModel 格式
- 🔄 **ONNX 支持**: .onnx 格式
- 🔄 **HDF5 支持**: .h5、.hdf5 格式
- 🔄 **性能优化**: 更好的大文件处理

### 未来计划（Phase 4+）
- 🔮 **MLOps 集成**: MLflow、DVC、Kubeflow
- 🔮 **云集成**: AWS S3、Google Cloud Storage
- 🔮 **Web 界面**: 基于浏览器的可视化
- 🔮 **Python 绑定**: 原生 Python 库

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