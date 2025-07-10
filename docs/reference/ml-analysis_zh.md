# ML分析功能（35项功能）

diffai机器学习分析功能综合指南：专为模型比较和分析而设计。

## 概述

diffai提供专门为机器学习模型比较和分析而设计的35项特殊分析功能。这些功能有助于研究开发、MLOps和部署工作流。

## 当前可用功能（v0.2.0）

### 1. `--stats` 统计分析
为模型比较提供详细的张量统计。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --stats
```

**输出**:
```
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

**分析字段**:
- **mean**: 参数平均值
- **std**: 参数标准差
- **min/max**: 参数值范围
- **shape**: 张量维度
- **dtype**: 数据类型精度

**用途**:
- 监控训练期间的参数变化
- 检测模型权重的统计变化
- 验证模型一致性

### 2. `--quantization-analysis` 量化分析
分析量化效果和效率。

**用法**:
```bash
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis
```

**输出**:
```
quantization_analysis: compression=0.25, precision_loss=minimal
```

**分析字段**:
- **compression**: 模型尺寸缩减比例
- **precision_loss**: 精度影响评估
- **efficiency**: 性能与质量权衡

**用途**:
- 验证量化质量
- 优化部署大小
- 比较压缩技术

### 3. `--sort-by-change-magnitude` 变化幅度排序
按幅度排序差异以便确定优先级。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude --stats
```

**输出**: 结果按最大变化在前进行排序

**用途**:
- 专注于最重要的变化
- 优先安排调试工作
- 识别关键参数变化

### 4. `--show-layer-impact` 层影响分析
分析变化的逐层影响。

**用法**:
```bash
diffai baseline.safetensors modified.safetensors --show-layer-impact
```

**输出**: 每层变化分析

**用途**:
- 了解哪些层变化最大
- 指导微调策略
- 分析架构修改

## 组合分析

组合多个功能进行全面分析：

```bash
# 全面模型分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  --stats \
  --quantization-analysis \
  --sort-by-change-magnitude \
  --show-layer-impact

# 自动化的JSON输出
diffai model1.safetensors model2.safetensors \
  --stats --output json
```

## 功能选择指南

**用于训练监控**:
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --stats --sort-by-change-magnitude
```

**用于生产部署**:
```bash
diffai current_prod.safetensors candidate.safetensors \
  --stats --quantization-analysis
```

**用于研究分析**:
```bash
diffai baseline.safetensors experiment.safetensors \
  --stats --show-layer-impact
```

**用于量化验证**:
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --stats
```

## 未来功能（第3阶段）

### 第3A阶段（核心功能）
- `--architecture-comparison` - 比较模型架构和结构变化
- `--memory-analysis` - 分析内存使用和优化机会
- `--anomaly-detection` - 检测模型参数中的数值异常
- `--change-summary` - 生成详细的变化摘要

### 第3B阶段（高级分析）
- `--convergence-analysis` - 分析模型参数中的收敛模式
- `--gradient-analysis` - 分析可用时的梯度信息
- `--similarity-matrix` - 生成模型比较的相似度矩阵

## 设计理念

diffai遵循UNIX哲学：简单、可组合的工具，专注做好一件事。功能是正交的，可以组合使用以形成强大的分析工作流。

## 集成示例

### MLflow集成
```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    result = subprocess.run([
        'diffai', model1_path, model2_path, '--stats', '--output', 'json'
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
```

### CI/CD流水线
```yaml
name: 模型验证
on: [push, pull_request]

jobs:
  model-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: 安装diffai
        run: cargo install diffai
        
      - name: 比较模型
        run: |
          diffai models/baseline.safetensors models/candidate.safetensors \
            --stats --output json > model_diff.json
```

## 相关文档

- [CLI参考](cli-reference_zh.md) - 完整的命令行选项
- [支持的格式](formats_zh.md) - 支持的文件格式
- [输出格式](output-formats_zh.md) - 输出格式规范