# ML模型比较指南

本指南涵盖diffai专门用于比较机器学习模型（包括PyTorch和Safetensors文件）的功能。

## 概述

diffai为AI/ML模型格式提供原生支持，允许您在张量级别而不是二进制文件级别比较模型。这使得在训练、微调、量化和部署过程中对模型变化进行有意义的分析成为可能。

## 支持的ML格式

### PyTorch模型
- **`.pt`文件**：PyTorch模型文件（使用Candle集成的pickle格式）
- **`.pth`文件**：PyTorch模型文件（替代扩展名）

### Safetensors模型
- **`.safetensors`文件**：HuggingFace Safetensors格式（推荐）

### 未来支持（第3阶段）
- **`.onnx`文件**：ONNX格式
- **`.h5`文件**：Keras/TensorFlow HDF5格式
- **`.pb`文件**：TensorFlow Protocol Buffer格式

## diffai分析内容

### 张量统计
对于模型中的每个张量，diffai计算并比较：

- **均值**：所有参数的平均值
- **标准差**：参数方差的度量
- **最小值**：最小参数值
- **最大值**：最大参数值
- **形状**：张量的维度和大小
- **数据类型**：精度级别（f16、f32、f64等）

### 结构变化
- **层添加/删除**：架构修改
- **形状变化**：张量维度修改
- **名称变化**：参数名称修改

### 训练进度分析
- **参数漂移**：参数随时间的变化
- **收敛指标**：训练稳定性指标
- **梯度流**：梯度健康性评估

## 基本用法

### 简单模型比较

```bash
# 比较两个Safetensors文件（自动运行综合分析）
diffai model1.safetensors model2.safetensors
```

**输出示例（完整分析）**：
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

### PyTorch模型比较

```bash
# 比较PyTorch模型文件（自动运行完整分析）
diffai model1.pt model2.pt

# 训练检查点比较
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt
```

### 输出格式

#### JSON输出
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

#### YAML输出
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

## 高级分析功能

### 1. 张量统计分析（自动运行）

训练进度的详细监控：

```bash
# 张量统计自动包含
diffai checkpoint_1.safetensors checkpoint_2.safetensors
```

### 2. 量化分析

量化效果分析：

```bash
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis
```

**输出**：
```
quantization_analysis: compression=0.25, speedup=1.8x, precision_loss=minimal
```

### 3. 变化幅度排序

优先显示最大变化：

```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

### 4. 层影响分析

按层分析变化：

```bash
diffai baseline.safetensors modified.safetensors --show-layer-impact
```

### 5. 架构比较

模型结构分析：

```bash
diffai model1.safetensors model2.safetensors --architecture-comparison
```

**输出**：
```
architecture_comparison: transformer->transformer, complexity=similar_complexity, migration=easy
```

### 6. 内存分析

内存使用分析：

```bash
diffai model1.safetensors model2.safetensors --memory-analysis
```

**输出**：
```
memory_analysis: delta=+12.5MB, peak=156.3MB, efficiency=0.85, recommendation=optimal
```

### 7. 异常检测

数值异常检测：

```bash
diffai model1.safetensors model2.safetensors --anomaly-detection
```

**输出**：
```
anomaldy_detection: type=none, severity=none, affected_layers=[], confidence=0.95
```

## 实践工作流

### 训练监控

```bash
# 监控每个epoch后的模型变化
diffai epoch_10.safetensors epoch_11.safetensors

# 关注最大变化
diffai epoch_10.safetensors epoch_11.safetensors --sort-by-change-magnitude
```

### 微调分析

```bash
# 比较基础模型和微调模型
diffai base_model.safetensors finetuned_model.safetensors --show-layer-impact
```

### 量化验证

```bash
# 量化前后质量评估
diffai original.safetensors quantized.safetensors --quantization-analysis
```

### 部署验证

```bash
# 生产部署前验证
diffai current_prod.safetensors candidate.safetensors
```

## 训练期间使用

### 过拟合检测

```bash
# 验证损失开始上升时的模型比较
diffai best_val_model.safetensors current_model.safetensors
```

### 收敛分析

```bash
# 分析连续epoch间的变化
diffai epoch_95.safetensors epoch_100.safetensors --convergence-analysis
```

### 梯度健康检查

```bash
# 检测梯度爆炸/消失
diffai prev_checkpoint.safetensors current_checkpoint.safetensors --gradient-analysis
```

## MLOps集成

### CI/CD流水线

```yaml
- name: Model regression test
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors --output json > model_diff.json
    
- name: Quantization validation
  run: |
    diffai fp32/model.safetensors quantized/model.safetensors --quantization-analysis
```

### 实验管理

```bash
# 实验结果比较
diffai experiments/baseline.safetensors experiments/variant_a.safetensors

# 多个实验候选的比较
for model in experiments/*.safetensors; do
  diffai baseline.safetensors "$model" --output json >> comparison_results.jsonl
done
```

### A/B测试

```bash
# 生产模型和候选模型比较
diffai production_v1.safetensors candidate_v2.safetensors
```

## 故障排除

### 常见问题

#### 张量形状不匹配
```bash
# 结构变化详细分析
diffai old_model.safetensors new_model.safetensors --architecture-comparison
```

#### 数值不稳定性
```bash
# NaN/Inf值检测
diffai stable_model.safetensors unstable_model.safetensors --anomaly-detection
```

#### 内存使用问题
```bash
# 内存效率分析
diffai small_model.safetensors large_model.safetensors --memory-analysis
```

## 优化建议

### 性能
- 对于大模型，明确指定`--format safetensors`
- JSON输出最适合自动化处理
- 层影响分析详细但可能耗时

### 精度
- 浮点比较调整`--epsilon`
- 量化模型设置适当的容差

## 相关内容

- [基本使用指南](basic-usage_zh.md) - diffai基本操作
- [科学数据分析](scientific-data_zh.md) - NumPy和MATLAB文件比较
- [CLI参考](../reference/cli-reference_zh.md) - 完整命令参考
- [ML分析功能](../reference/ml-analysis_zh.md) - 详细分析功能说明