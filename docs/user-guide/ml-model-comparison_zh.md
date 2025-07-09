# 机器学习模型比较指南

本指南介绍了diffai在比较机器学习模型方面的专业功能，包括PyTorch和Safetensors文件。

## 概述

diffai为AI/ML模型格式提供原生支持，让您能够在张量级别比较模型，而不是仅仅作为二进制文件。这使得在训练、微调、量化和部署过程中能够对模型变化进行有意义的分析。

## 支持的ML格式

### PyTorch模型
- **`.pt`文件**：PyTorch模型文件（pickle格式，集成Candle）
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
- **形状**：张量维度
- **数据类型**：参数精度（f32、f64等）
- **总参数数**：张量中的参数数量

### 模型架构
- **参数数量变化**：模型总参数
- **层添加/删除**：新增或删除的层
- **形状变化**：修改的层维度

## 基本模型比较

### 简单比较

```bash
# 比较两个PyTorch模型
diffai model1.pt model2.pt --stats

# 比较Safetensors模型（推荐）
diffai model1.safetensors model2.safetensors --stats

# 自动格式检测
diffai pretrained.safetensors finetuned.safetensors --stats
```

### 示例输出

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### 输出符号

| 符号 | 含义 | 描述 |
|------|------|------|
| `~` | 统计已变化 | 张量值改变但形状保持不变 |
| `+` | 已添加 | 新增张量/层 |
| `-` | 已删除 | 张量/层被删除 |

## 高级选项

### Epsilon容差

使用epsilon来忽略微小的浮点数差异：

```bash
# 忽略小于1e-6的差异
diffai model1.safetensors model2.safetensors --epsilon 1e-6

# 用于量化分析
diffai fp32_model.safetensors int8_model.safetensors --epsilon 0.1
```

### 输出格式

```bash
# JSON输出用于自动化
diffai model1.pt model2.pt --output json

# YAML输出便于阅读
diffai model1.pt model2.pt --output yaml

# 管道输出到文件进行处理
diffai model1.pt model2.pt --output json > changes.json
```

### 过滤结果

```bash
# 专注于特定层
diffai model1.safetensors model2.safetensors --path "classifier"

# 忽略时间戳或元数据
diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(timestamp|_metadata)"
```

## 常见用例

### 1. 微调分析

比较预训练模型与其微调版本：

```bash
diffai pretrained_bert.safetensors finetuned_bert.safetensors --stats

# 预期输出：注意力层的统计变化
# ~ bert.encoder.layer.11.attention.self.query.weight: mean=-0.0001→0.0023
# ~ classifier.weight: mean=0.0000→0.0145, std=0.0200→0.0890
```

**分析**：
- 早期层的小变化（特征提取保持相似）
- 最终层的较大变化（任务特定适应）

### 2. 量化影响评估

比较FP32和量化模型：

```bash
diffai model_fp32.safetensors model_int8.safetensors --epsilon 0.1

# 预期输出：受控的精度损失
# ~ conv1.weight: mean=0.0045→0.0043, std=0.2341→0.2298
# 未发现差异（在epsilon容差范围内）
```

**分析**：
- 小的统计变化表明量化成功
- 大的变化可能表明质量损失

### 3. 训练进度跟踪

比较训练过程中的检查点：

```bash
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt --stats

# 预期输出：收敛模式
# ~ layers.0.weight: mean=-0.0012→0.0034, std=1.2341→0.8907
# ~ layers.1.bias: mean=0.1234→0.0567, std=0.4567→0.3210
```

**分析**：
- 标准差递减表明收敛
- 均值变化显示学习方向

### 4. 架构比较

比较不同的模型架构：

```bash
diffai resnet50.safetensors efficientnet_b0.safetensors --stats

# 预期输出：结构差异
# ~ features.conv1.weight: shape=[64, 3, 7, 7] -> [32, 3, 3, 3]
# + features.mbconv.expand_conv.weight: shape=[96, 32, 1, 1]
# - features.layer4.2.downsample.0.weight: shape=[2048, 1024, 1, 1]
```

**分析**：
- 形状变化表明不同的层大小
- 添加/删除的张量显示架构创新

## 性能优化

### 内存考虑

对于大型模型（>1GB），考虑：

```bash
# 使用流模式（未来功能）
diffai --stream huge_model1.safetensors huge_model2.safetensors

# 专注分析特定部分
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# 使用更高的epsilon进行更快比较
diffai model1.safetensors model2.safetensors --epsilon 1e-3
```

### 速度优化

```bash
# 并行处理（未来功能）
diffai --threads 8 model1.safetensors model2.safetensors

# 跳过统计计算，仅进行形状分析
diffai --shape-only model1.safetensors model2.safetensors
```

## 集成示例

### MLflow集成

```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    # 运行diffai比较
    result = subprocess.run([
        'diffai', model1_path, model2_path, '--output', 'json'
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    # 记录到MLflow
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
        
        # 统计变化类型
        stats_changes = len([d for d in diff_data if 'TensorStatsChanged' in d])
        shape_changes = len([d for d in diff_data if 'TensorShapeChanged' in d])
        
        mlflow.log_metric("stats_changes", stats_changes)
        mlflow.log_metric("shape_changes", shape_changes)
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
            --output json > model_diff.json
            
      - name: 分析变化
        run: |
          # 如果关键层发生变化则失败
          if jq -e '.[] | select(.TensorShapeChanged and (.TensorShapeChanged[0] | contains("classifier")))' model_diff.json; then
            echo "严重：检测到关键层形状变化"
            exit 1
          fi
          
          # 如果参数变化过多则警告
          changes=$(jq length model_diff.json)
          if [ "$changes" -gt 10 ]; then
            echo "警告：检测到许多参数变化：$changes"
          fi
```

### Git预提交钩子

```bash
#!/bin/bash
# .git/hooks/pre-commit

model_files=$(git diff --cached --name-only | grep -E '\.(pt|pth|safetensors)$')

for file in $model_files; do
    if [ -f "$file" ]; then
        echo "分析模型变化：$file"
        
        # 与前一版本比较
        git show HEAD:"$file" > /tmp/old_model
        
        diffai /tmp/old_model "$file" --output json > /tmp/model_diff.json
        
        # 检查重大变化
        shape_changes=$(jq '[.[] | select(.TensorShapeChanged)] | length' /tmp/model_diff.json)
        
        if [ "$shape_changes" -gt 0 ]; then
            echo "警告：在$file中检测到架构变化"
            diffai /tmp/old_model "$file"
            
            read -p "继续提交？(y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        rm -f /tmp/old_model /tmp/model_diff.json
    fi
done
```

## 故障排除

### 常见问题

#### 1. "解析失败"错误

```bash
# 检查文件格式
file model.safetensors

# 验证文件完整性
diffai --check model.safetensors

# 尝试明确格式
diffai --format safetensors model1.safetensors model2.safetensors
```

#### 2. 大模型内存问题

```bash
# 使用更高的epsilon来降低精度
diffai --epsilon 1e-3 large1.safetensors large2.safetensors

# 专注于特定层
diffai --path "tensor.classifier" large1.safetensors large2.safetensors
```

#### 3. 二进制文件错误

```bash
# 确保文件是实际的模型文件，而非损坏的
ls -la model*.safetensors

# 检查文件是否压缩
file model.safetensors

# 如果压缩则尝试解压
gunzip model.safetensors.gz
```

## 最佳实践

### 1. 选择Epsilon值

| 用例 | 推荐Epsilon | 原因 |
|------|------------|------|
| 精确比较 | 无epsilon | 检测所有变化 |
| 训练进度 | 1e-6 | 忽略数值噪声 |
| 量化分析 | 0.01-0.1 | 考虑精度损失 |
| 架构检查 | 1e-3 | 专注结构变化 |

### 2. 输出格式选择

- **CLI**：人工审查和调试
- **JSON**：自动化和脚本编写
- **YAML**：配置文件和文档

### 3. 性能提示

- 使用`--path`专注于相关层的分析
- 设置适当的epsilon值以避免噪声
- 选择比较策略时考虑模型大小

## 高级ML分析功能

diffai提供28个高级机器学习分析功能，用于全面的模型评估：

### 1. 学习进度分析（`--learning-progress`）

分析模型检查点之间的训练进展：

```bash
# 比较训练检查点
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress

# 输出示例：
# + learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

**分析信息：**
- **trend**：`improving`、`degrading`或`stable`
- **magnitude**：变化幅度（0.0-1.0）
- **speed**：收敛速度（0.0-1.0）

**用例：**
- 监控训练进度
- 检测学习平台期
- 优化训练计划

### 2. 收敛分析（`--convergence-analysis`）

评估模型稳定性和收敛状态：

```bash
# 分析检查点之间的收敛
diffai model_before.safetensors model_after.safetensors --convergence-analysis

# 输出示例：
# + convergence_analysis: status=stable, stability=0.0234, action="Continue training"
```

**分析信息：**
- **status**：`converged`、`diverging`、`oscillating`、`stable`
- **stability**：参数变化的方差（较低=更稳定）
- **action**：推荐的下一步

**用例：**
- 确定何时停止训练
- 检测训练不稳定性
- 优化超参数

### 3. 异常检测（`--anomaly-detection`）

检测模型权重中的异常模式：

```bash
# 检测训练异常
diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection

# 输出示例：
# anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

**检测到的异常：**
- **梯度爆炸**：极大的权重值
- **梯度消失**：接近零的梯度
- **权重分布偏移**：异常的统计模式
- **NaN/Inf值**：无效的数值

**用例：**
- 调试训练问题
- 验证模型健康
- 防止部署损坏的模型

### 4. 内存分析（`--memory-analysis`）

分析内存使用和模型效率：

```bash
# 比较模型内存占用
diffai small_model.safetensors large_model.safetensors --memory-analysis

# 输出示例：
# memory_analysis: delta=+2.7MB, gpu_est=4.5MB, efficiency=0.25
```

**分析信息：**
- **delta**：模型间的内存差异
- **gpu_est**：估计的GPU内存需求
- **efficiency**：每MB参数比率

**用例：**
- 为部署约束优化
- 比较架构效率
- 规划硬件需求

### 5. 架构比较（`--architecture-comparison`）

比较模型架构和结构：

```bash
# 比较不同架构
diffai resnet.safetensors transformer.safetensors --architecture-comparison

# 输出示例：
# architecture_comparison: type1=cnn, type2=transformer, differences=15
```

**分析信息：**
- **架构类型**：CNN、RNN、Transformer、MLP等
- **层差异**：添加、删除或修改的层
- **参数分布**：参数如何分配

**用例：**
- 评估架构变化
- 比较模型族
- 设计决策验证

### 6. 多功能分析

结合多个功能进行全面分析：

```bash
# 全面训练分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection \
  --memory-analysis \
  --stats

# 输出示例：
# + learning_progress: trend=improving, magnitude=0.0432, speed=0.75
# + convergence_analysis: status=stable, stability=0.0156
# memory_analysis: delta=+0.1MB, efficiency=0.89
# 张量统计和详细分析...
```

### 7. 生产部署功能

生产环境的基本功能：

```bash
# 生产就绪检查
diffai production.safetensors candidate.safetensors \
  --anomaly-detection \
  --memory-analysis \
  --deployment-readiness

# 回归测试
diffai baseline.safetensors new_version.safetensors \
  --regression-test \
  --alert-on-degradation
```

### 8. 研发功能

研究工作流的高级分析：

```bash
# 超参数影响分析
diffai model_lr_001.safetensors model_lr_0001.safetensors \
  --hyperparameter-impact \
  --learning-rate-analysis

# 架构效率分析
diffai efficient_model.safetensors baseline_model.safetensors \
  --param-efficiency-analysis \
  --architecture-comparison
```

## 功能选择指南

**用于训练监控：**
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress --convergence-analysis --anomaly-detection
```

**用于生产部署：**
```bash
diffai current_prod.safetensors candidate.safetensors \
  --anomaly-detection --memory-analysis --deployment-readiness
```

**用于研究分析：**
```bash
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison --hyperparameter-impact --stats
```

**用于量化验证：**
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --memory-analysis --performance-impact-estimate
```

## 全部28个高级功能

### 学习与收敛分析（4功能）
- `--learning-progress` - 跟踪检查点间的学习进度
- `--convergence-analysis` - 分析收敛稳定性和模式
- `--anomaly-detection` - 检测训练异常（梯度爆炸、梯度消失）
- `--gradient-analysis` - 分析梯度特征和流动

### 架构与性能分析（4功能）
- `--architecture-comparison` - 比较模型架构和结构变化
- `--param-efficiency-analysis` - 分析模型间的参数效率
- `--memory-analysis` - 分析内存使用和优化机会
- `--inference-speed-estimate` - 估计推理速度和性能特征

### MLOps与部署支持（7功能）
- `--deployment-readiness` - 评估部署就绪性和兼容性
- `--regression-test` - 执行自动回归测试
- `--risk-assessment` - 评估部署风险和稳定性
- `--hyperparameter-impact` - 分析超参数对模型变化的影响
- `--learning-rate-analysis` - 分析学习率效果和优化
- `--alert-on-degradation` - 对超过阈值的性能退化发出警报
- `--performance-impact-estimate` - 估计变化的性能影响

### 实验与文档支持（4功能）
- `--generate-report` - 生成全面的分析报告
- `--markdown-output` - 以markdown格式输出结果用于文档
- `--include-charts` - 在输出中包含图表和可视化
- `--review-friendly` - 为人工审查者生成便于审查的输出

### 高级分析功能（6功能）
- `--embedding-analysis` - 分析嵌入层变化和语义漂移
- `--similarity-matrix` - 生成模型比较的相似度矩阵
- `--clustering-change` - 分析模型表示中的聚类变化
- `--attention-analysis` - 分析注意力机制模式（Transformer模型）
- `--head-importance` - 分析注意力头重要性和专门化
- `--attention-pattern-diff` - 比较模型间的注意力模式

### 其他分析功能（3功能）
- `--quantization-analysis` - 分析量化效果和效率
- `--sort-by-change-magnitude` - 按变化幅度排序差异以确定优先级
- `--change-summary` - 生成详细的变化摘要

## 下一步

- [基本用法](basic-usage.md) - 学习基本操作
- [科学数据分析](scientific-data.md) - NumPy和MATLAB文件比较
- [CLI参考](../reference/cli-reference.md) - 完整命令参考