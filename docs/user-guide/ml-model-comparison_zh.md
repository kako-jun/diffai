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

## ML分析功能

### 当前可用功能（v0.2.0）

diffai提供以下已实现的分析功能：

### 1. 统计分析（`--stats`）

为模型比较提供详细的张量统计：

```bash
# 使用详细统计比较训练检查点
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --stats

# 输出示例：
# ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
# ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

**分析信息：**
- **mean**：参数平均值
- **std**：参数标准差
- **min/max**：参数值范围
- **shape**：张量维度
- **dtype**：数据类型精度

**用例：**
- 监控训练期间的参数变化
- 检测模型权重的统计变化
- 验证模型一致性

### 2. 量化分析（`--quantization-analysis`）

分析量化效果和效率：

```bash
# 比较量化与全精度模型
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis

# 输出示例：
# quantization_analysis: compression=0.25, precision_loss=minimal
```

**分析信息：**
- **compression**：模型尺寸缩减比例
- **precision_loss**：精度影响评估
- **efficiency**：性能与质量权衡

**用例：**
- 验证量化质量
- 优化部署大小
- 比较压缩技术

### 3. 变化幅度排序（`--sort-by-change-magnitude`）

按幅度排序差异以便确定优先级：

```bash
# 按重要性排序变化
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude --stats

# 输出显示最大变化在前
```

**用例：**
- 专注于最重要的变化
- 优先安排调试工作
- 识别关键参数变化

### 4. 层影响分析（`--show-layer-impact`）

分析变化的逐层影响：

```bash
# 分析模型层间的影响
diffai baseline.safetensors modified.safetensors --show-layer-impact

# 输出显示每层变化分析
```

**用例：**
- 了解哪些层变化最大
- 指导微调策略
- 分析架构修改

### 5. 组合分析

结合多个功能进行全面分析：

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

**用于训练监控：**
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --stats --sort-by-change-magnitude
```

**用于生产部署：**
```bash
diffai current_prod.safetensors candidate.safetensors \
  --stats --quantization-analysis
```

**用于研究分析：**
```bash
diffai baseline.safetensors experiment.safetensors \
  --stats --show-layer-impact
```

**用于量化验证：**
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --stats
```

## 第3阶段功能（现已可用）

### 第3A阶段：核心分析功能

#### 架构比较（`--architecture-comparison`）
比较模型架构并检测结构变化：

```bash
diffai model1.safetensors model2.safetensors --architecture-comparison

# 输出示例：
# architecture_comparison: transformer->transformer, complexity=similar_complexity, migration=easy
```

**分析信息：**
- **架构类型检测**：Transformer、CNN、RNN或前馈网络
- **层深度比较**：层数和结构变化
- **参数计数分析**：大小比率和复杂性评估
- **迁移难度**：升级复杂性评估
- **兼容性评估**：跨架构兼容性

#### 内存分析（`--memory-analysis`）
分析内存使用和优化机会：

```bash
diffai model1.safetensors model2.safetensors --memory-analysis

# 输出示例：
# memory_analysis: delta=+12.5MB, peak=156.3MB, efficiency=0.85, recommendation=optimal
```

**分析信息：**
- **内存增量**：模型间的确切内存变化
- **峰值使用估算**：包括梯度和激活
- **GPU利用率**：估算GPU内存使用
- **优化机会**：梯度检查点、混合精度
- **内存泄漏检测**：识别异常大的张量

#### 异常检测（`--anomaly-detection`）
检测模型参数中的数值异常：

```bash
diffai model1.safetensors model2.safetensors --anomaly-detection

# 输出示例：
# anomaly_detection: type=none, severity=none, affected_layers=[], confidence=0.95
```

**分析信息：**
- **NaN/Inf检测**：数值不稳定性识别
- **梯度爆炸/消失**：参数变化幅度分析
- **死神经元**：零方差检测
- **根因分析**：建议的原因和解决方案
- **恢复概率**：训练恢复的可能性

#### 变化摘要（`--change-summary`）
生成详细的变化摘要：

```bash
diffai model1.safetensors model2.safetensors --change-summary

# 输出示例：
# change_summary: layers_changed=6, magnitude=0.15, patterns=[weight_updates, bias_adjustments]
```

**分析信息：**
- **变化幅度**：总体参数变化强度
- **变化模式**：检测到的修改类型
- **最大变化层**：按修改强度排名
- **结构vs参数变化**：变化类型分类
- **变化分布**：按层类型和功能

### 第3B阶段：高级分析功能

#### 收敛分析（`--convergence-analysis`）
分析模型参数中的收敛模式：

```bash
diffai model1.safetensors model2.safetensors --convergence-analysis

# 输出示例：
# convergence_analysis: status=converging, stability=0.92, early_stopping=continue
```

**分析信息：**
- **收敛状态**：已收敛、收敛中、停滞或发散
- **参数稳定性**：迭代间参数的稳定程度
- **停滞检测**：训练停滞的识别
- **早停建议**：何时停止训练
- **剩余迭代**：收敛所需的估计迭代数

#### 梯度分析（`--gradient-analysis`）
分析从参数变化估算的梯度信息：

```bash
diffai model1.safetensors model2.safetensors --gradient-analysis

# 输出示例：
# gradient_analysis: flow_health=healthy, norm=0.021, ratio=2.11, clipping=none
```

**分析信息：**
- **梯度流健康度**：总体梯度质量评估
- **梯度范数估算**：参数更新的幅度
- **问题层**：有梯度问题的层
- **裁剪建议**：建议的梯度裁剪值
- **学习率建议**：自适应LR推荐

#### 相似度矩阵（`--similarity-matrix`）
为模型比较生成相似度矩阵：

```bash
diffai model1.safetensors model2.safetensors --similarity-matrix

# 输出示例：
# similarity_matrix: dimensions=(6,6), mean_similarity=0.65, clustering=0.73
```

**分析信息：**
- **层间相似度**：余弦相似度矩阵
- **聚类系数**：相似度的聚类程度
- **异常值检测**：具有异常相似模式的层
- **矩阵质量评分**：总体相似度矩阵质量
- **相关模式**：块对角、层次结构

### 组合分析示例

```bash
# 综合第3阶段分析
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison \
  --memory-analysis \
  --anomaly-detection \
  --change-summary \
  --convergence-analysis \
  --gradient-analysis \
  --similarity-matrix

# MLOps集成的JSON输出
diffai model1.safetensors model2.safetensors \
  --architecture-comparison \
  --memory-analysis \
  --output json
```

### 设计理念
diffai遵循UNIX理念：简单、可组合的工具，专注做好一件事。第3阶段功能是正交的，可以组合使用以形成强大的分析工作流。

## 下一步

- [基本用法](basic-usage_zh.md) - 学习基本操作
- [科学数据分析](scientific-data_zh.md) - NumPy和MATLAB文件比较
- [CLI参考](../reference/cli-reference_zh.md) - 完整命令参考