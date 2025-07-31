# ML分析功能

diffai在比较PyTorch（.pt/.pth）或Safetensors（.safetensors）文件时，自动运行**11种专门的ML分析功能**。无需配置，遵循约定优于配置的原则。

## 自动执行

### ML分析运行时机
- **PyTorch文件（.pt/.pth）**：自动执行全部11项分析
- **Safetensors文件（.safetensors）**：自动执行全部11项分析
- **NumPy/MATLAB文件**：仅基础张量统计
- **其他格式**：通过diffx-core进行标准结构比较

### 零配置
```bash
# 自动运行全部11项ML分析功能
diffai baseline.safetensors finetuned.safetensors

# 无需标志 - diffai检测AI/ML文件并运行全面分析
```

## 11项ML分析功能

### 1. 学习率分析
**目的**：跟踪学习率变化和训练动态

**输出示例**：
```bash
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
```

**检测内容**：
- 学习率参数变化
- 训练调度调整
- 自适应学习率修改
- 趋势分析（递增/递减/稳定）

### 2. 优化器比较
**目的**：比较优化器状态和动量信息

**输出示例**：
```bash
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
```

**检测内容**：
- 优化器类型（Adam、SGD、AdamW、RMSprop）
- 动量缓冲区变化
- Beta参数演化
- 优化器状态一致性

### 3. 损失跟踪
**目的**：分析损失函数演化和收敛模式

**输出示例**：
```bash
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
```

**检测内容**：
- 损失趋势方向
- 改进率
- 收敛指标
- 训练稳定性

### 4. 准确率跟踪
**目的**：监控准确率变化和性能指标

**输出示例**：
```bash
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
```

**检测内容**：
- 准确率/F1/精确率/召回率变化
- 性能趋势分析
- 指标改进率
- 多指标支持

### 5. 模型版本分析
**目的**：识别模型版本控制和检查点信息

**输出示例**：
```bash
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
```

**检测内容**：
- 版本号变化
- 检查点进展
- 轮次/迭代跟踪
- 语义 vs 数字版本控制

### 6. 梯度分析
**目的**：分析梯度流、梯度消失/爆炸和稳定性

**输出示例**：
```bash
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
```

**检测内容**：
- 梯度流健康状态（healthy/warning/critical）
- 梯度消失检测（< 1e-7）
- 梯度爆炸检测（> 100）
- 梯度方差和稳定性
- 使用lawkit内存高效的增量统计

### 7. 量化分析
**目的**：检测混合精度（FP32/FP16/INT8/INT4）和压缩效果

**输出示例**：
```bash
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
```

**检测内容**：
- 混合精度使用（FP32、FP16、INT8、INT4）
- 压缩比
- 精度损失估计
- 模型量化覆盖范围
- 内存效率提升

### 8. 收敛分析
**目的**：学习曲线分析、平台期检测和优化轨迹

**输出示例**：
```bash
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
```

**检测内容**：
- 收敛状态（converging/converged/diverging）
- 学习曲线模式
- 训练中的平台期检测
- 稳定性评分（0.0-1.0）
- 优化轨迹健康状态

### 9. 激活分析
**目的**：分析激活函数使用和分布

**输出示例**：
```bash
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
```

**检测内容**：
- 激活函数类型（ReLU、GELU、Tanh、Sigmoid、Swish）
- 跨层使用分布
- 饱和风险评估
- 死神经元检测
- 现代激活支持

### 10. 注意力分析
**目的**：分析Transformer和注意力机制

**输出示例**：
```bash
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
```

**检测内容**：
- 多头注意力结构
- 注意力模式稳定性
- Transformer组件识别
- 注意力效率评分
- BERT/GPT/T5架构识别

### 11. 集成分析
**目的**：检测和分析集成模型结构

**输出示例**：
```bash
ensemble_analysis: ensemble_detected=false, model_type=feedforward
```

**检测内容**：
- 集成模型检测
- 组件模型计数
- 集成方法（装袋、提升、堆叠）
- 模型多样性评分
- 单一 vs 多模型分类

## 输出格式

### CLI输出（默认）
采用颜色编码和直观符号的人类可读格式。

### JSON输出（MLOps集成）
```bash
diffai model1.safetensors model2.safetensors --output json
```

```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing"
  },
  "gradient_analysis": {
    "flow_health": "healthy",
    "gradient_norm": 0.021069,
    "variance_change": "+15.3%"
  }
  // ... 包含全部11项分析
}
```

### YAML输出（报告）
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## 技术实现

### 内存效率
- **lawkit模式**：使用Welford算法的增量统计
- **流处理**：用于大型模型分析
- **diffx-core基础**：经过验证的diff引擎可靠性

### 错误处理
- **优雅降级**：在未找到特定模式时继续运行
- **健壮解析**：处理各种模型文件结构
- **后备机制**：分析无法完成时的默认值

### 性能优化
- **早期终止**：未检测到数据模式时跳过分析
- **批处理**：高效处理大型模型参数
- **内存限制**：大文件自动优化

## 用例

### 研究与开发
监控训练进度，检测收敛问题，分析架构变化。

### MLOps & CI/CD
自动模型验证，回归检测，性能监控。

### 模型优化
量化分析，内存使用跟踪，压缩评估。

### 实验跟踪  
比较模型变体，跟踪超参数效果，验证改进。

## 参见

- **[快速入门](quick-start_zh.md)** - 5分钟开始
- **[API参考](reference/api-reference_zh.md)** - 在代码中使用
- **[示例](examples/)** - 真实使用示例和输出
- **[技术详情](reference/ml-analysis-detailed_zh.md)** - 实现细节