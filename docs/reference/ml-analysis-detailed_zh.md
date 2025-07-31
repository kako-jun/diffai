# ML分析功能 - 技术参考

diffai在比较PyTorch（.pt/.pth）或Safetensors（.safetensors）文件时自动执行的11种专门ML分析功能的完整技术文档。

## 概述

diffai遵循**约定优于配置**原则：当检测到AI/ML文件时，全部11种ML分析功能自动运行，无需手动配置即可提供全面洞察。使用lawkit内存高效模式和diffx-core优化技术构建。

**自动触发条件：**
- **PyTorch文件（.pt/.pth）**：执行全部11项分析
- **Safetensors文件（.safetensors）**：执行全部11项分析  
- **NumPy/MATLAB文件**：仅基础张量统计
- **其他格式**：通过diffx-core进行标准结构比较

## 1. 学习率分析

**函数**：`analyze_learning_rate_changes()`  
**目的**：跟踪学习率变化和训练动态

### 检测逻辑
```rust
// 自动搜索模型数据中的学习率字段
let lr_fields = ["learning_rate", "lr", "step_size", "base_lr", "current_lr"];
// 分析优化器学习率参数的变化
```

### 输出格式
```bash
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
```

### JSON输出
```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing",
    "significance": "moderate"
  }
}
```

### 技术实现
- **算法**：带百分比计算的直接值比较
- **内存效率**：lawkit增量处理模式
- **阈值**：>5%变化被认为是重要的
- **错误处理**：找不到LR字段时的优雅回退

## 2. 优化器比较

**函数**：`analyze_optimizer_comparison()`  
**目的**：比较优化器状态和动量信息

### 检测逻辑
```rust
// 搜索优化器状态字典
let optimizer_fields = ["optimizer", "optimizer_state_dict", "optim", "momentum", "adam"];
// 分析动量、beta参数和状态演化
```

### 输出格式
```bash
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
```

### JSON输出
```json
{
  "optimizer_comparison": {
    "type": "Adam",
    "momentum_change": "+2.1%",
    "state_evolution": "stable",
    "beta1": 0.9,
    "beta2": 0.999
  }
}
```

### 技术实现
- **状态跟踪**：比较动量缓冲区和优化器参数
- **内存优化**：大型优化器状态的流式比较
- **支持的优化器**：Adam、SGD、AdamW、RMSprop（自动检测）

## 3. 损失跟踪

**函数**：`analyze_loss_tracking()`  
**目的**：分析损失函数演化和收敛模式

### 检测逻辑
```rust
// 自动检测损失相关字段
let loss_fields = ["loss", "train_loss", "val_loss", "epoch_loss", "step_loss"];
// 分析损失趋势和收敛指标
```

### 输出格式
```bash
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
```

### JSON输出
```json
{
  "loss_tracking": {
    "loss_trend": "decreasing",
    "improvement_rate": "15.2%",
    "convergence_score": 0.89,
    "stability": "high"
  }
}
```

### 技术实现
- **趋势分析**：计算损失值的变化方向
- **收敛评分**：在0.0-1.0范围内评估稳定性
- **改进率**：损失减少的百分比变化

## 4. 准确率跟踪

**函数**：`analyze_accuracy_tracking()`  
**目的**：监控准确率变化和性能指标

### 输出格式
```bash
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
```

### 技术实现
- **指标支持**：准确率、F1、精确率、召回率
- **趋势分析**：性能方向评估
- **多指标处理**：同时跟踪多个性能指标

## 5. 模型版本分析

**函数**：`analyze_model_version()`  
**目的**：识别模型版本控制和检查点信息

### 输出格式
```bash
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
```

### 技术实现
- **版本检测**：识别语义版本控制模式
- **检查点分析**：跟踪轮次/迭代进展
- **演化模式**：增量vs主要变更分类

## 6. 梯度分析

**函数**：`analyze_gradient_flow()`  
**目的**：分析梯度流、梯度消失/爆炸和稳定性

### 输出格式
```bash
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
```

### JSON输出
```json
{
  "gradient_analysis": {
    "flow_health": "healthy",
    "gradient_norm": 0.021069,
    "variance_change": "+15.3%",
    "vanishing_risk": "low",
    "exploding_risk": "low"
  }
}
```

### 技术实现
- **梯度范数**：通过L2范数计算梯度大小
- **健康评估**：healthy/warning/critical分类
- **阈值**：消失（< 1e-7）、爆炸（> 100）
- **lawkit统计**：内存高效的增量计算

## 7. 量化分析

**函数**：`analyze_quantization()`  
**目的**：检测混合精度（FP32/FP16/INT8/INT4）和压缩效果

### 输出格式
```bash
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
```

### JSON输出
```json
{
  "quantization_analysis": {
    "mixed_precision": "FP16+FP32",
    "compression": "12.5%",
    "precision_loss": "1.2%",
    "quantized_layers": 8,
    "bit_widths": [16, 32]
  }
}
```

### 技术实现
- **精度检测**：张量数据类型的自动分析
- **压缩比**：内存使用量比较
- **精度损失**：数值精度变化估算

## 8. 收敛分析

**函数**：`analyze_convergence()`  
**目的**：学习曲线分析、平台期检测、优化轨迹

### 输出格式
```bash
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
```

### 技术实现
- **收敛状态**：converging/converged/diverging分类
- **稳定性评分**：0.0-1.0范围
- **平台期检测**：识别学习停滞

## 9. 激活分析

**函数**：`analyze_activations()`  
**目的**：分析激活函数使用和分布

### 输出格式
```bash
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
```

### 技术实现
- **函数类型**：ReLU、GELU、Tanh、Sigmoid、Swish检测
- **使用分布**：跨层激活函数分析
- **饱和风险**：死神经元检测

## 10. 注意力分析

**函数**：`analyze_attention()`  
**目的**：分析Transformer和注意力机制

### 输出格式
```bash
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
```

### 技术实现
- **架构检测**：BERT/GPT/T5模式识别
- **头部分析**：多头注意力结构
- **效率评分**：注意力机制性能评估

## 11. 集成分析

**函数**：`analyze_ensemble()`  
**目的**：检测和分析集成模型结构

### 输出格式
```bash
ensemble_analysis: ensemble_detected=false, model_type=feedforward
```

### 技术实现
- **集成检测**：多模型结构识别
- **方法分析**：装袋、提升、堆叠
- **多样性评分**：模型间差异评估

## 内存优化和性能

### lawkit模式集成
- **增量统计**：Welford算法
- **流处理**：大规模模型支持
- **内存映射**：高效文件访问

### 并行处理
- **多线程**：独立分析的并行执行
- **批处理**：可配置批大小
- **早期终止**：跳过不适用的分析

## 错误处理

### 鲁棒性模式
- **优雅降级**：继续部分分析
- **回退值**：分析失败时的默认值
- **验证**：输入数据健全性检查

### 调试支持
- **详细日志**：`--verbose`选项
- **错误分类**：具体错误消息
- **性能跟踪**：执行时间和内存使用

## 自定义

### 配置选项
```rust
pub struct AnalysisOptions {
    pub weight_threshold: f64,      // 默认: 0.01
    pub gradient_threshold: f64,    // 默认: 1e-7
    pub convergence_window: usize,  // 默认: 10
    pub enable_all: bool,           // 默认: true
}
```

### 功能控制
- 单个功能启用/禁用
- 自定义阈值设置
- 输出格式自定义

## 参见

- **[ML分析概述](../ml-analysis_zh.md)** - 用户导向概述
- **[API参考](api-reference_zh.md)** - 编程接口
- **[CLI参考](cli-reference_zh.md)** - 命令行使用