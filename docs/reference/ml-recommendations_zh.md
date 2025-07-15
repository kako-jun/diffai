# ML推荐系统

> **基于11轴评估的AI/ML模型分析智能推荐系统**

ML推荐系统基于模型分析结果提供可操作的洞察，帮助用户理解检测到差异时应采取的行动。

## 概述

在比较ML模型（PyTorch、Safetensors）时，diffai会基于以下因素自动生成智能推荐：

- **11个评估轴** 覆盖ML工作流程的所有方面
- **3个优先级级别** (CRITICAL、WARNING、RECOMMENDATIONS)
- **33个预定义的自然英语消息** 支持动态值嵌入
- **行业最佳实践** 阈值和行动指南

## 评估矩阵

### 11个评估轴

1. **性能退化** - 模型准确率、推理速度、内存使用
2. **过拟合和泛化** - 训练/验证差距、泛化能力
3. **可重现性和实验管理** - 种子一致性、确定性操作
4. **生产部署风险** - 稳定性、破坏性变更、部署准备度
5. **数据漂移和分布偏移** - 输入分布变化、语义漂移
6. **计算效率和成本** - 训练成本、GPU使用、参数效率
7. **模型可解释性** - 注意力模式、特征重要性稳定性
8. **兼容性和集成** - ONNX导出、量化、API变更
9. **安全性和隐私** - 数据记忆化风险、模型大小增加
10. **MLOps工作流** - CI/CD集成、模型管理、收敛
11. **微调特定** - 灾难性遗忘、迁移学习效果

### 3个优先级级别

#### [CRITICAL] - 需要立即行动
- **性能退化**: >10%（严重级别：critical）
- **推理速度**: >3.0倍慢
- **内存使用**: >1000MB增加
- **过拟合风险**: >90%
- **可重现性**: <50%分数
- **梯度问题**: 梯度爆炸/消失
- **部署**: 阻塞状态

#### [WARNING] - 需要计划行动
- **性能退化**: >5%下降
- **推理速度**: >1.5倍慢
- **内存使用**: >500MB增加
- **过拟合风险**: >70%
- **可重现性**: <80%分数
- **部署准备度**: <60%

#### [RECOMMENDATIONS] - 建议改进
- **性能退化**: >2%变化
- **推理速度**: >1.2倍慢
- **内存使用**: >200MB增加
- **过拟合风险**: >50%
- **可重现性**: <95%分数
- **参数效率**: <80%

## 输出示例

### 关键问题
```
[CRITICAL]
• Model performance severely degraded by 15.2%. Stop deployment and investigate root cause.
• Inference speed critically degraded (3.2x slower). Identify and fix bottlenecks immediately.
• Memory usage increased critically (+1200MB). Risk of GPU memory exhaustion.
```

### 警告级别
```
[WARNING]
• Performance regression detected (7.3% drop). Run comprehensive validation before proceeding.
• Significant memory increase (+750MB). Consider model quantization or pruning.
• High overfitting risk (80%). Implement early stopping or increase regularization.
```

### 推荐事项
```
[RECOMMENDATIONS]
• Minor performance change (3.1%). Monitor metrics and validate on holdout set.
• Inference speed moderately affected (1.3x slower). Consider optimization opportunities.
• Parameter efficiency could be improved (75%). Consider optimization techniques.
```

## 阈值配置

### 基于行业的阈值

阈值基于AI/ML行业最佳实践设置：

| 指标 | CRITICAL | WARNING | RECOMMENDATIONS |
|------|----------|---------|-----------------|
| **性能下降** | >10% | >5% | >2% |
| **推理速度** | >3.0倍 | >1.5倍 | >1.2倍 |
| **内存增加** | >1000MB | >500MB | >200MB |
| **过拟合风险** | >90% | >70% | >50% |
| **可重现性** | <50% | <80% | <95% |
| **参数效率** | <30% | <60% | <80% |
| **注意力一致性** | <40% | <70% | <85% |
| **语义漂移** | >80% | >40% | >20% |
| **精度损失** | >10% | >3% | >1% |
| **参数更新** | >80% | >50% | >30% |
| **部署准备度** | hold | <60% | <80% |

## 输出格式

### 仅限CLI
推荐仅在**CLI输出格式**中显示，以保持结构化数据格式（JSON/YAML）的纯净性。

### 消息结构
每个推荐遵循以下模式：
```
[LEVEL]
• 包含具体值的问题描述。包含技术细节的推荐行动。
```

### 动态值嵌入
消息包含上下文信息：
- **百分比**: 性能变化、效率比
- **绝对值**: 内存使用、参数数量
- **倍数**: 速度比、压缩因子
- **计数**: 受影响层数、异常实例

## 与ML工作流程的集成

### CI/CD集成
```bash
# 退出代码反映最高优先级
diffai baseline.safetensors candidate.safetensors
echo $?  # 0=无问题，1=推荐，2=警告，3=关键
```

### MLOps流水线
```bash
# 用于自动处理的JSON输出
diffai model_v1.pt model_v2.pt --output json | jq '.recommendations[]'
```

### 开发工作流
```bash
# 带推荐的常规模型比较
diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt
# 获取模型开发进度的即时反馈
```

## 技术实现

### 消息预定义
所有33个消息模板都以自然英语预定义，确保：
- **一致性**: 统一的消息结构和语调
- **清晰度**: 专业、可操作的语言
- **完整性**: 问题描述 + 解决方案指导

### 评估顺序
1. **分析结果收集** 来自模型比较
2. **阈值评估** 跨越所有11个轴
3. **优先级级别确定** (CRITICAL > WARNING > RECOMMENDATIONS)
4. **消息生成** 带动态值嵌入
5. **输出格式化** 带彩色严重性指示器

### 性能考虑
- **无推荐生成时零开销**
- **阈值评估的最小处理**
- **消息生成的高效字符串格式化**
- **限制输出前5条推荐以提高可读性**

## 最佳实践

### 用户指南
1. **立即检查CRITICAL问题** - 这些表示阻塞问题
2. **计划WARNING问题** - 在下一个开发周期中解决
3. **考虑RECOMMENDATIONS** - 在资源允许时实施
4. **监控趋势** - 跟踪推荐模式随时间的变化

### 自动化指南
1. **设置CI/CD阈值** 基于你的风险承受度
2. **记录推荐** 用于趋势分析
3. **与警报系统集成** 用于CRITICAL问题
4. **使用结构化输出** 用于程序化处理

## 自定义

### 阈值调整
目前，阈值基于行业最佳实践硬编码。未来版本可能支持：
- 自定义阈值的配置文件
- 特定领域的阈值配置文件
- 基于模型特征的自适应阈值

### 消息自定义
系统使用预定义的英语消息。未来增强可能包括：
- 多语言支持
- 自定义消息模板
- 组织特定的术语