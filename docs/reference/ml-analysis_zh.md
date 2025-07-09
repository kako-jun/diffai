# ML分析功能

diffai的28个高级机器学习分析功能完整指南。

## 概述

diffai提供28个专门为机器学习模型比较和分析设计的专业分析功能。这些功能涵盖从研究开发到MLOps和部署的所有方面。

## 学习与收敛分析（4个功能）

### 1. `--learning-progress` 学习进度跟踪
跟踪和分析模型检查点之间的学习进度。

**用法**:
```bash
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress
```

**输出**:
```
+ learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

**分析字段**:
- **trend**: `improving`（改进）、`degrading`（退化）、`stable`（稳定）
- **magnitude**: 变化幅度 (0.0-1.0)
- **speed**: 收敛速度 (0.0-1.0)

### 2. `--convergence-analysis` 收敛分析
评估模型稳定性和收敛状态。

**用法**:
```bash
diffai model_before.safetensors model_after.safetensors --convergence-analysis
```

**输出**:
```
+ convergence_analysis: status=stable, stability=0.0234, action="继续训练"
```

### 3. `--anomaly-detection` 异常检测
检测训练期间的异常模式。

**用法**:
```bash
diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection
```

**输出**:
```
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### 4. `--gradient-analysis` 梯度分析
分析梯度特征和流动。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --gradient-analysis
```

## 架构与性能分析（4个功能）

### 5. `--architecture-comparison` 架构比较
比较模型结构和设计。

**用法**:
```bash
diffai resnet.safetensors transformer.safetensors --architecture-comparison
```

**输出**:
```
architecture_comparison: type1=cnn, type2=transformer, depth=50→24, differences=15
```

### 6. `--param-efficiency-analysis` 参数效率分析
分析模型间的参数效率。

**用法**:
```bash
diffai baseline.safetensors optimized.safetensors --param-efficiency-analysis
```

### 7. `--memory-analysis` 内存分析
分析内存使用情况和优化机会。

**用法**:
```bash
diffai small_model.safetensors large_model.safetensors --memory-analysis
```

### 8. `--inference-speed-estimate` 推理速度估计
估计推理速度和性能特征。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --inference-speed-estimate
```

## MLOps与部署支持（7个功能）

### 9. `--deployment-readiness` 部署就绪性评估
评估部署就绪性和兼容性。

**用法**:
```bash
diffai production.safetensors candidate.safetensors --deployment-readiness
```

**输出**:
```
[GRADUAL] deployment_readiness: readiness=0.75, strategy=gradual, risk=medium
```

### 10. `--regression-test` 回归测试
执行自动回归测试。

**用法**:
```bash
diffai baseline.safetensors new_version.safetensors --regression-test
```

### 11. `--risk-assessment` 风险评估
评估部署风险和稳定性。

**用法**:
```bash
diffai current.safetensors candidate.safetensors --risk-assessment
```

### 12. `--hyperparameter-impact` 超参数影响分析
分析超参数变化的影响。

**用法**:
```bash
diffai model_lr_001.safetensors model_lr_0001.safetensors --hyperparameter-impact
```

### 13. `--learning-rate-analysis` 学习率分析
分析学习率效果和优化。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --learning-rate-analysis
```

### 14. `--alert-on-degradation` 性能退化警报
对超过阈值的性能退化发出警报。

**用法**:
```bash
diffai baseline.safetensors new_model.safetensors --alert-on-degradation
```

### 15. `--performance-impact-estimate` 性能影响估计
估计变化的性能影响。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --performance-impact-estimate
```

## 实验与文档支持（4个功能）

### 16. `--generate-report` 生成综合报告
自动生成综合分析报告。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --generate-report
```

### 17. `--markdown-output` Markdown输出
以markdown格式生成报告。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --markdown-output
```

### 18. `--include-charts` 包含图表和可视化
在输出中包含可视化图表。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --include-charts
```

### 19. `--review-friendly` 审查友好输出
生成优化的人工审查输出。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --review-friendly
```

## 高级分析功能（6个功能）

### 20. `--embedding-analysis` 嵌入分析
分析嵌入层变化和语义漂移。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --embedding-analysis
```

### 21. `--similarity-matrix` 相似性矩阵
生成模型比较的相似性矩阵。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --similarity-matrix
```

### 22. `--clustering-change` 聚类变化分析
分析模型表示中的聚类变化。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --clustering-change
```

### 23. `--attention-analysis` 注意力分析
分析注意力机制模式（Transformer模型）。

**用法**:
```bash
diffai transformer1.safetensors transformer2.safetensors --attention-analysis
```

### 24. `--head-importance` 注意力头重要性
分析注意力头重要性和专业化。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --head-importance
```

### 25. `--attention-pattern-diff` 注意力模式差异
比较模型间的注意力模式。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --attention-pattern-diff
```

## 其他分析功能（3个功能）

### 26. `--quantization-analysis` 量化分析
分析量化效果和效率。

**用法**:
```bash
diffai fp32.safetensors quantized.safetensors --quantization-analysis
```

**输出**:
```
quantization_analysis: compression=75.0%, speedup=2.5x, precision_loss=2.0%, suitability=good
```

### 27. `--sort-by-change-magnitude` 按变化幅度排序
按幅度排序差异以设置优先级。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

### 28. `--change-summary` 变化摘要
生成详细的变化摘要。

**用法**:
```bash
diffai model1.safetensors model2.safetensors --change-summary
```

## 功能组合

### 训练监控
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection
```

### 生产部署
```bash
diffai current_prod.safetensors candidate.safetensors \
  --deployment-readiness \
  --risk-assessment \
  --regression-test
```

### 研究分析
```bash
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison \
  --embedding-analysis \
  --generate-report
```

### 量化验证
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis \
  --memory-analysis \
  --performance-impact-estimate
```

## 相关文档

- [CLI参考](cli-reference_zh.md) - 完整的命令行选项
- [支持的格式](formats_zh.md) - 支持的文件格式
- [输出格式](output-formats_zh.md) - 输出格式规范