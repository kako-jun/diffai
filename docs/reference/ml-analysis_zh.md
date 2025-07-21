# ML分析功能（35项功能）

diffai机器学习分析功能的综合指南：专为模型比较和分析而设计。

## 概述

diffai提供35个专门为机器学习模型比较和分析设计的特殊分析功能。这些功能有助于研究开发、MLOps和部署工作流程。

## 自动综合分析（v0.3.4+）

### 一体化ML分析
diffai自动为PyTorch和Safetensors文件提供综合分析。无需标志，30多个分析功能默认运行。

### 1. 张量统计分析
提供详细的张量统计。

### 2. `--quantization-analysis` 量化分析
分析量化效果和效率。

### 3. `--sort-by-change-magnitude` 变化幅度排序
按变化幅度排序差异。

### 4. `--show-layer-impact` 层影响分析
分析层级影响。

# 综合模型分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 用于自动化的JSON输出
diffai model1.safetensors model2.safetensors --output json

## 功能选择指南

### 5. `--architecture-comparison` 架构比较
分析架构差异。

### 6. `--memory-analysis` 内存分析
分析内存使用。

### 7. `--anomaly-detection` 异常检测
检测异常模式。

### 8. `--change-summary` 变更摘要
生成详细的变更摘要。

### 9. `--convergence-analysis` 收敛分析
分析收敛模式。

### 10. `--gradient-analysis` 梯度分析
分析梯度信息。

### 11. `--similarity-matrix` 相似性矩阵
生成相似性矩阵。

## 第3阶段功能（现已可用）

上述7个新功能（5-11）代表第3阶段功能，现已完全实现并可使用。

## 设计理念

diffai遵循UNIX哲学：简单、可组合的工具，专注做好一件事。

## 集成示例

### MLflow集成
展示MLflow集成示例。

### CI/CD管道
展示CI/CD管道使用示例。

## 相关内容

- [CLI参考](cli-reference_zh.md) - 完整命令参考
- [基本使用指南](../user-guide/basic-usage_zh.md) - diffai入门
- [ML模型比较指南](../user-guide/ml-model-comparison_zh.md) - 高级模型比较技术