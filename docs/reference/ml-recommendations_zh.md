# ML推荐系统

> **具有 11 轴评估的 AI/ML 模型分析智能推荐系统**

ML 推荐系统基于模型分析结果提供可操作的洞察，帮助用户理解在检测到差异时应采取的行动。

## Overview

When comparing ML models (PyTorch, Safetensors), diffai automatically generates intelligent recommendations based on:

- **11 evaluation axes** covering all aspects of ML workflows
- **3 priority levels** (CRITICAL, WARNING, RECOMMENDATIONS)
- **33 predefined natural English messages** with dynamic value embedding
- **Industry best practices** for thresholds and actions

## Evaluation Matrix

### 11 Evaluation Axes

1. **Performance Degradation** - Model accuracy, inference speed, memory usage
2. **Overfitting & Generalization** - Training/validation gaps, generalization ability
3. **Reproducibility & Experiment Management** - Seed consistency, deterministic operations
4. **Production Deployment Risk** - Stability, breaking changes, deployment readiness
5. **Data Drift & Distribution Shift** - Input distribution changes, semantic drift
6. **Computational Efficiency & Cost** - Training cost, GPU usage, parameter efficiency
7. **Model Interpretability** - Attention patterns, feature importance stability
8. **Compatibility & Integration** - ONNX export, quantization, API changes
9. **Security & Privacy** - Data memorization risks, model size increases
10. **MLOps Workflow** - CI/CD integration, model management, convergence
11. **Fine-tuning Specific** - Catastrophic forgetting, transfer learning effectiveness

### 3 Priority Levels

#### [CRITICAL] - Immediate Action Required
- **Performance degradation**: >10% with critical severity
- **Inference speed**: >3.0x slower
- **Memory usage**: >1000MB increase
- **Overfitting risk**: >90%
- **Reproducibility**: <50% score
- **Gradient issues**: Exploding/dead gradients
- **Deployment**: Blocked status

#### [WARNING] - Planned Action Needed
- **Performance degradation**: >5% drop
- **Inference speed**: >1.5x slower
- **Memory usage**: >500MB increase
- **Overfitting risk**: >70%
- **Reproducibility**: <80% score
- **Deployment readiness**: <60%

#### [RECOMMENDATIONS] - Improvement Suggested
- **Performance degradation**: >2% change
- **Inference speed**: >1.2x slower
- **Memory usage**: >200MB increase
- **Overfitting risk**: >50%
- **Reproducibility**: <95% score
- **Parameter efficiency**: <80%

## Example Output

### Critical Issues
```
[CRITICAL]
• Model performance severely degraded by 15.2%. Stop deployment and investigate root cause.
• Inference speed critically degraded (3.2x slower). Identify and fix bottlenecks immediately.
• Memory usage increased critically (+1200MB). Risk of GPU memory exhaustion.
```

### Warning Level
```
[WARNING]
• Performance regression detected (7.3% drop). Run comprehensive validation before proceeding.
• Significant memory increase (+750MB). Consider model quantization or pruning.
• High overfitting risk (80%). Implement early stopping or increase regularization.
```

### Recommendations
```
[RECOMMENDATIONS]
• Minor performance change (3.1%). Monitor metrics and validate on holdout set.
• Inference speed moderately affected (1.3x slower). Consider optimization opportunities.
• Parameter efficiency could be improved (75%). Consider optimization techniques.
```

## Threshold Configuration

### Industry-Based Thresholds

The thresholds are set based on AI/ML industry best practices:

| Metric | CRITICAL | WARNING | RECOMMENDATIONS |
|--------|----------|---------|-----------------|
| **Performance Drop** | >10% | >5% | >2% |
| **Inference Speed** | >3.0x | >1.5x | >1.2x |
| **Memory Increase** | >1000MB | >500MB | >200MB |
| **Overfitting Risk** | >90% | >70% | >50% |
| **Reproducibility** | <50% | <80% | <95% |
| **Parameter Efficiency** | <30% | <60% | <80% |
| **Attention Consistency** | <40% | <70% | <85% |
| **Semantic Drift** | >80% | >40% | >20% |
| **Precision Loss** | >10% | >3% | >1% |
| **Parameter Updates** | >80% | >50% | >30% |
| **Deployment Readiness** | hold | <60% | <80% |

## Output Format

### CLI Only
Recommendations are displayed **only in CLI output format** to maintain the purity of structured data formats (JSON/YAML).

### Message Structure
Each recommendation follows the pattern:
```
[LEVEL]
• Problem description with specific values. Recommended action with technical details.
```

### Dynamic Value Embedding
Messages include contextual information:
- **Percentages**: Performance changes, efficiency ratios
- **Absolute values**: Memory usage, parameter counts
- **Multipliers**: Speed ratios, compression factors
- **Counts**: Affected layers, anomaly instances

## Integration with ML Workflows

### CI/CD Integration
```bash
# Exit codes reflect highest priority level
diffai baseline.safetensors candidate.safetensors
echo $?  # 0=no issues, 1=recommendations, 2=warnings, 3=critical
```

### MLOps Pipeline
```bash
# JSON output for automated processing
diffai model_v1.pt model_v2.pt --output json | jq '.recommendations[]'
```

### Development Workflow
```bash
# Regular model comparison with recommendations
diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt
# Get immediate feedback on model development progress
```

## Technical Implementation

### Message Predefinition
All 33 message templates are predefined in natural English to ensure:
- **Consistency**: Uniform message structure and tone
- **Clarity**: Professional, actionable language
- **Completeness**: Problem description + solution guidance

### Evaluation Order
1. **Analysis results collection** from model comparison
2. **Threshold evaluation** across all 11 axes
3. **Priority level determination** (CRITICAL > WARNING > RECOMMENDATIONS)
4. **Message generation** with dynamic value embedding
5. **Output formatting** with colored severity indicators

### Performance Considerations
- **Zero overhead** when no recommendations are generated
- **Minimal processing** for threshold evaluation
- **Efficient string formatting** for message generation
- **Limited output** to top 5 recommendations for readability

## Best Practices

### For Users
1. **Review CRITICAL issues immediately** - These indicate blocking problems
2. **Plan for WARNING issues** - Address in next development cycle
3. **Consider RECOMMENDATIONS** - Implement when resources allow
4. **Monitor trends** - Track recommendation patterns over time

### For Automation
1. **Set CI/CD thresholds** based on your risk tolerance
2. **Log recommendations** for trend analysis
3. **Integrate with alerting** systems for CRITICAL issues
4. **Use structured output** for programmatic processing

## Customization

### Threshold Adjustment
Currently, thresholds are hardcoded based on industry best practices. Future versions may support:
- Configuration files for custom thresholds
- Domain-specific threshold profiles
- Adaptive thresholds based on model characteristics

### Message Customization
The system uses predefined English messages. Future enhancements may include:
- Multi-language support
- Custom message templates
- Organization-specific terminology