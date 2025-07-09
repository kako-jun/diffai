# CLI参考手册

diffai v0.2.0 完整命令行参考手册 - AI/ML专用差分工具。

## 命令概要

```
diffai [选项] <输入文件1> <输入文件2>
```

## 描述

diffai是专门为AI/ML工作流设计的差分工具，能够理解模型结构、张量统计和科学数据。它比较PyTorch模型、Safetensors文件、NumPy数组、MATLAB矩阵和结构化数据文件，专注于语义变化而非格式差异。

## 参数

### 必需参数

#### `<输入文件1>`
第一个要比较的输入文件或目录。

- **类型**: 文件路径或目录路径
- **格式**: PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)、MATLAB (.mat)、JSON、YAML、TOML、XML、INI、CSV
- **特殊用法**: 使用 `-` 表示从标准输入读取

#### `<输入文件2>`
第二个要比较的输入文件或目录。

- **类型**: 文件路径或目录路径
- **格式**: 与输入文件1相同
- **特殊用法**: 使用 `-` 表示从标准输入读取

**示例**:
```bash
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
diffai config.json config_new.json
diffai - config.json < input.json
```

## 选项

### 基本选项

#### `-f, --format <格式>`
显式指定输入文件格式。

- **可能的值**: `json`、`yaml`、`toml`、`ini`、`xml`、`csv`、`safetensors`、`pytorch`、`numpy`、`npz`、`matlab`
- **默认值**: 从文件扩展名自动检测
- **示例**: `--format safetensors`

#### `-o, --output <输出格式>`
选择输出格式。

- **可能的值**: `cli`、`json`、`yaml`、`unified`
- **默认值**: `cli`
- **示例**: `--output json`

#### `-r, --recursive`
递归比较目录。

- **示例**: `diffai dir1/ dir2/ --recursive`

#### `--stats`
显示ML模型和科学数据的详细统计信息。

- **示例**: `diffai model.safetensors model2.safetensors --stats`

### 高级选项

#### `--path <路径>`
按特定路径过滤差异。

- **示例**: `--path "config.users[0].name"`
- **格式**: 类JSONPath语法

#### `--ignore-keys-regex <正则表达式>`
忽略匹配正则表达式的键。

- **示例**: `--ignore-keys-regex "^id$"`
- **格式**: 标准正则表达式模式

#### `--epsilon <浮点数>`
设置浮点数比较的容差。

- **示例**: `--epsilon 0.001`
- **默认值**: 机器精度

#### `--array-id-key <键>`
指定用于识别数组元素的键。

- **示例**: `--array-id-key "id"`
- **用途**: 用于结构化数组比较

#### `--sort-by-change-magnitude`
按变化幅度排序差异（仅限ML模型）。

- **示例**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

## ML分析功能

### 学习与收敛分析

#### `--learning-progress`
跟踪训练检查点之间的学习进度。

- **输出**: 学习趋势、参数更新幅度、收敛速度
- **示例**: `diffai checkpoint_epoch_1.safetensors checkpoint_epoch_10.safetensors --learning-progress`

#### `--convergence-analysis`
分析收敛稳定性和模式。

- **输出**: 收敛状态、参数稳定性分析
- **示例**: `diffai baseline.safetensors converged.safetensors --convergence-analysis`

#### `--anomaly-detection`
检测训练异常（梯度爆炸、梯度消失）。

- **输出**: 异常类型、严重性、受影响的层、建议措施
- **示例**: `diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection`

#### `--gradient-analysis`
分析梯度特征和流动。

- **输出**: 梯度流健康状况、范数估计、问题层
- **示例**: `diffai model_before.pt model_after.pt --gradient-analysis`

### 架构与性能分析

#### `--architecture-comparison`
比较模型架构和结构变化。

- **输出**: 架构类型、深度变化、结构差异
- **示例**: `diffai baseline_arch.safetensors improved_arch.safetensors --architecture-comparison`

#### `--param-efficiency-analysis`
分析模型间的参数效率。

- **输出**: 参数效率指标、优化建议
- **示例**: `diffai large_model.pt optimized_model.pt --param-efficiency-analysis`

#### `--memory-analysis`
分析内存使用情况和优化机会。

- **输出**: 内存增量、GPU估计、效率评分
- **示例**: `diffai model_v1.safetensors model_v2.safetensors --memory-analysis`

#### `--inference-speed-estimate`
估计推理速度和性能特征。

- **输出**: 速度比例、FLOPS比例、瓶颈分析
- **示例**: `diffai original.pt optimized.pt --inference-speed-estimate`

### MLOps与部署支持

#### `--deployment-readiness`
评估部署就绪性和兼容性。

- **输出**: 就绪性评分、部署策略、风险评估
- **示例**: `diffai candidate_model.safetensors production_model.safetensors --deployment-readiness`

#### `--regression-test`
执行自动回归测试。

- **输出**: 测试结果、性能比较、回归指标
- **示例**: `diffai baseline.pt new_version.pt --regression-test`

#### `--risk-assessment`
评估部署风险和稳定性。

- **输出**: 风险级别、稳定性指标、缓解建议
- **示例**: `diffai stable_model.safetensors experimental_model.safetensors --risk-assessment`

#### `--hyperparameter-impact`
分析超参数对模型变化的影响。

- **输出**: 超参数敏感性、影响分析
- **示例**: `diffai config_v1.json config_v2.json --hyperparameter-impact`

#### `--learning-rate-analysis`
分析学习率效果和优化。

- **输出**: 学习率有效性、优化模式
- **示例**: `diffai lr_001.safetensors lr_01.safetensors --learning-rate-analysis`

#### `--alert-on-degradation`
对超过阈值的性能退化发出警报。

- **输出**: 退化警报、阈值违反
- **示例**: `diffai production.pt candidate.pt --alert-on-degradation`

#### `--performance-impact-estimate`
估计模型变化的性能影响。

- **输出**: 性能增量、影响估计、优化建议
- **示例**: `diffai baseline.safetensors optimized.safetensors --performance-impact-estimate`

### 实验与文档支持

#### `--generate-report`
生成综合分析报告。

- **输出**: 包含多项指标的详细分析报告
- **示例**: `diffai experiment_baseline.safetensors experiment_result.safetensors --generate-report`

#### `--markdown-output`
以markdown格式输出结果用于文档。

- **输出**: markdown格式的分析结果
- **示例**: `diffai model_v1.pt model_v2.pt --markdown-output`

#### `--include-charts`
在输出中包含图表和可视化。

- **输出**: 图表分析元数据（可视化功能即将推出）
- **示例**: `diffai data_v1.npy data_v2.npy --include-charts`

#### `--review-friendly`
生成对人工审查友好的输出。

- **输出**: 人类可读的分析摘要
- **示例**: `diffai pr_baseline.safetensors pr_changes.safetensors --review-friendly`

### 高级分析功能

#### `--embedding-analysis`
分析嵌入层变化和语义漂移。

- **输出**: 嵌入漂移分析、语义变化检测
- **示例**: `diffai embeddings_v1.safetensors embeddings_v2.safetensors --embedding-analysis`

#### `--similarity-matrix`
生成模型比较的相似性矩阵。

- **输出**: 相似性矩阵、相关性分析
- **示例**: `diffai model_a.pt model_b.pt --similarity-matrix`

#### `--clustering-change`
分析模型表示中的聚类变化。

- **输出**: 聚类分析、表示变化
- **示例**: `diffai representation_v1.safetensors representation_v2.safetensors --clustering-change`

#### `--attention-analysis`
分析注意力机制模式（Transformer模型）。

- **输出**: 注意力模式分析、机制评估
- **示例**: `diffai transformer_v1.safetensors transformer_v2.safetensors --attention-analysis`

#### `--head-importance`
分析注意力头的重要性和专业化。

- **输出**: 头重要性排名、专业化分析
- **示例**: `diffai attention_baseline.pt attention_pruned.pt --head-importance`

#### `--attention-pattern-diff`
比较模型间的注意力模式。

- **输出**: 注意力模式差异、行为变化
- **示例**: `diffai pattern_v1.safetensors pattern_v2.safetensors --attention-pattern-diff`

### 其他分析功能

#### `--quantization-analysis`
分析量化效果和效率。

- **输出**: 压缩比例、加速估计、精度损失
- **示例**: `diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis`

#### `--sort-by-change-magnitude`
按幅度排序差异以设置优先级。

- **输出**: 按幅度排序的差异列表
- **示例**: `diffai model1.pt model2.pt --sort-by-change-magnitude`

#### `--change-summary`
生成详细的变化摘要。

- **输出**: 综合变化分析和摘要
- **示例**: `diffai version_a.safetensors version_b.safetensors --change-summary`

## 输出示例

### CLI输出（默认）

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### 高级分析输出

```bash
$ diffai baseline.safetensors improved.safetensors --deployment-readiness --architecture-comparison
deployment_readiness: readiness=0.92, strategy=blue_green, risk=low, timeline=ready_for_immediate_deployment
architecture_comparison: type1=feedforward, type2=feedforward, depth=3->3, differences=0
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

### 科学数据分析

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy --stats
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB文件比较

```bash
$ diffai simulation_v1.mat simulation_v2.mat --stats
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### JSON输出

```bash
$ diffai model_v1.safetensors model_v2.safetensors --output json
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

### YAML输出

```bash
$ diffai model_v1.safetensors model_v2.safetensors --output yaml
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

## 退出代码

- **0**: 成功 - 发现差异或无差异
- **1**: 错误 - 无效参数或文件访问问题
- **2**: 致命错误 - 内部处理失败

## 环境变量

- **DIFFAI_CONFIG**: 配置文件路径
- **DIFFAI_LOG_LEVEL**: 日志级别（error、warn、info、debug）
- **DIFFAI_MAX_MEMORY**: 最大内存使用量（MB）

## 配置文件

diffai支持TOML格式的配置文件。将配置文件放置在：

- Unix: `~/.config/diffx/config.toml`
- Windows: `%APPDATA%/diffx/config.toml`
- 当前目录: `.diffx.toml`

配置文件示例：
```toml
[diffai]
default_output = "cli"
default_format = "auto"
epsilon = 0.001
sort_by_magnitude = false

[ml_analysis]
enable_all = false
learning_progress = true
convergence_analysis = true
memory_analysis = true
```

## 性能考虑

- **大文件**: diffai对GB+文件使用流式处理
- **内存使用**: 通过 `DIFFAI_MAX_MEMORY` 配置内存限制
- **并行处理**: 多文件比较自动并行化
- **缓存**: 重复比较的智能缓存

## 故障排除

### 常见问题

1. **"二进制文件差异"消息**: 使用 `--format` 指定文件类型
2. **内存不足**: 设置 `DIFFAI_MAX_MEMORY` 环境变量
3. **处理缓慢**: 仅在需要时对大模型使用 `--stats`
4. **缺少依赖**: 确保正确安装Rust工具链

### 调试模式

启用调试输出：
```bash
DIFFAI_LOG_LEVEL=debug diffai model1.safetensors model2.safetensors
```

## 相关文档

- [基本使用指南](../user-guide/basic-usage.md)
- [ML模型比较指南](../user-guide/ml-model-comparison.md)
- [科学数据分析指南](../user-guide/scientific-data.md)
- [输出格式参考](output-formats_zh.md)
- [支持的格式参考](formats_zh.md)