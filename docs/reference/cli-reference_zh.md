# CLI参考手册

diffai v0.3.4 完整命令行参考手册 - 具有简化界面的AI/ML专用差分工具

## 命令概要

```
diffai [选项] <输入文件1> <输入文件2>
```

## 描述

diffai是专门为AI/ML工作流设计的差分工具，能够理解模型结构、张量统计和科学数据。它比较PyTorch模型、Safetensors文件、NumPy数组、MATLAB矩阵和结构化数据文件，专注于语义变化而非格式差异。

## 参数

### 必需参数

#### `<输入文件1>`
第一个要比较的输入文件或目录

- **类型**: 文件路径或目录路径
- **格式**: PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)、MATLAB (.mat)、JSON、YAML、TOML、XML、INI、CSV
- **特殊用法**: 使用 `-` 表示从标准输入读取

#### `<输入文件2>`
第二个要比较的输入文件或目录

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
显式指定输入文件格式

- **可能的值**: `json`、`yaml`、`toml`、`ini`、`xml`、`csv`、`safetensors`、`pytorch`、`numpy`、`npz`、`matlab`
- **默认值**: 从文件扩展名自动检测
- **示例**: `--format safetensors`

#### `-o, --output <输出格式>`
选择输出格式

- **可能的值**: `cli`、`json`、`yaml`、`unified`
- **默认值**: `cli`
- **示例**: `--output json`

#### `-r, --recursive`
递归比较目录

- **示例**: `diffai dir1/ dir2/ --recursive`


### 高级选项

#### `--path <路径>`
按特定路径过滤差异

- **示例**: `--path "config.users[0].name"`
- **格式**: 类JSONPath语法

#### `--ignore-keys-regex <正则表达式>`
忽略匹配正则表达式的键

- **示例**: `--ignore-keys-regex "^id$"`
- **格式**: 标准正则表达式模式

#### `--epsilon <浮点数>`
设置浮点数比较的容差

- **示例**: `--epsilon 0.001`
- **默认值**: 机器精度

#### `--array-id-key <键>`
指定用于识别数组元素的键

- **示例**: `--array-id-key "id"`
- **用途**: 用于结构化数组比较

#### `-v, --verbose`
显示详细的处理信息，包括性能指标、配置详情和诊断输出。

- **示例**: `diffai model1.safetensors model2.safetensors --verbose`
- **用途**: 调试分析过程和性能

#### `--no-color`
禁用彩色输出，以提高与脚本、管道或不支持ANSI颜色的终端的兼容性。

- **示例**: `diffai config.json config.new.json --no-color`
- **用途**: 无颜色格式的纯文本输出
- **注意**: 在CI/CD环境和自动化脚本中特别有用

## ML分析功能

### ML分析（PyTorch/Safetensors文件自动执行）

**对于PyTorch（.pt/.pth）和Safetensors（.safetensors）文件，diffai会自动执行包括以下内容的综合分析：**

#### 综合分析套件（30+功能）

- **基础统计**: 每个张量的平均值、标准差、最小/最大值、形状、数据类型
- **量化分析**: 压缩比例、精度损失分析
- **架构比较**: 结构检测、层深度比较、迁移评估
- **内存分析**: 内存增量、峰值使用估算、优化建议
- **异常检测**: NaN/Inf检测、梯度爆炸/消失分析
- **收敛分析**: 参数稳定性、早停建议
- **梯度分析**: 梯度流健康度、范数估算、问题层
- **变化摘要**: 幅度分析、模式、层排名
- **相似度矩阵**: 层间相似度、聚类系数
- **部署准备**: 生产部署安全性评估
- **风险评估**: 变化影响评估
- **性能影响**: 速度和效率分析
- **参数效率**: 优化机会
- **回归测试**: 质量保证验证
- **学习进度**: 训练进度跟踪
- **嵌入分析**: 语义漂移检测
- **注意力分析**: Transformer注意力模式分析
- **统计显著性**: 变化显著性测试
- **迁移学习分析**: 微调效果
- **集成分析**: 多模型比较
- **超参数影响**: 配置变化效果
- **学习率分析**: 优化调度效果
- **等等更多...**

**🎯 无需标志** - 为了获得最佳用户体验，所有分析都会自动执行。

**示例**: 只需运行 `diffai model1.safetensors model2.safetensors` 即可获得综合分析。

## 输出示例

### CLI输出（默认 - 完整分析）

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low, timeline=ready_for_immediate_deployment
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### 综合分析的好处

- **30+分析功能**自动运行
- **无需选择选项** - 默认获得所有洞察
- **相同处理时间** - 无性能损失
- **生产就绪洞察** - 部署准备、风险评估等

### 科学数据分析（自动）

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB文件比较（自动）

```bash
$ diffai simulation_v1.mat simulation_v2.mat
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

- **0**: 成功 - 找到差异或无差异
- **1**: 错误 - 无效参数或文件访问问题
- **2**: 致命错误 - 内部处理失败

## 环境变量

- **DIFFAI_LOG_LEVEL**: 日志级别 (error, warn, info, debug)
- **DIFFAI_MAX_MEMORY**: 最大内存使用量 (MB为单位)

## 性能考虑

- **大文件**: diffai对GB+文件使用流处理
- **内存使用**: 可通过`DIFFAI_MAX_MEMORY`配置内存限制
- **并行处理**: 多文件比较的自动并行化
- **缓存**: 重复比较的智能缓存

## 故障排除

### 常见问题

1. **"Binary files differ"消息**: 使用`--format`指定文件类型
2. **内存不足**: 设置`DIFFAI_MAX_MEMORY`环境变量
3. **处理缓慢**: 分析已针对大模型自动优化
4. **缺少依赖**: 确保Rust工具链已正确安装

### 调试模式

启用调试输出：
```bash
DIFFAI_LOG_LEVEL=debug diffai model1.safetensors model2.safetensors
```

## 相关文档

- [基本用法指南](../user-guide/basic-usage_zh.md)
- [ML模型比较指南](../user-guide/ml-model-comparison_zh.md)
- [科学数据分析指南](../user-guide/scientific-data_zh.md)
- [输出格式参考](output-formats_zh.md)
- [支持格式参考](formats_zh.md)