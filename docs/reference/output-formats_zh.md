# 输出格式

diffai支持的输出格式及其规范。

## 概述

diffai支持四种不同的输出格式以满足各种用例：

1. **CLI** - 人类可读格式（默认）
2. **JSON** - 机器处理和自动化
3. **YAML** - 配置文件和人类可读的结构化数据
4. **Unified** - Git集成和传统diff格式

## CLI输出格式

### 概述
默认使用的人类可读彩色输出格式。

### 特性
- **彩色显示**: 按变化类型色彩编码
- **符号显示**: 直观的符号（`+`、`-`、`~`、`□`）
- **层次显示**: 嵌套结构表示
- **ML专用符号**: 针对ML分析结果的专门显示

### 用法
```bash
diffai model1.safetensors model2.safetensors --output cli
# 或
diffai model1.safetensors model2.safetensors  # 默认
```

### 示例输出
```
~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
+ new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- old_layer.bias: shape=[256], dtype=f32, params=256
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### 符号含义

#### 基本符号
| 符号 | 含义 | 颜色 | 描述 |
|------|------|------|------|
| `~` | 修改 | 蓝色 | 值改变但结构相同 |
| `+` | 添加 | 绿色 | 新增元素 |
| `-` | 删除 | 红色 | 删除元素 |
| `□` | 形状改变 | 黄色 | 张量形状改变 |

#### ML分析符号
| 符号 | 含义 | 颜色 | 描述 |
|------|------|------|------|
| `◦` | 分析结果 | 青色 | ML分析功能结果 |
| `[CRITICAL]` | 严重警报 | 红色 | 检测到严重问题 |
| `[WARNING]` | 警告 | 黄色 | 需要注意 |
| `[ALERT]` | 警报 | 红色 | 超过阈值 |
| `[HIGH]` | 高优先级 | 红色 | 高优先级问题 |
| `[HOLD]` | 暂停部署 | 红色 | 建议暂停部署 |
| `[GRADUAL]` | 渐进部署 | 黄色 | 建议渐进部署 |

## JSON输出格式

### 概述
适合机器处理和自动化的结构化数据格式。

### 特性
- **结构化数据**: 易于程序化解析
- **类型信息**: 保留完整的类型信息
- **API集成**: 易于与RESTful API集成
- **自动化**: 最适合CI/CD和脚本编写

### 用法
```bash
diffai model1.safetensors model2.safetensors --output json
```

### 示例输出
```json
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518, "shape": [64], "dtype": "f32", "total_params": 64},
      {"mean": 0.0017, "std": 0.0647, "shape": [64], "dtype": "f32", "total_params": 64}
    ]
  },
  {
    "TensorAdded": [
      "new_layer.weight",
      {"mean": 0.0123, "std": 0.0456, "shape": [64, 64], "dtype": "f32", "total_params": 4096}
    ]
  },
  {
    "AnomalyDetection": [
      "anomaly_detection",
      {
        "anomaly_type": "gradient_explosion",
        "severity": "critical",
        "affected_layers": ["layer1", "layer2"],
        "recommended_action": "降低学习率"
      }
    ]
  }
]
```

### 数据结构类型

#### 基本变化类型
- **Added**: `["key", value]` - 添加的元素
- **Removed**: `["key", value]` - 删除的元素
- **Modified**: `["key", old_value, new_value]` - 修改的元素
- **TypeChanged**: `["key", old_type, new_type]` - 类型改变

#### ML专用类型
- **TensorStatsChanged**: 张量统计改变
- **TensorShapeChanged**: 张量形状改变
- **TensorAdded**: 张量添加
- **TensorRemoved**: 张量删除
- **LearningProgress**: 学习进度分析
- **ConvergenceAnalysis**: 收敛分析
- **AnomalyDetection**: 异常检测
- **MemoryAnalysis**: 内存分析
- **DeploymentReadiness**: 部署就绪性
- **RiskAssessment**: 风险评估

## YAML输出格式

### 概述
适合配置文件和文档的人类可读结构化数据格式。

### 特性
- **可读性**: 人类友好的格式
- **注释**: 支持注释
- **层次结构**: 清晰的层次表示
- **配置文件**: 可以用作配置文件

### 用法
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

### 示例输出
```yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
    shape: [64]
    dtype: f32
    total_params: 64
  - mean: 0.0017
    std: 0.0647
    shape: [64]
    dtype: f32
    total_params: 64

- AnomalyDetection:
  - anomaly_detection
  - anomaly_type: gradient_explosion
    severity: critical
    affected_layers:
      - layer1
      - layer2
    recommended_action: 降低学习率
```

## Unified输出格式

### 概述
Git集成和与传统diff工具兼容。

### 特性
- **Git集成**: 与git diff兼容
- **合并工具**: 与三路合并工具协作
- **传统格式**: 与现有diff工具兼容
- **补丁应用**: 可以用git apply应用

### 用法
```bash
diffai config1.json config2.json --output unified
```

### 示例输出
```diff
--- config1.json
+++ config2.json
@@ -1,5 +1,6 @@
 {
   "model": {
-    "layers": 12,
+    "layers": 24,
     "hidden_size": 768
   },
+  "optimizer": "adam"
 }
```

## 输出格式选择指南

### 按用例

| 用例 | 推荐格式 | 原因 |
|------|----------|------|
| 人工审查 | CLI | 彩色、直观符号 |
| 自动化/脚本 | JSON | 机器可处理 |
| 配置文件 | YAML | 可读性、注释支持 |
| 文档 | YAML | 人类友好 |
| Git集成 | Unified | 现有工具兼容性 |
| API集成 | JSON | 标准数据交换格式 |
| 报告生成 | JSON | 结构化数据处理 |

### 按环境

| 环境 | 推荐格式 | 原因 |
|------|----------|------|
| 交互式命令行 | CLI | 即时理解 |
| CI/CD管道 | JSON | 自动化检查 |
| 开发环境 | CLI | 快速调试 |
| 生产环境 | JSON | 日志和监控 |
| 研究/实验 | YAML | 结果文档 |

## 高级用法示例

### 管道处理
```bash
# 输出JSON并用jq处理
diffai model1.safetensors model2.safetensors --output json | \
  jq '.[] | select(.TensorStatsChanged)'

# 输出YAML并保存到文件
diffai config1.yaml config2.yaml --output yaml > changes.yaml
```

### 条件逻辑
```bash
# 检查是否存在变化
if diffai model1.safetensors model2.safetensors --output json | jq -e 'length > 0'; then
  echo "检测到变化"
fi
```

### 多格式生成
```bash
# 生成人类和机器可读格式
diffai model1.safetensors model2.safetensors > human_readable.txt
diffai model1.safetensors model2.safetensors --output json > machine_readable.json
```

## 配置和自定义

### 配置文件
```toml
[output]
default = "cli"
json_pretty = true
yaml_flow = false
cli_colors = true
unified_context = 3

[colors]
added = "green"
removed = "red"
modified = "blue"
warning = "yellow"
error = "red"
```

### 环境变量
```bash
export DIFFAI_OUTPUT_FORMAT="json"
export DIFFAI_CLI_COLORS="true"
export DIFFAI_JSON_PRETTY="true"
export DIFFAI_YAML_FLOW="false"
```

## 相关文档

- [CLI参考](cli-reference_zh.md) - 完整的命令行选项
- [支持的格式](formats_zh.md) - 输入文件格式
- [ML分析功能](ml-analysis_zh.md) - 机器学习分析功能