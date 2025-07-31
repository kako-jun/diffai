# 文件格式与输出选项

diffai专门支持AI/ML和科学计算文件格式，为PyTorch和Safetensors文件提供自动分析。

## 支持的输入格式

### AI/ML模型格式（自动ML分析）

#### PyTorch模型（.pt、.pth）
- **自动ML分析**：自动运行全部11项功能
- **支持**：模型状态字典、检查点、完整模型
- **集成**：使用Candle进行PyTorch格式解析
- **内存**：高效处理大型模型（GB+文件）

```bash
# 自动全面分析
diffai baseline_model.pt finetuned_model.pt
# 输出：全部11项ML分析功能 + 张量统计
```

#### Safetensors（.safetensors）
- **自动ML分析**：自动运行全部11项功能  
- **支持**：HuggingFace标准格式
- **性能**：快速加载和内存高效解析
- **安全性**：安全的张量存储格式

```bash
# 自动全面分析
diffai model_v1.safetensors model_v2.safetensors
# 输出：全部11项ML分析功能 + 张量统计
```

### 科学数据格式（基础分析）

#### NumPy数组（.npy、.npz）
- **分析**：仅张量统计（无ML特定分析）
- **支持**：单一数组（.npy）和归档（.npz）
- **数据类型**：包括复数在内的所有NumPy数据类型
- **统计**：形状、均值、标准差、最小值、最大值、内存使用

```bash
# 基础张量分析
diffai experiment_data_v1.npy experiment_data_v2.npy
# 输出：形状、统计、数据类型变化
```

#### MATLAB文件（.mat）
- **分析**：仅张量统计（无ML特定分析）
- **支持**：带变量检测的MATLAB矩阵文件
- **数据类型**：double、single、复数、逻辑数组
- **变量**：多变量文件支持

```bash
# 基础张量分析
diffai simulation_v1.mat simulation_v2.mat
# 输出：变量变化、矩阵统计
```

### 格式检测
diffai根据以下内容自动检测文件格式：
1. **文件扩展名**（.pt、.safetensors、.npy、.mat）
2. **魔术字节**和文件头
3. **内容结构**分析

## 输出格式

### CLI输出（默认）
采用颜色编码和直观符号的人类可读格式。

#### 基础符号
| 符号 | 含义 | 颜色 | 描述 |
|------|------|------|------|
| `~` | 修改 | 蓝色 | 值已更改但结构相同 |
| `+` | 添加 | 绿色 | 添加了新元素 |
| `-` | 删除 | 红色 | 删除了元素 |
| `□` | 形状变化 | 黄色 | 张量形状已更改 |

#### ML分析符号
| 符号 | 含义 | 颜色 | 描述 |
|------|------|------|------|
| `◦` | 分析结果 | 青色 | ML分析功能结果 |
| `[CRITICAL]` | 严重警报 | 红色 | 检测到严重问题 |
| `[WARNING]` | 警告 | 黄色 | 需要注意 |

#### CLI输出示例
```bash
diffai model1.safetensors model2.safetensors

learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
+ new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- old_layer.bias: shape=[256], dtype=f32, params=256
```

### JSON输出
用于自动化和MLOps集成的结构化格式。

```bash
diffai model1.safetensors model2.safetensors --output json
```

**JSON结构示例**：
```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing"
  },
  "tensor_changes": [
    {
      "path": "fc1.weight",
      "change_type": "modified",
      "old_stats": {"mean": -0.0002, "std": 0.0514},
      "new_stats": {"mean": -0.0001, "std": 0.0716}
    }
  ]
}
```

### YAML输出
用于报告和文档的人类可读结构化格式。

```bash
diffai model1.safetensors model2.safetensors --output yaml
```

**YAML结构示例**：
```yaml
learning_rate_analysis:
  old: 0.001
  new: 0.0015
  change: "+50.0%"
  trend: "increasing"

gradient_analysis:
  flow_health: "healthy"
  gradient_norm: 0.021069
  variance_change: "+15.3%"
```

### 详细模式
用于调试和分析的全面诊断信息。

```bash
diffai model1.safetensors model2.safetensors --verbose
```

**详细输出包括**：
- 配置诊断
- 文件分析详情
- 性能指标
- 处理上下文
- 内存使用信息

## 格式特定功能

### PyTorch集成
- **状态字典**：自动解析模型状态
- **优化器状态**：分析优化器参数
- **检查点**：完整检查点比较支持
- **自定义对象**：处理自定义PyTorch对象

### Safetensors优势
- **安全性**：无任意代码执行风险
- **性能**：快速内存映射加载
- **跨平台**：跨语言一致性
- **元数据**：丰富的张量元数据支持

### NumPy功能
- **数组**：单一数组文件（.npy）
- **归档**：多数组文件（.npz）
- **复数**：完整复数数据支持
- **内存视图**：高效大数组处理

### MATLAB支持
- **变量**：多变量MATLAB文件
- **数据类型**：所有MATLAB数值类型
- **单元数组**：基础单元数组支持
- **稀疏矩阵**：稀疏矩阵检测

## 性能考虑

### 内存使用
- **流处理**：大文件分块处理
- **内存映射**：Safetensors的高效内存使用
- **垃圾收集**：自动内存清理
- **批处理**：可配置批大小

### 文件大小限制
- **无硬限制**：处理任意大小的文件
- **自动优化**：大文件的内存优化
- **进度报告**：大文件的进度指示
- **分块处理**：将大比较分解为块

### 速度优化
- **早期终止**：未检测到模式时停止
- **并行处理**：有益时使用多线程
- **缓存**：中间结果缓存
- **高效算法**：lawkit增量统计

## 错误处理

### 不支持的格式
diffai专注于AI/ML格式。对于通用结构化数据：
```bash
# 对于JSON、XML、CSV等 - 使用diffx
diffx data1.json data2.json
```

### 损坏文件
- **优雅降级**：可能时进行部分分析
- **错误恢复**：继续使用可用数据
- **清晰消息**：描述性错误消息
- **验证**：处理前的文件格式验证

### 大文件处理
- **内存监控**：自动内存使用跟踪
- **优化**：自动性能优化
- **进度**：长操作的进度报告
- **中断**：优雅处理用户中断

## 集成示例

### Python集成
```python
import diffai
import json

# 比较PyTorch模型
result = diffai.diff_files("model1.pt", "model2.pt", output_format="json")
analysis = json.loads(result)

# 访问ML分析结果
lr_analysis = analysis["learning_rate_analysis"]
gradient_info = analysis["gradient_analysis"]
```

### JavaScript集成
```javascript
const diffai = require('diffai-js');

// 比较Safetensors模型
const result = diffai.diffFiles("model1.safetensors", "model2.safetensors", {
    outputFormat: "json"
});

const analysis = JSON.parse(result);
console.log(analysis.quantization_analysis);
```

### 命令行自动化
```bash
#!/bin/bash
# CI/CD模型验证脚本

RESULT=$(diffai baseline.safetensors candidate.safetensors --output json)
ACCURACY_CHANGE=$(echo "$RESULT" | jq -r '.accuracy_tracking.accuracy_delta')

if [[ "$ACCURACY_CHANGE" == +* ]]; then
    echo "模型改进: $ACCURACY_CHANGE"
    exit 0
else
    echo "模型退化: $ACCURACY_CHANGE"
    exit 1
fi
```

## 参见

- **[快速入门](quick-start_zh.md)** - 快速开始
- **[ML分析](ml-analysis_zh.md)** - 了解11种分析功能
- **[API参考](reference/api-reference_zh.md)** - 编程接口
- **[示例](examples/)** - 真实使用示例