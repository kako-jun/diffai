# diffai 统一API参考

*diffai-python 和 diffai-js 语言绑定API文档*

## 概述

diffai 提供用于比较AI/ML模型文件和张量的统一API。支持PyTorch（.pt、.pth）、Safetensors、NumPy（.npy、.npz）和MATLAB（.mat）格式，专门针对机器学习用例进行分析。

## 主函数

### `diff(old, new, options)`

比较两个AI/ML模型结构或张量，返回包含ML特定分析的差异。

#### 参数

- `old` (Value): 原始/旧模型或张量数据
- `new` (Value): 新/更新的模型或张量数据
- `options` (DiffOptions, optional): 比较的配置选项

#### 返回值

- `Result<Vec<DiffResult>, Error>`: 包含ML特定变化的差异向量

#### 示例

```rust
use diffai_core::{diff, DiffOptions};
use serde_json::json;

// 模型元数据比较示例
let old = json!({
    "model_name": "bert-base",
    "layers": {
        "encoder.layer.0.attention.self.query.weight": [768, 768],
        "encoder.layer.0.attention.self.query.bias": [768]
    }
});

let new = json!({
    "model_name": "bert-base-finetuned",
    "layers": {
        "encoder.layer.0.attention.self.query.weight": [768, 768],
        "encoder.layer.0.attention.self.query.bias": [768]
    }
});

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    scientific_precision: Some(true),
    ..Default::default()
};

let results = diff(&old, &new, Some(&options))?;
```

## 选项

### DiffOptions 结构体

```rust
pub struct DiffOptions {
    // 数值比较
    pub epsilon: Option<f64>,
    
    // 数组比较
    pub array_id_key: Option<String>,
    
    // 过滤
    pub ignore_keys_regex: Option<String>,
    pub path_filter: Option<String>,
    
    // 输出控制
    pub output_format: Option<OutputFormat>,
    pub show_unchanged: Option<bool>,
    pub show_types: Option<bool>,
    
    // 内存优化
    pub use_memory_optimization: Option<bool>,
    pub batch_size: Option<usize>,
    
    // diffai特定选项
    pub ml_analysis_enabled: Option<bool>,
    pub tensor_comparison_mode: Option<String>,
    pub model_format: Option<String>,
    pub scientific_precision: Option<bool>,
    pub weight_threshold: Option<f64>,
    pub gradient_analysis: Option<bool>,
    pub statistical_summary: Option<bool>,
    pub verbose: Option<bool>,
    pub no_color: Option<bool>,
}
```

### 选项详情

#### ML特定选项

- **`ml_analysis_enabled`**: 启用ML特定分析（权重变化、梯度流等）
  - 默认值: `true`
  
- **`tensor_comparison_mode`**: 张量比较方式
  - 选项: `"element-wise"`, `"statistical"`, `"structural"`
  - 默认值: `"element-wise"`
  
- **`model_format`**: 优化解析的预期模型格式
  - 选项: `"pytorch"`, `"safetensors"`, `"numpy"`, `"matlab"`, `"auto"`
  - 默认值: `"auto"`
  
- **`scientific_precision`**: 数值输出使用科学记数法
  - 默认值: `false`
  
- **`weight_threshold`**: 报告的最小权重变化（有助于过滤噪音）
  - 默认值: `1e-6`
  
- **`gradient_analysis`**: 特别分析梯度相关张量
  - 默认值: `false`
  
- **`statistical_summary`**: 包含张量变化的统计摘要
  - 默认值: `false`

#### 通用选项（从统一API继承）

- **`epsilon`**: 数值比较容差
  - 默认值: `1e-9`（ML用例的更高精度）
  
- **`ignore_keys_regex`**: 要忽略的键（对时间戳、随机种子等有用）
  - 示例: `"^(timestamp|random_seed|training_step)"`
  
- **`show_unchanged`**: 在输出中包含未更改的层
  - 默认值: `false`
  
- **`use_memory_optimization`**: 为大型模型（>1GB）启用
  - 默认值: 文件>100MB时为 `true`

## 结果类型

### DiffResult 枚举（ML增强）

```rust
pub enum DiffResult {
    // 标准差异
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, String, String),
    
    // ML特定差异
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    WeightSignificantChange(String, f64, Statistics),
    LayerAdded(String, LayerInfo),
    LayerRemoved(String, LayerInfo),
    ArchitectureChanged(String, String),
    PrecisionChanged(String, String, String),
}
```

### ML特定结果类型

- **`TensorShapeChanged(path, old_shape, new_shape)`**: 张量维度变化
- **`WeightSignificantChange(path, magnitude, stats)`**: 包含统计信息的重要权重变化
- **`LayerAdded/Removed(path, info)`**: 神经网络层修改
- **`ArchitectureChanged(old_arch, new_arch)`**: 模型架构变化
- **`PrecisionChanged(path, old_precision, new_precision)`**: 数据类型变化（例如float32到float16）

### Statistics 结构体

```rust
pub struct Statistics {
    pub mean_change: f64,
    pub std_dev: f64,
    pub max_change: f64,
    pub min_change: f64,
    pub changed_elements: usize,
    pub total_elements: usize,
}
```

## 语言绑定

### Python

```python
import diffai_python

# 基本模型比较（用户自己加载模型）
results = diffai_python.diff(old_model, new_model)

# 带ML特定选项
results = diffai_python.diff(
    old_model,
    new_model,
    ml_analysis_enabled=True,
    tensor_comparison_mode="statistical",
    weight_threshold=1e-5,
    statistical_summary=True,
    scientific_precision=True
)

# 用户应使用适当的库加载模型（torch等）
# old_model = torch.load("model_epoch_1.pt")
# new_model = torch.load("model_epoch_10.pt")
# results = diffai_python.diff(old_model, new_model)
```

### TypeScript/JavaScript

```typescript
import { diff, DiffOptions } from 'diffai-js';

// 基本用法 - 用户自己加载模型
const results = await diff(oldModel, newModel);

// 带ML特定选项
const options: DiffOptions = {
    diffaiOptions: {
        mlAnalysisEnabled: true,
        tensorComparisonMode: 'statistical',
        scientificPrecision: true
    },
    epsilon: 1e-5,
    showTypes: true
};
const results = await diff(oldModel, newModel, options);
```

## 示例

### 比较PyTorch模型

```rust
use diffai_core::{diff, DiffOptions};

// 用户使用适当的ML库加载模型
let old_model = /* 使用PyTorch/candle/tch库加载 */;
let new_model = /* 使用PyTorch/candle/tch库加载 */;

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    weight_threshold: Some(0.001),
    statistical_summary: Some(true),
    ..Default::default()
};

let results = diff(&old_model, &new_model, Some(&options))?;
```

### 分析训练进度

```rust
let options = DiffOptions {
    tensor_comparison_mode: Some("statistical".to_string()),
    gradient_analysis: Some(true),
    show_unchanged: Some(false),
    ..Default::default()
};

// 比较检查点以查看训练进度
// 用户应使用适当的ML库加载检查点
let checkpoint_1 = /* 使用PyTorch/candle/tch库加载 */;
let checkpoint_10 = /* 使用PyTorch/candle/tch库加载 */;

let results = diff(&checkpoint_1, &checkpoint_10, Some(&options))?;
```

### 比较不同精度

```rust
let options = DiffOptions {
    epsilon: Some(1e-3), // 精度差异的更高容差
    scientific_precision: Some(true),
    ..Default::default()
};

// 用户应使用适当的ML库加载模型
let float32_model = /* 使用PyTorch/candle/tch库加载 */;
let float16_model = /* 使用PyTorch/candle/tch库加载 */;

let results = diff(&float32_model, &float16_model, Some(&options))?;
```

## 性能考虑

- **大型模型**: 为超过1GB的模型启用 `use_memory_optimization`
- **批处理**: 根据可用内存调整 `batch_size`（默认：1000个张量）
- **统计模式**: 使用 `tensor_comparison_mode: "statistical"` 进行大型张量的快速比较
- **过滤**: 使用 `path_filter` 专注于特定层或组件

## 错误处理

库为以下情况提供详细错误：
- 不支持的模型格式
- 损坏的模型文件
- 内存分配失败
- 不兼容的张量形状
- 精度损失警告

## 最佳实践

1. **设置适当的epsilon**: 比较不同精度的模型时使用更高的值（1e-3）
2. **使用权重阈值**: 过滤无关紧要的变化以专注于重要差异
3. **启用统计摘要**: 对于大型模型，统计摘要提供更好的洞察
4. **内存优化**: 对生产模型始终启用
5. **层过滤**: 在调试期间使用 `path_filter` 检查特定层