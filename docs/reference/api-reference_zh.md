# API参考 - diffai-core

提供AI/ML模型差异功能的`diffai-core` Rust crate的完整API文档。

## 概述

`diffai-core` crate是diffai生态系统的核心，为AI/ML模型文件和张量提供专门的差异操作。它可以嵌入到其他Rust应用程序中，添加ML特定的比较功能。

**统一API设计**：核心API公开了一个单一主函数`diff()`，用于所有比较操作，并具有自动全面分析。

## 安装

将`diffai-core`添加到您的`Cargo.toml`：

```toml
[dependencies]
diffai-core = "0.2.0"
```

### 功能标志

```toml
[dependencies]
diffai-core = { version = "0.2.0", features = ["all-formats"] }
```

可用功能：
- `pytorch`（默认） - PyTorch模型支持
- `safetensors`（默认） - Safetensors格式支持  
- `numpy`（默认） - NumPy数组支持
- `matlab` - MATLAB文件支持
- `all-formats` - 启用所有格式解析器

## 公共API

### 核心类型

#### `DiffResult`

表示两个AI/ML模型或张量之间的单个差异。

```rust
#[derive(Debug, PartialEq, Serialize)]
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

### 核心函数

#### `diff()`

计算两个AI/ML模型或张量之间差异的主要函数。这是所有比较操作的统一API入口点。

```rust
pub fn diff(
    old: &Value,
    new: &Value,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>, Error>
```

**参数：**
- `old`：原始/基线模型或张量数据
- `new`：新的/更新的模型或张量数据
- `options`：比较的可选配置选项

**返回值：** `Result<Vec<DiffResult>, Error>`表示找到的所有差异

#### DiffOptions结构体

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

**示例：**
```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let old_model = json!({
        "model_name": "bert-base",
        "layers": {
            "encoder.layer.0.attention.self.query.weight": [768, 768],
            "encoder.layer.0.attention.self.query.bias": [768]
        }
    });
    
    let new_model = json!({
        "model_name": "bert-base-finetuned",
        "layers": {
            "encoder.layer.0.attention.self.query.weight": [768, 768],
            "encoder.layer.0.attention.self.query.bias": [768]
        }
    });
    
    let options = DiffOptions {
        ml_analysis_enabled: Some(true),
        weight_threshold: Some(0.001),
        statistical_summary: Some(true),
        ..Default::default()
    };
    
    let differences = diff(&old_model, &new_model, Some(&options))?;
    
    for diff_result in differences {
        match diff_result {
            DiffResult::WeightSignificantChange(path, magnitude, stats) => {
                println!("在{}发现重要权重变化: magnitude={}", path, magnitude);
                println!("统计: mean_change={}, std_dev={}", stats.mean_change, stats.std_dev);
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

## 高级用法

### 自定义比较逻辑

#### ML特定分析

启用机器学习特定分析功能：

```rust
use diffai_core::{diff, DiffOptions};
use serde_json::json;

let old_checkpoint = json!({
    "epoch": 1,
    "model_state_dict": { /* 模型权重 */ },
    "optimizer_state_dict": { /* 优化器状态 */ }
});

let new_checkpoint = json!({
    "epoch": 10,
    "model_state_dict": { /* 更新的权重 */ },
    "optimizer_state_dict": { /* 更新的状态 */ }
});

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    tensor_comparison_mode: Some("statistical".to_string()),
    gradient_analysis: Some(true),
    statistical_summary: Some(true),
    ..Default::default()
};

let differences = diff(&old_checkpoint, &new_checkpoint, Some(&options))?;
```

#### 精度感知比较

处理不同数值精度的模型：

```rust
let options = DiffOptions {
    epsilon: Some(1e-3), // 精度差异的更高容差
    scientific_precision: Some(true),
    weight_threshold: Some(1e-4),
    ..Default::default()
};

let differences = diff(&float32_model, &float16_model, Some(&options))?;
```

### 使用不同模型格式

#### 加载和比较模型

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use std::fs;

// 用户应使用适当的ML库加载模型
fn compare_pytorch_models(
    model1_path: &str,
    model2_path: &str,
    options: Option<&DiffOptions>
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    // 示例：用户将使用candle、tch或其他PyTorch绑定
    // 将实际模型数据加载到serde_json::Value中
    
    // 这只是占位符 - 实际实现将使用ML库
    let old_content = fs::read_to_string(model1_path)?;
    let new_content = fs::read_to_string(model2_path)?;
    
    let old: Value = serde_json::from_str(&old_content)?;
    let new: Value = serde_json::from_str(&new_content)?;
    
    Ok(diff(&old, &new, options)?)
}
```

### 集成模式

#### 训练进度分析

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;

struct TrainingAnalyzer {
    pub weight_changes: Vec<(String, f64)>,
    pub architecture_changes: Vec<String>,
    pub precision_changes: Vec<(String, String, String)>,
}

impl TrainingAnalyzer {
    pub fn analyze_checkpoints(
        &mut self,
        checkpoint1: &Value,
        checkpoint2: &Value
    ) -> Result<(), Box<dyn std::error::Error>> {
        let options = DiffOptions {
            ml_analysis_enabled: Some(true),
            tensor_comparison_mode: Some("statistical".to_string()),
            statistical_summary: Some(true),
            ..Default::default()
        };
        
        let differences = diff(checkpoint1, checkpoint2, Some(&options))?;
        
        for diff_result in differences {
            match diff_result {
                DiffResult::WeightSignificantChange(path, magnitude, _) => {
                    self.weight_changes.push((path, magnitude));
                }
                DiffResult::ArchitectureChanged(old_arch, new_arch) => {
                    self.architecture_changes.push(
                        format!("{} -> {}", old_arch, new_arch)
                    );
                }
                DiffResult::PrecisionChanged(path, old_prec, new_prec) => {
                    self.precision_changes.push((path, old_prec, new_prec));
                }
                _ => {}
            }
        }
        
        Ok(())
    }
}
```

#### 异步模型比较

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tasks = vec![
        compare_models_async("model_v1.pt", "model_v2.pt"),
        compare_models_async("model_v2.pt", "model_v3.pt"),
    ];
    
    let results = futures::future::try_join_all(tasks).await?;
    
    for (i, diffs) in results.into_iter().enumerate() {
        println!("模型对 {}: {} 个差异", i + 1, diffs.len());
    }
    
    Ok(())
}

async fn compare_models_async(
    file1: &str,
    file2: &str
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    let content1 = tokio::fs::read_to_string(file1).await?;
    let content2 = tokio::fs::read_to_string(file2).await?;
    
    let result = tokio::task::spawn_blocking(move || {
        // 在实际使用中，使用ML库解析模型文件
        let old: Value = serde_json::from_str(&content1)?;
        let new: Value = serde_json::from_str(&content2)?;
        
        let options = DiffOptions {
            ml_analysis_enabled: Some(true),
            use_memory_optimization: Some(true),
            ..Default::default()
        };
        
        diff(&old, &new, Some(&options))
    }).await??;
    
    Ok(result)
}
```

## 错误处理

### 错误类型

库使用`anyhow::Error`进行错误处理：

```rust
use diffai_core::{diff, DiffOptions};
use anyhow::Result;

fn handle_model_errors() -> Result<()> {
    // ... 加载模型 ...
    
    match diff(&old_model, &new_model, None) {
        Ok(differences) => {
            println!("发现 {} 个差异", differences.len());
        }
        Err(e) => {
            eprintln!("模型比较错误: {}", e);
            
            // 检查特定错误类型
            if e.to_string().contains("memory") {
                eprintln!("考虑启用内存优化");
            }
        }
    }
    
    Ok(())
}
```

## 性能考虑

### 内存使用

对于大型模型：

```rust
use diffai_core::{diff, DiffOptions, DiffResult};

fn process_large_models(
    old: &Value,
    new: &Value
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    let options = DiffOptions {
        use_memory_optimization: Some(true),
        batch_size: Some(500), // 大张量的小批量
        tensor_comparison_mode: Some("statistical".to_string()),
        ..Default::default()
    };
    
    Ok(diff(old, new, Some(&options))?)
}
```

### 优化技巧

1. **使用内存优化** - 对于>1GB的模型
2. **设置适当的epsilon** - 根据精度要求
3. **使用统计模式** - 更快地比较大张量
4. **过滤路径** - 专注于特定层或组件
5. **调整批大小** - 基于可用内存

## 测试

### 单元测试

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    
    #[test]
    fn test_weight_change_detection() {
        let old = json!({
            "weights": {
                "layer1": [1.0, 2.0, 3.0],
                "layer2": [4.0, 5.0, 6.0]
            }
        });
        
        let new = json!({
            "weights": {
                "layer1": [1.1, 2.1, 3.1],
                "layer2": [4.0, 5.0, 6.0]
            }
        });
        
        let options = DiffOptions {
            ml_analysis_enabled: Some(true),
            weight_threshold: Some(0.05),
            ..Default::default()
        };
        
        let diffs = diff(&old, &new, Some(&options)).unwrap();
        
        // 应该检测到layer1中的重大变化
        assert!(diffs.iter().any(|d| matches!(d, 
            DiffResult::WeightSignificantChange(path, _, _) if path.contains("layer1")
        )));
    }
}
```

## 自动ML分析

**约定优于配置**：diffai-core在检测到PyTorch（.pt/.pth）或Safetensors（.safetensors）文件时，自动运行11种专门的ML分析功能：

1. **learning_rate_analysis** - 学习率跟踪和动态
2. **optimizer_comparison** - 优化器状态比较
3. **loss_tracking** - 损失函数演化分析
4. **accuracy_tracking** - 性能指标监控
5. **model_version_analysis** - 版本和检查点检测
6. **gradient_analysis** - 梯度流和稳定性分析
7. **quantization_analysis** - 混合精度检测（FP32/FP16/INT8/INT4）
8. **convergence_analysis** - 学习曲线和收敛模式
9. **activation_analysis** - 激活函数使用分析
10. **attention_analysis** - Transformer和注意力机制
11. **ensemble_analysis** - 集成模型检测

**触发条件：**
- **PyTorch/Safetensors**：自动运行全部11项分析
- **NumPy/MATLAB**：仅基础张量统计
- **其他格式**：标准结构比较

## 版本兼容性

- **0.3.16**：具有自动ML分析的当前稳定版本
- **基于**：diffx-core v0.6.x，提供经过验证的差异可靠性
- **最低Rust版本**：1.70.0
- **依赖项**：当前版本请参见`Cargo.toml`

## 参见

- 命令行使用请参考[CLI参考](cli-reference_zh.md)
- 实用示例请参考[ML模型比较指南](../user-guide/ml-model-comparison_zh.md)
- 语言绑定请参考[统一API参考](../bindings/unified-api_zh.md)