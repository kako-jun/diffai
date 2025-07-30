# API Reference - diffai-core

Complete API documentation for the `diffai-core` Rust crate, providing AI/ML model diff functionality.

## Overview

The `diffai-core` crate is the heart of the diffai ecosystem, providing specialized diff operations for AI/ML model files and tensors. It can be embedded in other Rust applications to add ML-specific comparison capabilities.

**Unified API Design**: The core API exposes a single main function `diff()` for all comparison operations with automatic comprehensive analysis.

## Installation

Add `diffai-core` to your `Cargo.toml`:

```toml
[dependencies]
diffai-core = "0.2.0"
```

### Feature Flags

```toml
[dependencies]
diffai-core = { version = "0.2.0", features = ["all-formats"] }
```

Available features:
- `pytorch` (default) - PyTorch model support
- `safetensors` (default) - Safetensors format support  
- `numpy` (default) - NumPy array support
- `matlab` - MATLAB file support
- `all-formats` - Enable all format parsers

## Public API

### Core Types

#### `DiffResult`

Represents a single difference between two AI/ML models or tensors.

```rust
#[derive(Debug, PartialEq, Serialize)]
pub enum DiffResult {
    // Standard differences
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, String, String),
    
    // ML-specific differences
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    WeightSignificantChange(String, f64, Statistics),
    LayerAdded(String, LayerInfo),
    LayerRemoved(String, LayerInfo),
    ArchitectureChanged(String, String),
    PrecisionChanged(String, String, String),
}
```

### Core Functions

#### `diff()`

Primary function for computing differences between two AI/ML models or tensors. This is the unified API entry point for all comparison operations.

```rust
pub fn diff(
    old: &Value,
    new: &Value,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>, Error>
```

**Parameters:**
- `old`: Original/baseline model or tensor data
- `new`: New/updated model or tensor data
- `options`: Optional configuration options for the comparison

**Returns:** `Result<Vec<DiffResult>, Error>` representing all differences found

#### DiffOptions Structure

```rust
pub struct DiffOptions {
    // Numeric comparison
    pub epsilon: Option<f64>,
    
    // Array comparison
    pub array_id_key: Option<String>,
    
    // Filtering
    pub ignore_keys_regex: Option<String>,
    pub path_filter: Option<String>,
    
    // Output control
    pub output_format: Option<OutputFormat>,
    pub show_unchanged: Option<bool>,
    pub show_types: Option<bool>,
    
    // Memory optimization
    pub use_memory_optimization: Option<bool>,
    pub batch_size: Option<usize>,
    
    // diffai-specific options
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

**Example:**
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
                println!("Significant weight change at {}: magnitude={}", path, magnitude);
                println!("Stats: mean_change={}, std_dev={}", stats.mean_change, stats.std_dev);
            }
            _ => {}
        }
    }
    
    Ok(())
}
```

## Advanced Usage

### Custom Comparison Logic

#### ML-Specific Analysis

Enable machine learning specific analysis features:

```rust
use diffai_core::{diff, DiffOptions};
use serde_json::json;

let old_checkpoint = json!({
    "epoch": 1,
    "model_state_dict": { /* model weights */ },
    "optimizer_state_dict": { /* optimizer state */ }
});

let new_checkpoint = json!({
    "epoch": 10,
    "model_state_dict": { /* updated weights */ },
    "optimizer_state_dict": { /* updated state */ }
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

#### Precision-Aware Comparison

Handle models with different numerical precisions:

```rust
let options = DiffOptions {
    epsilon: Some(1e-3), // Higher tolerance for precision differences
    scientific_precision: Some(true),
    weight_threshold: Some(1e-4),
    ..Default::default()
};

let differences = diff(&float32_model, &float16_model, Some(&options))?;
```

### Working with Different Model Formats

#### Loading and Comparing Models

```rust
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use std::fs;

// Users should load models using appropriate ML libraries
fn compare_pytorch_models(
    model1_path: &str,
    model2_path: &str,
    options: Option<&DiffOptions>
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    // Example: Users would use candle, tch, or other PyTorch bindings
    // to load the actual model data into a serde_json::Value
    
    // This is just a placeholder - actual implementation would use ML libraries
    let old_content = fs::read_to_string(model1_path)?;
    let new_content = fs::read_to_string(model2_path)?;
    
    let old: Value = serde_json::from_str(&old_content)?;
    let new: Value = serde_json::from_str(&new_content)?;
    
    Ok(diff(&old, &new, options)?)
}
```

### Integration Patterns

#### Training Progress Analysis

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

#### Async Model Comparison

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
        println!("Model pair {}: {} differences", i + 1, diffs.len());
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
        // In real usage, use ML libraries to parse model files
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

## Error Handling

### Error Types

The library uses `anyhow::Error` for error handling:

```rust
use diffai_core::{diff, DiffOptions};
use anyhow::Result;

fn handle_model_errors() -> Result<()> {
    // ... load models ...
    
    match diff(&old_model, &new_model, None) {
        Ok(differences) => {
            println!("Found {} differences", differences.len());
        }
        Err(e) => {
            eprintln!("Model comparison error: {}", e);
            
            // Check for specific error types
            if e.to_string().contains("memory") {
                eprintln!("Consider enabling memory optimization");
            }
        }
    }
    
    Ok(())
}
```

## Performance Considerations

### Memory Usage

For large models:

```rust
use diffai_core::{diff, DiffOptions, DiffResult};

fn process_large_models(
    old: &Value,
    new: &Value
) -> Result<Vec<DiffResult>, Box<dyn std::error::Error>> {
    let options = DiffOptions {
        use_memory_optimization: Some(true),
        batch_size: Some(500), // Smaller batch for large tensors
        tensor_comparison_mode: Some("statistical".to_string()),
        ..Default::default()
    };
    
    Ok(diff(old, new, Some(&options))?)
}
```

### Optimization Tips

1. **Use memory optimization** for models >1GB
2. **Set appropriate epsilon** for your precision requirements
3. **Use statistical mode** for faster comparison of large tensors
4. **Filter paths** to focus on specific layers or components
5. **Adjust batch size** based on available memory

## Testing

### Unit Tests

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
        
        // Should detect significant changes in layer1
        assert!(diffs.iter().any(|d| matches!(d, 
            DiffResult::WeightSignificantChange(path, _, _) if path.contains("layer1")
        )));
    }
}
```

## Automatic ML Analysis

**Convention over Configuration**: diffai-core automatically runs 11 specialized ML analysis functions when it detects PyTorch (.pt/.pth) or Safetensors (.safetensors) files:

1. **learning_rate_analysis** - Learning rate tracking and dynamics
2. **optimizer_comparison** - Optimizer state comparison
3. **loss_tracking** - Loss function evolution analysis
4. **accuracy_tracking** - Performance metrics monitoring
5. **model_version_analysis** - Version and checkpoint detection
6. **gradient_analysis** - Gradient flow and stability analysis
7. **quantization_analysis** - Mixed precision detection (FP32/FP16/INT8/INT4)
8. **convergence_analysis** - Learning curve and convergence patterns
9. **activation_analysis** - Activation function usage analysis
10. **attention_analysis** - Transformer and attention mechanisms
11. **ensemble_analysis** - Ensemble model detection

**Trigger Conditions:**
- **PyTorch/Safetensors**: All 11 analyses run automatically
- **NumPy/MATLAB**: Basic tensor statistics only
- **Other formats**: Standard structural comparison

## Version Compatibility

- **0.3.16**: Current stable version with automatic ML analysis
- **Built on**: diffx-core v0.6.x for proven diff reliability
- **Minimum Rust version**: 1.70.0
- **Dependencies**: See `Cargo.toml` for current versions

## See Also

- [CLI Reference](cli-reference.md) for command-line usage
- [ML Model Comparison Guide](../user-guide/ml-model-comparison.md) for practical examples
- [Unified API Reference](../bindings/unified-api.md) for language bindings