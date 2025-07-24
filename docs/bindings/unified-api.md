# diffai Unified API Reference

*Language bindings API documentation for diffai-python and diffai-js*

## Overview

diffai provides a unified API for comparing AI/ML model files and tensors. It supports PyTorch (.pt, .pth), Safetensors, NumPy (.npy, .npz), and MATLAB (.mat) formats with specialized analysis for machine learning use cases.

## Main Function

### `diff(old, new, options)`

Compares two AI/ML model structures or tensors and returns the differences with ML-specific analysis.

#### Parameters

- `old` (Value): The original/old model or tensor data
- `new` (Value): The new/updated model or tensor data
- `options` (DiffOptions, optional): Configuration options for the comparison

#### Returns

- `Result<Vec<DiffResult>, Error>`: A vector of differences including ML-specific changes

#### Example

```rust
use diffai_core::{diff, DiffOptions};
use serde_json::json;

// Example comparing model metadata
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

## Options

### DiffOptions Structure

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

### Option Details

#### ML-Specific Options

- **`ml_analysis_enabled`**: Enable ML-specific analysis (weight changes, gradient flow, etc.)
  - Default: `true`
  
- **`tensor_comparison_mode`**: How to compare tensors
  - Options: `"element-wise"`, `"statistical"`, `"structural"`
  - Default: `"element-wise"`
  
- **`model_format`**: Expected model format for optimized parsing
  - Options: `"pytorch"`, `"safetensors"`, `"numpy"`, `"matlab"`, `"auto"`
  - Default: `"auto"`
  
- **`scientific_precision`**: Use scientific notation for numeric output
  - Default: `false`
  
- **`weight_threshold`**: Minimum weight change to report (helps filter noise)
  - Default: `1e-6`
  
- **`gradient_analysis`**: Analyze gradient-related tensors specially
  - Default: `false`
  
- **`statistical_summary`**: Include statistical summaries of tensor changes
  - Default: `false`

#### Common Options (inherited from unified API)

- **`epsilon`**: Numeric comparison tolerance
  - Default: `1e-9` (higher precision for ML use cases)
  
- **`ignore_keys_regex`**: Keys to ignore (useful for timestamps, random seeds)
  - Example: `"^(timestamp|random_seed|training_step)"`
  
- **`show_unchanged`**: Include unchanged layers in output
  - Default: `false`
  
- **`use_memory_optimization`**: Enable for large models (>1GB)
  - Default: `true` for files >100MB

## Result Types

### DiffResult Enum (ML-Enhanced)

```rust
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

### ML-Specific Result Types

- **`TensorShapeChanged(path, old_shape, new_shape)`**: Tensor dimensions changed
- **`WeightSignificantChange(path, magnitude, stats)`**: Significant weight changes with statistics
- **`LayerAdded/Removed(path, info)`**: Neural network layer modifications
- **`ArchitectureChanged(old_arch, new_arch)`**: Model architecture changes
- **`PrecisionChanged(path, old_precision, new_precision)`**: Data type changes (e.g., float32 to float16)

### Statistics Structure

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


## Language Bindings

### Python

```python
import diffai_python

# Basic model comparison (users load models themselves)
results = diffai_python.diff(old_model, new_model)

# With ML-specific options
results = diffai_python.diff(
    old_model,
    new_model,
    ml_analysis_enabled=True,
    tensor_comparison_mode="statistical",
    weight_threshold=1e-5,
    statistical_summary=True,
    scientific_precision=True
)

# Users should load models using appropriate libraries (torch, etc.)
# old_model = torch.load("model_epoch_1.pt")
# new_model = torch.load("model_epoch_10.pt")
# results = diffai_python.diff(old_model, new_model)
```

### TypeScript/JavaScript

```typescript
import { diff, DiffOptions } from 'diffai-js';

// Basic usage - users load models themselves
const results = await diff(oldModel, newModel);

// With ML-specific options
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

## Examples

### Comparing PyTorch Models

```rust
use diffai_core::{diff, DiffOptions};
// Users should load model data using appropriate libraries (e.g., candle, tch)

// Load models using external libraries
let old_model = /* load using PyTorch/candle/tch library */;
let new_model = /* load using PyTorch/candle/tch library */;

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    weight_threshold: Some(0.001),
    statistical_summary: Some(true),
    ..Default::default()
};

let results = diff(&old_model, &new_model, Some(&options))?;
```

### Analyzing Training Progress

```rust
let options = DiffOptions {
    tensor_comparison_mode: Some("statistical".to_string()),
    gradient_analysis: Some(true),
    show_unchanged: Some(false),
    ..Default::default()
};

// Compare checkpoints to see training progress
// Users should load checkpoints using appropriate ML libraries
let checkpoint_1 = /* load using PyTorch/candle/tch library */;
let checkpoint_10 = /* load using PyTorch/candle/tch library */;

let results = diff(&checkpoint_1, &checkpoint_10, Some(&options))?;
```

### Comparing Different Precisions

```rust
let options = DiffOptions {
    epsilon: Some(1e-3), // Higher tolerance for precision differences
    scientific_precision: Some(true),
    ..Default::default()
};

// Users should load models using appropriate ML libraries
let float32_model = /* load using PyTorch/candle/tch library */;
let float16_model = /* load using PyTorch/candle/tch library */;

let results = diff(&float32_model, &float16_model, Some(&options))?;
```

## Performance Considerations

- **Large Models**: Enable `use_memory_optimization` for models >1GB
- **Batch Processing**: Adjust `batch_size` based on available memory (default: 1000 tensors)
- **Statistical Mode**: Use `tensor_comparison_mode: "statistical"` for faster comparison of large tensors
- **Filtering**: Use `path_filter` to focus on specific layers or components

## Error Handling

The library provides detailed errors for:
- Unsupported model formats
- Corrupted model files
- Memory allocation failures
- Incompatible tensor shapes
- Precision loss warnings

## Best Practices

1. **Set appropriate epsilon**: Use higher values (1e-3) when comparing models with different precisions
2. **Use weight threshold**: Filter out insignificant changes to focus on important differences
3. **Enable statistical summary**: For large models, statistical summaries provide better insights
4. **Memory optimization**: Always enable for production models
5. **Layer filtering**: Use `path_filter` to examine specific layers during debugging