# diffai-core API Specification

## Overview

`diffai-core` is the core library for AI/ML file comparison. It provides parsing, diff computation, and ML-specific analysis for deep learning model files.

## Installation

```toml
[dependencies]
diffai-core = "0.3"
```

## Main Types

### DiffResult

```rust
pub struct DiffResult {
    pub key: String,
    pub diff_type: DiffType,
    pub old_value: Option<Value>,
    pub new_value: Option<Value>,
    pub old_stats: Option<TensorStats>,
    pub new_stats: Option<TensorStats>,
}

pub enum DiffType {
    Added,
    Removed,
    Modified,
    TypeChanged,
}
```

### TensorStats

```rust
pub struct TensorStats {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub num_elements: usize,
}
```

### DiffOptions

```rust
pub struct DiffOptions {
    pub epsilon: Option<f64>,
    pub ignore_keys_regex: Option<String>,
    pub array_id_key: Option<String>,
    pub path_filter: Option<String>,
}
```

### FileFormat

```rust
pub enum FileFormat {
    Pytorch,
    Safetensors,
    Numpy,
    Matlab,
}
```

## Main Functions

### diff_paths

Compare two files by path with automatic format detection.

```rust
pub fn diff_paths(
    path1: &str,
    path2: &str,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>>
```

### diff

Compare two parsed values.

```rust
pub fn diff(
    old: &Value,
    new: &Value,
    options: Option<&DiffOptions>,
) -> Vec<DiffResult>
```

### detect_format_from_path

Auto-detect file format from extension.

```rust
pub fn detect_format_from_path(path: &str) -> Option<FileFormat>
```

### parse_file_by_format

Parse file into JSON Value representation.

```rust
pub fn parse_file_by_format(
    path: &str,
    format: FileFormat,
) -> Result<Value>
```

## Format-Specific Parsers

### PyTorch

```rust
pub fn parse_pytorch_model(path: &str) -> Result<Value>
```

Parses `.pt` / `.pth` files. Returns tensor metadata including:
- Shape, dtype, statistics (mean, std, min, max)
- Optimizer state (if present)
- Training metadata (epoch, loss, etc.)

### Safetensors

```rust
pub fn parse_safetensors_model(path: &str) -> Result<Value>
```

Parses `.safetensors` files. Efficient memory-mapped loading.

### NumPy

```rust
pub fn parse_numpy_file(path: &str) -> Result<Value>
```

Parses `.npy` (single array) and `.npz` (archive) files.

### MATLAB

```rust
pub fn parse_matlab_file(path: &str) -> Result<Value>
```

Parses `.mat` files (v5 format).

## Output Formatting

### format_output

Format diff results for display.

```rust
pub fn format_output(
    results: &[DiffResult],
    format: OutputFormat,
) -> String

pub enum OutputFormat {
    Text,
    Json,
    Yaml,
}
```

## Example Usage

```rust
use diffai_core::{diff_paths, DiffOptions, OutputFormat, format_output};

fn main() -> Result<()> {
    let options = DiffOptions {
        epsilon: Some(0.001),
        ignore_keys_regex: None,
        array_id_key: None,
        path_filter: None,
    };

    let results = diff_paths(
        "model_v1.pt",
        "model_v2.pt",
        Some(&options),
    )?;

    let output = format_output(&results, OutputFormat::Json);
    println!("{}", output);

    Ok(())
}
```

## ML Analysis Module

The `ml_analysis` module provides automatic analysis for deep learning models:

- Learning rate tracking
- Gradient health analysis
- Quantization detection
- Convergence patterns
- Attention mechanism analysis

These analyses run automatically when comparing PyTorch/Safetensors files.
