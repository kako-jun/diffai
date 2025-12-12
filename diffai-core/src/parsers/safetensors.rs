use anyhow::Result;
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};
use std::path::Path;

/// Parse SafeTensors model file - FOR INTERNAL USE ONLY (diffai-specific)
pub fn parse_safetensors_model(file_path: &Path) -> Result<Value> {
    let buffer = std::fs::read(file_path)?;
    let safetensors = SafeTensors::deserialize(&buffer)?;

    let mut result = serde_json::Map::new();
    let mut tensors = serde_json::Map::new();

    for tensor_name in safetensors.names() {
        let tensor_view = safetensors.tensor(tensor_name)?;
        let mut tensor_info = serde_json::Map::new();

        tensor_info.insert(
            "shape".to_string(),
            Value::Array(
                tensor_view
                    .shape()
                    .iter()
                    .map(|&s| Value::Number(s.into()))
                    .collect(),
            ),
        );
        tensor_info.insert(
            "dtype".to_string(),
            Value::String(format!("{:?}", tensor_view.dtype())),
        );

        // Calculate tensor statistics from raw data
        if let Some(stats) = compute_tensor_stats(&tensor_view) {
            tensor_info.insert("data_summary".to_string(), stats);
        }

        tensors.insert(tensor_name.to_string(), Value::Object(tensor_info));
    }

    result.insert(
        "model_type".to_string(),
        Value::String("safetensors".to_string()),
    );
    result.insert("tensors".to_string(), Value::Object(tensors));

    Ok(Value::Object(result))
}

fn compute_tensor_stats(tensor: &safetensors::tensor::TensorView) -> Option<Value> {
    let data = tensor.data();
    let dtype = tensor.dtype();

    // Convert raw bytes to f64 values based on dtype
    let values: Vec<f64> = match dtype {
        Dtype::F32 => {
            let floats: &[f32] = bytemuck::cast_slice(data);
            floats.iter().map(|&x| x as f64).collect()
        }
        Dtype::F64 => {
            let floats: &[f64] = bytemuck::cast_slice(data);
            floats.to_vec()
        }
        Dtype::F16 => {
            // F16 needs special handling - use half crate or convert manually
            // For simplicity, skip F16 stats for now
            return None;
        }
        Dtype::BF16 => {
            // BF16 needs special handling
            return None;
        }
        Dtype::I32 => {
            let ints: &[i32] = bytemuck::cast_slice(data);
            ints.iter().map(|&x| x as f64).collect()
        }
        Dtype::I64 => {
            let ints: &[i64] = bytemuck::cast_slice(data);
            ints.iter().map(|&x| x as f64).collect()
        }
        Dtype::I16 => {
            let ints: &[i16] = bytemuck::cast_slice(data);
            ints.iter().map(|&x| x as f64).collect()
        }
        Dtype::I8 => data.iter().map(|&x| x as i8 as f64).collect(),
        Dtype::U8 => data.iter().map(|&x| x as f64).collect(),
        Dtype::U16 => {
            let ints: &[u16] = bytemuck::cast_slice(data);
            ints.iter().map(|&x| x as f64).collect()
        }
        Dtype::U32 => {
            let ints: &[u32] = bytemuck::cast_slice(data);
            ints.iter().map(|&x| x as f64).collect()
        }
        Dtype::U64 => {
            let ints: &[u64] = bytemuck::cast_slice(data);
            ints.iter().map(|&x| x as f64).collect()
        }
        _ => return None,
    };

    if values.is_empty() {
        return None;
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Some(json!({
        "mean": mean,
        "std": std,
        "min": min,
        "max": max
    }))
}
