use anyhow::Result;
use safetensors::SafeTensors;
use serde_json::Value;
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

        tensors.insert(tensor_name.to_string(), Value::Object(tensor_info));
    }

    result.insert(
        "model_type".to_string(),
        Value::String("safetensors".to_string()),
    );
    result.insert("tensors".to_string(), Value::Object(tensors));

    Ok(Value::Object(result))
}
