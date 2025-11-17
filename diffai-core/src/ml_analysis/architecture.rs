use serde_json::Value;

use crate::types::DiffResult;
use crate::diff::extract_tensor_shape;

pub fn analyze_model_architecture_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    let old_arch = extract_model_architecture(old_model);
    let new_arch = extract_model_architecture(new_model);

    if old_arch != new_arch {
        results.push(DiffResult::ModelArchitectureChanged(
            "model".to_string(),
            old_arch,
            new_arch,
        ));
    }
}

pub(crate) fn extract_model_architecture(model: &Value) -> String {
    if let Value::Object(obj) = model {
        let mut architecture_info = Vec::new();
        let mut layer_count = 0;
        let mut total_params = 0;
        let mut layer_types = std::collections::HashSet::new();

        // Analyze model structure
        for (key, value) in obj {
            if key.contains("weight") || key.contains("bias") {
                layer_count += 1;

                // Extract layer type from key (e.g., "conv1.weight" -> "conv")
                if let Some(layer_type) = extract_layer_type(key) {
                    layer_types.insert(layer_type);
                }

                // Count parameters
                if let Some(shape) = extract_tensor_shape(value) {
                    let param_count: usize = shape.iter().product();
                    total_params += param_count;
                }
            }
        }

        architecture_info.push(format!("layers: {}", layer_count));
        architecture_info.push(format!("parameters: {}", total_params));
        if !layer_types.is_empty() {
            let mut types: Vec<_> = layer_types.into_iter().collect();
            types.sort();
            architecture_info.push(format!("types: [{}]", types.join(", ")));
        }

        format!("{{{}}}", architecture_info.join(", "))
    } else {
        "unknown".to_string()
    }
}

pub(crate) fn extract_layer_type(key: &str) -> Option<String> {
    // Extract layer type from parameter names
    // e.g., "features.0.weight" -> "conv", "classifier.weight" -> "linear"
    if key.contains("conv") {
        Some("conv".to_string())
    } else if key.contains("linear") || key.contains("fc") || key.contains("classifier") {
        Some("linear".to_string())
    } else if key.contains("norm") || key.contains("bn") {
        Some("norm".to_string())
    } else if key.contains("attention") || key.contains("attn") {
        Some("attention".to_string())
    } else if key.contains("embedding") || key.contains("embed") {
        Some("embedding".to_string())
    } else {
        // Generic layer type based on position
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() > 1 {
            Some(parts[0].to_string())
        } else {
            None
        }
    }
}
