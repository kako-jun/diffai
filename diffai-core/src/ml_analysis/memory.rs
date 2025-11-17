use serde_json::Value;

use crate::types::DiffResult;

// Memory Analysis - standard feature for all ML formats
pub fn analyze_memory_usage_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    let old_memory = calculate_model_memory_usage(old_model);
    let new_memory = calculate_model_memory_usage(new_model);

    if old_memory != new_memory {
        // Create a comprehensive memory analysis result
        let memory_change = new_memory as f64 - old_memory as f64;
        let memory_change_percent = if old_memory > 0 {
            (memory_change / old_memory as f64) * 100.0
        } else {
            0.0
        };

        // Use ModelArchitectureChanged variant for memory analysis
        let _memory_analysis = format!(
            "memory: {} → {} bytes ({:+.1}%)",
            old_memory, new_memory, memory_change_percent
        );

        results.push(DiffResult::ModelArchitectureChanged(
            "memory_analysis".to_string(),
            format!("memory_usage: {} bytes", old_memory),
            format!("memory_usage: {} bytes", new_memory),
        ));

        // Add detailed breakdown if significant change
        if memory_change.abs() > 1024.0 {
            // More than 1KB change
            let breakdown = create_memory_breakdown(old_model, new_model);
            if !breakdown.is_empty() {
                results.push(DiffResult::ModelArchitectureChanged(
                    "memory_breakdown".to_string(),
                    "previous".to_string(),
                    breakdown,
                ));
            }
        }
    }
}

// Calculate estimated memory usage of a model
pub(crate) fn calculate_model_memory_usage(model: &Value) -> usize {
    match model {
        Value::Object(obj) => {
            let mut total_memory = 0;

            // Base object overhead
            total_memory += std::mem::size_of::<serde_json::Map<String, Value>>();

            for (key, value) in obj {
                // Key memory
                total_memory += key.len();

                // Value memory
                total_memory += calculate_value_memory(value);
            }

            total_memory
        }
        _ => calculate_value_memory(model),
    }
}

// Calculate memory usage of a single Value
pub(crate) fn calculate_value_memory(value: &Value) -> usize {
    match value {
        Value::Null => std::mem::size_of::<Value>(),
        Value::Bool(_) => std::mem::size_of::<bool>(),
        Value::Number(_) => std::mem::size_of::<f64>(), // Assume f64
        Value::String(s) => s.len() + std::mem::size_of::<String>(),
        Value::Array(arr) => {
            let mut size = std::mem::size_of::<Vec<Value>>();
            for elem in arr {
                size += calculate_value_memory(elem);
            }
            size
        }
        Value::Object(obj) => {
            let mut size = std::mem::size_of::<serde_json::Map<String, Value>>();
            for (key, val) in obj {
                size += key.len() + calculate_value_memory(val);
            }
            size
        }
    }
}

// Create a detailed memory breakdown
pub(crate) fn create_memory_breakdown(old_model: &Value, new_model: &Value) -> String {
    let mut breakdown = Vec::new();

    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Analyze tensor memory usage
        let old_tensor_memory = calculate_tensor_memory(old_obj);
        let new_tensor_memory = calculate_tensor_memory(new_obj);

        if old_tensor_memory != new_tensor_memory {
            let change = new_tensor_memory as i64 - old_tensor_memory as i64;
            breakdown.push(format!(
                "tensors: {:+} bytes ({} → {})",
                change, old_tensor_memory, new_tensor_memory
            ));
        }

        // Analyze metadata memory
        let old_meta_memory = calculate_metadata_memory(old_obj);
        let new_meta_memory = calculate_metadata_memory(new_obj);

        if old_meta_memory != new_meta_memory {
            let change = new_meta_memory as i64 - old_meta_memory as i64;
            breakdown.push(format!(
                "metadata: {:+} bytes ({} → {})",
                change, old_meta_memory, new_meta_memory
            ));
        }
    }

    breakdown.join(", ")
}

// Calculate memory used by tensor data
pub(crate) fn calculate_tensor_memory(obj: &serde_json::Map<String, Value>) -> usize {
    let mut tensor_memory = 0;

    for (key, value) in obj {
        if key.contains("weight") || key.contains("bias") || key.contains("data") {
            // Estimate tensor memory based on shape and dtype
            if let Value::Object(tensor_obj) = value {
                if let Some(shape_value) = tensor_obj.get("shape") {
                    if let Value::Array(shape_arr) = shape_value {
                        let element_count: usize = shape_arr
                            .iter()
                            .filter_map(|v| v.as_u64())
                            .map(|x| x as usize)
                            .product();

                        // Assume 4 bytes per element (float32)
                        let dtype_size = if let Some(dtype) = tensor_obj.get("dtype") {
                            estimate_dtype_size(dtype)
                        } else {
                            4
                        };

                        tensor_memory += element_count * dtype_size;
                    }
                }
            } else {
                // For non-structured tensors, use value memory
                tensor_memory += calculate_value_memory(value);
            }
        }
    }

    tensor_memory
}

// Calculate memory used by metadata
pub(crate) fn calculate_metadata_memory(obj: &serde_json::Map<String, Value>) -> usize {
    let mut meta_memory = 0;

    for (key, value) in obj {
        if !key.contains("weight") && !key.contains("bias") && !key.contains("data") {
            meta_memory += key.len() + calculate_value_memory(value);
        }
    }

    meta_memory
}

// Estimate bytes per element based on dtype
pub(crate) fn estimate_dtype_size(dtype: &Value) -> usize {
    if let Value::String(dtype_str) = dtype {
        match dtype_str.to_lowercase().as_str() {
            s if s.contains("float64") || s.contains("f64") => 8,
            s if s.contains("float32") || s.contains("f32") => 4,
            s if s.contains("float16") || s.contains("f16") => 2,
            s if s.contains("int64") || s.contains("i64") => 8,
            s if s.contains("int32") || s.contains("i32") => 4,
            s if s.contains("int16") || s.contains("i16") => 2,
            s if s.contains("int8") || s.contains("i8") => 1,
            s if s.contains("uint64") || s.contains("u64") => 8,
            s if s.contains("uint32") || s.contains("u32") => 4,
            s if s.contains("uint16") || s.contains("u16") => 2,
            s if s.contains("uint8") || s.contains("u8") => 1,
            s if s.contains("bool") => 1,
            _ => 4, // Default to 4 bytes (float32)
        }
    } else {
        4 // Default
    }
}
