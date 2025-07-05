use serde::Serialize;
use serde_json::Value;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;
// use ini::Ini;
use anyhow::{Result, anyhow};
use quick_xml::de::from_str;
use csv::ReaderBuilder;
// AI/ML dependencies
use candle_core::Device;
use safetensors::SafeTensors;

#[derive(Debug, PartialEq, Serialize)]
pub enum DiffResult {
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, Value, Value),
    // AI/ML specific diff results
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    TensorStatsChanged(String, TensorStats, TensorStats),
    ModelArchitectureChanged(String, ModelInfo, ModelInfo),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub total_params: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelInfo {
    pub total_parameters: usize,
    pub layer_count: usize,
    pub layer_types: HashMap<String, usize>,
    pub model_size_bytes: usize,
}

pub fn diff(
    v1: &Value,
    v2: &Value,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) -> Vec<DiffResult> {
    let mut results = Vec::new();

    // Handle root level type or value change first
    if !values_are_equal(v1, v2, epsilon) {
        let type_match = match (v1, v2) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(_), Value::Bool(_)) => true,
            (Value::Number(_), Value::Number(_)) => true,
            (Value::String(_), Value::String(_)) => true,
            (Value::Array(_), Value::Array(_)) => true,
            (Value::Object(_), Value::Object(_)) => true,
            _ => false,
        };

        if !type_match {
            results.push(DiffResult::TypeChanged("".to_string(), v1.clone(), v2.clone()));
            return results; // If root type changed, no further diffing needed
        } else if v1.is_object() && v2.is_object() {
            diff_objects("", v1.as_object().unwrap(), v2.as_object().unwrap(), &mut results, ignore_keys_regex, epsilon, array_id_key);
        } else if v1.is_array() && v2.is_array() {
            diff_arrays("", v1.as_array().unwrap(), v2.as_array().unwrap(), &mut results, ignore_keys_regex, epsilon, array_id_key);
        } else {
            // Simple value modification at root
            results.push(DiffResult::Modified("".to_string(), v1.clone(), v2.clone()));
            return results;
        }
    }

    results
}

fn diff_recursive(
    path: &str,
    v1: &Value,
    v2: &Value,
    results: &mut Vec<DiffResult>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) {
    match (v1, v2) {
        (Value::Object(map1), Value::Object(map2)) => {
            diff_objects(path, map1, map2, results, ignore_keys_regex, epsilon, array_id_key);
        }
        (Value::Array(arr1), Value::Array(arr2)) => {
            diff_arrays(path, arr1, arr2, results, ignore_keys_regex, epsilon, array_id_key);
        }
        _ => { /* Should not happen if called correctly from diff_objects/diff_arrays */ }
    }
}

fn diff_objects(
    path: &str,
    map1: &serde_json::Map<String, Value>,
    map2: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) {
    // Check for modified or removed keys
    for (key, value1) in map1 {
        let current_path = if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };
        if let Some(regex) = ignore_keys_regex {
            if regex.is_match(key) {
                continue;
            }
        }
        match map2.get(key) {
            Some(value2) => {
                // Recurse for nested objects/arrays
                if value1.is_object() && value2.is_object() || value1.is_array() && value2.is_array() {
                    diff_recursive(&current_path, value1, value2, results, ignore_keys_regex, epsilon, array_id_key);
                } else if !values_are_equal(value1, value2, epsilon) {
                    let type_match = match (value1, value2) {
                        (Value::Null, Value::Null) => true,
                        (Value::Bool(_), Value::Bool(_)) => true,
                        (Value::Number(_), Value::Number(_)) => true,
                        (Value::String(_), Value::String(_)) => true,
                        (Value::Array(_), Value::Array(_)) => true,
                        (Value::Object(_), Value::Object(_)) => true,
                        _ => false,
                    };

                    if !type_match {
                        results.push(DiffResult::TypeChanged(current_path, value1.clone(), value2.clone()));
                    } else {
                        results.push(DiffResult::Modified(current_path, value1.clone(), value2.clone()));
                    }
                }
            }
            None => {
                results.push(DiffResult::Removed(current_path, value1.clone()));
            }
        }
    }

    // Check for added keys
    for (key, value2) in map2 {
        if !map1.contains_key(key) {
            let current_path = if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };
            results.push(DiffResult::Added(current_path, value2.clone()));
        }
    }
}

fn diff_arrays(
    path: &str,
    arr1: &Vec<Value>,
    arr2: &Vec<Value>,
    results: &mut Vec<DiffResult>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) {
    if let Some(id_key) = array_id_key {
        let mut map1: HashMap<Value, &Value> = HashMap::new();
        let mut no_id_elements1: Vec<(usize, &Value)> = Vec::new();
        for (i, val) in arr1.iter().enumerate() {
            if let Some(id_val) = val.get(id_key) {
                map1.insert(id_val.clone(), val);
            } else {
                no_id_elements1.push((i, val));
            }
        }

        let mut map2: HashMap<Value, &Value> = HashMap::new();
        let mut no_id_elements2: Vec<(usize, &Value)> = Vec::new();
        for (i, val) in arr2.iter().enumerate() {
            if let Some(id_val) = val.get(id_key) {
                map2.insert(id_val.clone(), val);
            } else {
                no_id_elements2.push((i, val));
            }
        }

        // Check for modified or removed elements
        for (id_val, val1) in &map1 {
            let current_path = format!("{}[{}={}]", path, id_key, id_val);
            match map2.get(&id_val) {
                Some(val2) => {
                    // Recurse for nested objects/arrays
                    if val1.is_object() && val2.is_object() || val1.is_array() && val2.is_array() {
                        diff_recursive(&current_path, val1, val2, results, ignore_keys_regex, epsilon, array_id_key);
                    } else if !values_are_equal(val1, val2, epsilon) {
                        let type_match = match (val1, val2) {
                            (Value::Null, Value::Null) => true,
                            (Value::Bool(_), Value::Bool(_)) => true,
                            (Value::Number(_), Value::Number(_)) => true,
                            (Value::String(_), Value::String(_)) => true,
                            (Value::Array(_), Value::Array(_)) => true,
                            (Value::Object(_), Value::Object(_)) => true,
                            _ => false,
                        };

                        if !type_match {
                            results.push(DiffResult::TypeChanged(current_path, (*val1).clone(), (*val2).clone()));
                        } else {
                            results.push(DiffResult::Modified(current_path, (*val1).clone(), (*val2).clone()));
                        }
                    }
                }
                None => {
                    results.push(DiffResult::Removed(current_path, (*val1).clone()));
                }
            }
        }

        // Check for added elements with ID
        for (id_val, val2) in map2 {
            if !map1.contains_key(&id_val) {
                let current_path = format!("{}[{}={}]", path, id_key, id_val);
                results.push(DiffResult::Added(current_path, val2.clone()));
            }
        }

        // Handle elements without ID using index-based comparison
        let max_len = no_id_elements1.len().max(no_id_elements2.len());
        for i in 0..max_len {
            match (no_id_elements1.get(i), no_id_elements2.get(i)) {
                (Some((idx1, val1)), Some((_idx2, val2))) => {
                    let current_path = format!("{}[{}]", path, idx1);
                    if val1.is_object() && val2.is_object() || val1.is_array() && val2.is_array() {
                        diff_recursive(&current_path, val1, val2, results, ignore_keys_regex, epsilon, array_id_key);
                    } else if !values_are_equal(val1, val2, epsilon) {
                        let type_match = match (val1, val2) {
                            (Value::Null, Value::Null) => true,
                            (Value::Bool(_), Value::Bool(_)) => true,
                            (Value::Number(_), Value::Number(_)) => true,
                            (Value::String(_), Value::String(_)) => true,
                            (Value::Array(_), Value::Array(_)) => true,
                            (Value::Object(_), Value::Object(_)) => true,
                            _ => false,
                        };

                        if !type_match {
                            results.push(DiffResult::TypeChanged(current_path, (*val1).clone(), (*val2).clone()));
                        } else {
                            results.push(DiffResult::Modified(current_path, (*val1).clone(), (*val2).clone()));
                        }
                    }
                }
                (Some((idx1, val1)), None) => {
                    let current_path = format!("{}[{}]", path, idx1);
                    results.push(DiffResult::Removed(current_path, (*val1).clone()));
                }
                (None, Some((idx2, val2))) => {
                    let current_path = format!("{}[{}]", path, idx2);
                    results.push(DiffResult::Added(current_path, (*val2).clone()));
                }
                (None, None) => break,
            }
        }
    } else {
        // Fallback to index-based comparison if no id_key is provided
        let max_len = arr1.len().max(arr2.len());
        for i in 0..max_len {
            let current_path = format!("{}[{}]", path, i);
            match (arr1.get(i), arr2.get(i)) {
                (Some(val1), Some(val2)) => {
                    // Recurse for nested objects/arrays within arrays
                    if val1.is_object() && val2.is_object() || val1.is_array() && val2.is_array() {
                        diff_recursive(&current_path, val1, val2, results, ignore_keys_regex, epsilon, array_id_key);
                    } else if !values_are_equal(val1, val2, epsilon) {
                        let type_match = match (val1, val2) {
                            (Value::Null, Value::Null) => true,
                            (Value::Bool(_), Value::Bool(_)) => true,
                            (Value::Number(_), Value::Number(_)) => true,
                            (Value::String(_), Value::String(_)) => true,
                            (Value::Array(_), Value::Array(_)) => true,
                            (Value::Object(_), Value::Object(_)) => true,
                            _ => false,
                        };

                        if !type_match {
                            results.push(DiffResult::TypeChanged(current_path, val1.clone(), val2.clone()));
                        } else {
                            results.push(DiffResult::Modified(current_path, val1.clone(), val2.clone()));
                        }
                    }
                }
                (Some(val1), None) => {
                    results.push(DiffResult::Removed(current_path, val1.clone()));
                }
                (None, Some(val2)) => {
                    results.push(DiffResult::Added(current_path, val2.clone()));
                }
                (None, None) => { /* Should not happen */ }
            }
        }
    }
}

fn values_are_equal(v1: &Value, v2: &Value, epsilon: Option<f64>) -> bool {
    if let (Some(e), Value::Number(n1), Value::Number(n2)) = (epsilon, v1, v2) {
        if let (Some(f1), Some(f2)) = (n1.as_f64(), n2.as_f64()) {
            return (f1 - f2).abs() < e;
        }
    }
    v1 == v2
}

pub fn value_type_name(value: &Value) -> &str {
    match value {
        Value::Null => "Null",
        Value::Bool(_) => "Boolean",
        Value::Number(_) => "Number",
        Value::String(_) => "String",
        Value::Array(_) => "Array",
        Value::Object(_) => "Object",
    }
}

pub fn parse_ini(content: &str) -> Result<Value> {
    use configparser::ini::Ini;
    
    let mut ini = Ini::new();
    ini.read(content.to_string())
        .map_err(|e| anyhow!("Failed to parse INI: {}", e))?;
    
    let mut root_map = serde_json::Map::new();

    for section_name in ini.sections() {
        let mut section_map = serde_json::Map::new();
        
        if let Some(section) = ini.get_map_ref().get(&section_name) {
            for (key, value) in section {
                if let Some(v) = value {
                    section_map.insert(key.clone(), Value::String(v.clone()));
                } else {
                    section_map.insert(key.clone(), Value::Null);
                }
            }
        }
        
        root_map.insert(section_name, Value::Object(section_map));
    }

    Ok(Value::Object(root_map))
}

pub fn parse_xml(content: &str) -> Result<Value> {
    let value: Value = from_str(content)?;
    Ok(value)
}

pub fn parse_csv(content: &str) -> Result<Value> {
    let mut reader = ReaderBuilder::new().from_reader(content.as_bytes());
    let mut records = Vec::new();

    let headers = reader.headers()?.clone();
    let has_headers = !headers.is_empty();

    for result in reader.into_records() {
        let record = result?;
        if has_headers {
            let mut obj = serde_json::Map::new();
            for (i, header) in headers.iter().enumerate() {
                if let Some(value) = record.get(i) {
                    obj.insert(header.to_string(), Value::String(value.to_string()));
                }
            }
            records.push(Value::Object(obj));
        } else {
            let mut arr = Vec::new();
            for field in record.iter() {
                arr.push(Value::String(field.to_string()));
            }
            records.push(Value::Array(arr));
        }
    }
    Ok(Value::Array(records))
}

// ============================================================================
// AI/ML File Format Support
// ============================================================================

/// Parse a PyTorch model file (.pth, .pt) and extract tensor information
pub fn parse_pytorch_model(file_path: &Path) -> Result<HashMap<String, TensorStats>> {
    let _device = Device::Cpu;
    let mut model_tensors = HashMap::new();
    
    // Try to load as safetensors first (more efficient)
    if let Ok(data) = std::fs::read(file_path) {
        if let Ok(safetensors) = SafeTensors::deserialize(&data) {
            for (name, tensor_view) in safetensors.tensors() {
                let shape: Vec<usize> = tensor_view.shape().to_vec();
                let dtype = match tensor_view.dtype() {
                    safetensors::Dtype::F32 => "f32".to_string(),
                    safetensors::Dtype::F64 => "f64".to_string(),
                    safetensors::Dtype::I32 => "i32".to_string(),
                    safetensors::Dtype::I64 => "i64".to_string(),
                    _ => "unknown".to_string(),
                };
                
                // Calculate basic statistics
                let total_params = shape.iter().product();
                let stats = TensorStats {
                    mean: 0.0, // TODO: Calculate actual mean from tensor data
                    std: 0.0,  // TODO: Calculate actual std from tensor data  
                    min: 0.0,  // TODO: Calculate actual min from tensor data
                    max: 0.0,  // TODO: Calculate actual max from tensor data
                    shape,
                    dtype,
                    total_params,
                };
                
                model_tensors.insert(name.to_string(), stats);
            }
            return Ok(model_tensors);
        }
    }
    
    // If safetensors parsing fails, try PyTorch pickle format
    // Note: This is a simplified implementation
    // In practice, you'd need to use candle's PyTorch loading capabilities
    Err(anyhow!("Failed to parse PyTorch model file: {}", file_path.display()))
}

/// Parse a Safetensors file (.safetensors) and extract tensor information  
pub fn parse_safetensors_model(file_path: &Path) -> Result<HashMap<String, TensorStats>> {
    let data = std::fs::read(file_path)?;
    let safetensors = SafeTensors::deserialize(&data)?;
    let mut model_tensors = HashMap::new();
    
    for (name, tensor_view) in safetensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = match tensor_view.dtype() {
            safetensors::Dtype::F32 => "f32".to_string(),
            safetensors::Dtype::F64 => "f64".to_string(),
            safetensors::Dtype::I32 => "i32".to_string(),
            safetensors::Dtype::I64 => "i64".to_string(),
            _ => "unknown".to_string(),
        };
        
        let total_params = shape.iter().product();
        
        // Extract raw data and calculate statistics
        let data_slice = tensor_view.data();
        let (mean, std, min, max) = match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let float_data: &[f32] = bytemuck::cast_slice(data_slice);
                calculate_f32_stats(float_data)
            },
            safetensors::Dtype::F64 => {
                let float_data: &[f64] = bytemuck::cast_slice(data_slice);
                calculate_f64_stats(float_data)
            },
            _ => (0.0, 0.0, 0.0, 0.0), // Skip non-float types for now
        };
        
        let stats = TensorStats {
            mean,
            std,
            min,
            max,
            shape,
            dtype,
            total_params,
        };
        
        model_tensors.insert(name.to_string(), stats);
    }
    
    Ok(model_tensors)
}

/// Compare two PyTorch/Safetensors models and return differences
pub fn diff_ml_models(
    model1_path: &Path,
    model2_path: &Path,
    epsilon: Option<f64>,
) -> Result<Vec<DiffResult>> {
    let model1_tensors = if model1_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        parse_safetensors_model(model1_path)?
    } else {
        parse_pytorch_model(model1_path)?
    };
    
    let model2_tensors = if model2_path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        parse_safetensors_model(model2_path)?
    } else {
        parse_pytorch_model(model2_path)?
    };
    
    let mut results = Vec::new();
    let eps = epsilon.unwrap_or(1e-6);
    
    // Check for added tensors
    for (name, stats) in &model2_tensors {
        if !model1_tensors.contains_key(name) {
            results.push(DiffResult::Added(
                format!("tensor.{}", name),
                serde_json::to_value(stats)?,
            ));
        }
    }
    
    // Check for removed tensors
    for (name, stats) in &model1_tensors {
        if !model2_tensors.contains_key(name) {
            results.push(DiffResult::Removed(
                format!("tensor.{}", name),
                serde_json::to_value(stats)?,
            ));
        }
    }
    
    // Check for modified tensors
    for (name, stats1) in &model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Check shape changes
            if stats1.shape != stats2.shape {
                results.push(DiffResult::TensorShapeChanged(
                    format!("tensor.{}", name),
                    stats1.shape.clone(),
                    stats2.shape.clone(),
                ));
            }
            
            // Check statistical changes (with epsilon tolerance)
            if (stats1.mean - stats2.mean).abs() > eps ||
               (stats1.std - stats2.std).abs() > eps ||
               (stats1.min - stats2.min).abs() > eps ||
               (stats1.max - stats2.max).abs() > eps {
                results.push(DiffResult::TensorStatsChanged(
                    format!("tensor.{}", name),
                    stats1.clone(),
                    stats2.clone(),
                ));
            }
        }
    }
    
    Ok(results)
}

// Helper functions for statistical calculations
fn calculate_f32_stats(data: &[f32]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    let mean = sum / data.len() as f64;
    
    let variance: f64 = data.iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / data.len() as f64;
    
    let std = variance.sqrt();
    let min = data.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as f64;
    let max = data.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() as f64;
    
    (mean, std, min, max)
}

fn calculate_f64_stats(data: &[f64]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    
    let sum: f64 = data.iter().sum();
    let mean = sum / data.len() as f64;
    
    let variance: f64 = data.iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>() / data.len() as f64;
    
    let std = variance.sqrt();
    let min = data.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = data.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    
    (mean, std, min, max)
}
