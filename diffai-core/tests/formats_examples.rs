use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library PyTorch format processing
#[test]
fn test_core_pytorch_format() {
    let v1 = parse_json(r#"{"pytorch": {"layers": 5, "params": 1000}}"#);
    let v2 = parse_json(r#"{"pytorch": {"layers": 8, "params": 1500}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_layers_diff = results.iter().any(|r| format!("{:?}", r).contains("layers"));
    let has_params_diff = results.iter().any(|r| format!("{:?}", r).contains("params"));
    assert!(has_layers_diff);
    assert!(has_params_diff);
}

/// Test case 2: Core library Safetensors format processing
#[test]
fn test_core_safetensors_format() {
    let v1 = parse_json(r#"{"safetensors": {"version": "1.0", "secure": true}}"#);
    let v2 = parse_json(r#"{"safetensors": {"version": "1.1", "secure": true}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_version_diff = results.iter().any(|r| format!("{:?}", r).contains("version"));
    assert!(has_version_diff);
}

/// Test case 3: Core library NumPy format processing
#[test]
fn test_core_numpy_format() {
    let v1 = parse_json(r#"{"numpy": {"array": [1, 2, 3], "dtype": "int32"}}"#);
    let v2 = parse_json(r#"{"numpy": {"array": [1, 2, 4], "dtype": "int32"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_array_diff = results.iter().any(|r| format!("{:?}", r).contains("array"));
    assert!(has_array_diff);
}

/// Test case 4: Core library NPZ format processing
#[test]
fn test_core_npz_format() {
    let v1 = parse_json(r#"{"npz": {"data1": [1, 2, 3], "data2": [4, 5, 6]}}"#);
    let v2 = parse_json(r#"{"npz": {"data1": [1, 2, 3], "data2": [4, 5, 7]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_data2_diff = results.iter().any(|r| format!("{:?}", r).contains("data2"));
    assert!(has_data2_diff);
}

/// Test case 5: Core library MATLAB format processing
#[test]
fn test_core_matlab_format() {
    let v1 = parse_json(r#"{"matlab": {"variables": {"result": 0.85}, "complex": true}}"#);
    let v2 = parse_json(r#"{"matlab": {"variables": {"result": 0.90}, "complex": true}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_result_diff = results.iter().any(|r| format!("{:?}", r).contains("result"));
    assert!(has_result_diff);
}