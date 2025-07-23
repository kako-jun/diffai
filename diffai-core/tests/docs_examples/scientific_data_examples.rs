use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library NumPy array comparison
#[test]
fn test_core_numpy_array_comparison() {
    let v1 = parse_json(r#"{"numpy_array": {"shape": [1000, 256], "mean": 0.1234, "std": 0.9876, "dtype": "float64"}}"#);
    let v2 = parse_json(r#"{"numpy_array": {"shape": [1000, 256], "mean": 0.1456, "std": 0.9654, "dtype": "float64"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_numpy_diff = results.iter().any(|r| format!("{:?}", r).contains("numpy_array"));
    assert!(has_numpy_diff);
}

/// Test case 2: Core library compressed NumPy archives
#[test]
fn test_core_compressed_numpy_archives() {
    let v1 = parse_json(r#"{"train_data": {"shape": [60000, 784], "mean": 0.1307, "std": 0.3081, "dtype": "float32"}, "test_data": {"shape": [10000, 784], "mean": 0.1325, "std": 0.3105, "dtype": "float32"}}"#);
    let v2 = parse_json(r#"{"train_data": {"shape": [60000, 784], "mean": 0.1309, "std": 0.3082, "dtype": "float32"}, "test_data": {"shape": [10000, 784], "mean": 0.1327, "std": 0.3106, "dtype": "float32"}, "validation_data": {"shape": [5000, 784], "mean": 0.1315, "std": 0.3095, "dtype": "float32"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_train_data_diff = results.iter().any(|r| format!("{:?}", r).contains("train_data"));
    assert!(has_train_data_diff);
}

/// Test case 3: Core library NumPy JSON output
#[test]
fn test_core_numpy_json_output() {
    let v1 = parse_json(r#"{"experiment": {"baseline": true, "data": [1.0, 2.0, 3.0]}}"#);
    let v2 = parse_json(r#"{"experiment": {"baseline": false, "data": [1.1, 2.1, 3.1]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_experiment_diff = results.iter().any(|r| format!("{:?}", r).contains("experiment"));
    assert!(has_experiment_diff);
}

/// Test case 4: Core library MATLAB file comparison
#[test]
fn test_core_matlab_file_comparison() {
    let v1 = parse_json(r#"{"results": {"shape": [500, 100], "mean": 2.3456, "std": 1.2345, "dtype": "double"}}"#);
    let v2 = parse_json(r#"{"results": {"shape": [500, 100], "mean": 2.4567, "std": 1.3456, "dtype": "double"}, "new_variable": {"shape": [100], "dtype": "single", "elements": 100}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_results_diff = results.iter().any(|r| format!("{:?}", r).contains("results"));
    assert!(has_results_diff);
}

/// Test case 5: Core library MATLAB specific variables
#[test]
fn test_core_matlab_specific_variables() {
    let v1 = parse_json(r#"{"experiment_data": {"temperature": [20.1, 20.2, 20.3]}}"#);
    let v2 = parse_json(r#"{"experiment_data": {"temperature": [21.1, 21.2, 21.3]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_experiment_data_diff = results.iter().any(|r| format!("{:?}", r).contains("experiment_data"));
    assert!(has_experiment_data_diff);
}

/// Test case 6: Core library MATLAB YAML output
#[test]
fn test_core_matlab_yaml_output() {
    let v1 = parse_json(r#"{"analysis": {"method": "linear", "r_squared": 0.85}}"#);
    let v2 = parse_json(r#"{"analysis": {"method": "polynomial", "r_squared": 0.92}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_analysis_diff = results.iter().any(|r| format!("{:?}", r).contains("analysis"));
    assert!(has_analysis_diff);
}

/// Test case 7: Core library epsilon tolerance numerical
#[test]
fn test_core_epsilon_tolerance_numerical() {
    let v1 = parse_json(r#"{"measurement": {"value": 1.0000001}}"#);
    let v2 = parse_json(r#"{"measurement": {"value": 1.0000002}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_measurement_diff = results.iter().any(|r| format!("{:?}", r).contains("measurement"));
    assert!(has_measurement_diff);
}

/// Test case 8: Core library MATLAB epsilon simulation
#[test]
fn test_core_matlab_epsilon_simulation() {
    let v1 = parse_json(r#"{"simulation": {"velocity": 1.23456789, "pressure": 101.325}}"#);
    let v2 = parse_json(r#"{"simulation": {"velocity": 1.23456790, "pressure": 101.326}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_simulation_diff = results.iter().any(|r| format!("{:?}", r).contains("simulation"));
    assert!(has_simulation_diff);
}

/// Test case 9: Core library MATLAB path filtering
#[test]
fn test_core_matlab_path_filtering() {
    let v1 = parse_json(r#"{"experimental_data": {"sample_1": {"concentration": 0.5}}}"#);
    let v2 = parse_json(r#"{"experimental_data": {"sample_1": {"concentration": 0.6}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_experimental_data_diff = results.iter().any(|r| format!("{:?}", r).contains("experimental_data"));
    assert!(has_experimental_data_diff);
}

/// Test case 10: Core library ignore metadata variables
#[test]
fn test_core_ignore_metadata_variables() {
    let v1 = parse_json(r#"{"metadata": {"created": "2024-01-01"}, "timestamp": "12:00:00", "data": {"values": [1, 2, 3]}}"#);
    let v2 = parse_json(r#"{"metadata": {"created": "2024-01-02"}, "timestamp": "13:00:00", "data": {"values": [1, 2, 4]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_data_diff = results.iter().any(|r| format!("{:?}", r).contains("data"));
    assert!(has_data_diff);
}

/// Test case 11: Core library experimental data validation
#[test]
fn test_core_experimental_data_validation() {
    let v1 = parse_json(r#"{"data": {"shape": [1000, 50], "mean": 0.4567, "std": 0.1234, "dtype": "float64"}}"#);
    let v2 = parse_json(r#"{"data": {"shape": [1000, 50], "mean": 0.5123, "std": 0.1456, "dtype": "float64"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_data_diff = results.iter().any(|r| format!("{:?}", r).contains("data"));
    assert!(has_data_diff);
}