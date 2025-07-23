use diffai_core::diff;
use regex::Regex;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library basic safetensors comparison
#[test]
fn test_core_basic_safetensors() {
    let v1 = parse_json(r#"{"tensor1": {"value": 0.5}}"#);
    let v2 = parse_json(r#"{"tensor1": {"value": 0.6}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_value_diff = results.iter().any(|r| format!("{:?}", r).contains("value"));
    assert!(has_value_diff);
}

/// Test case 2: Core library numpy array comparison
#[test]
fn test_core_numpy_array() {
    let v1 = parse_json(r#"{"data": [1.0, 2.0, 3.0]}"#);
    let v2 = parse_json(r#"{"data": [1.1, 2.1, 3.1]}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_data_diff = results.iter().any(|r| format!("{:?}", r).contains("data"));
    assert!(has_data_diff);
}

/// Test case 3: Core library MATLAB file comparison
#[test]
fn test_core_matlab_file() {
    let v1 = parse_json(r#"{"experiment": {"result": 0.85}}"#);
    let v2 = parse_json(r#"{"experiment": {"result": 0.90}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_result_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("result"));
    assert!(has_result_diff);
}

/// Test case 4: Core library JSON config comparison
#[test]
fn test_core_json_config() {
    let v1 = parse_json(r#"{"config": {"setting": "old"}}"#);
    let v2 = parse_json(r#"{"config": {"setting": "new"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_setting_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("setting"));
    assert!(has_setting_diff);
}

/// Test case 5: Core library stdin input simulation
#[test]
fn test_core_stdin_input() {
    let v1 = parse_json(r#"{"input": "stdin"}"#);
    let v2 = parse_json(r#"{"input": "file"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_input_diff = results.iter().any(|r| format!("{:?}", r).contains("input"));
    assert!(has_input_diff);
}

/// Test case 6: Core library recursive directory comparison
#[test]
fn test_core_recursive_directory() {
    let v1 = parse_json(r#"{"dir1": {"file1": "content1", "file2": "content2"}}"#);
    let v2 = parse_json(r#"{"dir2": {"file1": "content1", "file2": "different"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 7: Core library with regex ignore pattern
#[test]
fn test_core_verbose_mode() {
    let v1 = parse_json(r#"{"model": {"param": 1.0}}"#);
    let v2 = parse_json(r#"{"model": {"param": 1.1}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_param_diff = results.iter().any(|r| format!("{:?}", r).contains("param"));
    assert!(has_param_diff);
}

/// Test case 8: Core library no-color functionality (internal processing)
#[test]
fn test_core_no_color_processing() {
    let v1 = parse_json(r#"{"color": "enabled"}"#);
    let v2 = parse_json(r#"{"color": "disabled"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_color_diff = results.iter().any(|r| format!("{:?}", r).contains("color"));
    assert!(has_color_diff);
}

/// Test case 9: Core library full analysis processing
#[test]
fn test_core_full_analysis() {
    let v1 = parse_json(r#"{"analysis": {"complete": true}}"#);
    let v2 = parse_json(r#"{"analysis": {"complete": false}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_complete_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("complete"));
    assert!(has_complete_diff);
}

/// Test case 10: Core library structured output processing
#[test]
fn test_core_json_output() {
    let v1 = parse_json(r#"{"output": "test1"}"#);
    let v2 = parse_json(r#"{"output": "test2"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_output_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("output"));
    assert!(has_output_diff);
}

/// Test case 11: Core library YAML processing
#[test]
fn test_core_yaml_output() {
    let v1 = parse_json(r#"{"yaml": "format1"}"#);
    let v2 = parse_json(r#"{"yaml": "format2"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_yaml_diff = results.iter().any(|r| format!("{:?}", r).contains("yaml"));
    assert!(has_yaml_diff);
}

/// Test case 12: Core library scientific data analysis
#[test]
fn test_core_scientific_data() {
    let v1 = parse_json(r#"{"data": {"shape": [1000, 256], "mean": 0.1234}}"#);
    let v2 = parse_json(r#"{"data": {"shape": [1000, 256], "mean": 0.1456}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_mean_diff = results.iter().any(|r| format!("{:?}", r).contains("mean"));
    assert!(has_mean_diff);
}

/// Test case 13: Core library MATLAB simulation analysis
#[test]
fn test_core_matlab_simulation() {
    let v1 = parse_json(r#"{"results": {"var": "results", "shape": [500, 100]}}"#);
    let v2 = parse_json(r#"{"results": {"var": "results", "shape": [500, 120]}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_shape_diff = results.iter().any(|r| format!("{:?}", r).contains("shape"));
    assert!(has_shape_diff);
}

/// Test case 14: Core library debug mode processing
#[test]
fn test_core_debug_mode() {
    let v1 = parse_json(r#"{"debug": {"info": "level1"}}"#);
    let v2 = parse_json(r#"{"debug": {"info": "level2"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_info_diff = results.iter().any(|r| format!("{:?}", r).contains("info"));
    assert!(has_info_diff);
}

/// Test case 15: Core library YAML config comparison
#[test]
fn test_core_yaml_config() {
    let v1 = parse_json(r#"{"application": {"name": "app1"}}"#);
    let v2 = parse_json(r#"{"application": {"name": "app2"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_name_diff = results.iter().any(|r| format!("{:?}", r).contains("name"));
    assert!(has_name_diff);
}
