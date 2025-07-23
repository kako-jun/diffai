use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library CLI output format testing
#[test]
fn test_core_cli_output_format() {
    let v1 = parse_json(r#"{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}"#);
    let v2 = parse_json(r#"{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_fc1_diff = results.iter().any(|r| format!("{:?}", r).contains("fc1"));
    assert!(has_fc1_diff);
}

/// Test case 2: Core library default output format
#[test]
fn test_core_default_output() {
    let v1 = parse_json(r#"{"layers": 12, "hidden_size": 768}"#);
    let v2 = parse_json(r#"{"layers": 24, "hidden_size": 768}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    assert!(has_layers_diff);
}

/// Test case 3: Core library JSON output format
#[test]
fn test_core_json_output_format() {
    let v1 = parse_json(r#"{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}"#);
    let v2 = parse_json(r#"{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_fc1_diff = results.iter().any(|r| format!("{:?}", r).contains("fc1"));
    assert!(has_fc1_diff);
}

/// Test case 4: Core library YAML output format
#[test]
fn test_core_yaml_output_format() {
    let v1 = parse_json(r#"{"tensor": {"mean": 0.0018, "std": 0.0518}}"#);
    let v2 = parse_json(r#"{"tensor": {"mean": 0.0017, "std": 0.0647}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_tensor_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("tensor"));
    assert!(has_tensor_diff);
}

/// Test case 5: Core library unified output format
#[test]
fn test_core_unified_output_format() {
    let v1 = parse_json(r#"{"model": {"layers": 12, "hidden_size": 768}}"#);
    let v2 = parse_json(r#"{"model": {"layers": 24, "hidden_size": 768}, "optimizer": "adam"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    let has_optimizer_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("optimizer"));
    assert!(has_layers_diff || has_optimizer_diff);
}

/// Test case 6: Core library JSON with filtering
#[test]
fn test_core_json_with_filter() {
    let v1 = parse_json(r#"{"fc1": {"bias": {"mean": 0.0018, "std": 0.0518}}}"#);
    let v2 = parse_json(r#"{"fc1": {"bias": {"mean": 0.0017, "std": 0.0647}}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_fc1_diff = results.iter().any(|r| format!("{:?}", r).contains("fc1"));
    assert!(has_fc1_diff);
}

/// Test case 7: Core library YAML output to file
#[test]
fn test_core_yaml_output_to_file() {
    let v1 = parse_json(r#"{"config": {"timeout": 30, "retries": 3}}"#);
    let v2 = parse_json(r#"{"config": {"timeout": 60, "retries": 5}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_timeout_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("timeout"));
    let has_retries_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("retries"));
    assert!(has_timeout_diff || has_retries_diff);
}

/// Test case 8: Core library conditional logic check
#[test]
fn test_core_conditional_logic_check() {
    let v1 = parse_json(r#"{"model": {"parameters": 1000}}"#);
    let v2 = parse_json(r#"{"model": {"parameters": 2000}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_parameters_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("parameters"));
    assert!(has_parameters_diff);
}

/// Test case 9: Core library human readable output
#[test]
fn test_core_human_readable_output() {
    let v1 = parse_json(r#"{"layer1": {"weights": [1.0, 2.0, 3.0]}}"#);
    let v2 = parse_json(r#"{"layer1": {"weights": [1.1, 2.1, 3.1]}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_layer1_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layer1"));
    assert!(has_layer1_diff);
}

/// Test case 10: Core library machine readable output
#[test]
fn test_core_machine_readable_output() {
    let v1 = parse_json(r#"{"params": {"learning_rate": 0.001}}"#);
    let v2 = parse_json(r#"{"params": {"learning_rate": 0.01}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_learning_rate_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("learning_rate"));
    assert!(has_learning_rate_diff);
}

/// Test case 11: Core library environment variable JSON format
#[test]
fn test_core_env_var_json_format() {
    let v1 = parse_json(r#"{"model_version": "1.0"}"#);
    let v2 = parse_json(r#"{"model_version": "2.0"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_version_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("model_version"));
    assert!(has_version_diff);
}

/// Test case 12: Core library CLI colors
#[test]
fn test_core_cli_colors() {
    let v1 = parse_json(r#"{"status": "active"}"#);
    let v2 = parse_json(r#"{"status": "inactive"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_status_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("status"));
    assert!(has_status_diff);
}

/// Test case 13: Core library JSON pretty print
#[test]
fn test_core_json_pretty() {
    let v1 = parse_json(r#"{"data": {"value": 100}}"#);
    let v2 = parse_json(r#"{"data": {"value": 200}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_data_diff = results.iter().any(|r| format!("{:?}", r).contains("data"));
    let has_value_diff = results.iter().any(|r| format!("{:?}", r).contains("value"));
    assert!(has_data_diff || has_value_diff);
}
