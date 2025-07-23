use diffai_core::diff;
use serde_json::Value;
use std::collections::HashMap;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library basic diff functionality
#[test]
fn test_core_basic_diff() {
    let v1 = parse_json(r#"{"model": {"layers": 2, "params": 1000}}"#);
    let v2 = parse_json(r#"{"model": {"layers": 3, "params": 1500}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    // Check that differences were found
    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    let has_params_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("params"));
    assert!(has_layers_diff);
    assert!(has_params_diff);
}

/// Test case 2: Model statistics comparison
#[test]
fn test_core_model_stats() {
    let v1 = parse_json(r#"{"fc1": {"bias": 0.001, "weight": 0.5}}"#);
    let v2 = parse_json(r#"{"fc1": {"bias": 0.002, "weight": 0.6}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_bias_diff = results.iter().any(|r| format!("{:?}", r).contains("bias"));
    let has_weight_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("weight"));
    assert!(has_bias_diff);
    assert!(has_weight_diff);
}

/// Test case 3: Epsilon tolerance testing
#[test]
fn test_core_epsilon_tolerance() {
    let v1 = parse_json(r#"{"value": 1.0001}"#);
    let v2 = parse_json(r#"{"value": 1.0002}"#);

    // Without epsilon - should find difference
    let results_no_epsilon = diff(&v1, &v2, None, None, None);
    assert!(!results_no_epsilon.is_empty());

    // With epsilon - should ignore small difference
    let results_with_epsilon = diff(&v1, &v2, None, Some(0.001), None);
    assert!(results_with_epsilon.is_empty());
}

/// Test case 4: Array comparison
#[test]
fn test_core_array_comparison() {
    let v1 = parse_json(r#"{"array": [1, 2, 3]}"#);
    let v2 = parse_json(r#"{"array": [1, 2, 4]}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_array_diff = results.iter().any(|r| format!("{:?}", r).contains("array"));
    assert!(has_array_diff);
}

/// Test case 5: Nested object comparison
#[test]
fn test_core_nested_objects() {
    let v1 = parse_json(r#"{"experiment": {"result": 0.75}}"#);
    let v2 = parse_json(r#"{"experiment": {"result": 0.80}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_result_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("result"));
    assert!(has_result_diff);
}

/// Test case 6: Layer comparison
#[test]
fn test_core_layer_diff() {
    let v1 = parse_json(r#"{"layers": {"conv1": {"filters": 32}}}"#);
    let v2 = parse_json(r#"{"layers": {"conv1": {"filters": 64}}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_filters_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("filters"));
    assert!(has_filters_diff);
}

/// Test case 7: Performance metrics
#[test]
fn test_core_performance_metrics() {
    let v1 = parse_json(r#"{"performance": {"f1": 0.80}}"#);
    let v2 = parse_json(r#"{"performance": {"f1": 0.85}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_f1_diff = results.iter().any(|r| format!("{:?}", r).contains("f1"));
    assert!(has_f1_diff);
}

/// Test case 8: Identical values should return no differences
#[test]
fn test_core_identical_values() {
    let v1 = parse_json(r#"{"fc1": {"bias": 0.001}}"#);
    let v2 = parse_json(r#"{"fc1": {"bias": 0.001}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(results.is_empty());
}

/// Test case 9: Multiple metric comparison
#[test]
fn test_core_multiple_metrics() {
    let v1 = parse_json(r#"{"metric": 0.7}"#);
    let v2 = parse_json(r#"{"metric": 0.8}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_metric_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("metric"));
    assert!(has_metric_diff);
}

/// Test case 10: Statistical data comparison
#[test]
fn test_core_statistical_data() {
    let v1 = parse_json(r#"{"data": {"mean": 0.1234, "std": 0.9876}}"#);
    let v2 = parse_json(r#"{"data": {"mean": 0.1456, "std": 0.9654}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_mean_diff = results.iter().any(|r| format!("{:?}", r).contains("mean"));
    let has_std_diff = results.iter().any(|r| format!("{:?}", r).contains("std"));
    assert!(has_mean_diff);
    assert!(has_std_diff);
}

/// Test case 11: Simulation results
#[test]
fn test_core_simulation_results() {
    let v1 = parse_json(r#"{"results": {"mean": 2.3456, "std": 1.2345}}"#);
    let v2 = parse_json(r#"{"results": {"mean": 2.4567, "std": 1.3456}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 12: Version and layer tracking
#[test]
fn test_core_version_tracking() {
    let v1 = parse_json(r#"{"version": "1.0", "layers": 5}"#);
    let v2 = parse_json(r#"{"version": "2.0", "layers": 8}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_version_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("version"));
    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    assert!(has_version_diff);
    assert!(has_layers_diff);
}

/// Test case 13: Checkpoint comparison
#[test]
fn test_core_checkpoint_comparison() {
    let v1 = parse_json(r#"{"epoch": 10, "loss": 0.5}"#);
    let v2 = parse_json(r#"{"epoch": 20, "loss": 0.3}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 14: Configuration comparison
#[test]
fn test_core_config_comparison() {
    let v1 = parse_json(r#"{"config": {"setting": "old"}}"#);
    let v2 = parse_json(r#"{"config": {"setting": "new"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 15: Array results
#[test]
fn test_core_array_results() {
    let v1 = parse_json(r#"{"results": [0.8, 0.85, 0.9]}"#);
    let v2 = parse_json(r#"{"results": [0.82, 0.87, 0.92]}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 16: Temperature and pressure data
#[test]
fn test_core_physical_measurements() {
    let v1 = parse_json(r#"{"temperature": 25.5, "pressure": 101.3}"#);
    let v2 = parse_json(r#"{"temperature": 26.0, "pressure": 102.1}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 17: Weight comparison
#[test]
fn test_core_weight_comparison() {
    let v1 = parse_json(r#"{"weights": {"layer1": 0.5, "layer2": 0.3}}"#);
    let v2 = parse_json(r#"{"weights": {"layer1": 0.6, "layer2": 0.4}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 18: Architecture type comparison
#[test]
fn test_core_architecture_type() {
    let v1 = parse_json(r#"{"architecture": {"type": "cnn", "layers": 5}}"#);
    let v2 = parse_json(r#"{"architecture": {"type": "cnn", "layers": 8}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 19: Status and version tracking
#[test]
fn test_core_status_version() {
    let v1 = parse_json(r#"{"status": "stable", "version": "1.0"}"#);
    let v2 = parse_json(r#"{"status": "testing", "version": "1.1"}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 20: Performance optimization metrics
#[test]
fn test_core_performance_optimization() {
    let v1 = parse_json(r#"{"performance": {"speed": 100, "memory": 512}}"#);
    let v2 = parse_json(r#"{"performance": {"speed": 150, "memory": 256}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 21: Complex nested structures
#[test]
fn test_core_complex_nested() {
    let v1 = parse_json(r#"{"model": {"layers": {"conv1": {"params": 1000}}}}"#);
    let v2 = parse_json(r#"{"model": {"layers": {"conv1": {"params": 1500}}}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 22: Training metrics
#[test]
fn test_core_training_metrics() {
    let v1 = parse_json(r#"{"training": {"accuracy": 0.85, "loss": 0.15}}"#);
    let v2 = parse_json(r#"{"training": {"accuracy": 0.90, "loss": 0.10}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 23: Model metadata
#[test]
fn test_core_model_metadata() {
    let v1 = parse_json(r#"{"metadata": {"created": "2024-01-01", "size": "10MB"}}"#);
    let v2 = parse_json(r#"{"metadata": {"created": "2024-01-02", "size": "12MB"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}

/// Test case 24: Final optimization comparison
#[test]
fn test_core_final_optimization() {
    let v1 = parse_json(r#"{"optimization": {"level": 1, "enabled": true}}"#);
    let v2 = parse_json(r#"{"optimization": {"level": 2, "enabled": true}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
}
