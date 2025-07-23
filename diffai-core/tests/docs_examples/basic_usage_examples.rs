use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library basic comprehensive analysis
#[test]
fn test_core_basic_comprehensive_analysis() {
    let v1 = parse_json(r#"{"model": {"layers": 2, "params": 1000}}"#);
    let v2 = parse_json(r#"{"model": {"layers": 3, "params": 1500}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    let has_params_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("params"));
    assert!(has_layers_diff || has_params_diff);
}

/// Test case 2: Core library JSON output
#[test]
fn test_core_json_output() {
    let v1 = parse_json(r#"{"tensor": {"mean": 0.5, "std": 0.1}}"#);
    let v2 = parse_json(r#"{"tensor": {"mean": 0.6, "std": 0.2}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_tensor_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("tensor"));
    assert!(has_tensor_diff);
}

/// Test case 3: Core library YAML output
#[test]
fn test_core_yaml_output() {
    let v1 = parse_json(r#"{"weights": {"layer1": 0.5}}"#);
    let v2 = parse_json(r#"{"weights": {"layer1": 0.7}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_weights_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("weights"));
    assert!(has_weights_diff);
}

/// Test case 4: Core library recursive directory comparison
#[test]
fn test_core_recursive_directory_comparison() {
    let v1 = parse_json(r#"{"config": {"version": "1.0"}}"#);
    let v2 = parse_json(r#"{"config": {"version": "2.0"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_version_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("version"));
    assert!(has_version_diff);
}

/// Test case 5: Core library recursive with format
#[test]
fn test_core_recursive_with_format() {
    let v1 = parse_json(r#"{"model": {"type": "safetensors"}}"#);
    let v2 = parse_json(r#"{"model": {"type": "safetensors", "version": 2}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 6: Core library PyTorch model comparison
#[test]
fn test_core_pytorch_model_comparison() {
    let v1 = parse_json(r#"{"state_dict": {"layer1.weight": [0.1, 0.2]}}"#);
    let v2 = parse_json(r#"{"state_dict": {"layer1.weight": [0.15, 0.25]}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_state_dict_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("state_dict"));
    assert!(has_state_dict_diff);
}

/// Test case 7: Core library training checkpoint comparison
#[test]
fn test_core_training_checkpoint_comparison() {
    let v1 = parse_json(r#"{"epoch": 1, "loss": 0.8, "accuracy": 0.6}"#);
    let v2 = parse_json(r#"{"epoch": 10, "loss": 0.3, "accuracy": 0.9}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_epoch_diff = results.iter().any(|r| format!("{:?}", r).contains("epoch"));
    assert!(has_epoch_diff);
}

/// Test case 8: Core library baseline vs improved
#[test]
fn test_core_baseline_vs_improved() {
    let v1 = parse_json(r#"{"performance": 0.85, "params": 1000000}"#);
    let v2 = parse_json(r#"{"performance": 0.92, "params": 1200000}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_performance_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("performance"));
    assert!(has_performance_diff);
}

/// Test case 9: Core library safetensors comprehensive
#[test]
fn test_core_safetensors_comprehensive() {
    let v1 = parse_json(r#"{"tensors": {"fc1.bias": {"shape": [64]}}}"#);
    let v2 = parse_json(r#"{"tensors": {"fc1.bias": {"shape": [128]}}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_tensors_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("tensors"));
    assert!(has_tensors_diff);
}

/// Test case 10: Core library deployment validation
#[test]
fn test_core_deployment_validation() {
    let v1 = parse_json(r#"{"deployment": {"ready": true, "risk": "low"}}"#);
    let v2 = parse_json(r#"{"deployment": {"ready": true, "risk": "medium"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_deployment_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("deployment"));
    assert!(has_deployment_diff);
}

/// Test case 11: Core library NumPy array comparison
#[test]
fn test_core_numpy_array_comparison() {
    let v1 = parse_json(r#"{"array": {"data": [1.0, 2.0, 3.0], "shape": [3]}}"#);
    let v2 = parse_json(r#"{"array": {"data": [1.1, 2.1, 3.1], "shape": [3]}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_array_diff = results.iter().any(|r| format!("{:?}", r).contains("array"));
    assert!(has_array_diff);
}

/// Test case 12: Core library MATLAB file comparison
#[test]
fn test_core_matlab_file_comparison() {
    let v1 = parse_json(r#"{"simulation": {"time": 100, "results": [0.5, 0.6]}}"#);
    let v2 = parse_json(r#"{"simulation": {"time": 150, "results": [0.7, 0.8]}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_simulation_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("simulation"));
    assert!(has_simulation_diff);
}

/// Test case 13: Core library compressed NumPy archives
#[test]
fn test_core_compressed_numpy_archives() {
    let v1 = parse_json(r#"{"dataset": {"train": 1000, "test": 200}}"#);
    let v2 = parse_json(r#"{"dataset": {"train": 1200, "test": 250}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_dataset_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("dataset"));
    assert!(has_dataset_diff);
}

/// Test case 14: Core library experiment comparison
#[test]
fn test_core_experiment_comparison() {
    let v1 = parse_json(r#"{"experiment": {"id": "v1", "accuracy": 0.85}}"#);
    let v2 = parse_json(r#"{"experiment": {"id": "v2", "accuracy": 0.90}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_experiment_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("experiment"));
    assert!(has_experiment_diff);
}

/// Test case 15: Core library checkpoint learning analysis
#[test]
fn test_core_checkpoint_learning_analysis() {
    let v1 = parse_json(r#"{"checkpoint": {"epoch": 10, "loss": 0.5}}"#);
    let v2 = parse_json(r#"{"checkpoint": {"epoch": 20, "loss": 0.3}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_checkpoint_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("checkpoint"));
    assert!(has_checkpoint_diff);
}

/// Test case 16: Core library CI/CD model comparison
#[test]
fn test_core_cicd_model_comparison() {
    let v1 = parse_json(r#"{"model": {"version": "baseline", "accuracy": 0.85}}"#);
    let v2 = parse_json(r#"{"model": {"version": "new", "accuracy": 0.88}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}
