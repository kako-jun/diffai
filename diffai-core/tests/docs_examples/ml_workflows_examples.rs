use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library model development improvement
#[test]
fn test_core_model_development_improvement() {
    let v1 = parse_json(r#"{"architecture": "resnet18", "layers": 18, "parameters": 11000000}"#);
    let v2 = parse_json(r#"{"architecture": "resnet34", "layers": 34, "parameters": 21000000}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_architecture_diff = results.iter().any(|r| format!("{:?}", r).contains("architecture"));
    assert!(has_architecture_diff);
}

/// Test case 2: Core library finetuning comparison
#[test]
fn test_core_finetuning_comparison() {
    let v1 = parse_json(r#"{"model": {"pretrained": true, "weights": {"classifier": [0.0, 0.0]}}}"#);
    let v2 = parse_json(r#"{"model": {"pretrained": false, "weights": {"classifier": [0.8, 0.9]}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_classifier_diff = results.iter().any(|r| format!("{:?}", r).contains("classifier"));
    assert!(has_classifier_diff);
}

/// Test case 3: Core library experiment results comparison
#[test]
fn test_core_experiment_results_comparison() {
    let v1 = parse_json(r#"{"experiment": {"id": "001", "accuracy": 0.85, "loss": 0.3}}"#);
    let v2 = parse_json(r#"{"experiment": {"id": "002", "accuracy": 0.88, "loss": 0.25}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_experiment_diff = results.iter().any(|r| format!("{:?}", r).contains("experiment"));
    assert!(has_experiment_diff);
}

/// Test case 4: Core library hyperparameter differences
#[test]
fn test_core_hyperparameter_differences() {
    let v1 = parse_json(r#"{"config": {"learning_rate": 0.01, "batch_size": 32, "epochs": 100}}"#);
    let v2 = parse_json(r#"{"config": {"learning_rate": 0.001, "batch_size": 64, "epochs": 150}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_learning_rate_diff = results.iter().any(|r| format!("{:?}", r).contains("learning_rate"));
    assert!(has_learning_rate_diff);
}

/// Test case 5: Core library quantization comparison
#[test]
fn test_core_quantization_comparison() {
    let v1 = parse_json(r#"{"model": {"precision": "fp32", "size_mb": 100, "weights": {"layer1": [0.123456]}}}"#);
    let v2 = parse_json(r#"{"model": {"precision": "int8", "size_mb": 25, "weights": {"layer1": [0.125]}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_precision_diff = results.iter().any(|r| format!("{:?}", r).contains("precision"));
    assert!(has_precision_diff);
}

/// Test case 6: Core library pruning effects
#[test]
fn test_core_pruning_effects() {
    let v1 = parse_json(r#"{"model": {"parameters": 1000000, "layers": {"conv1": {"weights": [0.1, 0.2, 0.3]}, "conv2": {"weights": [0.4, 0.5, 0.6]}}}}"#);
    let v2 = parse_json(r#"{"model": {"parameters": 500000, "layers": {"conv1": {"weights": [0.1, 0.0, 0.3]}, "conv2": {"weights": [0.0, 0.5, 0.0]}}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_parameters_diff = results.iter().any(|r| format!("{:?}", r).contains("parameters"));
    assert!(has_parameters_diff);
}

/// Test case 7: Core library workflow comparison
#[test]
fn test_core_workflow_comparison() {
    let v1 = parse_json(r#"{"workflow": {"baseline": true, "results": {"accuracy": 0.85, "f1": 0.82}}}"#);
    let v2 = parse_json(r#"{"workflow": {"baseline": false, "results": {"accuracy": 0.90, "f1": 0.88}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_workflow_diff = results.iter().any(|r| format!("{:?}", r).contains("workflow"));
    assert!(has_workflow_diff);
}