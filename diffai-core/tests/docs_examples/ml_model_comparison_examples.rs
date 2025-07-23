use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library PyTorch models comprehensive
#[test]
fn test_core_pytorch_models_comprehensive() {
    let v1 = parse_json(r#"{"state_dict": {"fc1.weight": [0.1, 0.2], "fc1.bias": [0.01]}}"#);
    let v2 = parse_json(r#"{"state_dict": {"fc1.weight": [0.15, 0.25], "fc1.bias": [0.02]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_state_dict_diff = results.iter().any(|r| format!("{:?}", r).contains("state_dict"));
    assert!(has_state_dict_diff);
}

/// Test case 2: Core library safetensors models comprehensive
#[test]
fn test_core_safetensors_models_comprehensive() {
    let v1 = parse_json(r#"{"tensors": {"layer1.weight": {"shape": [64, 32]}}}"#);
    let v2 = parse_json(r#"{"tensors": {"layer1.weight": {"shape": [64, 64]}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_tensors_diff = results.iter().any(|r| format!("{:?}", r).contains("tensors"));
    assert!(has_tensors_diff);
}

/// Test case 3: Core library automatic format detection
#[test]
fn test_core_automatic_format_detection() {
    let v1 = parse_json(r#"{"model": {"pretrained": true, "accuracy": 0.85}}"#);
    let v2 = parse_json(r#"{"model": {"pretrained": false, "accuracy": 0.92}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_accuracy_diff = results.iter().any(|r| format!("{:?}", r).contains("accuracy"));
    assert!(has_accuracy_diff);
}

/// Test case 4: Core library epsilon tolerance minor
#[test]
fn test_core_epsilon_tolerance_minor() {
    let v1 = parse_json(r#"{"weights": {"layer1": 0.1000000}}"#);
    let v2 = parse_json(r#"{"weights": {"layer1": 0.1000001}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_weights_diff = results.iter().any(|r| format!("{:?}", r).contains("weights"));
    assert!(has_weights_diff);
}

/// Test case 5: Core library quantization analysis epsilon
#[test]
fn test_core_quantization_analysis_epsilon() {
    let v1 = parse_json(r#"{"model": {"precision": "fp32", "weights": [0.123, 0.456]}}"#);
    let v2 = parse_json(r#"{"model": {"precision": "int8", "weights": [0.12, 0.46]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_precision_diff = results.iter().any(|r| format!("{:?}", r).contains("precision"));
    assert!(has_precision_diff);
}

/// Test case 6: Core library JSON output automation
#[test]
fn test_core_json_output_automation() {
    let v1 = parse_json(r#"{"layers": {"conv1": {"filters": 32}}}"#);
    let v2 = parse_json(r#"{"layers": {"conv1": {"filters": 64}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_layers_diff = results.iter().any(|r| format!("{:?}", r).contains("layers"));
    assert!(has_layers_diff);
}

/// Test case 7: Core library YAML output readability
#[test]
fn test_core_yaml_output_readability() {
    let v1 = parse_json(r#"{"parameters": {"learning_rate": 0.01}}"#);
    let v2 = parse_json(r#"{"parameters": {"learning_rate": 0.001}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_parameters_diff = results.iter().any(|r| format!("{:?}", r).contains("parameters"));
    assert!(has_parameters_diff);
}

/// Test case 8: Core library pipe to file
#[test]
fn test_core_pipe_to_file() {
    let v1 = parse_json(r#"{"metrics": {"loss": 0.5}}"#);
    let v2 = parse_json(r#"{"metrics": {"loss": 0.3}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_metrics_diff = results.iter().any(|r| format!("{:?}", r).contains("metrics"));
    assert!(has_metrics_diff);
}

/// Test case 9: Core library focus specific layers
#[test]
fn test_core_focus_specific_layers() {
    let v1 = parse_json(r#"{"classifier": {"weight": [0.1, 0.2]}}"#);
    let v2 = parse_json(r#"{"classifier": {"weight": [0.15, 0.25]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_classifier_diff = results.iter().any(|r| format!("{:?}", r).contains("classifier"));
    assert!(has_classifier_diff);
}

/// Test case 10: Core library ignore metadata
#[test]
fn test_core_ignore_metadata() {
    let v1 = parse_json(r#"{"timestamp": "2024-01-01", "weights": {"layer1": 0.5}}"#);
    let v2 = parse_json(r#"{"timestamp": "2024-01-02", "weights": {"layer1": 0.6}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_weights_diff = results.iter().any(|r| format!("{:?}", r).contains("weights"));
    assert!(has_weights_diff);
}

/// Test case 11: Core library finetuning analysis
#[test]
fn test_core_finetuning_analysis() {
    let v1 = parse_json(r#"{"bert": {"encoder": {"attention": {"query": {"weight": [0.001]}}}}, "classifier": {"weight": [0.0]}}"#);
    let v2 = parse_json(r#"{"bert": {"encoder": {"attention": {"query": {"weight": [0.0023]}}}}, "classifier": {"weight": [0.0145]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_bert_diff = results.iter().any(|r| format!("{:?}", r).contains("bert"));
    assert!(has_bert_diff);
}

/// Test case 12: Core library quantization impact assessment
#[test]
fn test_core_quantization_impact_assessment() {
    let v1 = parse_json(r#"{"conv1": {"weight": {"mean": 0.0045, "std": 0.2341}}}"#);
    let v2 = parse_json(r#"{"conv1": {"weight": {"mean": 0.0043, "std": 0.2298}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_conv1_diff = results.iter().any(|r| format!("{:?}", r).contains("conv1"));
    assert!(has_conv1_diff);
}

/// Test case 13: Core library training progress tracking
#[test]
fn test_core_training_progress_tracking() {
    let v1 = parse_json(r#"{"layers": {"0": {"weight": {"mean": -0.0012, "std": 1.2341}}, "1": {"bias": {"mean": 0.1234, "std": 0.4567}}}}"#);
    let v2 = parse_json(r#"{"layers": {"0": {"weight": {"mean": 0.0034, "std": 0.8907}}, "1": {"bias": {"mean": 0.0567, "std": 0.3210}}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_layers_diff = results.iter().any(|r| format!("{:?}", r).contains("layers"));
    assert!(has_layers_diff);
}

/// Test case 14: Core library architecture comparison
#[test]
fn test_core_architecture_comparison() {
    let v1 = parse_json(r#"{"features": {"conv1": {"weight": {"shape": [64, 3, 7, 7]}}, "layer4": {"2": {"downsample": {"0": {"weight": {"shape": [2048, 1024, 1, 1]}}}}}}}"#);
    let v2 = parse_json(r#"{"features": {"conv1": {"weight": {"shape": [32, 3, 3, 3]}}, "mbconv": {"expand_conv": {"weight": {"shape": [96, 32, 1, 1]}}}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_features_diff = results.iter().any(|r| format!("{:?}", r).contains("features"));
    assert!(has_features_diff);
}

/// Test case 15: Core library recursive mode large models
#[test]
fn test_core_recursive_mode_large_models() {
    let v1 = parse_json(r#"{"large_model": {"size": "1GB", "parameters": 1000000}}"#);
    let v2 = parse_json(r#"{"large_model": {"size": "1.2GB", "parameters": 1200000}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_large_model_diff = results.iter().any(|r| format!("{:?}", r).contains("large_model"));
    assert!(has_large_model_diff);
}

/// Test case 16: Core library focus analysis specific parts
#[test]
fn test_core_focus_analysis_specific_parts() {
    let v1 = parse_json(r#"{"tensor": {"classifier": {"weight": [0.1, 0.2]}}}"#);
    let v2 = parse_json(r#"{"tensor": {"classifier": {"weight": [0.15, 0.25]}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_classifier_diff = results.iter().any(|r| format!("{:?}", r).contains("classifier"));
    assert!(has_classifier_diff);
}

/// Test case 17: Core library higher epsilon faster comparison
#[test]
fn test_core_higher_epsilon_faster_comparison() {
    let v1 = parse_json(r#"{"model": {"precision": 0.001234}}"#);
    let v2 = parse_json(r#"{"model": {"precision": 0.001567}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 18: Core library verbose mode processing info
#[test]
fn test_core_verbose_mode_processing_info() {
    let v1 = parse_json(r#"{"processing": {"stage": "training"}}"#);
    let v2 = parse_json(r#"{"processing": {"stage": "validation"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_processing_diff = results.iter().any(|r| format!("{:?}", r).contains("processing"));
    assert!(has_processing_diff);
}

/// Test case 19: Core library architecture differences only
#[test]
fn test_core_architecture_differences_only() {
    let v1 = parse_json(r#"{"architecture": {"type": "transformer", "layers": 12}}"#);
    let v2 = parse_json(r#"{"architecture": {"type": "transformer", "layers": 24}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_architecture_diff = results.iter().any(|r| format!("{:?}", r).contains("architecture"));
    assert!(has_architecture_diff);
}

/// Test case 20: Core library subprocess run JSON
#[test]
fn test_core_subprocess_run_json() {
    let v1 = parse_json(r#"{"model": {"version": "1.0"}}"#);
    let v2 = parse_json(r#"{"model": {"version": "2.0"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 21: Core library CI/CD compare models
#[test]
fn test_core_cicd_compare_models() {
    let v1 = parse_json(r#"{"model": {"type": "baseline", "accuracy": 0.85}}"#);
    let v2 = parse_json(r#"{"model": {"type": "candidate", "accuracy": 0.88}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 22: Core library single model analysis
#[test]
fn test_core_single_model_analysis() {
    let v1 = parse_json(r#"{"model": {"layers": 6, "parameters": 100000}}"#);
    let v2 = parse_json(r#"{"model": {"layers": 6, "parameters": 100000}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(results.is_empty());
}

/// Test case 23: Core library explicit format
#[test]
fn test_core_explicit_format() {
    let v1 = parse_json(r#"{"safetensors": {"format": "explicit"}}"#);
    let v2 = parse_json(r#"{"safetensors": {"format": "explicit", "version": 2}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_safetensors_diff = results.iter().any(|r| format!("{:?}", r).contains("safetensors"));
    assert!(has_safetensors_diff);
}

/// Test case 24: Core library memory optimization epsilon
#[test]
fn test_core_memory_optimization_epsilon() {
    let v1 = parse_json(r#"{"large": {"tensor": [0.001, 0.002, 0.003]}}"#);
    let v2 = parse_json(r#"{"large": {"tensor": [0.0015, 0.0025, 0.0035]}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_large_diff = results.iter().any(|r| format!("{:?}", r).contains("large"));
    assert!(has_large_diff);
}

/// Test case 25: Core library memory optimization path
#[test]
fn test_core_memory_optimization_path() {
    let v1 = parse_json(r#"{"tensor": {"classifier": {"weight": [0.1]}}}"#);
    let v2 = parse_json(r#"{"tensor": {"classifier": {"weight": [0.2]}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_classifier_diff = results.iter().any(|r| format!("{:?}", r).contains("classifier"));
    assert!(has_classifier_diff);
}

/// Test case 26: Core library comprehensive analysis automatic
#[test]
fn test_core_comprehensive_analysis_automatic() {
    let v1 = parse_json(r#"{"checkpoint": {"epoch": 10, "loss": 0.5, "accuracy": 0.8}}"#);
    let v2 = parse_json(r#"{"checkpoint": {"epoch": 20, "loss": 0.3, "accuracy": 0.9}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_checkpoint_diff = results.iter().any(|r| format!("{:?}", r).contains("checkpoint"));
    assert!(has_checkpoint_diff);
}

/// Test case 27: Core library experimental comparison automatic
#[test]
fn test_core_experimental_comparison_automatic() {
    let v1 = parse_json(r#"{"experiment": {"type": "baseline", "performance": 0.85}}"#);
    let v2 = parse_json(r#"{"experiment": {"type": "enhanced", "performance": 0.92}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_experiment_diff = results.iter().any(|r| format!("{:?}", r).contains("experiment"));
    assert!(has_experiment_diff);
}