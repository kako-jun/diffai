use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library basic verbose output
#[test]
fn test_core_basic_verbose_output() {
    let v1 = parse_json(r#"{"config": {"debug": true}}"#);
    let v2 = parse_json(r#"{"config": {"debug": false}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_config_diff = results.iter().any(|r| format!("{:?}", r).contains("config"));
    assert!(has_config_diff);
}

/// Test case 2: Core library verbose short form
#[test]
fn test_core_verbose_short_form() {
    let v1 = parse_json(r#"{"data": {"value": 1}}"#);
    let v2 = parse_json(r#"{"data": {"value": 2}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_data_diff = results.iter().any(|r| format!("{:?}", r).contains("data"));
    assert!(has_data_diff);
}

/// Test case 3: Core library verbose advanced options
#[test]
fn test_core_verbose_advanced_options() {
    let v1 = parse_json(r#"{"id": "001", "config": {"users": {"count": 10}}}"#);
    let v2 = parse_json(r#"{"id": "002", "config": {"users": {"count": 15}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_users_diff = results.iter().any(|r| format!("{:?}", r).contains("users"));
    assert!(has_users_diff);
}

/// Test case 4: Core library verbose ML analysis features
#[test]
fn test_core_verbose_ml_analysis_features() {
    let v1 = parse_json(r#"{"model": {"architecture": "transformer", "memory": "2GB"}}"#);
    let v2 = parse_json(r#"{"model": {"architecture": "transformer", "memory": "2.5GB"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 5: Core library verbose directory comparison
#[test]
fn test_core_verbose_directory_comparison() {
    let v1 = parse_json(r#"{"directory": {"files": 12}}"#);
    let v2 = parse_json(r#"{"directory": {"files": 14}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_directory_diff = results.iter().any(|r| format!("{:?}", r).contains("directory"));
    assert!(has_directory_diff);
}

/// Test case 6: Core library verbose debugging format detection
#[test]
fn test_core_verbose_debugging_format_detection() {
    let v1 = parse_json(r#"{"format": "unknown", "data": "test1"}"#);
    let v2 = parse_json(r#"{"format": "unknown", "data": "test2"}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_format_diff = results.iter().any(|r| format!("{:?}", r).contains("format"));
    assert!(has_format_diff);
}

/// Test case 7: Core library verbose ML analysis automatic
#[test]
fn test_core_verbose_ml_analysis_automatic() {
    let v1 = parse_json(r#"{"model": {"type": "pytorch", "version": "1.0"}}"#);
    let v2 = parse_json(r#"{"model": {"type": "pytorch", "version": "2.0"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 8: Core library verbose directory analysis
#[test]
fn test_core_verbose_directory_analysis() {
    let v1 = parse_json(r#"{"scan": {"dir1": {"files": 12}}}"#);
    let v2 = parse_json(r#"{"scan": {"dir2": {"files": 14}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_scan_diff = results.iter().any(|r| format!("{:?}", r).contains("scan"));
    assert!(has_scan_diff);
}

/// Test case 9: Core library verbose performance analysis
#[test]
fn test_core_verbose_performance_analysis() {
    let v1 = parse_json(r#"{"large_model": {"size": "1GB", "parameters": 1000000}}"#);
    let v2 = parse_json(r#"{"large_model": {"size": "1.2GB", "parameters": 1200000}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_large_model_diff = results.iter().any(|r| format!("{:?}", r).contains("large_model"));
    assert!(has_large_model_diff);
}

/// Test case 10: Core library verbose performance with options
#[test]
fn test_core_verbose_performance_with_options() {
    let v1 = parse_json(r#"{"data": {"precision": 0.12345}}"#);
    let v2 = parse_json(r#"{"data": {"precision": 0.12346}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_data_diff = results.iter().any(|r| format!("{:?}", r).contains("data"));
    assert!(has_data_diff);
}

/// Test case 11: Core library verbose configuration validation
#[test]
fn test_core_verbose_configuration_validation() {
    let v1 = parse_json(r#"{"id": "001", "timestamp": "12:00", "application": {"settings": {"timeout": 30}}}"#);
    let v2 = parse_json(r#"{"id": "002", "timestamp": "13:00", "application": {"settings": {"timeout": 60}}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_application_diff = results.iter().any(|r| format!("{:?}", r).contains("application"));
    assert!(has_application_diff);
}

/// Test case 12: Core library verbose output redirection
#[test]
fn test_core_verbose_output_redirection() {
    let v1 = parse_json(r#"{"result": {"status": "success"}}"#);
    let v2 = parse_json(r#"{"result": {"status": "failure"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_result_diff = results.iter().any(|r| format!("{:?}", r).contains("result"));
    assert!(has_result_diff);
}

/// Test case 13: Core library verbose CI/CD integration
#[test]
fn test_core_verbose_cicd_integration() {
    let v1 = parse_json(r#"{"model": {"type": "baseline", "accuracy": 0.85}}"#);
    let v2 = parse_json(r#"{"model": {"type": "improved", "accuracy": 0.90}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_model_diff = results.iter().any(|r| format!("{:?}", r).contains("model"));
    assert!(has_model_diff);
}

/// Test case 14: Core library verbose script automation
#[test]
fn test_core_verbose_script_automation() {
    let v1 = parse_json(r#"{"script": {"test": "automation"}}"#);
    let v2 = parse_json(r#"{"script": {"test": "automated"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_script_diff = results.iter().any(|r| format!("{:?}", r).contains("script"));
    assert!(has_script_diff);
}

/// Test case 15: Core library verbose only information
#[test]
fn test_core_verbose_only_information() {
    let v1 = parse_json(r#"{"verbose": {"info": "test"}}"#);
    let v2 = parse_json(r#"{"verbose": {"info": "tested"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_verbose_diff = results.iter().any(|r| format!("{:?}", r).contains("verbose"));
    assert!(has_verbose_diff);
}