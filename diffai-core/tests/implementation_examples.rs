use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library implementation of ML models enhanced functionality
#[test]
fn test_core_ml_models_enhanced() {
    let v1 = parse_json(r#"{"model": {"enhanced": true, "version": "2.4"}}"#);
    let v2 = parse_json(r#"{"model": {"enhanced": true, "version": "2.5"}}"#);
    
    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());
    
    let has_version_diff = results.iter().any(|r| format!("{:?}", r).contains("version"));
    assert!(has_version_diff);
}