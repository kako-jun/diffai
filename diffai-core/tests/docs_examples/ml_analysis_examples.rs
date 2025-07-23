use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library comprehensive analysis (automatic)
#[test]
fn test_core_comprehensive_analysis() {
    let v1 = parse_json(r#"{"fc1": {"bias": 0.0018, "weight": -0.0002}}"#);
    let v2 = parse_json(r#"{"fc1": {"bias": 0.0017, "weight": -0.0001}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_bias_diff = results.iter().any(|r| format!("{:?}", r).contains("bias"));
    let has_weight_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("weight"));
    assert!(has_bias_diff);
    assert!(has_weight_diff);
}

/// Test case 2: Core library architecture comparison
#[test]
fn test_core_architecture_comparison() {
    let v1 = parse_json(r#"{"architecture": {"type": "transformer", "layers": 12}}"#);
    let v2 = parse_json(r#"{"architecture": {"type": "transformer", "layers": 24}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    assert!(has_layers_diff);
}

/// Test case 3: Core library structured output for automation
#[test]
fn test_core_json_automation() {
    let v1 = parse_json(r#"{"analysis": {"features": 30, "enabled": true}}"#);
    let v2 = parse_json(r#"{"analysis": {"features": 35, "enabled": true}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_features_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("features"));
    assert!(has_features_diff);
}
