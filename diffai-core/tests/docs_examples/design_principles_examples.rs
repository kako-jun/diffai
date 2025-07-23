use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library comprehensive ML analysis (automatic)
#[test]
fn test_core_comprehensive_ml_analysis() {
    let v1 = parse_json(r#"{"model": {"type": "pytorch", "layers": 5}}"#);
    let v2 = parse_json(r#"{"model": {"type": "pytorch", "layers": 8}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_layers_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("layers"));
    assert!(has_layers_diff);
}

/// Test case 2: Core library detailed diagnostics with comprehensive analysis
#[test]
fn test_core_verbose_comprehensive() {
    let v1 = parse_json(r#"{"diagnostics": {"enabled": true}}"#);
    let v2 = parse_json(r#"{"diagnostics": {"enabled": false}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_enabled_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("enabled"));
    assert!(has_enabled_diff);
}

/// Test case 3: Core library directory structure comparison
#[test]
fn test_core_directory_comparison() {
    let v1 = parse_json(r#"{"directory": {"files": 5, "subdirs": 2}}"#);
    let v2 = parse_json(r#"{"directory": {"files": 7, "subdirs": 3}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_files_diff = results.iter().any(|r| format!("{:?}", r).contains("files"));
    let has_subdirs_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("subdirs"));
    assert!(has_files_diff);
    assert!(has_subdirs_diff);
}

/// Test case 4: Core library ML analysis automatic features
#[test]
fn test_core_ml_analysis_automatic() {
    let v1 = parse_json(r#"{"ml": {"features": 30, "accuracy": 0.85}}"#);
    let v2 = parse_json(r#"{"ml": {"features": 35, "accuracy": 0.90}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_features_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("features"));
    let has_accuracy_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("accuracy"));
    assert!(has_features_diff);
    assert!(has_accuracy_diff);
}

/// Test case 5: Core library verbose ML analysis with debug info
#[test]
fn test_core_verbose_ml_analysis() {
    let v1 = parse_json(r#"{"debug": {"level": 1}}"#);
    let v2 = parse_json(r#"{"debug": {"level": 2}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_level_diff = results.iter().any(|r| format!("{:?}", r).contains("level"));
    assert!(has_level_diff);
}

/// Test case 6: Core library comprehensive analysis with structured output
#[test]
fn test_core_json_comprehensive_analysis() {
    let v1 = parse_json(r#"{"output": {"format": "cli"}}"#);
    let v2 = parse_json(r#"{"output": {"format": "json"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_format_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("format"));
    assert!(has_format_diff);
}
