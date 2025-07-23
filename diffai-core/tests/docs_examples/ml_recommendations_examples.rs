use diffai_core::diff;
use serde_json::Value;

// Helper function to parse JSON strings
fn parse_json(json_str: &str) -> Value {
    serde_json::from_str(json_str).expect("Failed to parse JSON")
}

/// Test case 1: Core library deployment recommendations
#[test]
fn test_core_deployment_recommendations() {
    let v1 = parse_json(r#"{"model": {"performance": 0.85, "memory": 512}}"#);
    let v2 = parse_json(r#"{"model": {"performance": 0.75, "memory": 1024}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_performance_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("performance"));
    let has_memory_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("memory"));
    assert!(has_performance_diff);
    assert!(has_memory_diff);
}

/// Test case 2: Core library JSON recommendations processing
#[test]
fn test_core_json_recommendations() {
    let v1 = parse_json(r#"{"recommendations": {"enabled": true, "level": "high"}}"#);
    let v2 = parse_json(r#"{"recommendations": {"enabled": true, "level": "critical"}}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_level_diff = results.iter().any(|r| format!("{:?}", r).contains("level"));
    assert!(has_level_diff);
}

/// Test case 3: Core library training progress recommendations
#[test]
fn test_core_training_progress() {
    let v1 = parse_json(r#"{"epoch": 10, "loss": 0.5, "accuracy": 0.80}"#);
    let v2 = parse_json(r#"{"epoch": 20, "loss": 0.3, "accuracy": 0.85}"#);

    let results = diff(&v1, &v2, None, None, None);
    assert!(!results.is_empty());

    let has_epoch_diff = results.iter().any(|r| format!("{:?}", r).contains("epoch"));
    let has_loss_diff = results.iter().any(|r| format!("{:?}", r).contains("loss"));
    let has_accuracy_diff = results
        .iter()
        .any(|r| format!("{:?}", r).contains("accuracy"));
    assert!(has_epoch_diff);
    assert!(has_loss_diff);
    assert!(has_accuracy_diff);
}
