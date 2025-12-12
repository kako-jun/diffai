use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_regularization_impact(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Dropout analysis
        if let Some((old_dropout, new_dropout)) = analyze_dropout_patterns(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "dropout_regularization".to_string(),
                old_dropout,
                new_dropout,
            ));
        }

        // Weight decay analysis
        if let Some((old_decay, new_decay)) = analyze_weight_decay_impact(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_decay_impact".to_string(),
                old_decay,
                new_decay,
            ));
        }

        // L1/L2 regularization analysis
        if let Some((old_l_reg, new_l_reg)) = analyze_l_regularization(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "l_regularization".to_string(),
                old_l_reg,
                new_l_reg,
            ));
        }
    }
}

// Activation Pattern Analysis - analyze activation function patterns
// Helper functions for regularization analysis
fn analyze_dropout_patterns(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_dropout = extract_dropout_rate(old_obj);
    let new_dropout = extract_dropout_rate(new_obj);

    if (old_dropout - new_dropout).abs() > 0.001 {
        Some((
            format!("dropout_rate: {old_dropout:.3}"),
            format!("dropout_rate: {new_dropout:.3}"),
        ))
    } else {
        None
    }
}

fn extract_dropout_rate(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("dropout")
        .or_else(|| obj.get("dropout_rate"))
        .or_else(|| obj.get("p"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn analyze_weight_decay_impact(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_decay = extract_weight_decay(old_obj);
    let new_decay = extract_weight_decay(new_obj);

    if (old_decay - new_decay).abs() > 1e-6 {
        Some((
            format!("weight_decay: {old_decay:.6}"),
            format!("weight_decay: {new_decay:.6}"),
        ))
    } else {
        None
    }
}

fn extract_weight_decay(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("weight_decay")
        .or_else(|| obj.get("l2_reg"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn analyze_l_regularization(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_l1 = extract_l1_reg(old_obj);
    let new_l1 = extract_l1_reg(new_obj);
    let old_l2 = extract_l2_reg(old_obj);
    let new_l2 = extract_l2_reg(new_obj);

    if (old_l1 - new_l1).abs() > 1e-6 || (old_l2 - new_l2).abs() > 1e-6 {
        Some((
            format!("l_reg: L1={old_l1:.6}, L2={old_l2:.6}"),
            format!("l_reg: L1={new_l1:.6}, L2={new_l2:.6}"),
        ))
    } else {
        None
    }
}

fn extract_l1_reg(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("l1_reg").and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn extract_l2_reg(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("l2_reg").and_then(|v| v.as_f64()).unwrap_or(0.0)
}
