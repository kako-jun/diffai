use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_batch_normalization_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Batch normalization layer detection and analysis
        if let Some((old_bn, new_bn)) = analyze_batch_norm_layers(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "batch_normalization_layers".to_string(),
                old_bn,
                new_bn,
            ));
        }

        // Batch normalization parameter analysis
        if let Some((old_params, new_params)) = analyze_batch_norm_parameters(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "batch_normalization_parameters".to_string(),
                old_params,
                new_params,
            ));
        }

        // Moving statistics analysis for batch norm
        if let Some((old_stats, new_stats)) = analyze_batch_norm_statistics(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "batch_normalization_statistics".to_string(),
                old_stats,
                new_stats,
            ));
        }
    }
}

// Regularization Impact Analysis - measure regularization technique effectiveness
// Helper functions for batch normalization analysis
fn analyze_batch_norm_layers(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_bn_count = count_batch_norm_layers(old_obj);
    let new_bn_count = count_batch_norm_layers(new_obj);

    if old_bn_count != new_bn_count {
        Some((
            format!("batch_norm_layers: {old_bn_count}"),
            format!("batch_norm_layers: {new_bn_count}"),
        ))
    } else {
        None
    }
}

fn count_batch_norm_layers(obj: &serde_json::Map<String, Value>) -> usize {
    let mut count = 0;
    for (key, _) in obj {
        if key.contains("batch_norm") || key.contains("bn") || key.contains("BatchNorm") {
            count += 1;
        }
    }
    count
}

fn analyze_batch_norm_parameters(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_params = extract_batch_norm_params(old_obj);
    let new_params = extract_batch_norm_params(new_obj);

    if old_params != new_params {
        Some((
            format!(
                "bn_params: momentum={:.3}, eps={:.6}",
                old_params.0, old_params.1
            ),
            format!(
                "bn_params: momentum={:.3}, eps={:.6}",
                new_params.0, new_params.1
            ),
        ))
    } else {
        None
    }
}

fn extract_batch_norm_params(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let momentum = obj.get("momentum").and_then(|v| v.as_f64()).unwrap_or(0.1);
    let eps = obj.get("eps").and_then(|v| v.as_f64()).unwrap_or(1e-5);
    (momentum, eps)
}

fn analyze_batch_norm_statistics(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_stats = extract_batch_norm_stats(old_obj);
    let new_stats = extract_batch_norm_stats(new_obj);

    if (old_stats.0 - new_stats.0).abs() > 0.01 || (old_stats.1 - new_stats.1).abs() > 0.01 {
        Some((
            format!(
                "bn_stats: running_mean={:.3}, running_var={:.3}",
                old_stats.0, old_stats.1
            ),
            format!(
                "bn_stats: running_mean={:.3}, running_var={:.3}",
                new_stats.0, new_stats.1
            ),
        ))
    } else {
        None
    }
}

fn extract_batch_norm_stats(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let running_mean = obj
        .get("running_mean")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let running_var = obj
        .get("running_var")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    (running_mean, running_var)
}
