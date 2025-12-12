use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_weight_distribution_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Detect significant weight changes
        detect_weight_significant_changes(old_obj, new_obj, results);

        // Weight distribution statistics
        if let Some((old_dist, new_dist)) = analyze_weight_distributions(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_distributions".to_string(),
                old_dist,
                new_dist,
            ));
        }

        // Weight initialization analysis
        if let Some((old_init, new_init)) = analyze_weight_initialization(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_initialization".to_string(),
                old_init,
                new_init,
            ));
        }

        // Weight sparsity analysis
        if let Some((old_sparsity, new_sparsity)) = analyze_weight_sparsity(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_sparsity".to_string(),
                old_sparsity,
                new_sparsity,
            ));
        }
    }
}

/// Detect significant weight changes
fn detect_weight_significant_changes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
) {
    // Check parameters container
    let container_keys = ["parameters", "weights", "state_dict", "model_state_dict"];

    for container_key in &container_keys {
        if let (Some(Value::Object(old_params)), Some(Value::Object(new_params))) =
            (old_obj.get(*container_key), new_obj.get(*container_key))
        {
            for (key, old_val) in old_params {
                if let Some(new_val) = new_params.get(key) {
                    // For "weights" container, check all entries
                    // For other containers, only check weight/bias keys
                    if *container_key == "weights" || key.contains("weight") || key.contains("bias")
                    {
                        check_weight_change(
                            &format!("{container_key}.{key}"),
                            old_val,
                            new_val,
                            results,
                        );
                    }
                }
            }
        }
    }

    // Check top-level weight keys
    for (key, old_val) in old_obj {
        if let Some(new_val) = new_obj.get(key) {
            if key.contains("weight") || key.contains("bias") {
                check_weight_change(key, old_val, new_val, results);
            }
        }
    }
}

fn check_weight_change(
    path: &str,
    old_val: &Value,
    new_val: &Value,
    results: &mut Vec<DiffResult>,
) {
    // Handle scalar weights
    if let (Some(old_f), Some(new_f)) = (old_val.as_f64(), new_val.as_f64()) {
        let magnitude = (new_f - old_f).abs();
        if magnitude >= 0.05 {
            results.push(DiffResult::WeightSignificantChange(
                path.to_string(),
                magnitude,
            ));
        }
    }

    // Handle array weights
    if let (Value::Array(old_arr), Value::Array(new_arr)) = (old_val, new_val) {
        let old_vals: Vec<f64> = old_arr.iter().filter_map(|v| v.as_f64()).collect();
        let new_vals: Vec<f64> = new_arr.iter().filter_map(|v| v.as_f64()).collect();

        if !old_vals.is_empty() && old_vals.len() == new_vals.len() {
            let max_diff = old_vals
                .iter()
                .zip(new_vals.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, |a, b| a.max(b));

            if max_diff >= 0.05 {
                results.push(DiffResult::WeightSignificantChange(
                    path.to_string(),
                    max_diff,
                ));
            }
        }
    }

    // Handle object with statistics
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_val, new_val) {
        // Check mean/std changes
        if let (Some(old_mean), Some(new_mean)) = (
            old_obj.get("mean").and_then(|v| v.as_f64()),
            new_obj.get("mean").and_then(|v| v.as_f64()),
        ) {
            let magnitude = (new_mean - old_mean).abs();
            // For weight objects, use relative change threshold (50% change)
            // or absolute threshold (0.01) for small values
            let relative_change = if old_mean.abs() > 1e-10 {
                magnitude / old_mean.abs()
            } else {
                magnitude
            };
            if magnitude >= 0.05 || (magnitude >= 0.005 && relative_change >= 0.5) {
                results.push(DiffResult::WeightSignificantChange(
                    path.to_string(),
                    magnitude,
                ));
            }
        }
    }
}

// Model Complexity Assessment - comprehensive model complexity evaluation
// Helper functions for weight distribution analysis
fn analyze_weight_distributions(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_stats = calculate_weight_stats(old_obj);
    let new_stats = calculate_weight_stats(new_obj);

    if (old_stats.0 - new_stats.0).abs() > 0.001 || (old_stats.1 - new_stats.1).abs() > 0.001 {
        Some((
            format!(
                "weight_stats: mean={:.4}, std={:.4}",
                old_stats.0, old_stats.1
            ),
            format!(
                "weight_stats: mean={:.4}, std={:.4}",
                new_stats.0, new_stats.1
            ),
        ))
    } else {
        None
    }
}

fn calculate_weight_stats(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let mean = obj
        .get("weight_mean")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let std = obj
        .get("weight_std")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    (mean, std)
}

fn analyze_weight_initialization(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_init = extract_weight_init_method(old_obj);
    let new_init = extract_weight_init_method(new_obj);

    if old_init != new_init {
        Some((
            format!("weight_init: {old_init}"),
            format!("weight_init: {new_init}"),
        ))
    } else {
        None
    }
}

fn extract_weight_init_method(obj: &serde_json::Map<String, Value>) -> String {
    obj.get("weight_init")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string()
}

fn analyze_weight_sparsity(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_sparsity = calculate_weight_sparsity(old_obj);
    let new_sparsity = calculate_weight_sparsity(new_obj);

    if (old_sparsity - new_sparsity).abs() > 0.01 {
        Some((
            format!("sparsity: {:.1}%", old_sparsity * 100.0),
            format!("sparsity: {:.1}%", new_sparsity * 100.0),
        ))
    } else {
        None
    }
}

fn calculate_weight_sparsity(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("weight_sparsity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}
