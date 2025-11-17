use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_weight_distribution_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
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

// Model Complexity Assessment - comprehensive model complexity evaluation
// Helper functions for weight distribution analysis
fn analyze_weight_distributions(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_stats = calculate_weight_stats(old_obj);
    let new_stats = calculate_weight_stats(new_obj);
    
    if (old_stats.0 - new_stats.0).abs() > 0.001 || (old_stats.1 - new_stats.1).abs() > 0.001 {
        Some((
            format!("weight_stats: mean={:.4}, std={:.4}", old_stats.0, old_stats.1),
            format!("weight_stats: mean={:.4}, std={:.4}", new_stats.0, new_stats.1),
        ))
    } else {
        None
    }
}

fn calculate_weight_stats(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let mean = obj.get("weight_mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let std = obj.get("weight_std").and_then(|v| v.as_f64()).unwrap_or(1.0);
    (mean, std)
}

fn analyze_weight_initialization(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_init = extract_weight_init_method(old_obj);
    let new_init = extract_weight_init_method(new_obj);
    
    if old_init != new_init {
        Some((
            format!("weight_init: {}", old_init),
            format!("weight_init: {}", new_init),
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

fn analyze_weight_sparsity(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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
