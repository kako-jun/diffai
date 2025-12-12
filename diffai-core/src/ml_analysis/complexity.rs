use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_model_complexity_assessment(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Parameter count analysis
        if let Some((old_params, new_params)) = analyze_parameter_count(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "parameter_count".to_string(),
                old_params,
                new_params,
            ));
        }

        // Computational complexity analysis
        if let Some((old_flops, new_flops)) = analyze_computational_complexity(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "computational_complexity".to_string(),
                old_flops,
                new_flops,
            ));
        }

        // Model depth and width analysis
        if let Some((old_arch, new_arch)) = analyze_model_architecture_complexity(old_obj, new_obj)
        {
            results.push(DiffResult::ModelArchitectureChanged(
                "architecture_complexity".to_string(),
                old_arch,
                new_arch,
            ));
        }
    }
}

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

// Helper functions for activation pattern analysis
fn analyze_activation_functions(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_activations = extract_activation_functions(old_obj);
    let new_activations = extract_activation_functions(new_obj);

    if old_activations != new_activations {
        Some((
            format!("activations: {}", old_activations.join(", ")),
            format!("activations: {}", new_activations.join(", ")),
        ))
    } else {
        None
    }
}

fn extract_activation_functions(obj: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut activations = Vec::new();
    for (key, _) in obj {
        if key.contains("activation")
            || key.contains("relu")
            || key.contains("sigmoid")
            || key.contains("tanh")
            || key.contains("gelu")
            || key.contains("swish")
        {
            activations.push(key.clone());
        }
    }
    activations.sort();
    activations
}

fn analyze_activation_saturation(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_saturation = calculate_activation_saturation(old_obj);
    let new_saturation = calculate_activation_saturation(new_obj);

    if (old_saturation - new_saturation).abs() > 0.01 {
        Some((
            format!("saturation: {:.2}%", old_saturation * 100.0),
            format!("saturation: {:.2}%", new_saturation * 100.0),
        ))
    } else {
        None
    }
}

fn calculate_activation_saturation(obj: &serde_json::Map<String, Value>) -> f64 {
    // Estimate saturation based on activation statistics
    obj.get("activation_stats")
        .and_then(|v| v.get("saturation"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn analyze_dead_neurons(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_dead = count_dead_neurons(old_obj);
    let new_dead = count_dead_neurons(new_obj);

    if old_dead != new_dead {
        Some((
            format!("dead_neurons: {old_dead}"),
            format!("dead_neurons: {new_dead}"),
        ))
    } else {
        None
    }
}

fn count_dead_neurons(obj: &serde_json::Map<String, Value>) -> usize {
    obj.get("dead_neurons")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize
}

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

// Helper functions for model complexity assessment
fn analyze_parameter_count(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_params = extract_parameter_count(old_obj);
    let new_params = extract_parameter_count(new_obj);

    if old_params != new_params {
        Some((
            format!("parameters: {}", format_number(old_params)),
            format!("parameters: {}", format_number(new_params)),
        ))
    } else {
        None
    }
}

fn extract_parameter_count(obj: &serde_json::Map<String, Value>) -> u64 {
    obj.get("parameter_count")
        .or_else(|| obj.get("num_parameters"))
        .or_else(|| obj.get("total_params"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
}

fn format_number(num: u64) -> String {
    if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

fn analyze_computational_complexity(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_flops = extract_flops(old_obj);
    let new_flops = extract_flops(new_obj);

    if old_flops != new_flops {
        Some((
            format!("flops: {}", format_number(old_flops)),
            format!("flops: {}", format_number(new_flops)),
        ))
    } else {
        None
    }
}

fn extract_flops(obj: &serde_json::Map<String, Value>) -> u64 {
    obj.get("flops")
        .or_else(|| obj.get("gflops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
}

fn analyze_model_architecture_complexity(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_depth = extract_model_depth(old_obj);
    let new_depth = extract_model_depth(new_obj);
    let old_width = extract_model_width(old_obj);
    let new_width = extract_model_width(new_obj);

    if old_depth != new_depth || old_width != new_width {
        Some((
            format!("architecture: depth={old_depth}, width={old_width}"),
            format!("architecture: depth={new_depth}, width={new_width}"),
        ))
    } else {
        None
    }
}

fn extract_model_depth(obj: &serde_json::Map<String, Value>) -> u32 {
    obj.get("depth")
        .or_else(|| obj.get("num_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

fn extract_model_width(obj: &serde_json::Map<String, Value>) -> u32 {
    obj.get("width")
        .or_else(|| obj.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

// ============================================================================
// PARSER FUNCTIONS - FOR INTERNAL USE ONLY
// ============================================================================
// These functions are public only for CLI and language bindings.
// External users should use the main diff() function with file reading.
