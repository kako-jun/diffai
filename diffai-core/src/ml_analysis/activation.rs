use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_activation_pattern_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Detect direct activation function changes
        detect_activation_function_changes(old_obj, new_obj, results);

        // Activation saturation analysis
        if let Some((old_saturation, new_saturation)) =
            analyze_activation_saturation(old_obj, new_obj)
        {
            results.push(DiffResult::ModelArchitectureChanged(
                "activation_saturation".to_string(),
                old_saturation,
                new_saturation,
            ));
        }

        // Dead neuron analysis
        if let Some((old_dead, new_dead)) = analyze_dead_neurons(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "dead_neurons".to_string(),
                old_dead,
                new_dead,
            ));
        }
    }
}

/// Detect activation function changes in various formats
fn detect_activation_function_changes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
) {
    // Common keys for activation functions
    let activation_keys = [
        "activation",
        "activation_fn",
        "activation_function",
        "act_fn",
        "nonlinearity",
        "hidden_act",
        "output_activation",
    ];

    // Check top-level activation keys
    for key in &activation_keys {
        if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
            let old_str = value_to_activation_string(old_val);
            let new_str = value_to_activation_string(new_val);
            if old_str != new_str {
                results.push(DiffResult::ActivationFunctionChanged(
                    key.to_string(),
                    old_str,
                    new_str,
                ));
            }
        }
    }

    // Check nested structures (model_config, network, variables, etc.)
    let nested_keys = ["model_config", "config", "network", "variables"];
    for nested_key in &nested_keys {
        if let (Some(Value::Object(old_nested)), Some(Value::Object(new_nested))) =
            (old_obj.get(*nested_key), new_obj.get(*nested_key))
        {
            for key in &activation_keys {
                if let (Some(old_val), Some(new_val)) = (old_nested.get(*key), new_nested.get(*key))
                {
                    let old_str = value_to_activation_string(old_val);
                    let new_str = value_to_activation_string(new_val);
                    if old_str != new_str {
                        results.push(DiffResult::ActivationFunctionChanged(
                            format!("{nested_key}.{key}"),
                            old_str,
                            new_str,
                        ));
                    }
                }
            }
        }
    }

    // Check variables.network for MATLAB format
    if let (Some(Value::Object(old_vars)), Some(Value::Object(new_vars))) =
        (old_obj.get("variables"), new_obj.get("variables"))
    {
        if let (Some(Value::Object(old_net)), Some(Value::Object(new_net))) =
            (old_vars.get("network"), new_vars.get("network"))
        {
            for key in &activation_keys {
                if let (Some(old_val), Some(new_val)) = (old_net.get(*key), new_net.get(*key)) {
                    let old_str = value_to_activation_string(old_val);
                    let new_str = value_to_activation_string(new_val);
                    if old_str != new_str {
                        results.push(DiffResult::ActivationFunctionChanged(
                            format!("variables.network.{key}"),
                            old_str,
                            new_str,
                        ));
                    }
                }
            }
        }
    }
}

fn value_to_activation_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        _ => val.to_string(),
    }
}

// Weight Distribution Analysis - statistical analysis of weight distributions
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
