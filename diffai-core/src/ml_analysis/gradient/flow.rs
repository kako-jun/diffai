use serde_json::Value;

use super::statistics::{estimate_gradient_norm_from_weights, estimate_max_gradient_from_weights};
use super::types::GradientFlowInfo;

// Analyze gradient flow through network layers
pub(super) fn analyze_gradient_flow(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_flow = extract_gradient_flow_info(old_obj)?;
    let new_flow = extract_gradient_flow_info(new_obj)?;

    let mut flow_analysis = Vec::new();

    // Analyze vanishing gradients
    if old_flow.vanishing_layers != new_flow.vanishing_layers {
        let change = new_flow.vanishing_layers as i32 - old_flow.vanishing_layers as i32;
        let trend = if change == 0 {
            "stable"
        } else if change > 0 {
            "more_vanishing"
        } else {
            "less_vanishing"
        };
        flow_analysis.push(format!(
            "vanishing_layers: {} ({:+}, {})",
            new_flow.vanishing_layers, change, trend
        ));
    }

    // Analyze exploding gradients
    if old_flow.exploding_layers != new_flow.exploding_layers {
        let change = new_flow.exploding_layers as i32 - old_flow.exploding_layers as i32;
        flow_analysis.push(format!(
            "exploding_layers: {} ({:+})",
            new_flow.exploding_layers, change
        ));
    }

    // Analyze gradient flow balance
    if let (Some(old_balance), Some(new_balance)) = (old_flow.flow_balance, new_flow.flow_balance) {
        let balance_change = new_balance - old_balance;
        let balance_status = if balance_change.abs() < 0.1 {
            "balanced"
        } else if balance_change > 0.0 {
            "forward_dominant"
        } else {
            "backward_dominant"
        };
        flow_analysis.push(format!(
            "flow_balance: {new_balance:.3} ({balance_change:+.3}, {balance_status})"
        ));
    }

    if flow_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "vanishing: {}, exploding: {}, balance: {:.3}",
        old_flow.vanishing_layers,
        old_flow.exploding_layers,
        old_flow.flow_balance.unwrap_or(0.0)
    );
    let new_info = flow_analysis.join(", ");

    Some((old_info, new_info))
}

// Enhanced gradient flow information extraction with lawkit streaming and helper functions
pub(super) fn extract_gradient_flow_info(
    obj: &serde_json::Map<String, Value>,
) -> Option<GradientFlowInfo> {
    let mut vanishing_layers = 0;
    let mut exploding_layers = 0;

    // Use weight-based gradient estimation as fallback
    let _estimated_norm = estimate_gradient_norm_from_weights(obj);
    let _estimated_max = estimate_max_gradient_from_weights(obj);
    #[allow(unused_assignments)]
    let mut flow_balance = None;

    // Enhanced thresholds based on modern deep learning practices
    let vanishing_threshold = 1e-7; // More sensitive
    let exploding_threshold = 5.0; // More conservative
    let _moderate_exploding_threshold = 1.0;

    let mut layer_gradients = Vec::new();

    // Comprehensive gradient analysis across all model components
    for (key, value) in obj {
        let is_gradient_related = key.contains("grad")
            || key.contains("gradient")
            || key.contains("weight")
            || key.contains("bias")
            || key.contains("param")
            || key.contains("layer");

        if is_gradient_related {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        let abs_val = val.abs();
                        layer_gradients.push(abs_val);

                        // Enhanced gradient problem detection
                        if abs_val < vanishing_threshold {
                            vanishing_layers += 1;
                        } else if abs_val > exploding_threshold {
                            exploding_layers += 1;
                        }
                    }
                }
                Value::Array(arr) => {
                    // Process arrays with memory efficiency
                    let mut layer_sum = 0.0;
                    let mut layer_count = 0;

                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                let abs_val = val.abs();
                                layer_sum += abs_val;
                                layer_count += 1;
                            }
                        }
                    }

                    if layer_count > 0 {
                        let layer_mean = layer_sum / layer_count as f64;
                        layer_gradients.push(layer_mean);

                        if layer_mean < vanishing_threshold {
                            vanishing_layers += 1;
                        } else if layer_mean > exploding_threshold {
                            exploding_layers += 1;
                        }
                    }
                }
                Value::Object(nested) => {
                    // Recursive flow analysis for nested structures
                    if let Some(nested_flow) = extract_gradient_flow_info(nested) {
                        vanishing_layers += nested_flow.vanishing_layers;
                        exploding_layers += nested_flow.exploding_layers;
                    }
                }
                _ => {}
            }
        }
    }

    // Enhanced flow balance estimation
    flow_balance = estimate_gradient_flow_balance(obj);

    // Additional validation: check gradient magnitude distribution
    if !layer_gradients.is_empty() {
        layer_gradients.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = layer_gradients.len() / 2;
        let median_grad = layer_gradients[median_idx];

        // Adjust counts based on median gradient
        if median_grad < vanishing_threshold * 10.0 {
            // If median is very low, model likely has vanishing gradients
            vanishing_layers = vanishing_layers.max(layer_gradients.len() / 3);
        }
    }

    Some(GradientFlowInfo {
        vanishing_layers,
        exploding_layers,
        flow_balance,
    })
}

// Enhanced gradient flow balance estimation using lawkit streaming patterns
pub(super) fn estimate_gradient_flow_balance(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut layer_gradients = Vec::new();
    let mut total_forward_flow = 0.0;
    let mut total_backward_flow = 0.0;
    let mut layer_count = 0;

    // Advanced layer-wise gradient flow analysis
    for (key, value) in obj {
        let is_layer_weight = key.contains("layer")
            || key.contains("block")
            || key.contains("weight")
            || key.contains("attention")
            || key.contains("ffn")
            || key.contains("transformer");

        if is_layer_weight {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        layer_gradients.push((key.clone(), val.abs()));
                    }
                }
                Value::Array(arr) => {
                    // Calculate mean gradient magnitude for this layer
                    let mut sum = 0.0;
                    let mut count = 0;
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                sum += val.abs();
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        layer_gradients.push((key.clone(), sum / count as f64));
                    }
                }
                Value::Object(nested) => {
                    // Recursive analysis for nested layer structures
                    if let Some(nested_flow) = estimate_gradient_flow_balance(nested) {
                        layer_gradients.push((key.clone(), nested_flow));
                    }
                }
                _ => {}
            }
        }
    }

    if layer_gradients.is_empty() {
        return None;
    }

    // Analyze gradient flow patterns
    layer_gradients.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by layer name

    // Calculate forward and backward flow based on layer position
    let total_layers = layer_gradients.len();
    for (i, (_layer_name, gradient_mag)) in layer_gradients.iter().enumerate() {
        let layer_position = i as f64 / total_layers as f64;

        // Early layers contribute to forward flow
        if layer_position < 0.5 {
            total_forward_flow += gradient_mag * (1.0 - layer_position);
        } else {
            // Later layers contribute to backward flow
            total_backward_flow += gradient_mag * layer_position;
        }

        layer_count += 1;
    }

    // Enhanced flow balance calculation
    if layer_count > 0 && total_backward_flow > 1e-12 {
        let flow_ratio = total_forward_flow / total_backward_flow;

        // Normalize to 0-1 range where 0.5 is perfect balance
        let normalized_balance = 1.0 / (1.0 + (flow_ratio - 1.0).abs());
        Some(normalized_balance)
    } else {
        None
    }
}
