use serde_json::Value;

use super::loss::extract_loss_trajectory;

#[derive(Debug)]
pub(crate) struct TrainingStabilityMetrics {
    pub(crate) gradient_variance: Option<f64>,
    pub(crate) loss_oscillation: f64,
    pub(crate) parameter_drift: f64,
    pub(crate) overall_score: f64,
}

/// Enhanced training stability analysis with statistical significance
pub(crate) fn analyze_training_stability_statistical(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_stability = calculate_training_stability_metrics(old_obj)?;
    let new_stability = calculate_training_stability_metrics(new_obj)?;

    let mut stability_changes = Vec::new();

    // Compare variance in gradients
    if let (Some(old_grad_var), Some(new_grad_var)) = (
        old_stability.gradient_variance,
        new_stability.gradient_variance,
    ) {
        let variance_change = (new_grad_var - old_grad_var) / old_grad_var.max(1e-8);
        if variance_change.abs() > 0.1 {
            stability_changes.push(format!(
                "gradient_variance: {:+.2}%",
                variance_change * 100.0
            ));
        }
    }

    // Compare loss oscillation
    if (old_stability.loss_oscillation - new_stability.loss_oscillation).abs() > 0.05 {
        let oscillation_change = new_stability.loss_oscillation - old_stability.loss_oscillation;
        stability_changes.push(format!("loss_oscillation: {oscillation_change:+.3}"));
    }

    // Compare overall stability score
    if (old_stability.overall_score - new_stability.overall_score).abs() > 0.05 {
        let score_change = new_stability.overall_score - old_stability.overall_score;
        stability_changes.push(format!("stability_score: {score_change:+.3}"));
    }

    if stability_changes.is_empty() {
        return None;
    }

    let old_info = format!(
        "oscillation: {:.3}, score: {:.3}",
        old_stability.loss_oscillation, old_stability.overall_score
    );
    let new_info = stability_changes.join(", ");

    Some((old_info, new_info))
}

/// Calculate training stability metrics
pub(crate) fn calculate_training_stability_metrics(
    obj: &serde_json::Map<String, Value>,
) -> Option<TrainingStabilityMetrics> {
    // Extract gradient variance if available
    let gradient_variance = extract_gradient_variance(obj);

    // Calculate loss oscillation
    let loss_trajectory = extract_loss_trajectory(obj)?;
    let loss_oscillation = if loss_trajectory.len() > 2 {
        calculate_oscillation_metric(&loss_trajectory)
    } else {
        0.0
    };

    // Calculate parameter drift
    let parameter_drift = calculate_parameter_drift(obj);

    // Overall stability score (higher is better)
    let overall_score = {
        let base_score = 1.0 - loss_oscillation.min(1.0);
        let gradient_penalty = gradient_variance.map_or(0.0, |gv| (gv * 0.1).min(0.3));
        let drift_penalty = (parameter_drift * 0.2).min(0.3);
        (base_score - gradient_penalty - drift_penalty).max(0.0)
    };

    Some(TrainingStabilityMetrics {
        gradient_variance,
        loss_oscillation,
        parameter_drift,
        overall_score,
    })
}

/// Analyze training stability from various metrics
pub(crate) fn analyze_training_stability(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let mut stability_factors = Vec::new();

    // Check gradient norms if available
    if let (Some(old_grad), Some(new_grad)) = (
        extract_gradient_norm(old_obj),
        extract_gradient_norm(new_obj),
    ) {
        let grad_change = (new_grad / old_grad - 1.0) * 100.0;
        let grad_stability = if grad_change.abs() < 10.0 {
            "stable"
        } else if grad_change.abs() < 50.0 {
            "moderate_variation"
        } else {
            "high_variation"
        };
        stability_factors.push(format!("gradient_norm: {grad_stability}"));
    }

    // Check learning rate stability
    if let (Some(old_lr), Some(new_lr)) = (
        extract_current_learning_rate(old_obj),
        extract_current_learning_rate(new_obj),
    ) {
        let lr_ratio = new_lr / old_lr;
        let lr_stability = if (lr_ratio - 1.0).abs() < 0.1 {
            "stable"
        } else if lr_ratio < 1.0 {
            "decreasing"
        } else {
            "increasing"
        };
        stability_factors.push(format!("learning_rate: {lr_stability}"));
    }

    // Check parameter magnitude changes
    if let (Some(old_params), Some(new_params)) = (
        estimate_parameter_magnitude(old_obj),
        estimate_parameter_magnitude(new_obj),
    ) {
        let param_change = ((new_params / old_params - 1.0) * 100.0).abs();
        let param_stability = if param_change < 1.0 {
            "stable"
        } else if param_change < 5.0 {
            "mild_change"
        } else {
            "significant_change"
        };
        stability_factors.push(format!("parameters: {param_stability}"));
    }

    if stability_factors.is_empty() {
        return None;
    }

    let old_info = "evaluating".to_string();
    let new_info = stability_factors.join(", ");

    Some((old_info, new_info))
}

// Helper functions

/// Extract gradient norm from model checkpoint
pub(super) fn extract_gradient_norm(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let grad_keys = ["grad_norm", "gradient_norm", "total_grad_norm"];
    for key in &grad_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

fn extract_gradient_variance(_obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Placeholder implementation
    None
}

fn calculate_oscillation_metric(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 3 {
        return 0.0;
    }

    let mut oscillations = 0;
    for i in 1..trajectory.len() - 1 {
        let prev_diff = trajectory[i] - trajectory[i - 1];
        let curr_diff = trajectory[i + 1] - trajectory[i];

        if prev_diff * curr_diff < 0.0 {
            oscillations += 1;
        }
    }

    oscillations as f64 / (trajectory.len() - 2) as f64
}

fn calculate_parameter_drift(_obj: &serde_json::Map<String, Value>) -> f64 {
    // Placeholder implementation
    0.1
}

fn extract_current_learning_rate(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let lr_keys = ["lr", "learning_rate", "current_lr"];
    for key in &lr_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

/// Estimate parameter magnitude (used by other modules)
pub(super) fn estimate_parameter_magnitude(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Simple heuristic based on detected weights
    let mut total_magnitude = 0.0;
    let mut count = 0;

    for (key, value) in obj {
        if key.contains("weight") || key.contains("bias") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    total_magnitude += val.abs();
                    count += 1;
                }
            }
        }
    }

    if count > 0 {
        Some(total_magnitude / count as f64)
    } else {
        None
    }
}
