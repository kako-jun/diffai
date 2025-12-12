use serde_json::Value;

use super::epoch::extract_epoch_info;
use super::learning_curves::calculate_convergence_rate;
use super::loss::extract_loss_trajectory;
use super::stability::{estimate_parameter_magnitude, extract_gradient_norm};

#[derive(Debug, Clone)]
pub(crate) struct OptimizationTrajectory {
    pub(crate) parameter_stability: f64,
    pub(crate) gradient_flow_health: f64,
    pub(crate) learning_efficiency: f64,
    pub(crate) overfitting_risk: f64,
    pub(crate) generalization_gap: Option<f64>,
}

/// Enhanced optimization trajectory analysis using helper functions
pub(crate) fn analyze_optimization_trajectory(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_trajectory = extract_optimization_trajectory(old_obj)?;
    let new_trajectory = extract_optimization_trajectory(new_obj)?;

    // Use helper functions for enhanced analysis
    let old_lr = extract_current_learning_rate(old_obj);
    let new_lr = extract_current_learning_rate(new_obj);
    let old_param_mag = estimate_parameter_magnitude(old_obj);
    let new_param_mag = estimate_parameter_magnitude(new_obj);
    let old_epoch = extract_epoch_info(old_obj);
    let new_epoch = extract_epoch_info(new_obj);

    let mut trajectory_changes = Vec::new();

    // Learning rate analysis
    if let (Some(old_rate), Some(new_rate)) = (old_lr, new_lr) {
        if (old_rate - new_rate).abs() > old_rate * 0.1 {
            let lr_change = ((new_rate - old_rate) / old_rate) * 100.0;
            trajectory_changes.push(format!("learning_rate: {lr_change:+.2}%"));
        }
    }

    // Parameter magnitude analysis
    if let (Some(old_mag), Some(new_mag)) = (old_param_mag, new_param_mag) {
        if (old_mag - new_mag).abs() > old_mag * 0.05 {
            let mag_change = ((new_mag - old_mag) / old_mag) * 100.0;
            trajectory_changes.push(format!("parameter_magnitude: {mag_change:+.2}%"));
        }
    }

    // Epoch progression analysis
    if let (Some(old_ep), Some(new_ep)) = (old_epoch, new_epoch) {
        if new_ep > old_ep {
            trajectory_changes.push(format!("epoch_progress: {old_ep} -> {new_ep}"));
        }
    }

    let stability_change = new_trajectory.parameter_stability - old_trajectory.parameter_stability;
    if stability_change.abs() > 0.05 {
        trajectory_changes.push(format!("param_stability: {stability_change:+.3}"));
    }

    let efficiency_change = new_trajectory.learning_efficiency - old_trajectory.learning_efficiency;
    if efficiency_change.abs() > 0.05 {
        trajectory_changes.push(format!("learning_efficiency: {efficiency_change:+.3}"));
    }

    if let (Some(old_gap), Some(new_gap)) = (
        old_trajectory.generalization_gap,
        new_trajectory.generalization_gap,
    ) {
        let gap_change = new_gap - old_gap;
        if gap_change.abs() > 0.02 {
            trajectory_changes.push(format!("generalization_gap: {gap_change:+.3}"));
        }
    }

    if trajectory_changes.is_empty() {
        return None;
    }

    let old_info = format!(
        "stability: {:.3}, efficiency: {:.3}",
        old_trajectory.parameter_stability, old_trajectory.learning_efficiency
    );
    let new_info = trajectory_changes.join(", ");

    Some((old_info, new_info))
}

/// Extract optimization trajectory from model checkpoint
pub(crate) fn extract_optimization_trajectory(
    obj: &serde_json::Map<String, Value>,
) -> Option<OptimizationTrajectory> {
    let parameter_stability = calculate_parameter_stability(obj);
    let gradient_flow_health = calculate_gradient_flow_health(obj);
    let learning_efficiency = calculate_learning_efficiency(obj);
    let overfitting_risk = calculate_overfitting_risk(obj);
    let generalization_gap = extract_generalization_gap(obj);

    Some(OptimizationTrajectory {
        parameter_stability,
        gradient_flow_health,
        learning_efficiency,
        overfitting_risk,
        generalization_gap,
    })
}

/// Detect oscillation pattern (shared function)
pub(super) fn detect_oscillation_pattern(trajectory: &[f64]) -> String {
    if trajectory.len() < 4 {
        return "insufficient_data".to_string();
    }

    let mut direction_changes = 0;
    for i in 1..trajectory.len() - 1 {
        let prev_trend = trajectory[i] - trajectory[i - 1];
        let curr_trend = trajectory[i + 1] - trajectory[i];

        if prev_trend * curr_trend < 0.0 {
            direction_changes += 1;
        }
    }

    let oscillation_rate = direction_changes as f64 / (trajectory.len() - 2) as f64;

    if oscillation_rate > 0.6 {
        "high_oscillation".to_string()
    } else if oscillation_rate > 0.3 {
        "moderate_oscillation".to_string()
    } else {
        "stable".to_string()
    }
}

/// Calculate smoothness score (shared function)
pub(super) fn calculate_smoothness_score(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 3 {
        return 1.0;
    }

    // Calculate second derivative (acceleration)
    let mut second_derivatives = Vec::new();
    for i in 1..trajectory.len() - 1 {
        let second_deriv = trajectory[i + 1] - 2.0 * trajectory[i] + trajectory[i - 1];
        second_derivatives.push(second_deriv.abs());
    }

    let mean_acceleration =
        second_derivatives.iter().sum::<f64>() / second_derivatives.len() as f64;

    // Higher smoothness = lower acceleration
    1.0 / (1.0 + mean_acceleration)
}

/// Calculate momentum indicator (shared function)
pub(super) fn calculate_momentum_indicator(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 2 {
        return 0.0;
    }

    let recent_change = trajectory.last().unwrap() - trajectory[trajectory.len() - 2];
    let overall_change = trajectory.last().unwrap() - trajectory.first().unwrap();

    if overall_change.abs() < 1e-8 {
        return 0.0;
    }

    // Momentum: how much of recent change aligns with overall trend
    recent_change / overall_change
}

/// Calculate saturation risk (shared function)
pub(super) fn calculate_saturation_risk(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 5 {
        return 0.0;
    }

    // Check if improvements are diminishing
    let recent_window = trajectory.len().min(5);
    let recent = &trajectory[trajectory.len() - recent_window..];

    let initial_rate = if trajectory.len() > recent_window {
        let early = &trajectory[0..recent_window];
        let early_improvement = (early.first().unwrap() - early.last().unwrap()).abs();
        early_improvement / recent_window as f64
    } else {
        return 0.0;
    };

    let recent_rate =
        (recent.first().unwrap() - recent.last().unwrap()).abs() / recent_window as f64;

    if initial_rate > 0.0 {
        1.0 - (recent_rate / initial_rate).min(1.0)
    } else {
        0.0
    }
}

// Helper functions

fn extract_current_learning_rate(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let lr_keys = ["lr", "learning_rate", "current_lr"];
    for key in &lr_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

fn calculate_parameter_stability(obj: &serde_json::Map<String, Value>) -> f64 {
    // Simplified: based on gradient norm if available
    extract_gradient_norm(obj).map_or(0.5, |norm| (1.0 / (1.0 + norm)).min(1.0))
}

fn calculate_gradient_flow_health(_obj: &serde_json::Map<String, Value>) -> f64 {
    // Placeholder implementation
    0.8
}

fn calculate_learning_efficiency(obj: &serde_json::Map<String, Value>) -> f64 {
    // Based on loss improvement rate
    if let Some(trajectory) = extract_loss_trajectory(obj) {
        calculate_convergence_rate(&trajectory).min(1.0)
    } else {
        0.5
    }
}

fn calculate_overfitting_risk(_obj: &serde_json::Map<String, Value>) -> f64 {
    // Placeholder implementation
    0.3
}

fn extract_generalization_gap(_obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Placeholder implementation
    None
}
