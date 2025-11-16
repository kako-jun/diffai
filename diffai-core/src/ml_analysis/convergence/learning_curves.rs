use serde_json::Value;

use super::plateau::detect_plateau;

#[derive(Debug, Clone)]
pub(crate) struct LearningCurveMetrics {
    pub(crate) loss_trajectory: Vec<f64>,
    pub(crate) accuracy_trajectory: Vec<f64>,
    pub(crate) learning_rate_schedule: Vec<f64>,
    pub(crate) gradient_norms: Vec<f64>,
    pub(crate) epochs: Vec<f64>,
    pub(crate) convergence_rate: f64,
    pub(crate) stability_score: f64,
    pub(crate) plateau_detected: bool,
    pub(crate) early_stopping_suggestion: Option<String>,
}

/// Enhanced learning curve analysis using lawkit incremental statistics
pub(crate) fn analyze_learning_curves_comprehensive(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_metrics = extract_learning_curve_metrics(old_obj)?;
    let new_metrics = extract_learning_curve_metrics(new_obj)?;

    let mut analysis_points = Vec::new();

    // Loss trajectory analysis
    if !old_metrics.loss_trajectory.is_empty() && !new_metrics.loss_trajectory.is_empty() {
        let loss_improvement = calculate_trajectory_improvement(&old_metrics.loss_trajectory, &new_metrics.loss_trajectory);
        analysis_points.push(format!("loss_trajectory_improvement: {:.4}", loss_improvement));
    }

    // Convergence rate comparison
    if (old_metrics.convergence_rate - new_metrics.convergence_rate).abs() > 0.001 {
        let rate_change = new_metrics.convergence_rate - old_metrics.convergence_rate;
        analysis_points.push(format!("convergence_rate_change: {:+.4}", rate_change));
    }

    // Stability score analysis
    if (old_metrics.stability_score - new_metrics.stability_score).abs() > 0.05 {
        let stability_change = new_metrics.stability_score - old_metrics.stability_score;
        analysis_points.push(format!("stability_change: {:+.3}", stability_change));
    }

    // Plateau detection
    if old_metrics.plateau_detected != new_metrics.plateau_detected {
        let plateau_status = if new_metrics.plateau_detected { "detected" } else { "resolved" };
        analysis_points.push(format!("plateau_status: {}", plateau_status));
    }

    // Early stopping suggestion
    if let Some(ref suggestion) = new_metrics.early_stopping_suggestion {
        analysis_points.push(format!("early_stopping: {}", suggestion));
    }

    if analysis_points.is_empty() {
        return None;
    }

    let old_info = format!(
        "convergence_rate: {:.4}, stability: {:.3}, plateau: {}",
        old_metrics.convergence_rate, old_metrics.stability_score, old_metrics.plateau_detected
    );
    let new_info = analysis_points.join(", ");

    Some((old_info, new_info))
}

/// Extract comprehensive learning curve metrics
pub(crate) fn extract_learning_curve_metrics(obj: &serde_json::Map<String, Value>) -> Option<LearningCurveMetrics> {
    let mut loss_trajectory = Vec::new();
    let mut accuracy_trajectory = Vec::new();
    let mut learning_rate_schedule = Vec::new();
    let mut gradient_norms = Vec::new();
    let mut epochs = Vec::new();

    // Extract historical data (lawkit streaming pattern)
    for (key, value) in obj {
        match value {
            Value::Array(arr) => {
                if key.contains("loss") && key.contains("history") {
                    loss_trajectory = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("accuracy") && key.contains("history") {
                    accuracy_trajectory = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("lr") && key.contains("history") {
                    learning_rate_schedule = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("grad") && key.contains("history") {
                    gradient_norms = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("epoch") && key.contains("history") {
                    epochs = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                }
            }
            Value::Number(num) => {
                if let Some(val) = num.as_f64() {
                    // Single point data
                    if key.contains("loss") {
                        loss_trajectory.push(val);
                    } else if key.contains("accuracy") {
                        accuracy_trajectory.push(val);
                    } else if key.contains("lr") || key.contains("learning_rate") {
                        learning_rate_schedule.push(val);
                    } else if key.contains("grad_norm") {
                        gradient_norms.push(val);
                    } else if key.contains("epoch") {
                        epochs.push(val);
                    }
                }
            }
            _ => {}
        }
    }

    if loss_trajectory.is_empty() {
        return None;
    }

    // Calculate convergence metrics using lawkit incremental statistics
    let convergence_rate = calculate_convergence_rate(&loss_trajectory);
    let stability_score = calculate_stability_score(&loss_trajectory);
    let plateau_detected = detect_plateau(&loss_trajectory);
    let early_stopping_suggestion = generate_early_stopping_suggestion(&loss_trajectory, &accuracy_trajectory);

    Some(LearningCurveMetrics {
        loss_trajectory,
        accuracy_trajectory,
        learning_rate_schedule,
        gradient_norms,
        epochs,
        convergence_rate,
        stability_score,
        plateau_detected,
        early_stopping_suggestion,
    })
}

// Helper functions

/// Enhanced convergence calculation using lawkit statistical methods
pub(super) fn calculate_convergence_rate(loss_trajectory: &[f64]) -> f64 {
    if loss_trajectory.len() < 2 {
        return 0.0;
    }

    // Use exponential moving average for smoothing (lawkit pattern)
    let mut smoothed_losses = Vec::new();
    let alpha = 0.1; // Smoothing factor
    smoothed_losses.push(loss_trajectory[0]);

    for &loss in &loss_trajectory[1..] {
        let smoothed = alpha * loss + (1.0 - alpha) * smoothed_losses.last().unwrap();
        smoothed_losses.push(smoothed);
    }

    // Calculate rate of improvement
    let initial_loss = smoothed_losses[0];
    let final_loss = *smoothed_losses.last().unwrap();

    if initial_loss <= 0.0 {
        return 0.0;
    }

    let improvement_ratio = (initial_loss - final_loss) / initial_loss;
    let epochs = smoothed_losses.len() as f64;

    // Normalize by number of epochs
    improvement_ratio / epochs
}

fn calculate_stability_score(loss_trajectory: &[f64]) -> f64 {
    if loss_trajectory.len() < 3 {
        return 1.0;
    }

    // Calculate variance and coefficient of variation
    let mean = loss_trajectory.iter().sum::<f64>() / loss_trajectory.len() as f64;
    let variance = loss_trajectory.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / loss_trajectory.len() as f64;
    let std_dev = variance.sqrt();

    // Stability score: lower coefficient of variation = higher stability
    if mean > 0.0 {
        1.0 / (1.0 + std_dev / mean)
    } else {
        0.0
    }
}

fn generate_early_stopping_suggestion(loss_trajectory: &[f64], accuracy_trajectory: &[f64]) -> Option<String> {
    if loss_trajectory.len() < 10 {
        return None;
    }

    let plateau_detected = detect_plateau(loss_trajectory);
    let convergence_rate = calculate_convergence_rate(loss_trajectory);

    // Check for overfitting indicators
    let has_accuracy = !accuracy_trajectory.is_empty();
    let overfitting_risk = if has_accuracy && accuracy_trajectory.len() >= 5 {
        // Simple overfitting detection: accuracy plateau while loss still decreasing
        let acc_stable = detect_plateau(accuracy_trajectory);
        let loss_decreasing = convergence_rate > 0.001;
        acc_stable && loss_decreasing
    } else {
        false
    };

    if plateau_detected && convergence_rate < 0.001 {
        Some("consider_early_stopping".to_string())
    } else if overfitting_risk {
        Some("overfitting_detected".to_string())
    } else if convergence_rate < 0.0001 {
        Some("minimal_improvement".to_string())
    } else {
        None
    }
}

fn calculate_trajectory_improvement(old_trajectory: &[f64], new_trajectory: &[f64]) -> f64 {
    if old_trajectory.is_empty() || new_trajectory.is_empty() {
        return 0.0;
    }

    let old_final = *old_trajectory.last().unwrap();
    let new_final = *new_trajectory.last().unwrap();

    if old_final > 0.0 {
        (old_final - new_final) / old_final
    } else {
        0.0
    }
}
