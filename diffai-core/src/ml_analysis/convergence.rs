use serde_json::Value;

use crate::types::DiffResult;

pub fn analyze_convergence_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Enhanced learning curve analysis
        let learning_curve_analysis = analyze_learning_curves_comprehensive(old_obj, new_obj);
        if let Some(curve_info) = learning_curve_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "learning_curve_analysis".to_string(),
                curve_info.0,
                curve_info.1,
            ));
        }
        
        // Enhanced convergence pattern detection using helper functions
        let convergence_patterns = analyze_convergence_patterns_advanced(old_obj, new_obj);
        if let Some(pattern_info) = convergence_patterns {
            results.push(DiffResult::ModelArchitectureChanged(
                "convergence_patterns".to_string(),
                pattern_info.0,
                pattern_info.1,
            ));
        }
        
        // Loss convergence analysis using helper function
        let loss_convergence = analyze_loss_convergence(old_obj, new_obj);
        if let Some(loss_info) = loss_convergence {
            results.push(DiffResult::ModelArchitectureChanged(
                "loss_convergence".to_string(),
                loss_info.0,
                loss_info.1,
            ));
        }
        
        // Training stability analysis using helper function  
        let training_stability = analyze_training_stability(old_obj, new_obj);
        if let Some(stability_info) = training_stability {
            results.push(DiffResult::ModelArchitectureChanged(
                "training_stability_detailed".to_string(),
                stability_info.0,
                stability_info.1,
            ));
        }
        
        // Epoch progression analysis using helper function
        let epoch_progression = analyze_epoch_progression(old_obj, new_obj);
        if let Some(epoch_info) = epoch_progression {
            results.push(DiffResult::ModelArchitectureChanged(
                "epoch_progression".to_string(),
                epoch_info.0,
                epoch_info.1,
            ));
        }
        
        // Training stability with statistical significance
        let stability_analysis = analyze_training_stability_statistical(old_obj, new_obj);
        if let Some(stability_info) = stability_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "training_stability".to_string(),
                stability_info.0,
                stability_info.1,
            ));
        }
        
        // Enhanced optimization trajectory analysis
        let optimization_analysis = analyze_optimization_trajectory(old_obj, new_obj);
        if let Some(opt_info) = optimization_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "optimization_trajectory".to_string(),
                opt_info.0,
                opt_info.1,
            ));
        }
        
        // Plateau detection and early stopping analysis
        let plateau_analysis = analyze_plateau_detection(old_obj, new_obj);
        if let Some(plateau_info) = plateau_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "plateau_detection".to_string(),
                plateau_info.0,
                plateau_info.1,
            ));
        }
    }
}

// Enhanced training stability analysis with statistical significance
fn analyze_training_stability_statistical(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_stability = calculate_training_stability_metrics(old_obj)?;
    let new_stability = calculate_training_stability_metrics(new_obj)?;
    
    let mut stability_changes = Vec::new();
    
    // Compare variance in gradients
    if let (Some(old_grad_var), Some(new_grad_var)) = (old_stability.gradient_variance, new_stability.gradient_variance) {
        let variance_change = (new_grad_var - old_grad_var) / old_grad_var.max(1e-8);
        if variance_change.abs() > 0.1 {
            stability_changes.push(format!("gradient_variance: {:+.2}%", variance_change * 100.0));
        }
    }
    
    // Compare loss oscillation
    if (old_stability.loss_oscillation - new_stability.loss_oscillation).abs() > 0.05 {
        let oscillation_change = new_stability.loss_oscillation - old_stability.loss_oscillation;
        stability_changes.push(format!("loss_oscillation: {:+.3}", oscillation_change));
    }
    
    // Compare overall stability score
    if (old_stability.overall_score - new_stability.overall_score).abs() > 0.05 {
        let score_change = new_stability.overall_score - old_stability.overall_score;
        stability_changes.push(format!("stability_score: {:+.3}", score_change));
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

#[derive(Debug)]
struct TrainingStabilityMetrics {
    gradient_variance: Option<f64>,
    loss_oscillation: f64,
    parameter_drift: f64,
    overall_score: f64,
}

fn calculate_training_stability_metrics(obj: &serde_json::Map<String, Value>) -> Option<TrainingStabilityMetrics> {
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

// Enhanced loss convergence analysis using helper functions
fn analyze_loss_convergence(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    // Enhanced analysis using loss history from helper functions
    let old_loss_history = extract_loss_history(old_obj).unwrap_or_else(|| {
        extract_loss_value(old_obj).map(|v| vec![v]).unwrap_or_default()
    });
    let new_loss_history = extract_loss_history(new_obj).unwrap_or_else(|| {
        extract_loss_value(new_obj).map(|v| vec![v]).unwrap_or_default()
    });
    
    if old_loss_history.is_empty() || new_loss_history.is_empty() {
        return None;
    }
    
    // Use helper functions for enhanced analysis
    let trend_analysis = analyze_loss_trend(&old_loss_history, &new_loss_history);
    let old_slope = calculate_trend_slope(&old_loss_history);
    let new_slope = calculate_trend_slope(&new_loss_history);
    
    // Basic loss comparison
    let old_loss = old_loss_history.last().unwrap_or(&0.0);
    let new_loss = new_loss_history.last().unwrap_or(&0.0);
    let loss_change = new_loss - old_loss;
    let loss_change_percent = if *old_loss != 0.0 {
        (loss_change / old_loss) * 100.0
    } else {
        0.0
    };
    
    // Determine convergence status
    let convergence_status = if loss_change < -0.001 {
        "improving"
    } else if loss_change > 0.001 {
        "diverging"
    } else {
        "stable"
    };
    
    // Enhanced analysis information
    let old_info = format!("loss: {:.6}, slope: {:.6}", old_loss, old_slope);
    let new_info = format!("loss: {:.6} ({:+.2}%), slope: {:.6}, trend: {}, status: {}", 
        new_loss, loss_change_percent, new_slope, trend_analysis, convergence_status);
    
    Some((old_info, new_info))
}

// Extract loss value from model checkpoint
fn extract_loss_value(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Try various common loss field names
    let loss_keys = ["loss", "train_loss", "training_loss", "val_loss", "validation_loss", 
                     "total_loss", "current_loss", "best_loss"];
    
    for key in &loss_keys {
        if let Some(loss_val) = obj.get(*key) {
            if let Value::Number(num) = loss_val {
                return num.as_f64();
            }
        }
    }
    
    // Look in nested structures
    if let Some(metrics) = obj.get("metrics") {
        if let Value::Object(metrics_obj) = metrics {
            for key in &loss_keys {
                if let Some(loss_val) = metrics_obj.get(*key) {
                    if let Value::Number(num) = loss_val {
                        return num.as_f64();
                    }
                }
            }
        }
    }
    
    None
}

// Extract loss history for trend analysis
fn extract_loss_history(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
    let history_keys = ["loss_history", "train_losses", "validation_losses", "loss_curve"];
    
    for key in &history_keys {
        if let Some(history_val) = obj.get(*key) {
            if let Value::Array(history_arr) = history_val {
                let mut losses = Vec::new();
                for item in history_arr {
                    if let Value::Number(num) = item {
                        if let Some(loss) = num.as_f64() {
                            losses.push(loss);
                        }
                    }
                }
                if !losses.is_empty() {
                    return Some(losses);
                }
            }
        }
    }
    
    None
}

// Analyze loss trend from historical data
fn analyze_loss_trend(old_history: &[f64], new_history: &[f64]) -> String {
    if old_history.is_empty() || new_history.is_empty() {
        return "insufficient_data".to_string();
    }
    
    // Calculate trend slope for recent history
    let old_trend = calculate_trend_slope(&old_history[old_history.len().saturating_sub(5)..]);
    let new_trend = calculate_trend_slope(&new_history[new_history.len().saturating_sub(5)..]);
    
    let trend_change = new_trend - old_trend;
    
    if trend_change < -0.01 {
        "accelerating_improvement".to_string()
    } else if trend_change > 0.01 {
        "slowing_improvement".to_string()
    } else if new_trend < -0.001 {
        "steady_improvement".to_string()
    } else if new_trend > 0.001 {
        "deteriorating".to_string()
    } else {
        "plateauing".to_string()
    }
}

// Calculate trend slope using simple linear regression
fn calculate_trend_slope(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
    let y_sum: f64 = values.iter().sum();
    let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let x_sq_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
    
    let denominator = n * x_sq_sum - x_sum * x_sum;
    if denominator.abs() < 1e-10 {
        return 0.0;
    }
    
    (n * xy_sum - x_sum * y_sum) / denominator
}

// Analyze training stability from various metrics
fn analyze_training_stability(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let mut stability_factors = Vec::new();
    
    // Check gradient norms if available
    if let (Some(old_grad), Some(new_grad)) = (
        extract_gradient_norm(old_obj), 
        extract_gradient_norm(new_obj)
    ) {
        let grad_change = (new_grad / old_grad - 1.0) * 100.0;
        let grad_stability = if grad_change.abs() < 10.0 {
            "stable"
        } else if grad_change.abs() < 50.0 {
            "moderate_variation"
        } else {
            "high_variation"
        };
        stability_factors.push(format!("gradient_norm: {}", grad_stability));
    }
    
    // Check learning rate stability
    if let (Some(old_lr), Some(new_lr)) = (
        extract_current_learning_rate(old_obj),
        extract_current_learning_rate(new_obj)
    ) {
        let lr_ratio = new_lr / old_lr;
        let lr_stability = if (lr_ratio - 1.0).abs() < 0.1 {
            "stable"
        } else if lr_ratio < 1.0 {
            "decreasing"
        } else {
            "increasing"
        };
        stability_factors.push(format!("learning_rate: {}", lr_stability));
    }
    
    // Check parameter magnitude changes
    if let (Some(old_params), Some(new_params)) = (
        estimate_parameter_magnitude(old_obj),
        estimate_parameter_magnitude(new_obj)
    ) {
        let param_change = ((new_params / old_params - 1.0) * 100.0).abs();
        let param_stability = if param_change < 1.0 {
            "stable"
        } else if param_change < 5.0 {
            "mild_change"
        } else {
            "significant_change"
        };
        stability_factors.push(format!("parameters: {}", param_stability));
    }
    
    if stability_factors.is_empty() {
        return None;
    }
    
    let old_info = "evaluating".to_string();
    let new_info = stability_factors.join(", ");
    
    Some((old_info, new_info))
}

// Analyze epoch progression patterns
fn analyze_epoch_progression(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_epoch = extract_epoch_info(old_obj)?;
    let new_epoch = extract_epoch_info(new_obj)?;
    
    if new_epoch <= old_epoch {
        return None; // No progression or regression
    }
    
    let epoch_diff = new_epoch - old_epoch;
    let progression_rate = if epoch_diff == 1.0 {
        "normal"
    } else if epoch_diff < 1.0 {
        "fractional"
    } else {
        "skipped_epochs"
    };
    
    let old_info = format!("epoch: {}", old_epoch);
    let new_info = format!("epoch: {}, progression: {} ({:+.1})", new_epoch, progression_rate, epoch_diff);
    
    Some((old_info, new_info))
}

// Enhanced convergence analysis structures with lawkit memory patterns
#[derive(Debug, Clone)]
struct LearningCurveMetrics {
    loss_trajectory: Vec<f64>,
    accuracy_trajectory: Vec<f64>,
    learning_rate_schedule: Vec<f64>,
    gradient_norms: Vec<f64>,
    epochs: Vec<f64>,
    convergence_rate: f64,
    stability_score: f64,
    plateau_detected: bool,
    early_stopping_suggestion: Option<String>,
}

#[derive(Debug, Clone)]
struct ConvergencePatterns {
    trend_direction: String,
    convergence_speed: String,
    oscillation_pattern: String,
    smoothness_score: f64,
    momentum_indicator: f64,
    saturation_risk: f64,
}

#[derive(Debug, Clone)]
struct OptimizationTrajectory {
    parameter_stability: f64,
    gradient_flow_health: f64,
    learning_efficiency: f64,
    overfitting_risk: f64,
    generalization_gap: Option<f64>,
}

#[derive(Debug, Clone)]
struct PlateauAnalysis {
    plateau_length: usize,
    plateau_start_epoch: Option<f64>,
    plateau_threshold: f64,
    recovery_probability: f64,
    recommended_action: String,
}

// Enhanced learning curve analysis using lawkit incremental statistics
fn analyze_learning_curves_comprehensive(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

// Extract comprehensive learning curve metrics
fn extract_learning_curve_metrics(obj: &serde_json::Map<String, Value>) -> Option<LearningCurveMetrics> {
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

// Helper functions for convergence analysis
fn extract_gradient_norm(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let grad_keys = ["grad_norm", "gradient_norm", "total_grad_norm"];
    for key in &grad_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

// Enhanced convergence calculation functions using lawkit statistical methods
fn calculate_convergence_rate(loss_trajectory: &[f64]) -> f64 {
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

fn detect_plateau(loss_trajectory: &[f64]) -> bool {
    if loss_trajectory.len() < 5 {
        return false;
    }
    
    // Check last N points for minimal change
    let window_size = (loss_trajectory.len() / 3).min(10).max(3);
    let recent_losses = &loss_trajectory[loss_trajectory.len() - window_size..];
    
    let min_loss = recent_losses.iter().copied().fold(f64::INFINITY, f64::min);
    let max_loss = recent_losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    // Plateau detected if variation is less than 1% of mean
    if min_loss > 0.0 {
        let variation_ratio = (max_loss - min_loss) / min_loss;
        variation_ratio < 0.01
    } else {
        false
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

fn extract_current_learning_rate(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let lr_keys = ["lr", "learning_rate", "current_lr"];
    for key in &lr_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

// Advanced convergence pattern analysis
fn analyze_convergence_patterns_advanced(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_patterns = extract_convergence_patterns(old_obj)?;
    let new_patterns = extract_convergence_patterns(new_obj)?;
    
    let mut pattern_changes = Vec::new();
    
    if old_patterns.trend_direction != new_patterns.trend_direction {
        pattern_changes.push(format!("trend: {} -> {}", old_patterns.trend_direction, new_patterns.trend_direction));
    }
    
    if old_patterns.convergence_speed != new_patterns.convergence_speed {
        pattern_changes.push(format!("speed: {} -> {}", old_patterns.convergence_speed, new_patterns.convergence_speed));
    }
    
    if (old_patterns.smoothness_score - new_patterns.smoothness_score).abs() > 0.1 {
        let smoothness_change = new_patterns.smoothness_score - old_patterns.smoothness_score;
        pattern_changes.push(format!("smoothness: {:+.2}", smoothness_change));
    }
    
    if pattern_changes.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "trend: {}, speed: {}, smoothness: {:.2}",
        old_patterns.trend_direction, old_patterns.convergence_speed, old_patterns.smoothness_score
    );
    let new_info = pattern_changes.join(", ");
    
    Some((old_info, new_info))
}

fn extract_convergence_patterns(obj: &serde_json::Map<String, Value>) -> Option<ConvergencePatterns> {
    let loss_trajectory = extract_loss_trajectory(obj)?;
    
    if loss_trajectory.len() < 3 {
        return None;
    }
    
    // Trend analysis
    let trend_direction = if loss_trajectory.first().unwrap() > loss_trajectory.last().unwrap() {
        "decreasing".to_string()
    } else if loss_trajectory.first().unwrap() < loss_trajectory.last().unwrap() {
        "increasing".to_string()
    } else {
        "stable".to_string()
    };
    
    // Speed analysis using helper functions
    let convergence_rate = calculate_convergence_rate(&loss_trajectory);
    let convergence_speed = if convergence_rate > 0.01 {
        "fast".to_string()
    } else if convergence_rate > 0.001 {
        "moderate".to_string()
    } else {
        "slow".to_string()
    };
    
    // Enhanced pattern analysis using helper functions
    let oscillation_pattern = detect_oscillation_pattern(&loss_trajectory);
    let smoothness_score = calculate_smoothness_score(&loss_trajectory);
    let momentum_indicator = calculate_momentum_indicator(&loss_trajectory);
    let saturation_risk = calculate_saturation_risk(&loss_trajectory);
    
    Some(ConvergencePatterns {
        trend_direction,
        convergence_speed,
        oscillation_pattern,
        smoothness_score,
        momentum_indicator,
        saturation_risk,
    })
}

// Enhanced optimization trajectory analysis using helper functions
fn analyze_optimization_trajectory(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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
            trajectory_changes.push(format!("learning_rate: {:+.2}%", lr_change));
        }
    }
    
    // Parameter magnitude analysis
    if let (Some(old_mag), Some(new_mag)) = (old_param_mag, new_param_mag) {
        if (old_mag - new_mag).abs() > old_mag * 0.05 {
            let mag_change = ((new_mag - old_mag) / old_mag) * 100.0;
            trajectory_changes.push(format!("parameter_magnitude: {:+.2}%", mag_change));
        }
    }
    
    // Epoch progression analysis
    if let (Some(old_ep), Some(new_ep)) = (old_epoch, new_epoch) {
        if new_ep > old_ep {
            trajectory_changes.push(format!("epoch_progress: {} -> {}", old_ep, new_ep));
        }
    }
    
    let stability_change = new_trajectory.parameter_stability - old_trajectory.parameter_stability;
    if stability_change.abs() > 0.05 {
        trajectory_changes.push(format!("param_stability: {:+.3}", stability_change));
    }
    
    let efficiency_change = new_trajectory.learning_efficiency - old_trajectory.learning_efficiency;
    if efficiency_change.abs() > 0.05 {
        trajectory_changes.push(format!("learning_efficiency: {:+.3}", efficiency_change));
    }
    
    if let (Some(old_gap), Some(new_gap)) = (old_trajectory.generalization_gap, new_trajectory.generalization_gap) {
        let gap_change = new_gap - old_gap;
        if gap_change.abs() > 0.02 {
            trajectory_changes.push(format!("generalization_gap: {:+.3}", gap_change));
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

// Plateau detection analysis
fn analyze_plateau_detection(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_plateau = extract_plateau_analysis(old_obj)?;
    let new_plateau = extract_plateau_analysis(new_obj)?;
    
    let mut plateau_changes = Vec::new();
    
    if old_plateau.plateau_length != new_plateau.plateau_length {
        let length_change = new_plateau.plateau_length as i32 - old_plateau.plateau_length as i32;
        plateau_changes.push(format!("plateau_length: {} ({:+})", new_plateau.plateau_length, length_change));
    }
    
    if (old_plateau.recovery_probability - new_plateau.recovery_probability).abs() > 0.1 {
        let recovery_change = new_plateau.recovery_probability - old_plateau.recovery_probability;
        plateau_changes.push(format!("recovery_probability: {:+.2}", recovery_change));
    }
    
    if old_plateau.recommended_action != new_plateau.recommended_action {
        plateau_changes.push(format!("action: {}", new_plateau.recommended_action));
    }
    
    if plateau_changes.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "length: {}, recovery_prob: {:.2}",
        old_plateau.plateau_length, old_plateau.recovery_probability
    );
    let new_info = plateau_changes.join(", ");
    
    Some((old_info, new_info))
}

fn estimate_parameter_magnitude(obj: &serde_json::Map<String, Value>) -> Option<f64> {
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

fn extract_epoch_info(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    if let Some(Value::Number(num)) = obj.get("epoch") {
        return num.as_f64();
    }
    None
}

// Helper functions for enhanced convergence analysis
fn extract_loss_trajectory(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
    // Try to find loss history
    for (key, value) in obj {
        if key.contains("loss") && key.contains("history") {
            if let Value::Array(arr) = value {
                let trajectory: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                if !trajectory.is_empty() {
                    return Some(trajectory);
                }
            }
        }
    }
    
    // Fallback to single loss value
    if let Some(loss) = extract_loss_value(obj) {
        Some(vec![loss])
    } else {
        None
    }
}

fn detect_oscillation_pattern(trajectory: &[f64]) -> String {
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

fn calculate_smoothness_score(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 3 {
        return 1.0;
    }
    
    // Calculate second derivative (acceleration)
    let mut second_derivatives = Vec::new();
    for i in 1..trajectory.len() - 1 {
        let second_deriv = trajectory[i + 1] - 2.0 * trajectory[i] + trajectory[i - 1];
        second_derivatives.push(second_deriv.abs());
    }
    
    let mean_acceleration = second_derivatives.iter().sum::<f64>() / second_derivatives.len() as f64;
    
    // Higher smoothness = lower acceleration
    1.0 / (1.0 + mean_acceleration)
}

fn calculate_momentum_indicator(trajectory: &[f64]) -> f64 {
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

fn calculate_saturation_risk(trajectory: &[f64]) -> f64 {
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
    
    let recent_rate = (recent.first().unwrap() - recent.last().unwrap()).abs() / recent_window as f64;
    
    if initial_rate > 0.0 {
        1.0 - (recent_rate / initial_rate).min(1.0)
    } else {
        0.0
    }
}

fn extract_optimization_trajectory(obj: &serde_json::Map<String, Value>) -> Option<OptimizationTrajectory> {
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

fn extract_plateau_analysis(obj: &serde_json::Map<String, Value>) -> Option<PlateauAnalysis> {
    let loss_trajectory = extract_loss_trajectory(obj)?;
    
    let plateau_length = calculate_plateau_length(&loss_trajectory);
    let plateau_start_epoch = find_plateau_start(&loss_trajectory);
    let plateau_threshold = 0.01; // 1% threshold
    let recovery_probability = calculate_recovery_probability(&loss_trajectory);
    let recommended_action = generate_plateau_recommendation(&loss_trajectory);
    
    Some(PlateauAnalysis {
        plateau_length,
        plateau_start_epoch,
        plateau_threshold,
        recovery_probability,
        recommended_action,
    })
}

// Simplified implementations for helper functions
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

fn calculate_plateau_length(trajectory: &[f64]) -> usize {
    if trajectory.len() < 3 {
        return 0;
    }
    
    let threshold = 0.01;
    let mut plateau_count = 0;
    
    for i in 1..trajectory.len() {
        let change_ratio = (trajectory[i] - trajectory[i - 1]).abs() / trajectory[i - 1].abs().max(1e-8);
        if change_ratio < threshold {
            plateau_count += 1;
        } else {
            plateau_count = 0; // Reset if significant change
        }
    }
    
    plateau_count
}

fn find_plateau_start(trajectory: &[f64]) -> Option<f64> {
    let plateau_length = calculate_plateau_length(trajectory);
    if plateau_length > 0 && trajectory.len() > plateau_length {
        Some((trajectory.len() - plateau_length) as f64)
    } else {
        None
    }
}

fn calculate_recovery_probability(trajectory: &[f64]) -> f64 {
    let plateau_length = calculate_plateau_length(trajectory);
    if plateau_length == 0 {
        return 1.0;
    }
    
    // Longer plateaus have lower recovery probability
    (1.0 / (1.0 + plateau_length as f64 * 0.1)).max(0.1)
}

fn generate_plateau_recommendation(trajectory: &[f64]) -> String {
    let plateau_length = calculate_plateau_length(trajectory);
    
    if plateau_length > 10 {
        "consider_lr_reduction".to_string()
    } else if plateau_length > 5 {
        "monitor_closely".to_string()
    } else {
        "continue_training".to_string()
    }
}
