use serde_json::Value;

/// Enhanced loss convergence analysis using helper functions
pub(crate) fn analyze_loss_convergence(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

/// Extract loss value from model checkpoint
pub(crate) fn extract_loss_value(obj: &serde_json::Map<String, Value>) -> Option<f64> {
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

/// Extract loss history for trend analysis
pub(crate) fn extract_loss_history(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
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

/// Analyze loss trend from historical data
pub(crate) fn analyze_loss_trend(old_history: &[f64], new_history: &[f64]) -> String {
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

/// Calculate trend slope using simple linear regression
pub(crate) fn calculate_trend_slope(values: &[f64]) -> f64 {
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

/// Extract loss trajectory (used by other modules)
pub(super) fn extract_loss_trajectory(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
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
