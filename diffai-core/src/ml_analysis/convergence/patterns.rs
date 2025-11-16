use serde_json::Value;

use super::loss::extract_loss_trajectory;
use super::learning_curves::calculate_convergence_rate;
use super::optimization::{calculate_momentum_indicator, calculate_saturation_risk, calculate_smoothness_score, detect_oscillation_pattern};

#[derive(Debug, Clone)]
pub(crate) struct ConvergencePatterns {
    pub(crate) trend_direction: String,
    pub(crate) convergence_speed: String,
    pub(crate) oscillation_pattern: String,
    pub(crate) smoothness_score: f64,
    pub(crate) momentum_indicator: f64,
    pub(crate) saturation_risk: f64,
}

/// Advanced convergence pattern analysis
pub(crate) fn analyze_convergence_patterns_advanced(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

/// Extract convergence patterns from model checkpoint
pub(crate) fn extract_convergence_patterns(obj: &serde_json::Map<String, Value>) -> Option<ConvergencePatterns> {
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
