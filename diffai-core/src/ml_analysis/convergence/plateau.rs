use serde_json::Value;

use super::loss::extract_loss_trajectory;

#[derive(Debug, Clone)]
pub(crate) struct PlateauAnalysis {
    pub(crate) plateau_length: usize,
    pub(crate) plateau_start_epoch: Option<f64>,
    pub(crate) plateau_threshold: f64,
    pub(crate) recovery_probability: f64,
    pub(crate) recommended_action: String,
}

/// Plateau detection and early stopping analysis
pub(crate) fn analyze_plateau_detection(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

/// Detect plateau in loss trajectory (shared function)
pub(super) fn detect_plateau(loss_trajectory: &[f64]) -> bool {
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

/// Extract plateau analysis from model checkpoint
pub(crate) fn extract_plateau_analysis(obj: &serde_json::Map<String, Value>) -> Option<PlateauAnalysis> {
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

/// Calculate plateau length
pub(crate) fn calculate_plateau_length(trajectory: &[f64]) -> usize {
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

/// Find plateau start epoch
pub(crate) fn find_plateau_start(trajectory: &[f64]) -> Option<f64> {
    let plateau_length = calculate_plateau_length(trajectory);
    if plateau_length > 0 && trajectory.len() > plateau_length {
        Some((trajectory.len() - plateau_length) as f64)
    } else {
        None
    }
}

/// Calculate recovery probability
pub(crate) fn calculate_recovery_probability(trajectory: &[f64]) -> f64 {
    let plateau_length = calculate_plateau_length(trajectory);
    if plateau_length == 0 {
        return 1.0;
    }

    // Longer plateaus have lower recovery probability
    (1.0 / (1.0 + plateau_length as f64 * 0.1)).max(0.1)
}

/// Generate plateau recommendation
pub(crate) fn generate_plateau_recommendation(trajectory: &[f64]) -> String {
    let plateau_length = calculate_plateau_length(trajectory);

    if plateau_length > 10 {
        "consider_lr_reduction".to_string()
    } else if plateau_length > 5 {
        "monitor_closely".to_string()
    } else {
        "continue_training".to_string()
    }
}
