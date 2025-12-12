use serde_json::Value;

use super::statistics::extract_gradient_statistics;

// Analyze gradient distribution patterns
pub(super) fn analyze_gradient_distributions(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_grad_stats = extract_gradient_statistics(old_obj)?;
    let new_grad_stats = extract_gradient_statistics(new_obj)?;

    let mut has_significant_change = false;

    // Analyze sparsity (percentage of near-zero gradients)
    let sparsity_change = match (old_grad_stats.sparsity, new_grad_stats.sparsity) {
        (Some(old_sparsity), Some(new_sparsity)) => {
            let change = new_sparsity - old_sparsity;
            if change.abs() >= 0.01 {
                has_significant_change = true;
            }
            Some(change)
        }
        _ => None,
    };

    // Analyze outlier gradients
    let outlier_change = match (old_grad_stats.outlier_count, new_grad_stats.outlier_count) {
        (Some(old_outliers), Some(new_outliers)) => {
            let change = new_outliers as i32 - old_outliers as i32;
            if change != 0 {
                has_significant_change = true;
            }
            Some(change)
        }
        _ => None,
    };

    // Only report if there's actual change
    if !has_significant_change {
        return None;
    }

    let old_info = format!(
        "sparsity: {:.1}%, outliers: {}",
        old_grad_stats.sparsity.unwrap_or(0.0) * 100.0,
        old_grad_stats.outlier_count.unwrap_or(0)
    );

    let mut new_parts = Vec::new();
    if let (Some(new_sparsity), Some(change)) = (new_grad_stats.sparsity, sparsity_change) {
        let trend = if change.abs() < 0.01 {
            "stable"
        } else if change > 0.0 {
            "more_sparse"
        } else {
            "less_sparse"
        };
        new_parts.push(format!(
            "sparsity: {:.1}% ({:+.1}%, {})",
            new_sparsity * 100.0,
            change * 100.0,
            trend
        ));
    }
    if let (Some(new_outliers), Some(change)) = (new_grad_stats.outlier_count, outlier_change) {
        new_parts.push(format!("outliers: {new_outliers} ({change:+})"));
    }

    if new_parts.is_empty() {
        return None;
    }

    Some((old_info, new_parts.join(", ")))
}
