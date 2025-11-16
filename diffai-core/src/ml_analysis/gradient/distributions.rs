use serde_json::Value;

use super::statistics::extract_gradient_statistics;

// Analyze gradient distribution patterns
pub(super) fn analyze_gradient_distributions(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_grad_stats = extract_gradient_statistics(old_obj)?;
    let new_grad_stats = extract_gradient_statistics(new_obj)?;

    let mut distribution_analysis = Vec::new();

    // Analyze sparsity (percentage of near-zero gradients)
    if let (Some(old_sparsity), Some(new_sparsity)) =
        (old_grad_stats.sparsity, new_grad_stats.sparsity)
    {
        let sparsity_change = new_sparsity - old_sparsity;
        let sparsity_trend = if sparsity_change.abs() < 0.01 {
            "stable"
        } else if sparsity_change > 0.0 {
            "more_sparse"
        } else {
            "less_sparse"
        };
        distribution_analysis.push(format!(
            "sparsity: {:.1}% ({:+.1}%, {})",
            new_sparsity * 100.0,
            sparsity_change * 100.0,
            sparsity_trend
        ));
    }

    // Analyze outlier gradients
    if let (Some(old_outliers), Some(new_outliers)) = (
        old_grad_stats.outlier_count,
        new_grad_stats.outlier_count,
    ) {
        let outlier_change = new_outliers as i32 - old_outliers as i32;
        distribution_analysis.push(format!(
            "outliers: {} ({:+})",
            new_outliers, outlier_change
        ));
    }

    if distribution_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "sparsity: {:.1}%, outliers: {}",
        old_grad_stats.sparsity.unwrap_or(0.0) * 100.0,
        old_grad_stats.outlier_count.unwrap_or(0)
    );
    let new_info = distribution_analysis.join(", ");

    Some((old_info, new_info))
}
