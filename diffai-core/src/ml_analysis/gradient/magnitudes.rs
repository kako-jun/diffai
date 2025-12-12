use serde_json::Value;

use super::statistics::extract_gradient_statistics;

// Analyze gradient magnitude patterns
pub(super) fn analyze_gradient_magnitudes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_grad_stats = extract_gradient_statistics(old_obj)?;
    let new_grad_stats = extract_gradient_statistics(new_obj)?;

    let mut magnitude_analysis = Vec::new();

    // Compare gradient norms
    if let (Some(old_norm), Some(new_norm)) = (old_grad_stats.total_norm, new_grad_stats.total_norm)
    {
        let norm_change = (new_norm / old_norm - 1.0) * 100.0;
        let norm_trend = if norm_change.abs() < 5.0 {
            "stable"
        } else if norm_change > 0.0 {
            "increasing"
        } else {
            "decreasing"
        };
        magnitude_analysis.push(format!(
            "total_norm: {new_norm:.6} ({norm_change:+.1}%, {norm_trend})"
        ));
    }

    // Compare max gradients
    if let (Some(old_max), Some(new_max)) =
        (old_grad_stats.max_gradient, new_grad_stats.max_gradient)
    {
        let max_change = (new_max / old_max - 1.0) * 100.0;
        magnitude_analysis.push(format!("max_gradient: {new_max:.6} ({max_change:+.1}%)"));
    }

    // Compare gradient variance
    if let (Some(old_var), Some(new_var)) = (old_grad_stats.variance, new_grad_stats.variance) {
        let var_change = (new_var / old_var - 1.0) * 100.0;
        magnitude_analysis.push(format!("variance: {new_var:.6} ({var_change:+.1}%)"));
    }

    if magnitude_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "norm: {:.6}, max: {:.6}, var: {:.6}",
        old_grad_stats.total_norm.unwrap_or(0.0),
        old_grad_stats.max_gradient.unwrap_or(0.0),
        old_grad_stats.variance.unwrap_or(0.0)
    );
    let new_info = magnitude_analysis.join(", ");

    Some((old_info, new_info))
}
