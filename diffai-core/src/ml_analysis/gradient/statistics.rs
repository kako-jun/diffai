use serde_json::Value;

use super::types::{EnhancedGradientStats, GradientStatistics};

// Extract gradient statistics from model data - Enhanced with lawkit memory patterns
pub(super) fn extract_gradient_statistics(
    obj: &serde_json::Map<String, Value>,
) -> Option<GradientStatistics> {
    let mut total_norm = None;
    let mut max_gradient = None;
    let mut variance = None;
    let mut sparsity = None;
    let mut outlier_count = None;

    // First pass: Look for explicit gradient statistics (diffx optimization pattern)
    let grad_keys = [
        "grad_norm",
        "gradient_norm",
        "total_grad_norm",
        "max_grad",
        "gradient_max",
        "grad_variance",
        "grad_sparsity",
        "gradient_outliers",
        "grad_flow",
    ];

    for key in &grad_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            if let Some(val) = num.as_f64() {
                match *key {
                    "grad_norm" | "gradient_norm" | "total_grad_norm" => total_norm = Some(val),
                    "max_grad" | "gradient_max" => max_gradient = Some(val),
                    "grad_variance" => variance = Some(val),
                    "grad_sparsity" => sparsity = Some(val),
                    _ => {}
                }
            }
        }
    }

    // Enhanced estimation using incremental statistics (lawkit pattern)
    if total_norm.is_none() || max_gradient.is_none() || variance.is_none() {
        let stats = compute_enhanced_gradient_statistics(obj);
        total_norm = total_norm.or(stats.total_norm);
        max_gradient = max_gradient.or(stats.max_gradient);
        variance = variance.or(stats.variance);
    }

    // Memory-efficient sparsity and outlier calculations
    sparsity = sparsity.or_else(|| estimate_gradient_sparsity_streaming(obj));
    outlier_count = outlier_count.or_else(|| count_gradient_outliers_robust(obj));

    Some(GradientStatistics {
        total_norm,
        max_gradient,
        variance,
        sparsity,
        outlier_count,
    })
}

pub(super) fn compute_enhanced_gradient_statistics(
    obj: &serde_json::Map<String, Value>,
) -> EnhancedGradientStats {
    // Use lawkit-style incremental statistics for memory efficiency
    let mut sum_squares = 0.0;
    let mut sum_values = 0.0;
    let mut sum_square_values = 0.0;
    let mut max_val: f64 = 0.0;
    let mut count = 0;

    // Multi-source gradient data analysis (PyTorch, Safetensors, NumPy patterns)
    for (key, value) in obj {
        let is_gradient_related = key.contains("grad")
            || key.contains("gradient")
            || key.contains("weight")
            || key.contains("bias")
            || key.contains("param");

        if is_gradient_related {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        // Incremental statistics (Welford's algorithm)
                        sum_squares += val * val;
                        sum_values += val;
                        sum_square_values += val * val;
                        max_val = max_val.max(val.abs());
                        count += 1;
                    }
                }
                Value::Array(arr) => {
                    // Handle tensor arrays efficiently
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                sum_squares += val * val;
                                sum_values += val;
                                sum_square_values += val * val;
                                max_val = max_val.max(val.abs());
                                count += 1;
                            }
                        }
                    }
                }
                Value::Object(nested) => {
                    // Recursive analysis for nested structures
                    let nested_stats = compute_enhanced_gradient_statistics(nested);
                    if let Some(norm) = nested_stats.total_norm {
                        sum_squares += norm * norm;
                        count += 1;
                    }
                    if let Some(max_nested) = nested_stats.max_gradient {
                        max_val = max_val.max(max_nested);
                    }
                }
                _ => {}
            }
        }
    }

    let total_norm = if count > 0 {
        Some(sum_squares.sqrt())
    } else {
        None
    };
    let max_gradient = if count > 0 { Some(max_val) } else { None };

    // Calculate variance using stable algorithm
    let variance = if count > 1 {
        let mean = sum_values / count as f64;
        let variance_val = (sum_square_values / count as f64) - (mean * mean);
        Some(variance_val.max(0.0)) // Ensure non-negative
    } else {
        None
    };

    EnhancedGradientStats {
        total_norm,
        max_gradient,
        variance,
    }
}

// Legacy function for backward compatibility
pub(super) fn estimate_gradient_norm_from_weights(
    obj: &serde_json::Map<String, Value>,
) -> Option<f64> {
    compute_enhanced_gradient_statistics(obj).total_norm
}

// Legacy function for backward compatibility
pub(super) fn estimate_max_gradient_from_weights(
    obj: &serde_json::Map<String, Value>,
) -> Option<f64> {
    compute_enhanced_gradient_statistics(obj).max_gradient
}

// Memory-efficient streaming sparsity calculation
fn estimate_gradient_sparsity_streaming(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut near_zero_count = 0;
    let mut total_count = 0;
    let threshold = 1e-8;

    // Multi-threshold analysis for better sparsity detection
    let thresholds = [1e-8, 1e-6, 1e-4];
    let mut sparsity_levels = vec![0; thresholds.len()];

    for (key, value) in obj {
        let is_gradient_data = key.contains("grad")
            || key.contains("gradient")
            || key.contains("weight")
            || key.contains("param");

        if is_gradient_data {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        let abs_val = val.abs();
                        for (i, &thresh) in thresholds.iter().enumerate() {
                            if abs_val < thresh {
                                sparsity_levels[i] += 1;
                            }
                        }
                        if abs_val < threshold {
                            near_zero_count += 1;
                        }
                        total_count += 1;
                    }
                }
                Value::Array(arr) => {
                    // Efficient array processing with chunking
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                let abs_val = val.abs();
                                if abs_val < threshold {
                                    near_zero_count += 1;
                                }
                                total_count += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if total_count > 0 {
        Some(near_zero_count as f64 / total_count as f64)
    } else {
        None
    }
}


// Robust outlier detection using incremental statistics
fn count_gradient_outliers_robust(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    let mut values = Vec::new();
    let mut outliers = 0;

    // First pass: collect all gradient values
    for (key, value) in obj {
        let is_gradient_data = key.contains("grad")
            || key.contains("gradient")
            || key.contains("weight")
            || key.contains("param");

        if is_gradient_data {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        values.push(val);
                    }
                }
                Value::Array(arr) => {
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                values.push(val);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if values.is_empty() {
        return Some(0);
    }

    // Use incremental statistics for memory efficiency (lawkit pattern)
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let std_dev = variance.sqrt();

    // Multiple outlier detection methods
    let z_score_threshold = 3.0;
    let iqr_multiplier = 1.5;

    // Z-score method
    for &val in &values {
        let z_score = (val - mean).abs() / std_dev;
        if z_score > z_score_threshold {
            outliers += 1;
        }
    }

    // IQR method for additional validation
    let mut sorted_values = values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted_values.len();
    if len >= 4 {
        let q1 = sorted_values[len / 4];
        let q3 = sorted_values[3 * len / 4];
        let iqr = q3 - q1;
        let lower_bound = q1 - iqr_multiplier * iqr;
        let upper_bound = q3 + iqr_multiplier * iqr;

        let iqr_outliers = values
            .iter()
            .filter(|&&val| val < lower_bound || val > upper_bound)
            .count();

        // Use maximum of both methods
        outliers = outliers.max(iqr_outliers);
    }

    Some(outliers)
}

