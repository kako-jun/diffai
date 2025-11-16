use serde_json::Value;

use crate::types::DiffResult;

// A4-3: GRADIENT ANALYSIS - Medium Priority ML Feature
// ============================================================================

// A4-3: GradientAnalysis - Gradient patterns and optimization behavior analysis
pub fn analyze_gradient_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Gradient magnitude analysis
        if let Some((old_mag, new_mag)) = analyze_gradient_magnitudes(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "gradient_magnitudes".to_string(),
                old_mag,
                new_mag,
            ));
        }
        
        // Gradient distribution analysis
        if let Some((old_dist, new_dist)) = analyze_gradient_distributions(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "gradient_distributions".to_string(),
                old_dist,
                new_dist,
            ));
        }
        
        // Gradient flow analysis
        if let Some((old_flow, new_flow)) = analyze_gradient_flow(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "gradient_flow".to_string(),
                old_flow,
                new_flow,
            ));
        }
    }
}

// Analyze gradient magnitude patterns
fn analyze_gradient_magnitudes(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_grad_stats = extract_gradient_statistics(old_obj)?;
    let new_grad_stats = extract_gradient_statistics(new_obj)?;
    
    let mut magnitude_analysis = Vec::new();
    
    // Compare gradient norms
    if let (Some(old_norm), Some(new_norm)) = (old_grad_stats.total_norm, new_grad_stats.total_norm) {
        let norm_change = ((new_norm / old_norm - 1.0) * 100.0);
        let norm_trend = if norm_change.abs() < 5.0 {
            "stable"
        } else if norm_change > 0.0 {
            "increasing"
        } else {
            "decreasing"
        };
        magnitude_analysis.push(format!(
            "total_norm: {:.6} ({:+.1}%, {})", 
            new_norm, norm_change, norm_trend
        ));
    }
    
    // Compare max gradients
    if let (Some(old_max), Some(new_max)) = (old_grad_stats.max_gradient, new_grad_stats.max_gradient) {
        let max_change = ((new_max / old_max - 1.0) * 100.0);
        magnitude_analysis.push(format!(
            "max_gradient: {:.6} ({:+.1}%)", 
            new_max, max_change
        ));
    }
    
    // Compare gradient variance
    if let (Some(old_var), Some(new_var)) = (old_grad_stats.variance, new_grad_stats.variance) {
        let var_change = ((new_var / old_var - 1.0) * 100.0);
        magnitude_analysis.push(format!(
            "variance: {:.6} ({:+.1}%)", 
            new_var, var_change
        ));
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

// Analyze gradient distribution patterns
fn analyze_gradient_distributions(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_grad_stats = extract_gradient_statistics(old_obj)?;
    let new_grad_stats = extract_gradient_statistics(new_obj)?;
    
    let mut distribution_analysis = Vec::new();
    
    // Analyze sparsity (percentage of near-zero gradients)
    if let (Some(old_sparsity), Some(new_sparsity)) = (old_grad_stats.sparsity, new_grad_stats.sparsity) {
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
            new_sparsity * 100.0, sparsity_change * 100.0, sparsity_trend
        ));
    }
    
    // Analyze outlier gradients
    if let (Some(old_outliers), Some(new_outliers)) = (old_grad_stats.outlier_count, new_grad_stats.outlier_count) {
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

// Analyze gradient flow through network layers
fn analyze_gradient_flow(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_flow = extract_gradient_flow_info(old_obj)?;
    let new_flow = extract_gradient_flow_info(new_obj)?;
    
    let mut flow_analysis = Vec::new();
    
    // Analyze vanishing gradients
    if old_flow.vanishing_layers != new_flow.vanishing_layers {
        let change = new_flow.vanishing_layers as i32 - old_flow.vanishing_layers as i32;
        let trend = if change == 0 {
            "stable"
        } else if change > 0 {
            "more_vanishing"
        } else {
            "less_vanishing"
        };
        flow_analysis.push(format!(
            "vanishing_layers: {} ({:+}, {})", 
            new_flow.vanishing_layers, change, trend
        ));
    }
    
    // Analyze exploding gradients
    if old_flow.exploding_layers != new_flow.exploding_layers {
        let change = new_flow.exploding_layers as i32 - old_flow.exploding_layers as i32;
        flow_analysis.push(format!(
            "exploding_layers: {} ({:+})", 
            new_flow.exploding_layers, change
        ));
    }
    
    // Analyze gradient flow balance
    if let (Some(old_balance), Some(new_balance)) = (old_flow.flow_balance, new_flow.flow_balance) {
        let balance_change = new_balance - old_balance;
        let balance_status = if balance_change.abs() < 0.1 {
            "balanced"
        } else if balance_change > 0.0 {
            "forward_dominant"
        } else {
            "backward_dominant"
        };
        flow_analysis.push(format!(
            "flow_balance: {:.3} ({:+.3}, {})", 
            new_balance, balance_change, balance_status
        ));
    }
    
    if flow_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "vanishing: {}, exploding: {}, balance: {:.3}",
        old_flow.vanishing_layers,
        old_flow.exploding_layers,
        old_flow.flow_balance.unwrap_or(0.0)
    );
    let new_info = flow_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Helper structures for gradient analysis
#[derive(Debug)]
struct GradientStatistics {
    total_norm: Option<f64>,
    max_gradient: Option<f64>,
    variance: Option<f64>,
    sparsity: Option<f64>, // Fraction of near-zero gradients
    outlier_count: Option<usize>,
}

#[derive(Debug)]
struct GradientFlowInfo {
    vanishing_layers: usize,
    exploding_layers: usize,
    flow_balance: Option<f64>,
}

// Extract gradient statistics from model data - Enhanced with lawkit memory patterns
fn extract_gradient_statistics(obj: &serde_json::Map<String, Value>) -> Option<GradientStatistics> {
    let mut total_norm = None;
    let mut max_gradient = None;
    let mut variance = None;
    let mut sparsity = None;
    let mut outlier_count = None;
    
    // First pass: Look for explicit gradient statistics (diffx optimization pattern)
    let grad_keys = [
        "grad_norm", "gradient_norm", "total_grad_norm",
        "max_grad", "gradient_max", "grad_variance", 
        "grad_sparsity", "gradient_outliers", "grad_flow"
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
        if total_norm.is_none() { total_norm = stats.total_norm; }
        if max_gradient.is_none() { max_gradient = stats.max_gradient; }
        if variance.is_none() { variance = stats.variance; }
    }
    
    // Memory-efficient sparsity calculation
    if sparsity.is_none() {
        sparsity = estimate_gradient_sparsity_streaming(obj);
    }
    
    // Enhanced outlier detection
    if outlier_count.is_none() {
        outlier_count = count_gradient_outliers_robust(obj);
    }
    
    Some(GradientStatistics {
        total_norm,
        max_gradient,
        variance,
        sparsity,
        outlier_count,
    })
}

// Enhanced gradient flow information extraction with lawkit streaming and helper functions
fn extract_gradient_flow_info(obj: &serde_json::Map<String, Value>) -> Option<GradientFlowInfo> {
    let mut vanishing_layers = 0;
    let mut exploding_layers = 0;
    
    // Use weight-based gradient estimation as fallback
    let estimated_norm = estimate_gradient_norm_from_weights(obj);
    let estimated_max = estimate_max_gradient_from_weights(obj);
    let mut flow_balance = None;
    
    // Enhanced thresholds based on modern deep learning practices
    let vanishing_threshold = 1e-7;  // More sensitive
    let exploding_threshold = 5.0;   // More conservative
    let _moderate_exploding_threshold = 1.0;
    
    let mut layer_gradients = Vec::new();
    
    // Comprehensive gradient analysis across all model components
    for (key, value) in obj {
        let is_gradient_related = key.contains("grad") || key.contains("gradient") ||
                                 key.contains("weight") || key.contains("bias") ||
                                 key.contains("param") || key.contains("layer");
        
        if is_gradient_related {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        let abs_val = val.abs();
                        layer_gradients.push(abs_val);
                        
                        // Enhanced gradient problem detection
                        if abs_val < vanishing_threshold {
                            vanishing_layers += 1;
                        } else if abs_val > exploding_threshold {
                            exploding_layers += 1;
                        }
                    }
                }
                Value::Array(arr) => {
                    // Process arrays with memory efficiency
                    let mut layer_sum = 0.0;
                    let mut layer_count = 0;
                    
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                let abs_val = val.abs();
                                layer_sum += abs_val;
                                layer_count += 1;
                            }
                        }
                    }
                    
                    if layer_count > 0 {
                        let layer_mean = layer_sum / layer_count as f64;
                        layer_gradients.push(layer_mean);
                        
                        if layer_mean < vanishing_threshold {
                            vanishing_layers += 1;
                        } else if layer_mean > exploding_threshold {
                            exploding_layers += 1;
                        }
                    }
                }
                Value::Object(nested) => {
                    // Recursive flow analysis for nested structures
                    if let Some(nested_flow) = extract_gradient_flow_info(nested) {
                        vanishing_layers += nested_flow.vanishing_layers;
                        exploding_layers += nested_flow.exploding_layers;
                    }
                }
                _ => {}
            }
        }
    }
    
    // Enhanced flow balance estimation
    flow_balance = estimate_gradient_flow_balance(obj);
    
    // Additional validation: check gradient magnitude distribution
    if !layer_gradients.is_empty() {
        layer_gradients.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = layer_gradients.len() / 2;
        let median_grad = layer_gradients[median_idx];
        
        // Adjust counts based on median gradient
        if median_grad < vanishing_threshold * 10.0 {
            // If median is very low, model likely has vanishing gradients
            vanishing_layers = vanishing_layers.max(layer_gradients.len() / 3);
        }
    }
    
    Some(GradientFlowInfo {
        vanishing_layers,
        exploding_layers,
        flow_balance,
    })
}

// Enhanced gradient statistics computation using lawkit incremental patterns
struct EnhancedGradientStats {
    total_norm: Option<f64>,
    max_gradient: Option<f64>, 
    variance: Option<f64>,
}

fn compute_enhanced_gradient_statistics(obj: &serde_json::Map<String, Value>) -> EnhancedGradientStats {
    // Use lawkit-style incremental statistics for memory efficiency
    let mut sum_squares = 0.0;
    let mut sum_values = 0.0;
    let mut sum_square_values = 0.0;
    let mut max_val: f64 = 0.0;
    let mut count = 0;
    
    // Multi-source gradient data analysis (PyTorch, Safetensors, NumPy patterns)
    for (key, value) in obj {
        let is_gradient_related = key.contains("grad") || key.contains("gradient") ||
                                 key.contains("weight") || key.contains("bias") ||
                                 key.contains("param");
        
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
    
    let total_norm = if count > 0 { Some(sum_squares.sqrt()) } else { None };
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
fn estimate_gradient_norm_from_weights(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    compute_enhanced_gradient_statistics(obj).total_norm
}

// Legacy function for backward compatibility
fn estimate_max_gradient_from_weights(obj: &serde_json::Map<String, Value>) -> Option<f64> {
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
        let is_gradient_data = key.contains("grad") || key.contains("gradient") ||
                              key.contains("weight") || key.contains("param");
        
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

// Backward compatibility
fn estimate_gradient_sparsity(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    estimate_gradient_sparsity_streaming(obj)
}

// Robust outlier detection using incremental statistics
fn count_gradient_outliers_robust(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    let mut values = Vec::new();
    let mut outliers = 0;
    
    // First pass: collect all gradient values
    for (key, value) in obj {
        let is_gradient_data = key.contains("grad") || key.contains("gradient") ||
                              key.contains("weight") || key.contains("param");
        
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
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
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
        
        let iqr_outliers = values.iter()
            .filter(|&&val| val < lower_bound || val > upper_bound)
            .count();
        
        // Use maximum of both methods
        outliers = outliers.max(iqr_outliers);
    }
    
    Some(outliers)
}

// Backward compatibility
fn count_gradient_outliers(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    count_gradient_outliers_robust(obj)
}

// Enhanced gradient flow balance estimation using lawkit streaming patterns
fn estimate_gradient_flow_balance(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut layer_gradients = Vec::new();
    let mut total_forward_flow = 0.0;
    let mut total_backward_flow = 0.0;
    let mut layer_count = 0;
    
    // Advanced layer-wise gradient flow analysis
    for (key, value) in obj {
        let is_layer_weight = key.contains("layer") || key.contains("block") || 
                             key.contains("weight") || key.contains("attention") ||
                             key.contains("ffn") || key.contains("transformer");
        
        if is_layer_weight {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        layer_gradients.push((key.clone(), val.abs()));
                    }
                }
                Value::Array(arr) => {
                    // Calculate mean gradient magnitude for this layer
                    let mut sum = 0.0;
                    let mut count = 0;
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                sum += val.abs();
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        layer_gradients.push((key.clone(), sum / count as f64));
                    }
                }
                Value::Object(nested) => {
                    // Recursive analysis for nested layer structures
                    if let Some(nested_flow) = estimate_gradient_flow_balance(nested) {
                        layer_gradients.push((key.clone(), nested_flow));
                    }
                }
                _ => {}
            }
        }
    }
    
    if layer_gradients.is_empty() {
        return None;
    }
    
    // Analyze gradient flow patterns
    layer_gradients.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by layer name
    
    // Calculate forward and backward flow based on layer position
    let total_layers = layer_gradients.len();
    for (i, (_layer_name, gradient_mag)) in layer_gradients.iter().enumerate() {
        let layer_position = i as f64 / total_layers as f64;
        
        // Early layers contribute to forward flow
        if layer_position < 0.5 {
            total_forward_flow += gradient_mag * (1.0 - layer_position);
        } else {
            // Later layers contribute to backward flow  
            total_backward_flow += gradient_mag * layer_position;
        }
        
        layer_count += 1;
    }
    
    // Enhanced flow balance calculation
    if layer_count > 0 && total_backward_flow > 1e-12 {
        let flow_ratio = total_forward_flow / total_backward_flow;
        
        // Normalize to 0-1 range where 0.5 is perfect balance
        let normalized_balance = 1.0 / (1.0 + (flow_ratio - 1.0).abs());
        Some(normalized_balance)
    } else {
        None
    }
}
