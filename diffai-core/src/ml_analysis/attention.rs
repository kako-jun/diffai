use serde_json::Value;

use crate::diff::{extract_tensor_data, extract_tensor_shape};
use crate::types::DiffResult;

// A5-1: ATTENTION ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-1: AttentionAnalysis - Transformer attention mechanism analysis
pub fn analyze_attention_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Attention head analysis
        if let Some((old_heads, new_heads)) = analyze_attention_heads(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "attention_heads".to_string(),
                old_heads,
                new_heads,
            ));
        }

        // Attention weight distribution analysis
        if let Some((old_dist, new_dist)) = analyze_attention_weight_distributions(old_obj, new_obj)
        {
            results.push(DiffResult::ModelArchitectureChanged(
                "attention_weight_distributions".to_string(),
                old_dist,
                new_dist,
            ));
        }

        // Multi-head attention analysis
        if let Some((old_mha, new_mha)) = analyze_multihead_attention(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "multihead_attention".to_string(),
                old_mha,
                new_mha,
            ));
        }
    }
}

// Analyze attention head configurations
fn analyze_attention_heads(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_heads = extract_attention_head_info(old_obj)?;
    let new_heads = extract_attention_head_info(new_obj)?;

    let mut head_analysis = Vec::new();

    // Compare number of attention heads
    if old_heads.num_heads != new_heads.num_heads {
        head_analysis.push(format!(
            "num_heads: {} -> {}",
            old_heads.num_heads, new_heads.num_heads
        ));
    }

    // Compare head dimensions
    if let (Some(old_dim), Some(new_dim)) = (old_heads.head_dim, new_heads.head_dim) {
        if old_dim != new_dim {
            head_analysis.push(format!("head_dim: {old_dim} -> {new_dim}"));
        }
    }

    // Compare attention patterns per head
    if old_heads.head_patterns != new_heads.head_patterns {
        let pattern_changes =
            compare_attention_patterns(&old_heads.head_patterns, &new_heads.head_patterns);
        if !pattern_changes.is_empty() {
            head_analysis.push(format!("patterns: {}", pattern_changes.join(", ")));
        }
    }

    if head_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "heads: {}, dim: {}, patterns: {}",
        old_heads.num_heads,
        old_heads.head_dim.unwrap_or(0),
        old_heads.head_patterns.len()
    );
    let new_info = head_analysis.join(", ");

    Some((old_info, new_info))
}

// Analyze attention weight distributions
fn analyze_attention_weight_distributions(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_dist = extract_attention_weight_distribution(old_obj)?;
    let new_dist = extract_attention_weight_distribution(new_obj)?;

    let mut distribution_analysis = Vec::new();

    // Compare attention sparsity
    if let (Some(old_sparsity), Some(new_sparsity)) = (old_dist.sparsity, new_dist.sparsity) {
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

    // Compare attention entropy
    if let (Some(old_entropy), Some(new_entropy)) = (old_dist.entropy, new_dist.entropy) {
        let entropy_change = (new_entropy / old_entropy - 1.0) * 100.0;
        let entropy_trend = if entropy_change.abs() < 5.0 {
            "stable"
        } else if entropy_change > 0.0 {
            "more_diverse"
        } else {
            "more_focused"
        };
        distribution_analysis.push(format!(
            "entropy: {new_entropy:.3} ({entropy_change:+.1}%, {entropy_trend})"
        ));
    }

    // Compare attention peak concentration
    if let (Some(old_peak), Some(new_peak)) =
        (old_dist.peak_concentration, new_dist.peak_concentration)
    {
        let peak_change = new_peak - old_peak;
        distribution_analysis.push(format!(
            "peak_concentration: {new_peak:.3} ({peak_change:+.3})"
        ));
    }

    if distribution_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "sparsity: {:.1}%, entropy: {:.3}, peak: {:.3}",
        old_dist.sparsity.unwrap_or(0.0) * 100.0,
        old_dist.entropy.unwrap_or(0.0),
        old_dist.peak_concentration.unwrap_or(0.0)
    );
    let new_info = distribution_analysis.join(", ");

    Some((old_info, new_info))
}

// Analyze multi-head attention configurations
fn analyze_multihead_attention(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(String, String)> {
    let old_mha = extract_multihead_attention_info(old_obj)?;
    let new_mha = extract_multihead_attention_info(new_obj)?;

    let mut mha_analysis = Vec::new();

    // Compare attention layers
    if old_mha.num_layers != new_mha.num_layers {
        mha_analysis.push(format!(
            "layers: {} -> {}",
            old_mha.num_layers, new_mha.num_layers
        ));
    }

    // Compare self-attention vs cross-attention ratio
    if let (Some(old_ratio), Some(new_ratio)) =
        (old_mha.self_attention_ratio, new_mha.self_attention_ratio)
    {
        let ratio_change = new_ratio - old_ratio;
        if ratio_change.abs() > 0.05 {
            mha_analysis.push(format!(
                "self_attention_ratio: {new_ratio:.2} ({ratio_change:+.2})"
            ));
        }
    }

    // Compare attention dropout
    if let (Some(old_dropout), Some(new_dropout)) =
        (old_mha.attention_dropout, new_mha.attention_dropout)
    {
        if (old_dropout - new_dropout).abs() > 0.001 {
            mha_analysis.push(format!("dropout: {old_dropout:.3} -> {new_dropout:.3}"));
        }
    }

    // Compare position encoding changes
    if old_mha.position_encoding != new_mha.position_encoding {
        mha_analysis.push(format!(
            "position_encoding: {} -> {}",
            old_mha.position_encoding, new_mha.position_encoding
        ));
    }

    if mha_analysis.is_empty() {
        return None;
    }

    let old_info = format!(
        "layers: {}, self_ratio: {:.2}, dropout: {:.3}, pos_enc: {}",
        old_mha.num_layers,
        old_mha.self_attention_ratio.unwrap_or(0.0),
        old_mha.attention_dropout.unwrap_or(0.0),
        old_mha.position_encoding
    );
    let new_info = mha_analysis.join(", ");

    Some((old_info, new_info))
}

// Helper structures for attention analysis
#[derive(Debug)]
struct AttentionHeadInfo {
    num_heads: usize,
    head_dim: Option<usize>,
    head_patterns: Vec<String>,
}

#[derive(Debug)]
struct AttentionWeightDistribution {
    sparsity: Option<f64>,
    entropy: Option<f64>,
    peak_concentration: Option<f64>,
}

#[derive(Debug)]
struct MultiHeadAttentionInfo {
    num_layers: usize,
    self_attention_ratio: Option<f64>,
    attention_dropout: Option<f64>,
    position_encoding: String,
}

// Extract attention head information
fn extract_attention_head_info(obj: &serde_json::Map<String, Value>) -> Option<AttentionHeadInfo> {
    let mut num_heads = 0;
    let mut head_dim = None;
    let mut head_patterns = Vec::new();

    // Look for attention-related keys
    for (key, value) in obj {
        if key.contains("attention") || key.contains("attn") {
            // Count attention heads
            if key.contains("head") || key.contains("multi_head") {
                if let Some(shape) = extract_tensor_shape(value) {
                    if shape.len() >= 2 {
                        num_heads = shape[0]; // First dimension often represents heads
                        head_dim = Some(shape[1]); // Second dimension often represents head dimension
                    }
                }
            }

            // Extract attention patterns
            if key.contains("weight")
                || key.contains("query")
                || key.contains("key")
                || key.contains("value")
            {
                head_patterns.push(extract_attention_pattern_type(key));
            }
        }
    }

    // If no explicit heads found, estimate from common patterns
    if num_heads == 0 {
        num_heads = estimate_attention_heads_from_weights(obj);
    }

    if num_heads > 0 {
        Some(AttentionHeadInfo {
            num_heads,
            head_dim,
            head_patterns,
        })
    } else {
        None
    }
}

// Extract attention weight distribution statistics
fn extract_attention_weight_distribution(
    obj: &serde_json::Map<String, Value>,
) -> Option<AttentionWeightDistribution> {
    let mut sparsity = None;
    let mut entropy = None;
    let mut peak_concentration = None;

    // Look for attention weights and calculate statistics
    for (key, value) in obj {
        if key.contains("attention") && key.contains("weight") {
            if let Some(data) = extract_tensor_data(value) {
                // Calculate sparsity (fraction of near-zero weights)
                let near_zero_count = data.iter().filter(|&&x| x.abs() < 1e-6).count();
                sparsity = Some(near_zero_count as f64 / data.len() as f64);

                // Calculate entropy (measure of attention distribution)
                entropy = calculate_attention_entropy(&data);

                // Calculate peak concentration (max attention weight)
                peak_concentration = data
                    .iter()
                    .map(|x| x.abs())
                    .fold(0.0f64, |a, b| a.max(b))
                    .into();

                break; // Use first attention weight tensor found
            }
        }
    }

    Some(AttentionWeightDistribution {
        sparsity,
        entropy,
        peak_concentration,
    })
}

// Extract multi-head attention configuration
fn extract_multihead_attention_info(
    obj: &serde_json::Map<String, Value>,
) -> Option<MultiHeadAttentionInfo> {
    let mut num_layers = 0;
    let mut self_attention_ratio = None;
    let mut attention_dropout = None;
    let mut position_encoding = "unknown".to_string();

    // Count attention layers
    for key in obj.keys() {
        if key.contains("layer") && key.contains("attention") {
            num_layers += 1;
        }
    }

    // Look for attention configuration
    if let Some(Value::Number(dropout)) = obj.get("attention_dropout") {
        attention_dropout = dropout.as_f64();
    }

    // Detect position encoding type
    if obj.contains_key("position_embeddings") || obj.contains_key("pos_embed") {
        position_encoding = "learned".to_string();
    } else if obj
        .keys()
        .any(|k| k.contains("sinusoidal") || k.contains("sin_pos"))
    {
        position_encoding = "sinusoidal".to_string();
    } else if obj
        .keys()
        .any(|k| k.contains("relative") || k.contains("rel_pos"))
    {
        position_encoding = "relative".to_string();
    }

    // Estimate self-attention ratio
    let self_attn_count = obj
        .keys()
        .filter(|k| k.contains("self_attn") || k.contains("self_attention"))
        .count();
    let cross_attn_count = obj
        .keys()
        .filter(|k| k.contains("cross_attn") || k.contains("cross_attention"))
        .count();
    let total_attn = self_attn_count + cross_attn_count;
    if total_attn > 0 {
        self_attention_ratio = Some(self_attn_count as f64 / total_attn as f64);
    }

    if num_layers > 0 {
        Some(MultiHeadAttentionInfo {
            num_layers,
            self_attention_ratio,
            attention_dropout,
            position_encoding,
        })
    } else {
        None
    }
}

// Helper functions for attention analysis
fn compare_attention_patterns(old_patterns: &[String], new_patterns: &[String]) -> Vec<String> {
    let mut changes = Vec::new();

    let old_set: std::collections::HashSet<_> = old_patterns.iter().collect();
    let new_set: std::collections::HashSet<_> = new_patterns.iter().collect();

    // Find added patterns
    for pattern in new_set.difference(&old_set) {
        changes.push(format!("+{pattern}"));
    }

    // Find removed patterns
    for pattern in old_set.difference(&new_set) {
        changes.push(format!("-{pattern}"));
    }

    changes
}

fn extract_attention_pattern_type(key: &str) -> String {
    if key.contains("query") || key.contains("q_proj") {
        "query".to_string()
    } else if key.contains("key") || key.contains("k_proj") {
        "key".to_string()
    } else if key.contains("value") || key.contains("v_proj") {
        "value".to_string()
    } else if key.contains("output") || key.contains("o_proj") {
        "output".to_string()
    } else {
        "generic".to_string()
    }
}

fn estimate_attention_heads_from_weights(obj: &serde_json::Map<String, Value>) -> usize {
    // Heuristic: look for common multi-head attention patterns
    for (key, value) in obj {
        if key.contains("multi_head") || key.contains("mha") {
            if let Some(shape) = extract_tensor_shape(value) {
                if shape.len() >= 3 && shape[0] > 1 && shape[0] <= 32 {
                    return shape[0]; // Reasonable number of heads
                }
            }
        }
    }

    // Default estimation based on common architectures
    if obj.keys().any(|k| k.contains("transformer")) {
        return 8; // Common default
    }

    0
}

fn calculate_attention_entropy(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }

    // Normalize to probability distribution
    let sum: f64 = data.iter().map(|x| x.abs()).sum();
    if sum == 0.0 {
        return Some(0.0);
    }

    let mut entropy = 0.0;
    for &value in data {
        let prob = value.abs() / sum;
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }

    Some(entropy)
}
