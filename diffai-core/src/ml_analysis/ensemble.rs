use serde_json::Value;

use crate::types::DiffResult;

// A5-2: ENSEMBLE ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-2: EnsembleAnalysis - Multiple model combination and ensemble method analysis
pub fn analyze_ensemble_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Ensemble composition analysis
        if let Some((old_comp, new_comp)) = analyze_ensemble_composition(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "ensemble_composition".to_string(),
                old_comp,
                new_comp,
            ));
        }
        
        // Ensemble voting strategy analysis
        if let Some((old_vote, new_vote)) = analyze_ensemble_voting_strategy(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "ensemble_voting_strategy".to_string(),
                old_vote,
                new_vote,
            ));
        }
        
        // Model weight distribution analysis
        if let Some((old_weights, new_weights)) = analyze_ensemble_model_weights(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "ensemble_model_weights".to_string(),
                old_weights,
                new_weights,
            ));
        }
    }
}

// Analyze ensemble composition changes
fn analyze_ensemble_composition(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_ensemble = extract_ensemble_composition(old_obj)?;
    let new_ensemble = extract_ensemble_composition(new_obj)?;
    
    let mut composition_analysis = Vec::new();
    
    // Compare number of models in ensemble
    if old_ensemble.num_models != new_ensemble.num_models {
        composition_analysis.push(format!(
            "num_models: {} -> {}",
            old_ensemble.num_models, new_ensemble.num_models
        ));
    }
    
    // Compare model types
    let old_types: std::collections::HashSet<_> = old_ensemble.model_types.iter().collect();
    let new_types: std::collections::HashSet<_> = new_ensemble.model_types.iter().collect();
    
    if old_types != new_types {
        let added_types: Vec<_> = new_types.difference(&old_types).collect();
        let removed_types: Vec<_> = old_types.difference(&new_types).collect();
        
        let mut type_changes = Vec::new();
        if !added_types.is_empty() {
            type_changes.push(format!("+{}", added_types.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",")));
        }
        if !removed_types.is_empty() {
            type_changes.push(format!("-{}", removed_types.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",")));
        }
        if !type_changes.is_empty() {
            composition_analysis.push(format!("model_types: {}", type_changes.join(", ")));
        }
    }
    
    // Compare ensemble method
    if old_ensemble.ensemble_method != new_ensemble.ensemble_method {
        composition_analysis.push(format!(
            "method: {} -> {}",
            old_ensemble.ensemble_method, new_ensemble.ensemble_method
        ));
    }
    
    if composition_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "models: {}, types: [{}], method: {}",
        old_ensemble.num_models,
        old_ensemble.model_types.join(", "),
        old_ensemble.ensemble_method
    );
    let new_info = composition_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Analyze ensemble voting strategy changes
fn analyze_ensemble_voting_strategy(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_voting = extract_ensemble_voting_info(old_obj)?;
    let new_voting = extract_ensemble_voting_info(new_obj)?;
    
    let mut voting_analysis = Vec::new();
    
    // Compare voting type
    if old_voting.voting_type != new_voting.voting_type {
        voting_analysis.push(format!(
            "voting_type: {} -> {}",
            old_voting.voting_type, new_voting.voting_type
        ));
    }
    
    // Compare consensus threshold
    if let (Some(old_threshold), Some(new_threshold)) = (old_voting.consensus_threshold, new_voting.consensus_threshold) {
        if (old_threshold - new_threshold).abs() > 0.01 {
            voting_analysis.push(format!(
                "consensus_threshold: {:.2} -> {:.2}",
                old_threshold, new_threshold
            ));
        }
    }
    
    // Compare weighted voting
    if old_voting.weighted_voting != new_voting.weighted_voting {
        voting_analysis.push(format!(
            "weighted_voting: {} -> {}",
            old_voting.weighted_voting, new_voting.weighted_voting
        ));
    }
    
    // Compare confidence calibration
    if old_voting.confidence_calibration != new_voting.confidence_calibration {
        voting_analysis.push(format!(
            "confidence_calibration: {} -> {}",
            old_voting.confidence_calibration, new_voting.confidence_calibration
        ));
    }
    
    if voting_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "type: {}, threshold: {:.2}, weighted: {}, calibrated: {}",
        old_voting.voting_type,
        old_voting.consensus_threshold.unwrap_or(0.0),
        old_voting.weighted_voting,
        old_voting.confidence_calibration
    );
    let new_info = voting_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Analyze ensemble model weight distribution
fn analyze_ensemble_model_weights(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_weights = extract_ensemble_model_weights(old_obj)?;
    let new_weights = extract_ensemble_model_weights(new_obj)?;
    
    let mut weight_analysis = Vec::new();
    
    // Compare weight distribution entropy
    let old_entropy = calculate_weight_entropy(&old_weights.weights);
    let new_entropy = calculate_weight_entropy(&new_weights.weights);
    
    if let (Some(old_ent), Some(new_ent)) = (old_entropy, new_entropy) {
        let entropy_change = ((new_ent / old_ent - 1.0) * 100.0);
        if entropy_change.abs() > 5.0 {
            let entropy_trend = if entropy_change > 0.0 {
                "more_diverse"
            } else {
                "more_concentrated"
            };
            weight_analysis.push(format!(
                "entropy: {:.3} ({:+.1}%, {})",
                new_ent, entropy_change, entropy_trend
            ));
        }
    }
    
    // Compare dominant model
    if let (Some(old_dom), Some(new_dom)) = (&old_weights.dominant_model, &new_weights.dominant_model) {
        if old_dom != new_dom {
            weight_analysis.push(format!(
                "dominant_model: {} -> {}",
                old_dom, new_dom
            ));
        }
    }
    
    // Compare weight variance
    let old_variance = calculate_weight_variance(&old_weights.weights);
    let new_variance = calculate_weight_variance(&new_weights.weights);
    
    if old_variance > 0.0 && new_variance > 0.0 {
        let variance_change = ((new_variance / old_variance - 1.0) * 100.0);
        if variance_change.abs() > 10.0 {
            weight_analysis.push(format!(
                "weight_variance: {:.4} ({:+.1}%)",
                new_variance, variance_change
            ));
        }
    }
    
    if weight_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "entropy: {:.3}, dominant: {}, variance: {:.4}",
        old_entropy.unwrap_or(0.0),
        old_weights.dominant_model.as_deref().unwrap_or("unknown"),
        old_variance
    );
    let new_info = weight_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Helper structures for ensemble analysis
#[derive(Debug)]
struct EnsembleComposition {
    num_models: usize,
    model_types: Vec<String>,
    ensemble_method: String,
}

#[derive(Debug)]
struct EnsembleVotingInfo {
    voting_type: String,
    consensus_threshold: Option<f64>,
    weighted_voting: bool,
    confidence_calibration: bool,
}

#[derive(Debug)]
struct EnsembleModelWeights {
    weights: Vec<f64>,
    dominant_model: Option<String>,
}

// Extract ensemble composition information
fn extract_ensemble_composition(obj: &serde_json::Map<String, Value>) -> Option<EnsembleComposition> {
    let mut num_models = 0;
    let mut model_types = Vec::new();
    let mut ensemble_method = "unknown".to_string();
    
    // Look for ensemble-specific keys
    for (key, value) in obj {
        if key.contains("ensemble") || key.contains("committee") {
            // Count models in ensemble
            if key.contains("models") || key.contains("members") {
                if let Value::Array(models) = value {
                    num_models = models.len();
                    
                    // Extract model types
                    for model in models {
                        if let Value::Object(model_obj) = model {
                            if let Some(Value::String(model_type)) = model_obj.get("type") {
                                model_types.push(model_type.clone());
                            } else {
                                model_types.push("unknown".to_string());
                            }
                        }
                    }
                } else if let Value::Number(count) = value {
                    if let Some(count_val) = count.as_u64() {
                        num_models = count_val as usize;
                    }
                }
            }
            
            // Detect ensemble method
            if key.contains("method") || key.contains("strategy") {
                if let Value::String(method) = value {
                    ensemble_method = method.clone();
                }
            }
        }
        
        // Infer ensemble from multiple model references
        if key.contains("model_") || (key.contains("classifier_") && key.len() > 12) {
            num_models += 1;
            model_types.push(infer_model_type_from_key(key));
        }
    }
    
    // Infer ensemble method from keys
    if ensemble_method == "unknown" {
        if obj.contains_key("voting") || obj.contains_key("vote") {
            ensemble_method = "voting".to_string();
        } else if obj.contains_key("stacking") || obj.contains_key("stack") {
            ensemble_method = "stacking".to_string();
        } else if obj.contains_key("bagging") || obj.contains_key("bootstrap") {
            ensemble_method = "bagging".to_string();
        } else if obj.contains_key("boosting") || obj.contains_key("boost") {
            ensemble_method = "boosting".to_string();
        }
    }
    
    if num_models > 1 {
        Some(EnsembleComposition {
            num_models,
            model_types,
            ensemble_method,
        })
    } else {
        None
    }
}

// Extract ensemble voting information
fn extract_ensemble_voting_info(obj: &serde_json::Map<String, Value>) -> Option<EnsembleVotingInfo> {
    let mut voting_type = "majority".to_string();
    let mut consensus_threshold = None;
    let mut weighted_voting = false;
    let mut confidence_calibration = false;
    
    // Look for voting configuration
    for (key, value) in obj {
        if key.contains("voting") || key.contains("consensus") {
            if key.contains("type") || key.contains("method") {
                if let Value::String(v_type) = value {
                    voting_type = v_type.clone();
                }
            } else if key.contains("threshold") || key.contains("min") {
                if let Value::Number(threshold) = value {
                    consensus_threshold = threshold.as_f64();
                }
            } else if key.contains("weight") {
                weighted_voting = true;
            }
        }
        
        if key.contains("calibration") || key.contains("confidence") {
            confidence_calibration = true;
        }
    }
    
    // Infer voting type from method names
    if obj.contains_key("soft_voting") || obj.contains_key("probability_voting") {
        voting_type = "soft".to_string();
    } else if obj.contains_key("hard_voting") || obj.contains_key("majority_voting") {
        voting_type = "hard".to_string();
    }
    
    Some(EnsembleVotingInfo {
        voting_type,
        consensus_threshold,
        weighted_voting,
        confidence_calibration,
    })
}

// Extract ensemble model weights
fn extract_ensemble_model_weights(obj: &serde_json::Map<String, Value>) -> Option<EnsembleModelWeights> {
    let mut weights = Vec::new();
    let mut dominant_model = None;
    
    // Look for explicit ensemble weights
    if let Some(Value::Array(weight_array)) = obj.get("ensemble_weights") {
        for weight_val in weight_array {
            if let Value::Number(weight) = weight_val {
                if let Some(w) = weight.as_f64() {
                    weights.push(w);
                }
            }
        }
    } else if let Some(Value::Array(weight_array)) = obj.get("model_weights") {
        for weight_val in weight_array {
            if let Value::Number(weight) = weight_val {
                if let Some(w) = weight.as_f64() {
                    weights.push(w);
                }
            }
        }
    } else {
        // Infer weights from model performance or confidence scores
        for (key, value) in obj {
            if key.contains("model_") && (key.contains("weight") || key.contains("confidence") || key.contains("score")) {
                if let Value::Number(weight) = value {
                    if let Some(w) = weight.as_f64() {
                        weights.push(w);
                    }
                }
            }
        }
    }
    
    // Find dominant model (highest weight)
    if !weights.is_empty() {
        let max_weight = weights.iter().fold(0.0f64, |a, &b| a.max(b));
        if let Some(max_idx) = weights.iter().position(|&x| x == max_weight) {
            dominant_model = Some(format!("model_{}", max_idx));
        }
    }
    
    if !weights.is_empty() {
        Some(EnsembleModelWeights {
            weights,
            dominant_model,
        })
    } else {
        None
    }
}

// Helper functions for ensemble analysis
fn infer_model_type_from_key(key: &str) -> String {
    if key.contains("svm") || key.contains("support_vector") {
        "svm".to_string()
    } else if key.contains("tree") || key.contains("forest") || key.contains("rf") {
        "tree".to_string()
    } else if key.contains("neural") || key.contains("mlp") || key.contains("nn") {
        "neural".to_string()
    } else if key.contains("naive_bayes") || key.contains("nb") {
        "naive_bayes".to_string()
    } else if key.contains("logistic") || key.contains("lr") {
        "logistic".to_string()
    } else if key.contains("xgb") || key.contains("gradient_boost") {
        "gradient_boosting".to_string()
    } else {
        "unknown".to_string()
    }
}

fn calculate_weight_entropy(weights: &[f64]) -> Option<f64> {
    if weights.is_empty() {
        return None;
    }
    
    let sum: f64 = weights.iter().sum();
    if sum == 0.0 {
        return Some(0.0);
    }
    
    let mut entropy = 0.0;
    for &weight in weights {
        if weight > 0.0 {
            let prob = weight / sum;
            entropy -= prob * prob.log2();
        }
    }
    
    Some(entropy)
}

fn calculate_weight_variance(weights: &[f64]) -> f64 {
    if weights.len() <= 1 {
        return 0.0;
    }
    
    let mean: f64 = weights.iter().sum::<f64>() / weights.len() as f64;
    let variance: f64 = weights.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (weights.len() - 1) as f64;
    
    variance
}
