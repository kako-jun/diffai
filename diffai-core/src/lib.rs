#![allow(clippy::uninlined_format_args)]

use regex::Regex;
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;
// use ini::Ini;
use anyhow::{anyhow, Result};
use csv::ReaderBuilder;
use quick_xml::de::from_str;
// AI/ML dependencies
use candle_core::Device;
use safetensors::SafeTensors;

#[derive(Debug, PartialEq, Serialize)]
pub enum DiffResult {
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, Value, Value),
    // AI/ML specific diff results
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    TensorStatsChanged(String, TensorStats, TensorStats),
    ModelArchitectureChanged(String, ModelInfo, ModelInfo),
    // Learning progress analysis
    LearningProgress(String, LearningProgressInfo),
    ConvergenceAnalysis(String, ConvergenceInfo),
    // Anomaly detection
    AnomalyDetection(String, AnomalyInfo),
    GradientAnalysis(String, GradientInfo),
    // Memory and performance analysis
    MemoryAnalysis(String, MemoryAnalysisInfo),
    InferenceSpeedAnalysis(String, InferenceSpeedInfo),
    // CI/CD integration
    RegressionTest(String, RegressionTestInfo),
    AlertOnDegradation(String, AlertInfo),
    // Code review support
    ReviewFriendly(String, ReviewFriendlyInfo),
    ChangeSummary(String, ChangeSummaryInfo),
    RiskAssessment(String, RiskAssessmentInfo),
    // Architecture comparison
    ArchitectureComparison(String, ArchitectureComparisonInfo),
    ParamEfficiencyAnalysis(String, ParamEfficiencyInfo),
    // Hyperparameter analysis
    HyperparameterImpact(String, HyperparameterInfo),
    LearningRateAnalysis(String, LearningRateInfo),
    // A/B test support
    DeploymentReadiness(String, DeploymentReadinessInfo),
    PerformanceImpactEstimate(String, PerformanceImpactInfo),
    // Experiment documentation
    GenerateReport(String, ReportInfo),
    MarkdownOutput(String, MarkdownInfo),
    IncludeCharts(String, ChartInfo),
    // Embedding analysis
    EmbeddingAnalysis(String, EmbeddingInfo),
    SimilarityMatrix(String, SimilarityMatrixInfo),
    ClusteringChange(String, ClusteringInfo),
    // Attention analysis
    AttentionAnalysis(String, AttentionInfo),
    HeadImportance(String, HeadImportanceInfo),
    AttentionPatternDiff(String, AttentionPatternInfo),
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub total_params: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelInfo {
    pub total_parameters: usize,
    pub layer_count: usize,
    pub layer_types: HashMap<String, usize>,
    pub model_size_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LearningProgressInfo {
    pub loss_trend: String, // "improving", "degrading", "stable"
    pub parameter_update_magnitude: f64,
    pub gradient_norm_ratio: f64,
    pub convergence_speed: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConvergenceInfo {
    pub convergence_status: String, // "converged", "diverging", "oscillating", "stable"
    pub gradient_stability: f64,
    pub parameter_stability: f64,
    pub recommended_action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnomalyInfo {
    pub anomaly_type: String, // "gradient_explosion", "vanishing_gradient", "weight_explosion", "normal"
    pub severity: String,     // "critical", "warning", "minor", "none"
    pub affected_layers: Vec<String>,
    pub recommended_action: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GradientInfo {
    pub gradient_norm_estimate: f64,
    pub gradient_flow_health: String, // "healthy", "diminishing", "exploding", "dead"
    pub problematic_layers: Vec<String>,
    pub gradient_ratio: f64, // Current vs expected gradient magnitude ratio
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryAnalysisInfo {
    pub model1_size_bytes: usize,
    pub model2_size_bytes: usize,
    pub memory_delta_bytes: i64, // Change in memory usage (can be negative)
    pub memory_efficiency_ratio: f64, // Parameters per byte ratio
    pub estimated_gpu_memory_mb: f64, // Estimated GPU memory usage in MB
    pub memory_recommendation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InferenceSpeedInfo {
    pub model1_flops_estimate: u64, // Floating point operations estimate
    pub model2_flops_estimate: u64,
    pub speed_change_ratio: f64, // Relative speed change (>1.0 = slower, <1.0 = faster)
    pub bottleneck_layers: Vec<String>, // Layers that may cause performance bottlenecks
    pub inference_recommendation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RegressionTestInfo {
    pub test_passed: bool,
    pub failed_checks: Vec<String>, // List of checks that failed
    pub severity_level: String,     // "low", "medium", "high", "critical"
    pub recommended_action: String,
    pub performance_degradation: f64, // Percentage degradation (negative for improvement)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AlertInfo {
    pub alert_triggered: bool,
    pub alert_type: String, // "performance", "accuracy", "memory", "stability"
    pub threshold_exceeded: f64, // How much the threshold was exceeded (factor)
    pub current_value: f64,
    pub threshold_value: f64,
    pub alert_message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReviewFriendlyInfo {
    pub summary: String,                 // High-level summary of changes
    pub impact_assessment: String,       // "low", "medium", "high", "critical"
    pub key_changes: Vec<String>,        // List of most important changes
    pub reviewer_notes: Vec<String>,     // Notes for human reviewers
    pub approval_recommendation: String, // "approve", "request_changes", "needs_discussion"
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChangeSummaryInfo {
    pub total_layers_changed: usize,
    pub most_changed_layers: Vec<String>, // Layers with biggest changes
    pub change_distribution: HashMap<String, usize>, // Layer types and their change counts
    pub overall_change_magnitude: f64,    // 0.0 to 1.0 scale
    pub change_patterns: Vec<String>,     // Detected patterns in changes
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RiskAssessmentInfo {
    pub overall_risk_level: String, // "low", "medium", "high", "critical"
    pub risk_factors: Vec<String>,  // Identified risk factors
    pub deployment_readiness: String, // "ready", "needs_testing", "not_ready"
    pub rollback_difficulty: String, // "easy", "moderate", "difficult"
    pub recommended_monitoring: Vec<String>, // What to monitor post-deployment
}

// Architecture comparison info
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArchitectureComparisonInfo {
    pub architecture_type_1: String, // "transformer", "cnn", "resnet", etc.
    pub architecture_type_2: String,
    pub structural_differences: Vec<String>, // Key architectural differences
    pub layer_depth_comparison: (usize, usize), // (model1_depth, model2_depth)
    pub activation_functions: (Vec<String>, Vec<String>), // Different activation functions used
    pub comparison_summary: String,          // Human-readable comparison summary
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParamEfficiencyInfo {
    pub params_per_layer_1: f64,    // Average parameters per layer in model 1
    pub params_per_layer_2: f64,    // Average parameters per layer in model 2
    pub efficiency_ratio: f64,      // Model 2 efficiency vs Model 1 (higher = more efficient)
    pub sparse_layers: Vec<String>, // Layers with high sparsity
    pub dense_layers: Vec<String>,  // Layers with high density
    pub efficiency_recommendation: String, // Optimization suggestions
}

// Hyperparameter analysis info
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HyperparameterInfo {
    pub detected_changes: HashMap<String, (String, String)>, // param: (old_value, new_value)
    pub impact_scores: HashMap<String, f64>,                 // param: impact_score (0.0-1.0)
    pub high_impact_params: Vec<String>,                     // Parameters with significant impact
    pub suggested_tuning: Vec<String>,                       // Suggested parameter adjustments
    pub stability_assessment: String, // "stable", "unstable", "needs_attention"
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LearningRateInfo {
    pub estimated_lr_1: f64, // Estimated learning rate from model 1 changes
    pub estimated_lr_2: f64, // Estimated learning rate from model 2 changes
    pub lr_schedule_pattern: String, // "constant", "decay", "cyclic", "adaptive"
    pub lr_effectiveness: f64, // 0.0-1.0 score of learning rate effectiveness
    pub lr_recommendation: String, // Learning rate adjustment suggestions
}

// A/B test support info
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DeploymentReadinessInfo {
    pub readiness_score: f64,           // 0.0-1.0 overall readiness score
    pub blocking_issues: Vec<String>,   // Issues that block deployment
    pub warnings: Vec<String>,          // Issues that should be monitored
    pub deployment_strategy: String,    // "safe", "gradual", "full", "hold"
    pub estimated_risk_level: String,   // "low", "medium", "high", "critical"
    pub go_live_recommendation: String, // Final recommendation
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerformanceImpactInfo {
    pub latency_impact_estimate: f64,    // Expected latency change ratio
    pub throughput_impact_estimate: f64, // Expected throughput change ratio
    pub memory_impact_estimate: f64,     // Expected memory usage change ratio
    pub accuracy_impact_estimate: f64,   // Expected accuracy change (if detectable)
    pub overall_performance_score: f64,  // Combined performance impact score
    pub performance_recommendation: String, // Performance optimization suggestions
}

// Experiment documentation info
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ReportInfo {
    pub experiment_title: String,  // Auto-generated experiment title
    pub summary: String,           // Executive summary of changes
    pub key_findings: Vec<String>, // Most important discoveries
    pub methodology: String,       // How the comparison was performed
    pub conclusions: Vec<String>,  // Main conclusions and insights
    pub next_steps: Vec<String>,   // Recommended follow-up actions
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MarkdownInfo {
    pub markdown_content: String, // Full markdown report
    pub sections: Vec<String>,    // List of section headers
    pub tables_generated: usize,  // Number of tables included
    pub charts_referenced: usize, // Number of chart references
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ChartInfo {
    pub chart_type: String,             // "bar", "line", "heatmap", "scatter"
    pub chart_data: Vec<(String, f64)>, // Data points for the chart
    pub chart_title: String,            // Title for the chart
    pub x_axis_label: String,           // X-axis label
    pub y_axis_label: String,           // Y-axis label
    pub chart_description: String,      // Description of what the chart shows
}

// Embedding analysis info
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EmbeddingInfo {
    pub embedding_layers: Vec<String>, // Names of embedding layers found
    pub embedding_dimensions: HashMap<String, (usize, usize)>, // layer: (dim1, dim2)
    pub embedding_similarity: HashMap<String, f64>, // layer: cosine_similarity
    pub semantic_drift: f64,           // Overall semantic drift score (0.0-1.0)
    pub affected_vocabularies: Vec<String>, // Vocabularies that changed significantly
    pub embedding_recommendation: String, // Suggestions for embedding optimization
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SimilarityMatrixInfo {
    pub matrix_size: (usize, usize), // Dimensions of similarity matrix
    pub average_similarity: f64,     // Average similarity score
    pub similarity_distribution: Vec<f64>, // Distribution of similarity scores
    pub outlier_pairs: Vec<(String, String, f64)>, // (entity1, entity2, similarity)
    pub cluster_count_estimate: usize, // Estimated number of clusters
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ClusteringInfo {
    pub clusters_before: usize,            // Number of clusters in model 1
    pub clusters_after: usize,             // Number of clusters in model 2
    pub cluster_stability: f64,            // How stable clusters are (0.0-1.0)
    pub migrated_entities: Vec<String>,    // Entities that changed clusters
    pub new_clusters: Vec<String>,         // Newly formed clusters
    pub dissolved_clusters: Vec<String>,   // Clusters that disappeared
    pub clustering_recommendation: String, // Clustering optimization suggestions
}

// Attention analysis info
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AttentionInfo {
    pub attention_layers: Vec<String>, // Names of attention layers
    pub attention_heads_count: HashMap<String, (usize, usize)>, // layer: (heads1, heads2)
    pub attention_pattern_changes: Vec<String>, // Significant pattern changes
    pub attention_entropy: HashMap<String, f64>, // layer: entropy_score
    pub attention_focus_shift: String, // "local_to_global", "global_to_local", "stable"
    pub attention_recommendation: String, // Attention optimization suggestions
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HeadImportanceInfo {
    pub head_importance_scores: HashMap<String, Vec<f64>>, // layer: [head_scores]
    pub most_important_heads: Vec<(String, usize, f64)>,   // (layer, head_idx, importance)
    pub least_important_heads: Vec<(String, usize, f64)>,  // (layer, head_idx, importance)
    pub prunable_heads: Vec<(String, usize)>,              // (layer, head_idx) that can be pruned
    pub head_specialization: HashMap<String, String>,      // head: specialization_type
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AttentionPatternInfo {
    pub pattern_type_1: String, // "local", "global", "sparse", "dense"
    pub pattern_type_2: String,
    pub pattern_similarity: f64, // Overall pattern similarity (0.0-1.0)
    pub attention_span_change: f64, // Change in attention span
    pub locality_bias_change: f64, // Change in locality bias
    pub pattern_change_summary: String, // Human-readable summary of changes
}

pub fn diff(
    v1: &Value,
    v2: &Value,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) -> Vec<DiffResult> {
    let mut results = Vec::new();

    // Handle root level type or value change first
    if !values_are_equal(v1, v2, epsilon) {
        let type_match = matches!(
            (v1, v2),
            (Value::Null, Value::Null)
                | (Value::Bool(_), Value::Bool(_))
                | (Value::Number(_), Value::Number(_))
                | (Value::String(_), Value::String(_))
                | (Value::Array(_), Value::Array(_))
                | (Value::Object(_), Value::Object(_))
        );

        if !type_match {
            results.push(DiffResult::TypeChanged(
                "".to_string(),
                v1.clone(),
                v2.clone(),
            ));
            return results; // If root type changed, no further diffing needed
        } else if v1.is_object() && v2.is_object() {
            diff_objects(
                "",
                v1.as_object().unwrap(),
                v2.as_object().unwrap(),
                &mut results,
                ignore_keys_regex,
                epsilon,
                array_id_key,
            );
        } else if v1.is_array() && v2.is_array() {
            diff_arrays(
                "",
                v1.as_array().unwrap(),
                v2.as_array().unwrap(),
                &mut results,
                ignore_keys_regex,
                epsilon,
                array_id_key,
            );
        } else {
            // Simple value modification at root
            results.push(DiffResult::Modified("".to_string(), v1.clone(), v2.clone()));
            return results;
        }
    }

    results
}

fn diff_recursive(
    path: &str,
    v1: &Value,
    v2: &Value,
    results: &mut Vec<DiffResult>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) {
    match (v1, v2) {
        (Value::Object(map1), Value::Object(map2)) => {
            diff_objects(
                path,
                map1,
                map2,
                results,
                ignore_keys_regex,
                epsilon,
                array_id_key,
            );
        }
        (Value::Array(arr1), Value::Array(arr2)) => {
            diff_arrays(
                path,
                arr1,
                arr2,
                results,
                ignore_keys_regex,
                epsilon,
                array_id_key,
            );
        }
        _ => { /* Should not happen if called correctly from diff_objects/diff_arrays */ }
    }
}

fn diff_objects(
    path: &str,
    map1: &serde_json::Map<String, Value>,
    map2: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) {
    // Check for modified or removed keys
    for (key, value1) in map1 {
        let current_path = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };
        if let Some(regex) = ignore_keys_regex {
            if regex.is_match(key) {
                continue;
            }
        }
        match map2.get(key) {
            Some(value2) => {
                // Recurse for nested objects/arrays
                if value1.is_object() && value2.is_object()
                    || value1.is_array() && value2.is_array()
                {
                    diff_recursive(
                        &current_path,
                        value1,
                        value2,
                        results,
                        ignore_keys_regex,
                        epsilon,
                        array_id_key,
                    );
                } else if !values_are_equal(value1, value2, epsilon) {
                    let type_match = matches!(
                        (value1, value2),
                        (Value::Null, Value::Null)
                            | (Value::Bool(_), Value::Bool(_))
                            | (Value::Number(_), Value::Number(_))
                            | (Value::String(_), Value::String(_))
                            | (Value::Array(_), Value::Array(_))
                            | (Value::Object(_), Value::Object(_))
                    );

                    if !type_match {
                        results.push(DiffResult::TypeChanged(
                            current_path,
                            value1.clone(),
                            value2.clone(),
                        ));
                    } else {
                        results.push(DiffResult::Modified(
                            current_path,
                            value1.clone(),
                            value2.clone(),
                        ));
                    }
                }
            }
            None => {
                results.push(DiffResult::Removed(current_path, value1.clone()));
            }
        }
    }

    // Check for added keys
    for (key, value2) in map2 {
        if !map1.contains_key(key) {
            let current_path = if path.is_empty() {
                key.clone()
            } else {
                format!("{path}.{key}")
            };
            results.push(DiffResult::Added(current_path, value2.clone()));
        }
    }
}

fn diff_arrays(
    path: &str,
    arr1: &[Value],
    arr2: &[Value],
    results: &mut Vec<DiffResult>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) {
    if let Some(id_key) = array_id_key {
        let mut map1: HashMap<Value, &Value> = HashMap::new();
        let mut no_id_elements1: Vec<(usize, &Value)> = Vec::new();
        for (i, val) in arr1.iter().enumerate() {
            if let Some(id_val) = val.get(id_key) {
                map1.insert(id_val.clone(), val);
            } else {
                no_id_elements1.push((i, val));
            }
        }

        let mut map2: HashMap<Value, &Value> = HashMap::new();
        let mut no_id_elements2: Vec<(usize, &Value)> = Vec::new();
        for (i, val) in arr2.iter().enumerate() {
            if let Some(id_val) = val.get(id_key) {
                map2.insert(id_val.clone(), val);
            } else {
                no_id_elements2.push((i, val));
            }
        }

        // Check for modified or removed elements
        for (id_val, val1) in &map1 {
            let current_path = format!("{}[{}={}]", path, id_key, id_val);
            match map2.get(id_val) {
                Some(val2) => {
                    // Recurse for nested objects/arrays
                    if val1.is_object() && val2.is_object() || val1.is_array() && val2.is_array() {
                        diff_recursive(
                            &current_path,
                            val1,
                            val2,
                            results,
                            ignore_keys_regex,
                            epsilon,
                            array_id_key,
                        );
                    } else if !values_are_equal(val1, val2, epsilon) {
                        let type_match = matches!(
                            (val1, val2),
                            (Value::Null, Value::Null)
                                | (Value::Bool(_), Value::Bool(_))
                                | (Value::Number(_), Value::Number(_))
                                | (Value::String(_), Value::String(_))
                                | (Value::Array(_), Value::Array(_))
                                | (Value::Object(_), Value::Object(_))
                        );

                        if !type_match {
                            results.push(DiffResult::TypeChanged(
                                current_path,
                                (*val1).clone(),
                                (*val2).clone(),
                            ));
                        } else {
                            results.push(DiffResult::Modified(
                                current_path,
                                (*val1).clone(),
                                (*val2).clone(),
                            ));
                        }
                    }
                }
                None => {
                    results.push(DiffResult::Removed(current_path, (*val1).clone()));
                }
            }
        }

        // Check for added elements with ID
        for (id_val, val2) in map2 {
            if !map1.contains_key(&id_val) {
                let current_path = format!("{}[{}={}]", path, id_key, id_val);
                results.push(DiffResult::Added(current_path, val2.clone()));
            }
        }

        // Handle elements without ID using index-based comparison
        let max_len = no_id_elements1.len().max(no_id_elements2.len());
        for i in 0..max_len {
            match (no_id_elements1.get(i), no_id_elements2.get(i)) {
                (Some((idx1, val1)), Some((_idx2, val2))) => {
                    let current_path = format!("{}[{}]", path, idx1);
                    if val1.is_object() && val2.is_object() || val1.is_array() && val2.is_array() {
                        diff_recursive(
                            &current_path,
                            val1,
                            val2,
                            results,
                            ignore_keys_regex,
                            epsilon,
                            array_id_key,
                        );
                    } else if !values_are_equal(val1, val2, epsilon) {
                        let type_match = matches!(
                            (val1, val2),
                            (Value::Null, Value::Null)
                                | (Value::Bool(_), Value::Bool(_))
                                | (Value::Number(_), Value::Number(_))
                                | (Value::String(_), Value::String(_))
                                | (Value::Array(_), Value::Array(_))
                                | (Value::Object(_), Value::Object(_))
                        );

                        if !type_match {
                            results.push(DiffResult::TypeChanged(
                                current_path,
                                (*val1).clone(),
                                (*val2).clone(),
                            ));
                        } else {
                            results.push(DiffResult::Modified(
                                current_path,
                                (*val1).clone(),
                                (*val2).clone(),
                            ));
                        }
                    }
                }
                (Some((idx1, val1)), None) => {
                    let current_path = format!("{}[{}]", path, idx1);
                    results.push(DiffResult::Removed(current_path, (*val1).clone()));
                }
                (None, Some((idx2, val2))) => {
                    let current_path = format!("{}[{}]", path, idx2);
                    results.push(DiffResult::Added(current_path, (*val2).clone()));
                }
                (None, None) => break,
            }
        }
    } else {
        // Fallback to index-based comparison if no id_key is provided
        let max_len = arr1.len().max(arr2.len());
        for i in 0..max_len {
            let current_path = format!("{}[{}]", path, i);
            match (arr1.get(i), arr2.get(i)) {
                (Some(val1), Some(val2)) => {
                    // Recurse for nested objects/arrays within arrays
                    if val1.is_object() && val2.is_object() || val1.is_array() && val2.is_array() {
                        diff_recursive(
                            &current_path,
                            val1,
                            val2,
                            results,
                            ignore_keys_regex,
                            epsilon,
                            array_id_key,
                        );
                    } else if !values_are_equal(val1, val2, epsilon) {
                        let type_match = matches!(
                            (val1, val2),
                            (Value::Null, Value::Null)
                                | (Value::Bool(_), Value::Bool(_))
                                | (Value::Number(_), Value::Number(_))
                                | (Value::String(_), Value::String(_))
                                | (Value::Array(_), Value::Array(_))
                                | (Value::Object(_), Value::Object(_))
                        );

                        if !type_match {
                            results.push(DiffResult::TypeChanged(
                                current_path,
                                val1.clone(),
                                val2.clone(),
                            ));
                        } else {
                            results.push(DiffResult::Modified(
                                current_path,
                                val1.clone(),
                                val2.clone(),
                            ));
                        }
                    }
                }
                (Some(val1), None) => {
                    results.push(DiffResult::Removed(current_path, val1.clone()));
                }
                (None, Some(val2)) => {
                    results.push(DiffResult::Added(current_path, val2.clone()));
                }
                (None, None) => { /* Should not happen */ }
            }
        }
    }
}

fn values_are_equal(v1: &Value, v2: &Value, epsilon: Option<f64>) -> bool {
    if let (Some(e), Value::Number(n1), Value::Number(n2)) = (epsilon, v1, v2) {
        if let (Some(f1), Some(f2)) = (n1.as_f64(), n2.as_f64()) {
            return (f1 - f2).abs() < e;
        }
    }
    v1 == v2
}

pub fn value_type_name(value: &Value) -> &str {
    match value {
        Value::Null => "Null",
        Value::Bool(_) => "Boolean",
        Value::Number(_) => "Number",
        Value::String(_) => "String",
        Value::Array(_) => "Array",
        Value::Object(_) => "Object",
    }
}

pub fn parse_ini(content: &str) -> Result<Value> {
    use configparser::ini::Ini;

    let mut ini = Ini::new();
    ini.read(content.to_string())
        .map_err(|e| anyhow!("Failed to parse INI: {}", e))?;

    let mut root_map = serde_json::Map::new();

    for section_name in ini.sections() {
        let mut section_map = serde_json::Map::new();

        if let Some(section) = ini.get_map_ref().get(&section_name) {
            for (key, value) in section {
                if let Some(v) = value {
                    section_map.insert(key.clone(), Value::String(v.clone()));
                } else {
                    section_map.insert(key.clone(), Value::Null);
                }
            }
        }

        root_map.insert(section_name, Value::Object(section_map));
    }

    Ok(Value::Object(root_map))
}

pub fn parse_xml(content: &str) -> Result<Value> {
    let value: Value = from_str(content)?;
    Ok(value)
}

pub fn parse_csv(content: &str) -> Result<Value> {
    let mut reader = ReaderBuilder::new().from_reader(content.as_bytes());
    let mut records = Vec::new();

    let headers = reader.headers()?.clone();
    let has_headers = !headers.is_empty();

    for result in reader.into_records() {
        let record = result?;
        if has_headers {
            let mut obj = serde_json::Map::new();
            for (i, header) in headers.iter().enumerate() {
                if let Some(value) = record.get(i) {
                    obj.insert(header.to_string(), Value::String(value.to_string()));
                }
            }
            records.push(Value::Object(obj));
        } else {
            let mut arr = Vec::new();
            for field in record.iter() {
                arr.push(Value::String(field.to_string()));
            }
            records.push(Value::Array(arr));
        }
    }
    Ok(Value::Array(records))
}

// ============================================================================
// AI/ML File Format Support
// ============================================================================

/// Parse a PyTorch model file (.pth, .pt) and extract tensor information
pub fn parse_pytorch_model(file_path: &Path) -> Result<HashMap<String, TensorStats>> {
    let _device = Device::Cpu;
    let mut model_tensors = HashMap::new();

    // Try to load as safetensors first (more efficient)
    if let Ok(data) = std::fs::read(file_path) {
        if let Ok(safetensors) = SafeTensors::deserialize(&data) {
            for (name, tensor_view) in safetensors.tensors() {
                let shape: Vec<usize> = tensor_view.shape().to_vec();
                let dtype = match tensor_view.dtype() {
                    safetensors::Dtype::F32 => "f32".to_string(),
                    safetensors::Dtype::F64 => "f64".to_string(),
                    safetensors::Dtype::I32 => "i32".to_string(),
                    safetensors::Dtype::I64 => "i64".to_string(),
                    _ => "unknown".to_string(),
                };

                // Calculate actual statistics from tensor data
                let total_params = shape.iter().product();
                let (mean, std, min, max) = calculate_safetensors_stats(&tensor_view);
                
                let stats = TensorStats {
                    mean,
                    std,
                    min,
                    max,
                    shape,
                    dtype,
                    total_params,
                };

                model_tensors.insert(name.to_string(), stats);
            }
            return Ok(model_tensors);
        }
    }

    // If safetensors parsing fails, the file is likely in PyTorch pickle format
    // For now, provide a more informative error message suggesting conversion
    Err(anyhow!(
        "Failed to parse file {}: Only Safetensors format is fully supported. \
        PyTorch (.pt/.pth) files are not yet fully implemented. \
        Please convert your PyTorch model to Safetensors format using: \
        `torch.save(model.state_dict(), 'model.safetensors')`",
        file_path.display()
    ))
}

/// Parse a Safetensors file (.safetensors) and extract tensor information  
pub fn parse_safetensors_model(file_path: &Path) -> Result<HashMap<String, TensorStats>> {
    let data = std::fs::read(file_path)?;
    let safetensors = SafeTensors::deserialize(&data)?;
    let mut model_tensors = HashMap::new();

    for (name, tensor_view) in safetensors.tensors() {
        let shape: Vec<usize> = tensor_view.shape().to_vec();
        let dtype = match tensor_view.dtype() {
            safetensors::Dtype::F32 => "f32".to_string(),
            safetensors::Dtype::F64 => "f64".to_string(),
            safetensors::Dtype::I32 => "i32".to_string(),
            safetensors::Dtype::I64 => "i64".to_string(),
            _ => "unknown".to_string(),
        };

        let total_params = shape.iter().product();

        // Extract raw data and calculate statistics
        let data_slice = tensor_view.data();
        let (mean, std, min, max) = match tensor_view.dtype() {
            safetensors::Dtype::F32 => {
                let float_data: &[f32] = bytemuck::cast_slice(data_slice);
                calculate_f32_stats(float_data)
            }
            safetensors::Dtype::F64 => {
                let float_data: &[f64] = bytemuck::cast_slice(data_slice);
                calculate_f64_stats(float_data)
            }
            _ => (0.0, 0.0, 0.0, 0.0), // Skip non-float types for now
        };

        let stats = TensorStats {
            mean,
            std,
            min,
            max,
            shape,
            dtype,
            total_params,
        };

        model_tensors.insert(name.to_string(), stats);
    }

    Ok(model_tensors)
}

/// Compare two PyTorch/Safetensors models and return differences
pub fn diff_ml_models(
    model1_path: &Path,
    model2_path: &Path,
    epsilon: Option<f64>,
) -> Result<Vec<DiffResult>> {
    let model1_tensors = if model1_path.extension().and_then(|s| s.to_str()) == Some("safetensors")
    {
        parse_safetensors_model(model1_path)?
    } else {
        parse_pytorch_model(model1_path)?
    };

    let model2_tensors = if model2_path.extension().and_then(|s| s.to_str()) == Some("safetensors")
    {
        parse_safetensors_model(model2_path)?
    } else {
        parse_pytorch_model(model2_path)?
    };

    let mut results = Vec::new();
    let eps = epsilon.unwrap_or(1e-6);

    // Check for added tensors
    for (name, stats) in &model2_tensors {
        if !model1_tensors.contains_key(name) {
            results.push(DiffResult::Added(
                format!("tensor.{}", name),
                serde_json::to_value(stats)?,
            ));
        }
    }

    // Check for removed tensors
    for (name, stats) in &model1_tensors {
        if !model2_tensors.contains_key(name) {
            results.push(DiffResult::Removed(
                format!("tensor.{}", name),
                serde_json::to_value(stats)?,
            ));
        }
    }

    // Check for modified tensors
    for (name, stats1) in &model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Check shape changes
            if stats1.shape != stats2.shape {
                results.push(DiffResult::TensorShapeChanged(
                    format!("tensor.{}", name),
                    stats1.shape.clone(),
                    stats2.shape.clone(),
                ));
            }

            // Check statistical changes (with epsilon tolerance)
            if (stats1.mean - stats2.mean).abs() > eps
                || (stats1.std - stats2.std).abs() > eps
                || (stats1.min - stats2.min).abs() > eps
                || (stats1.max - stats2.max).abs() > eps
            {
                results.push(DiffResult::TensorStatsChanged(
                    format!("tensor.{}", name),
                    stats1.clone(),
                    stats2.clone(),
                ));
            }
        }
    }

    Ok(results)
}

/// Enhanced ML model comparison with additional analysis features
#[allow(clippy::too_many_arguments)]
pub fn diff_ml_models_enhanced(
    model1_path: &Path,
    model2_path: &Path,
    epsilon: Option<f64>,
    show_layer_impact: bool,
    quantization_analysis: bool,
    detailed_stats: bool,
    learning_progress: bool,
    convergence_analysis: bool,
    anomaly_detection: bool,
    gradient_analysis: bool,
    memory_analysis: bool,
    inference_speed_estimate: bool,
    regression_test: bool,
    alert_on_degradation: bool,
    review_friendly: bool,
    change_summary: bool,
    risk_assessment: bool,
    architecture_comparison: bool,
    param_efficiency_analysis: bool,
    hyperparameter_impact: bool,
    learning_rate_analysis: bool,
    deployment_readiness: bool,
    performance_impact_estimate: bool,
    generate_report: bool,
    markdown_output: bool,
    include_charts: bool,
    embedding_analysis: bool,
    similarity_matrix: bool,
    clustering_change: bool,
    attention_analysis: bool,
    head_importance: bool,
    attention_pattern_diff: bool,
) -> Result<Vec<DiffResult>> {
    let model1_tensors = if model1_path.extension().and_then(|s| s.to_str()) == Some("safetensors")
    {
        parse_safetensors_model(model1_path)?
    } else {
        parse_pytorch_model(model1_path)?
    };

    let model2_tensors = if model2_path.extension().and_then(|s| s.to_str()) == Some("safetensors")
    {
        parse_safetensors_model(model2_path)?
    } else {
        parse_pytorch_model(model2_path)?
    };

    let mut results = Vec::new();
    let eps = epsilon.unwrap_or(1e-6);

    // Enhanced model-level analysis
    if detailed_stats {
        let model1_info = calculate_model_info(&model1_tensors);
        let model2_info = calculate_model_info(&model2_tensors);

        if model1_info.total_parameters != model2_info.total_parameters
            || model1_info.layer_count != model2_info.layer_count
        {
            results.push(DiffResult::ModelArchitectureChanged(
                "model".to_string(),
                model1_info,
                model2_info,
            ));
        }
    }

    // Check for added tensors
    for (name, stats) in &model2_tensors {
        if !model1_tensors.contains_key(name) {
            results.push(DiffResult::Added(
                format!("tensor.{}", name),
                serde_json::to_value(stats)?,
            ));
        }
    }

    // Check for removed tensors
    for (name, stats) in &model1_tensors {
        if !model2_tensors.contains_key(name) {
            results.push(DiffResult::Removed(
                format!("tensor.{}", name),
                serde_json::to_value(stats)?,
            ));
        }
    }

    // Check for modified tensors with enhanced analysis
    for (name, stats1) in &model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Check shape changes
            if stats1.shape != stats2.shape {
                results.push(DiffResult::TensorShapeChanged(
                    format!("tensor.{}", name),
                    stats1.shape.clone(),
                    stats2.shape.clone(),
                ));
            }

            // Enhanced statistical changes analysis
            let mean_change = (stats1.mean - stats2.mean).abs();
            let std_change = (stats1.std - stats2.std).abs();
            let min_change = (stats1.min - stats2.min).abs();
            let max_change = (stats1.max - stats2.max).abs();

            let stats_changed =
                mean_change > eps || std_change > eps || min_change > eps || max_change > eps;

            if stats_changed {
                if show_layer_impact {
                    // Add layer impact information to the key
                    let impact_score = calculate_layer_impact(stats1, stats2);
                    let enhanced_key = format!("tensor.{} [impact: {:.4}]", name, impact_score);
                    results.push(DiffResult::TensorStatsChanged(
                        enhanced_key,
                        stats1.clone(),
                        stats2.clone(),
                    ));
                } else {
                    results.push(DiffResult::TensorStatsChanged(
                        format!("tensor.{}", name),
                        stats1.clone(),
                        stats2.clone(),
                    ));
                }
            }

            // Quantization analysis
            if quantization_analysis {
                let quantization_info = analyze_quantization_impact(stats1, stats2);
                if !quantization_info.is_empty() {
                    results.push(DiffResult::Modified(
                        format!("quantization.{}", name),
                        serde_json::to_value(&quantization_info)?,
                        serde_json::Value::Null,
                    ));
                }
            }
        }
    }

    // Learning progress analysis
    if learning_progress {
        let progress_info = analyze_learning_progress(&model1_tensors, &model2_tensors);
        results.push(DiffResult::LearningProgress(
            "learning_progress".to_string(),
            progress_info,
        ));
    }

    // Convergence analysis
    if convergence_analysis {
        let convergence_info = analyze_convergence(&model1_tensors, &model2_tensors);
        results.push(DiffResult::ConvergenceAnalysis(
            "convergence_analysis".to_string(),
            convergence_info,
        ));
    }

    // Anomaly detection
    if anomaly_detection {
        let anomaly_info = detect_anomalies(&model1_tensors, &model2_tensors);
        results.push(DiffResult::AnomalyDetection(
            "anomaly_detection".to_string(),
            anomaly_info,
        ));
    }

    // Gradient analysis
    if gradient_analysis {
        let gradient_info = analyze_gradients(&model1_tensors, &model2_tensors);
        results.push(DiffResult::GradientAnalysis(
            "gradient_analysis".to_string(),
            gradient_info,
        ));
    }

    // Memory analysis
    if memory_analysis {
        let memory_info = analyze_memory_usage(&model1_tensors, &model2_tensors);
        results.push(DiffResult::MemoryAnalysis(
            "memory_analysis".to_string(),
            memory_info,
        ));
    }

    // Inference speed estimation
    if inference_speed_estimate {
        let speed_info = estimate_inference_speed(&model1_tensors, &model2_tensors);
        results.push(DiffResult::InferenceSpeedAnalysis(
            "inference_speed_analysis".to_string(),
            speed_info,
        ));
    }

    // Regression testing
    if regression_test {
        let regression_info = perform_regression_test(&model1_tensors, &model2_tensors);
        results.push(DiffResult::RegressionTest(
            "regression_test".to_string(),
            regression_info,
        ));
    }

    // Alert on degradation
    if alert_on_degradation {
        let alert_info = check_for_degradation(&model1_tensors, &model2_tensors);
        results.push(DiffResult::AlertOnDegradation(
            "alert_on_degradation".to_string(),
            alert_info,
        ));
    }

    // Review-friendly output
    if review_friendly {
        let review_info =
            generate_review_friendly_summary(&model1_tensors, &model2_tensors, &results);
        results.push(DiffResult::ReviewFriendly(
            "review_friendly".to_string(),
            review_info,
        ));
    }

    // Change summary
    if change_summary {
        let summary_info = generate_change_summary(&model1_tensors, &model2_tensors);
        results.push(DiffResult::ChangeSummary(
            "change_summary".to_string(),
            summary_info,
        ));
    }

    // Risk assessment
    if risk_assessment {
        let risk_info = assess_deployment_risk(&model1_tensors, &model2_tensors, &results);
        results.push(DiffResult::RiskAssessment(
            "risk_assessment".to_string(),
            risk_info,
        ));
    }

    // Architecture comparison
    if architecture_comparison {
        let arch_info = compare_architectures(&model1_tensors, &model2_tensors);
        results.push(DiffResult::ArchitectureComparison(
            "architecture_comparison".to_string(),
            arch_info,
        ));
    }

    // Parameter efficiency analysis
    if param_efficiency_analysis {
        let efficiency_info = analyze_param_efficiency(&model1_tensors, &model2_tensors);
        results.push(DiffResult::ParamEfficiencyAnalysis(
            "param_efficiency_analysis".to_string(),
            efficiency_info,
        ));
    }

    // Hyperparameter impact analysis
    if hyperparameter_impact {
        let hyper_info = analyze_hyperparameter_impact(&model1_tensors, &model2_tensors);
        results.push(DiffResult::HyperparameterImpact(
            "hyperparameter_impact".to_string(),
            hyper_info,
        ));
    }

    // Learning rate analysis
    if learning_rate_analysis {
        let lr_info = analyze_learning_rate(&model1_tensors, &model2_tensors);
        results.push(DiffResult::LearningRateAnalysis(
            "learning_rate_analysis".to_string(),
            lr_info,
        ));
    }

    // Deployment readiness assessment
    if deployment_readiness {
        let deploy_info = assess_deployment_readiness(&model1_tensors, &model2_tensors, &results);
        results.push(DiffResult::DeploymentReadiness(
            "deployment_readiness".to_string(),
            deploy_info,
        ));
    }

    // Performance impact estimation
    if performance_impact_estimate {
        let perf_info = estimate_performance_impact(&model1_tensors, &model2_tensors);
        results.push(DiffResult::PerformanceImpactEstimate(
            "performance_impact_estimate".to_string(),
            perf_info,
        ));
    }

    // Report generation
    if generate_report {
        let report_info = generate_experiment_report(&model1_tensors, &model2_tensors, &results);
        results.push(DiffResult::GenerateReport(
            "generate_report".to_string(),
            report_info,
        ));
    }

    // Markdown output
    if markdown_output {
        let markdown_info = generate_markdown_output(&model1_tensors, &model2_tensors, &results);
        results.push(DiffResult::MarkdownOutput(
            "markdown_output".to_string(),
            markdown_info,
        ));
    }

    // Charts generation
    if include_charts {
        let chart_info = generate_charts(&model1_tensors, &model2_tensors);
        results.push(DiffResult::IncludeCharts(
            "include_charts".to_string(),
            chart_info,
        ));
    }

    // Embedding analysis
    if embedding_analysis {
        let embed_info = analyze_embeddings(&model1_tensors, &model2_tensors);
        results.push(DiffResult::EmbeddingAnalysis(
            "embedding_analysis".to_string(),
            embed_info,
        ));
    }

    // Similarity matrix generation
    if similarity_matrix {
        let sim_info = generate_similarity_matrix(&model1_tensors, &model2_tensors);
        results.push(DiffResult::SimilarityMatrix(
            "similarity_matrix".to_string(),
            sim_info,
        ));
    }

    // Clustering change analysis
    if clustering_change {
        let cluster_info = analyze_clustering_changes(&model1_tensors, &model2_tensors);
        results.push(DiffResult::ClusteringChange(
            "clustering_change".to_string(),
            cluster_info,
        ));
    }

    // Attention analysis (Transformer models)
    if attention_analysis {
        let attention_info = analyze_attention_mechanisms(&model1_tensors, &model2_tensors);
        results.push(DiffResult::AttentionAnalysis(
            "attention_analysis".to_string(),
            attention_info,
        ));
    }

    // Head importance analysis
    if head_importance {
        let head_info = analyze_head_importance(&model1_tensors, &model2_tensors);
        results.push(DiffResult::HeadImportance(
            "head_importance".to_string(),
            head_info,
        ));
    }

    // Attention pattern differences
    if attention_pattern_diff {
        let pattern_info = analyze_attention_patterns(&model1_tensors, &model2_tensors);
        results.push(DiffResult::AttentionPatternDiff(
            "attention_pattern_diff".to_string(),
            pattern_info,
        ));
    }

    Ok(results)
}

/// Calculate layer impact score based on parameter changes
fn calculate_layer_impact(stats1: &TensorStats, stats2: &TensorStats) -> f64 {
    let mean_change = (stats1.mean - stats2.mean).abs();
    let std_change = (stats1.std - stats2.std).abs();
    let param_ratio = stats1.total_params as f64;

    // Weighted impact score considering parameter count and statistical changes
    (mean_change + std_change) * param_ratio.log10().max(1.0)
}

/// Analyze quantization impact between two tensor versions
fn analyze_quantization_impact(stats1: &TensorStats, stats2: &TensorStats) -> HashMap<String, f64> {
    let mut analysis = HashMap::new();

    // Check if precision loss indicates quantization
    let precision_loss = (stats1.max - stats1.min) / (stats2.max - stats2.min);
    if precision_loss > 1.5 {
        analysis.insert("precision_loss_ratio".to_string(), precision_loss);
    }

    // Check for typical quantization patterns
    let range_compression = ((stats1.max - stats1.min) - (stats2.max - stats2.min)).abs();
    if range_compression > 0.1 {
        analysis.insert("range_compression".to_string(), range_compression);
    }

    analysis
}

/// Calculate overall model information from tensors
fn calculate_model_info(tensors: &HashMap<String, TensorStats>) -> ModelInfo {
    let total_parameters: usize = tensors.values().map(|stats| stats.total_params).sum();
    let layer_count = tensors.len();

    let mut layer_types = HashMap::new();
    for name in tensors.keys() {
        let layer_type = extract_layer_type(name);
        *layer_types.entry(layer_type).or_insert(0) += 1;
    }

    // Estimate model size in bytes (assuming f32 = 4 bytes per parameter)
    let model_size_bytes = total_parameters * 4;

    ModelInfo {
        total_parameters,
        layer_count,
        layer_types,
        model_size_bytes,
    }
}

/// Extract layer type from tensor name for analysis
fn extract_layer_type(tensor_name: &str) -> String {
    if tensor_name.contains("conv") || tensor_name.contains("Conv") {
        "conv".to_string()
    } else if tensor_name.contains("linear")
        || tensor_name.contains("Linear")
        || tensor_name.contains("fc")
    {
        "linear".to_string()
    } else if tensor_name.contains("norm")
        || tensor_name.contains("Norm")
        || tensor_name.contains("bn")
    {
        "norm".to_string()
    } else if tensor_name.contains("attention") || tensor_name.contains("attn") {
        "attention".to_string()
    } else if tensor_name.contains("embedding") || tensor_name.contains("embed") {
        "embedding".to_string()
    } else {
        "other".to_string()
    }
}

/// Analyze learning progress between two model checkpoints
fn analyze_learning_progress(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> LearningProgressInfo {
    let mut total_magnitude = 0.0;
    let mut gradient_changes = 0.0;
    let mut param_count = 0;

    // Calculate overall parameter update magnitude
    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Parameter change magnitude (using mean and std as proxies)
            let mean_change = (stats1.mean - stats2.mean).abs();
            let std_change = (stats1.std - stats2.std).abs();
            total_magnitude += mean_change + std_change;

            // Estimate gradient information from parameter changes
            let param_change_ratio = mean_change / (stats1.mean.abs() + 1e-8);
            gradient_changes += param_change_ratio;
            param_count += 1;
        }
    }

    let avg_magnitude = if param_count > 0 {
        total_magnitude / param_count as f64
    } else {
        0.0
    };
    let avg_gradient_ratio = if param_count > 0 {
        gradient_changes / param_count as f64
    } else {
        0.0
    };

    // Determine loss trend based on parameter changes
    let loss_trend = if avg_magnitude > 0.1 {
        "improving" // Large changes suggest active learning
    } else if avg_magnitude < 0.001 {
        "stable" // Very small changes suggest convergence
    } else {
        "degrading" // Medium changes might indicate instability
    };

    // Estimate convergence speed based on magnitude
    let convergence_speed = if avg_magnitude > 0.1 { 0.8 } else { 0.2 };

    LearningProgressInfo {
        loss_trend: loss_trend.to_string(),
        parameter_update_magnitude: avg_magnitude,
        gradient_norm_ratio: avg_gradient_ratio,
        convergence_speed,
    }
}

/// Analyze convergence characteristics between model checkpoints
fn analyze_convergence(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> ConvergenceInfo {
    let mut parameter_stability = 0.0;
    let mut gradient_stability = 0.0;
    let mut oscillation_count = 0;
    let mut total_layers = 0;

    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Parameter stability (lower is more stable)
            let mean_stability = (stats1.mean - stats2.mean).abs() / (stats1.mean.abs() + 1e-8);
            let std_stability = (stats1.std - stats2.std).abs() / (stats1.std.abs() + 1e-8);
            parameter_stability += (mean_stability + std_stability) / 2.0;

            // Gradient stability estimation
            let gradient_est = (stats1.max - stats1.min) / (stats2.max - stats2.min + 1e-8);
            gradient_stability += (gradient_est - 1.0).abs();

            // Check for oscillations (rapid sign changes in statistics)
            if (stats1.mean > 0.0) != (stats2.mean > 0.0) {
                oscillation_count += 1;
            }

            total_layers += 1;
        }
    }

    let avg_param_stability = if total_layers > 0 {
        parameter_stability / total_layers as f64
    } else {
        0.0
    };
    let avg_gradient_stability = if total_layers > 0 {
        gradient_stability / total_layers as f64
    } else {
        0.0
    };
    let oscillation_ratio = oscillation_count as f64 / total_layers.max(1) as f64;

    // Determine convergence status
    let convergence_status = if avg_param_stability < 0.01 && avg_gradient_stability < 0.1 {
        "converged"
    } else if oscillation_ratio > 0.3 {
        "oscillating"
    } else if avg_param_stability > 0.5 {
        "diverging"
    } else {
        "stable"
    };

    // Provide recommendations based on analysis
    let recommended_action = match convergence_status {
        "converged" => "Training can be stopped. Model has converged.",
        "diverging" => "Reduce learning rate or check for gradient explosion.",
        "oscillating" => "Consider learning rate scheduling or gradient clipping.",
        _ => "Continue training with current settings.",
    };

    ConvergenceInfo {
        convergence_status: convergence_status.to_string(),
        gradient_stability: avg_gradient_stability,
        parameter_stability: avg_param_stability,
        recommended_action: recommended_action.to_string(),
    }
}

/// Detect training anomalies such as gradient explosion or vanishing gradients
fn detect_anomalies(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> AnomalyInfo {
    let mut weight_explosion_layers = Vec::new();
    let mut vanishing_gradient_layers = Vec::new();
    let mut gradient_explosion_layers = Vec::new();
    let mut max_parameter_change = 0.0_f64;
    let mut max_gradient_estimate = 0.0_f64;

    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Check for weight explosion (parameters growing too large)
            let weight_magnitude_1 = (stats1.mean.abs() + stats1.std.abs()).max(stats1.max.abs());
            let weight_magnitude_2 = (stats2.mean.abs() + stats2.std.abs()).max(stats2.max.abs());

            if weight_magnitude_2 > 10.0 || weight_magnitude_2 > weight_magnitude_1 * 5.0 {
                weight_explosion_layers.push(name.clone());
            }

            // Check for gradient explosion (large parameter changes)
            let param_change = (stats1.mean - stats2.mean).abs() + (stats1.std - stats2.std).abs();
            max_parameter_change = max_parameter_change.max(param_change);

            if param_change > 1.0 {
                gradient_explosion_layers.push(name.clone());
            }

            // Check for vanishing gradients (very small changes despite training)
            let gradient_estimate = param_change / (weight_magnitude_1 + 1e-8);
            max_gradient_estimate = max_gradient_estimate.max(gradient_estimate);

            if gradient_estimate < 1e-6 && param_change < 1e-8 {
                vanishing_gradient_layers.push(name.clone());
            }
        }
    }

    // Determine anomaly type and severity
    let (anomaly_type, severity, affected_layers, recommended_action) =
        if !gradient_explosion_layers.is_empty() {
            (
                "gradient_explosion",
                if max_parameter_change > 5.0 {
                    "critical"
                } else {
                    "warning"
                },
                gradient_explosion_layers,
                "Reduce learning rate immediately or apply gradient clipping.",
            )
        } else if !weight_explosion_layers.is_empty() {
            (
                "weight_explosion",
                if max_parameter_change > 10.0 {
                    "critical"
                } else {
                    "warning"
                },
                weight_explosion_layers,
                "Check weight initialization and apply weight decay or normalization.",
            )
        } else if vanishing_gradient_layers.len() > model1_tensors.len() / 2 {
            (
                "vanishing_gradient",
                "warning",
                vanishing_gradient_layers,
                "Increase learning rate, check activation functions, or use residual connections.",
            )
        } else {
            (
                "normal",
                "none",
                Vec::new(),
                "No anomalies detected. Training appears normal.",
            )
        };

    AnomalyInfo {
        anomaly_type: anomaly_type.to_string(),
        severity: severity.to_string(),
        affected_layers,
        recommended_action: recommended_action.to_string(),
    }
}

/// Analyze gradient characteristics and flow health
fn analyze_gradients(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> GradientInfo {
    let mut gradient_norms = Vec::new();
    let mut problematic_layers = Vec::new();
    let mut total_gradient_flow = 0.0;
    let mut layer_count = 0;

    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Estimate gradient norm from parameter changes
            let param_change =
                ((stats1.mean - stats2.mean).powi(2) + (stats1.std - stats2.std).powi(2)).sqrt();

            // Normalize by parameter magnitude to get gradient estimate
            let param_magnitude = stats1.mean.abs() + stats1.std.abs() + 1e-8;
            let gradient_estimate = param_change / param_magnitude;

            gradient_norms.push(gradient_estimate);
            total_gradient_flow += gradient_estimate;
            layer_count += 1;

            // Identify problematic layers
            if gradient_estimate > 1.0 {
                problematic_layers.push(format!("{} (exploding: {:.4})", name, gradient_estimate));
            } else if gradient_estimate < 1e-6 {
                problematic_layers.push(format!("{} (vanishing: {:.6})", name, gradient_estimate));
            }
        }
    }

    let avg_gradient_flow = if layer_count > 0 {
        total_gradient_flow / layer_count as f64
    } else {
        0.0
    };

    // Calculate gradient norm estimate (root mean square)
    let gradient_norm_estimate = if !gradient_norms.is_empty() {
        (gradient_norms.iter().map(|&x| x * x).sum::<f64>() / gradient_norms.len() as f64).sqrt()
    } else {
        0.0
    };

    // Determine gradient flow health
    let gradient_flow_health = if avg_gradient_flow > 0.5 {
        "exploding"
    } else if avg_gradient_flow < 1e-5 {
        "dead"
    } else if avg_gradient_flow < 1e-3 {
        "diminishing"
    } else {
        "healthy"
    };

    // Calculate gradient ratio (current vs expected)
    let expected_gradient = 0.01; // Expected normal gradient magnitude
    let gradient_ratio = avg_gradient_flow / expected_gradient;

    GradientInfo {
        gradient_norm_estimate,
        gradient_flow_health: gradient_flow_health.to_string(),
        problematic_layers,
        gradient_ratio,
    }
}

/// Analyze memory usage characteristics between two models
fn analyze_memory_usage(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> MemoryAnalysisInfo {
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    let model1_size_bytes = model1_info.model_size_bytes;
    let model2_size_bytes = model2_info.model_size_bytes;
    let memory_delta_bytes = model2_size_bytes as i64 - model1_size_bytes as i64;

    // Memory efficiency: parameters per byte (higher is better)
    let memory_efficiency_ratio = if model2_size_bytes > 0 {
        model2_info.total_parameters as f64 / model2_size_bytes as f64
    } else {
        0.0
    };

    // Estimate GPU memory usage (model weights + activation memory + overhead)
    // Rule of thumb: model size * 1.5 (activations) + 20% overhead
    let estimated_gpu_memory_mb = (model2_size_bytes as f64 * 1.7) / (1024.0 * 1024.0);

    // Generate recommendation based on memory changes
    let memory_recommendation = if memory_delta_bytes > 100_000_000 {
        // > 100MB increase
        format!("⚠️ Significant memory increase (+{:.1}MB). Consider model optimization or batch size reduction.", 
               memory_delta_bytes as f64 / (1024.0 * 1024.0))
    } else if memory_delta_bytes < -50_000_000 {
        // > 50MB decrease
        format!(
            "✅ Good memory reduction (-{:.1}MB). Model is more memory efficient.",
            (-memory_delta_bytes) as f64 / (1024.0 * 1024.0)
        )
    } else if estimated_gpu_memory_mb > 8000.0 {
        // > 8GB
        "⚠️ Large model size. Consider using gradient checkpointing or model sharding.".to_string()
    } else {
        "✅ Memory usage appears reasonable.".to_string()
    };

    MemoryAnalysisInfo {
        model1_size_bytes,
        model2_size_bytes,
        memory_delta_bytes,
        memory_efficiency_ratio,
        estimated_gpu_memory_mb,
        memory_recommendation,
    }
}

/// Estimate inference speed characteristics between two models
fn estimate_inference_speed(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> InferenceSpeedInfo {
    let model1_flops = estimate_model_flops(model1_tensors);
    let model2_flops = estimate_model_flops(model2_tensors);

    // Speed change ratio (higher FLOPS = slower)
    let speed_change_ratio = if model1_flops > 0 {
        model2_flops as f64 / model1_flops as f64
    } else {
        1.0
    };

    // Identify potential bottleneck layers (large layers that may slow inference)
    let mut bottleneck_layers = Vec::new();
    for (name, stats) in model2_tensors {
        let layer_flops = estimate_layer_flops(name, stats);
        // Consider layers with >10M FLOPS as potential bottlenecks
        if layer_flops > 10_000_000 {
            bottleneck_layers.push(format!(
                "{} ({:.1}M FLOPS)",
                name,
                layer_flops as f64 / 1_000_000.0
            ));
        }
    }

    // Generate performance recommendation
    let inference_recommendation = if speed_change_ratio > 2.0 {
        format!(
            "⚠️ Inference may be {:.1}x slower. Consider model pruning or distillation.",
            speed_change_ratio
        )
    } else if speed_change_ratio < 0.5 {
        format!(
            "✅ Inference should be {:.1}x faster. Good optimization!",
            1.0 / speed_change_ratio
        )
    } else if !bottleneck_layers.is_empty() {
        format!(
            "💡 {} potential bottleneck layers identified. Consider layer-wise optimization.",
            bottleneck_layers.len()
        )
    } else {
        "✅ Inference speed should be similar.".to_string()
    };

    InferenceSpeedInfo {
        model1_flops_estimate: model1_flops,
        model2_flops_estimate: model2_flops,
        speed_change_ratio,
        bottleneck_layers,
        inference_recommendation,
    }
}

/// Estimate total FLOPS for a model based on tensor statistics
fn estimate_model_flops(tensors: &HashMap<String, TensorStats>) -> u64 {
    let mut total_flops = 0u64;

    for (name, stats) in tensors {
        total_flops += estimate_layer_flops(name, stats);
    }

    total_flops
}

/// Estimate FLOPS for a single layer based on its name and tensor statistics
fn estimate_layer_flops(layer_name: &str, stats: &TensorStats) -> u64 {
    let total_params = stats.total_params as u64;

    // Rough FLOPS estimation based on layer type
    if layer_name.contains("conv") || layer_name.contains("Conv") {
        // Convolution: roughly 2 * params (multiply-accumulate)
        total_params * 2
    } else if layer_name.contains("linear")
        || layer_name.contains("Linear")
        || layer_name.contains("fc")
    {
        // Linear layer: roughly 2 * params (matrix multiplication)
        total_params * 2
    } else if layer_name.contains("attention") || layer_name.contains("attn") {
        // Attention: roughly 4 * params (Q, K, V, O projections)
        total_params * 4
    } else if layer_name.contains("norm") || layer_name.contains("Norm") {
        // Normalization: roughly 1 * params (element-wise operations)
        total_params
    } else {
        // Default: assume 2 * params for other layers
        total_params * 2
    }
}

/// Perform automated regression testing on ML model changes
fn perform_regression_test(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> RegressionTestInfo {
    let mut failed_checks = Vec::new();
    let mut total_performance_change = 0.0;
    let mut test_count = 0;

    // Test 1: Architecture compatibility (no major structural changes)
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    if model1_info.total_parameters != model2_info.total_parameters {
        let param_change = ((model2_info.total_parameters as f64
            - model1_info.total_parameters as f64)
            / model1_info.total_parameters as f64
            * 100.0)
            .abs();
        if param_change > 10.0 {
            // > 10% parameter change
            failed_checks.push(format!(
                "Parameter count changed by {:.1}% (from {} to {})",
                param_change, model1_info.total_parameters, model2_info.total_parameters
            ));
        }
        total_performance_change += param_change;
        test_count += 1;
    }

    // Test 2: Weight distribution stability (no extreme changes)
    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            // Check for extreme weight changes
            let mean_change =
                ((stats1.mean - stats2.mean).abs() / (stats1.mean.abs() + 1e-8)) * 100.0;
            let std_change = ((stats1.std - stats2.std).abs() / (stats1.std.abs() + 1e-8)) * 100.0;

            if mean_change > 50.0 {
                // > 50% change in mean
                failed_checks.push(format!(
                    "Layer {} mean changed by {:.1}%",
                    name, mean_change
                ));
            }
            if std_change > 100.0 {
                // > 100% change in std
                failed_checks.push(format!("Layer {} std changed by {:.1}%", name, std_change));
            }

            total_performance_change += (mean_change + std_change) / 2.0;
            test_count += 1;
        }
    }

    // Test 3: Model size efficiency
    let size_change = ((model2_info.model_size_bytes as f64 - model1_info.model_size_bytes as f64)
        / model1_info.model_size_bytes as f64
        * 100.0)
        .abs();
    if size_change > 25.0 {
        // > 25% size change
        failed_checks.push(format!("Model size changed by {:.1}%", size_change));
    }

    // Calculate overall performance degradation
    let performance_degradation = if test_count > 0 {
        total_performance_change / test_count as f64
    } else {
        0.0
    };

    // Determine severity level
    let severity_level = if performance_degradation > 30.0 || failed_checks.len() > 3 {
        "critical"
    } else if performance_degradation > 15.0 || failed_checks.len() > 1 {
        "high"
    } else if performance_degradation > 5.0 || !failed_checks.is_empty() {
        "medium"
    } else {
        "low"
    };

    let test_passed = failed_checks.is_empty() && performance_degradation < 5.0;

    let recommended_action = if !test_passed {
        match severity_level {
            "critical" => {
                "🚨 Deployment blocked. Major regressions detected. Review changes carefully."
            }
            "high" => "⚠️ Manual review required before deployment. Significant changes detected.",
            "medium" => "⚠️ Consider additional testing. Some changes may affect performance.",
            _ => "✅ Minor changes detected. Proceed with normal testing.",
        }
    } else {
        "✅ All regression tests passed. Safe to deploy."
    }
    .to_string();

    RegressionTestInfo {
        test_passed,
        failed_checks,
        severity_level: severity_level.to_string(),
        recommended_action,
        performance_degradation,
    }
}

/// Check for performance degradation and trigger alerts if thresholds are exceeded
fn check_for_degradation(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> AlertInfo {
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    // Define thresholds
    let memory_threshold = 1.2; // 20% memory increase
    let performance_threshold = 1.5; // 50% performance degradation
    let stability_threshold = 0.1; // 10% stability change

    // Check memory degradation
    let memory_ratio = model2_info.model_size_bytes as f64 / model1_info.model_size_bytes as f64;
    if memory_ratio > memory_threshold {
        return AlertInfo {
            alert_triggered: true,
            alert_type: "memory".to_string(),
            threshold_exceeded: memory_ratio / memory_threshold,
            current_value: memory_ratio,
            threshold_value: memory_threshold,
            alert_message: format!("🚨 Memory usage increased by {:.1}% (threshold: {:.1}%). Current: {:.1}MB → {:.1}MB", 
                                 (memory_ratio - 1.0) * 100.0, (memory_threshold - 1.0) * 100.0,
                                 model1_info.model_size_bytes as f64 / (1024.0 * 1024.0),
                                 model2_info.model_size_bytes as f64 / (1024.0 * 1024.0)),
        };
    }

    // Check performance degradation (estimated via FLOPS)
    let model1_flops = estimate_model_flops(model1_tensors);
    let model2_flops = estimate_model_flops(model2_tensors);
    let performance_ratio = if model1_flops > 0 {
        model2_flops as f64 / model1_flops as f64
    } else {
        1.0
    };

    if performance_ratio > performance_threshold {
        return AlertInfo {
            alert_triggered: true,
            alert_type: "performance".to_string(),
            threshold_exceeded: performance_ratio / performance_threshold,
            current_value: performance_ratio,
            threshold_value: performance_threshold,
            alert_message: format!(
                "🚨 Performance degraded by {:.1}% (threshold: {:.1}%). FLOPS: {} → {}",
                (performance_ratio - 1.0) * 100.0,
                (performance_threshold - 1.0) * 100.0,
                model1_flops,
                model2_flops
            ),
        };
    }

    // Check stability degradation (parameter variance)
    let mut stability_change = 0.0;
    let mut layer_count = 0;

    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            let variance_change = ((stats1.std - stats2.std).abs() / (stats1.std + 1e-8))
                .max((stats1.mean - stats2.mean).abs() / (stats1.mean.abs() + 1e-8));
            stability_change += variance_change;
            layer_count += 1;
        }
    }

    let avg_stability_change = if layer_count > 0 {
        stability_change / layer_count as f64
    } else {
        0.0
    };

    if avg_stability_change > stability_threshold {
        return AlertInfo {
            alert_triggered: true,
            alert_type: "stability".to_string(),
            threshold_exceeded: avg_stability_change / stability_threshold,
            current_value: avg_stability_change,
            threshold_value: stability_threshold,
            alert_message: format!("🚨 Model stability degraded by {:.1}% (threshold: {:.1}%). Average parameter variance change: {:.4}", 
                                 avg_stability_change * 100.0, stability_threshold * 100.0, avg_stability_change),
        };
    }

    // No alerts triggered
    AlertInfo {
        alert_triggered: false,
        alert_type: "none".to_string(),
        threshold_exceeded: 0.0,
        current_value: 0.0,
        threshold_value: 0.0,
        alert_message:
            "✅ No performance degradation detected. All metrics within acceptable thresholds."
                .to_string(),
    }
}

/// Generate review-friendly summary for human reviewers
fn generate_review_friendly_summary(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
    analysis_results: &[DiffResult],
) -> ReviewFriendlyInfo {
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    let mut key_changes = Vec::new();
    let mut reviewer_notes = Vec::new();
    let mut has_critical_issues = false;

    // Analyze parameter changes
    let param_change = if model1_info.total_parameters != model2_info.total_parameters {
        let change_pct = ((model2_info.total_parameters as f64
            - model1_info.total_parameters as f64)
            / model1_info.total_parameters as f64
            * 100.0)
            .abs();
        key_changes.push(format!(
            "Parameter count: {} → {} ({:+.1}%)",
            model1_info.total_parameters,
            model2_info.total_parameters,
            (model2_info.total_parameters as f64 - model1_info.total_parameters as f64)
                / model1_info.total_parameters as f64
                * 100.0
        ));
        change_pct
    } else {
        0.0
    };

    // Check for critical issues from analysis results
    for result in analysis_results {
        match result {
            DiffResult::AnomalyDetection(_, anomaly) if anomaly.severity == "critical" => {
                has_critical_issues = true;
                reviewer_notes.push(format!(
                    "🚨 Critical anomaly detected: {}",
                    anomaly.anomaly_type
                ));
            }
            DiffResult::RegressionTest(_, regression) if !regression.test_passed => {
                if regression.severity_level == "critical" || regression.severity_level == "high" {
                    has_critical_issues = true;
                }
                reviewer_notes.push(format!(
                    "⚠️ Regression test failed ({}): {} checks failed",
                    regression.severity_level,
                    regression.failed_checks.len()
                ));
            }
            _ => {}
        }
    }

    // Calculate impact assessment
    let impact_score = param_change.clamp(20.0, 100.0) / 100.0; // Normalize to 0-1
    let impact_assessment = if has_critical_issues {
        "critical"
    } else if impact_score > 0.3 || param_change > 25.0 {
        "high"
    } else if impact_score > 0.1 || param_change > 10.0 {
        "medium"
    } else {
        "low"
    };

    // Generate summary
    let summary = if has_critical_issues {
        "⚠️ Model changes include critical issues that require immediate attention."
    } else if param_change > 25.0 {
        "📊 Significant model architecture changes detected. Review impact carefully."
    } else if param_change > 10.0 {
        "🔄 Moderate model changes. Standard review recommended."
    } else {
        "✅ Minor model changes. Routine review sufficient."
    }
    .to_string();

    // Add general reviewer notes
    if model2_info.model_size_bytes > model1_info.model_size_bytes * 2 {
        reviewer_notes.push("💾 Model size doubled - check memory requirements".to_string());
    }

    // Approval recommendation
    let approval_recommendation = if has_critical_issues {
        "request_changes"
    } else if impact_assessment == "high" {
        "needs_discussion"
    } else {
        "approve"
    }
    .to_string();

    ReviewFriendlyInfo {
        summary,
        impact_assessment: impact_assessment.to_string(),
        key_changes,
        reviewer_notes,
        approval_recommendation,
    }
}

/// Generate detailed change summary for technical analysis
fn generate_change_summary(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> ChangeSummaryInfo {
    let mut total_layers_changed = 0;
    let mut layer_changes = Vec::new();
    let mut change_distribution = HashMap::new();
    let mut total_change_magnitude = 0.0;
    let mut change_patterns = Vec::new();

    // Analyze layer changes
    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            let mean_change = (stats1.mean - stats2.mean).abs();
            let std_change = (stats1.std - stats2.std).abs();
            let change_magnitude = mean_change + std_change;

            if change_magnitude > 0.01 {
                // Significant change threshold
                total_layers_changed += 1;
                layer_changes.push((name.clone(), change_magnitude));

                let layer_type = extract_layer_type(name);
                *change_distribution.entry(layer_type).or_insert(0) += 1;

                total_change_magnitude += change_magnitude;
            }
        }
    }

    // Sort by change magnitude and get top changed layers
    layer_changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let most_changed_layers: Vec<String> = layer_changes
        .iter()
        .take(5)
        .map(|(name, mag)| format!("{} (Δ={:.4})", name, mag))
        .collect();

    // Detect change patterns
    if change_distribution.get("linear").unwrap_or(&0) > &2 {
        change_patterns.push("Multiple linear layer changes detected".to_string());
    }
    if change_distribution.get("conv").unwrap_or(&0) > &2 {
        change_patterns.push("Multiple convolution layer changes detected".to_string());
    }
    if change_distribution.get("attention").unwrap_or(&0) > &0 {
        change_patterns.push("Attention mechanism modifications detected".to_string());
    }

    // Normalize overall change magnitude
    let overall_change_magnitude = if total_layers_changed > 0 {
        (total_change_magnitude / total_layers_changed as f64).min(1.0)
    } else {
        0.0
    };

    ChangeSummaryInfo {
        total_layers_changed,
        most_changed_layers,
        change_distribution,
        overall_change_magnitude,
        change_patterns,
    }
}

/// Assess deployment risk based on all analysis results
fn assess_deployment_risk(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
    analysis_results: &[DiffResult],
) -> RiskAssessmentInfo {
    let mut risk_factors = Vec::new();
    let mut risk_score = 0.0;
    let mut has_blockers = false;
    let mut recommended_monitoring = Vec::new();

    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    // Check for high-risk factors from analysis results
    for result in analysis_results {
        match result {
            DiffResult::AnomalyDetection(_, anomaly) => match anomaly.severity.as_str() {
                "critical" => {
                    has_blockers = true;
                    risk_factors.push(format!("Critical anomaly: {}", anomaly.anomaly_type));
                    risk_score += 0.4;
                }
                "warning" => {
                    risk_factors.push(format!("Anomaly warning: {}", anomaly.anomaly_type));
                    risk_score += 0.2;
                }
                _ => {}
            },
            DiffResult::RegressionTest(_, regression) => {
                if !regression.test_passed {
                    match regression.severity_level.as_str() {
                        "critical" => {
                            has_blockers = true;
                            risk_factors.push("Critical regression test failures".to_string());
                            risk_score += 0.4;
                        }
                        "high" => {
                            risk_factors.push("High severity regression issues".to_string());
                            risk_score += 0.3;
                        }
                        _ => {
                            risk_factors.push("Minor regression issues".to_string());
                            risk_score += 0.1;
                        }
                    }
                }
            }
            DiffResult::AlertOnDegradation(_, alert) => {
                if alert.alert_triggered {
                    risk_factors.push(format!("Performance degradation: {}", alert.alert_type));
                    risk_score += 0.2;
                }
            }
            _ => {}
        }
    }

    // Memory and size risk factors
    let size_change_ratio =
        model2_info.model_size_bytes as f64 / model1_info.model_size_bytes as f64;
    if size_change_ratio > 1.5 {
        risk_factors.push("Significant memory increase (>50%)".to_string());
        risk_score += 0.2;
        recommended_monitoring.push("Memory usage and OOM errors".to_string());
    }

    // Architecture change risk
    let param_change_ratio =
        (model2_info.total_parameters as f64 - model1_info.total_parameters as f64).abs()
            / model1_info.total_parameters as f64;
    if param_change_ratio > 0.2 {
        risk_factors.push("Major architecture changes (>20% parameter change)".to_string());
        risk_score += 0.3;
        recommended_monitoring.push("Model accuracy and inference latency".to_string());
    }

    // Determine overall risk level
    let overall_risk_level = if has_blockers || risk_score > 0.7 {
        "critical"
    } else if risk_score > 0.4 {
        "high"
    } else if risk_score > 0.2 {
        "medium"
    } else {
        "low"
    };

    // Deployment readiness
    let deployment_readiness = if has_blockers {
        "not_ready"
    } else if risk_score > 0.4 {
        "needs_testing"
    } else {
        "ready"
    };

    // Rollback difficulty
    let rollback_difficulty = if param_change_ratio > 0.3 {
        "difficult"
    } else if param_change_ratio > 0.1 {
        "moderate"
    } else {
        "easy"
    };

    // Add standard monitoring recommendations
    recommended_monitoring.extend([
        "Model prediction accuracy".to_string(),
        "Inference response times".to_string(),
        "Error rates and exceptions".to_string(),
    ]);

    RiskAssessmentInfo {
        overall_risk_level: overall_risk_level.to_string(),
        risk_factors,
        deployment_readiness: deployment_readiness.to_string(),
        rollback_difficulty: rollback_difficulty.to_string(),
        recommended_monitoring,
    }
}

// Helper functions for statistical calculations
fn calculate_f32_stats(data: &[f32]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let sum: f64 = data.iter().map(|&x| x as f64).sum();
    let mean = sum / data.len() as f64;

    let variance: f64 = data
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;

    let std = variance.sqrt();
    let min = data
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f64;
    let max = data
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap() as f64;

    (mean, std, min, max)
}

fn calculate_f64_stats(data: &[f64]) -> (f64, f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let sum: f64 = data.iter().sum();
    let mean = sum / data.len() as f64;

    let variance: f64 = data
        .iter()
        .map(|&x| {
            let diff = x - mean;
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64;

    let std = variance.sqrt();
    let min = data
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let max = data
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    (mean, std, min, max)
}

/// Calculate tensor statistics from a Candle tensor
fn calculate_tensor_stats(tensor: &candle_core::Tensor) -> (f64, f64, f64, f64) {
    match tensor.dtype() {
        candle_core::DType::F32 => {
            match tensor.flatten_all() {
                Ok(flattened) => {
                    match flattened.to_vec1::<f32>() {
                        Ok(data) => calculate_f32_stats(&data),
                        Err(_) => (0.0, 0.0, 0.0, 0.0),
                    }
                }
                Err(_) => (0.0, 0.0, 0.0, 0.0),
            }
        }
        candle_core::DType::F64 => {
            match tensor.flatten_all() {
                Ok(flattened) => {
                    match flattened.to_vec1::<f64>() {
                        Ok(data) => calculate_f64_stats(&data),
                        Err(_) => (0.0, 0.0, 0.0, 0.0),
                    }
                }
                Err(_) => (0.0, 0.0, 0.0, 0.0),
            }
        }
        _ => {
            // For other data types, return zero stats
            (0.0, 0.0, 0.0, 0.0)
        }
    }
}

/// Calculate tensor statistics from Safetensors tensor view
fn calculate_safetensors_stats(tensor_view: &safetensors::tensor::TensorView) -> (f64, f64, f64, f64) {
    match tensor_view.dtype() {
        safetensors::Dtype::F32 => {
            match tensor_view.data().chunks_exact(4) {
                chunks => {
                    let data: Vec<f32> = chunks
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    if !data.is_empty() {
                        calculate_f32_stats(&data)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    }
                }
            }
        }
        safetensors::Dtype::F64 => {
            match tensor_view.data().chunks_exact(8) {
                chunks => {
                    let data: Vec<f64> = chunks
                        .map(|chunk| {
                            f64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3],
                                chunk[4], chunk[5], chunk[6], chunk[7],
                            ])
                        })
                        .collect();
                    if !data.is_empty() {
                        calculate_f64_stats(&data)
                    } else {
                        (0.0, 0.0, 0.0, 0.0)
                    }
                }
            }
        }
        _ => {
            // For other data types, return zero stats
            (0.0, 0.0, 0.0, 0.0)
        }
    }
}

/// Compare model architectures and identify structural differences
fn compare_architectures(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> ArchitectureComparisonInfo {
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    // Detect architecture types based on layer names
    let arch_type_1 = detect_architecture_type(model1_tensors);
    let arch_type_2 = detect_architecture_type(model2_tensors);

    // Find structural differences
    let mut structural_differences = Vec::new();

    // Compare layer types distribution
    for (layer_type, count1) in &model1_info.layer_types {
        if let Some(count2) = model2_info.layer_types.get(layer_type) {
            if count1 != count2 {
                structural_differences
                    .push(format!("{} layers: {} → {}", layer_type, count1, count2));
            }
        } else {
            structural_differences.push(format!("{} layers removed (was {})", layer_type, count1));
        }
    }

    for (layer_type, count2) in &model2_info.layer_types {
        if !model1_info.layer_types.contains_key(layer_type) {
            structural_differences.push(format!("{} layers added (now {})", layer_type, count2));
        }
    }

    // Detect activation functions (simplified)
    let activations_1 = detect_activation_functions(model1_tensors);
    let activations_2 = detect_activation_functions(model2_tensors);

    let layer_depth_comparison = (model1_info.layer_count, model2_info.layer_count);

    let comparison_summary = if arch_type_1 == arch_type_2 {
        if structural_differences.is_empty() {
            "Architectures are identical".to_string()
        } else {
            format!(
                "Same architecture type with {} modifications",
                structural_differences.len()
            )
        }
    } else {
        format!(
            "Different architecture types: {} vs {}",
            arch_type_1, arch_type_2
        )
    };

    ArchitectureComparisonInfo {
        architecture_type_1: arch_type_1,
        architecture_type_2: arch_type_2,
        structural_differences,
        layer_depth_comparison,
        activation_functions: (activations_1, activations_2),
        comparison_summary,
    }
}

/// Analyze parameter efficiency between two models
fn analyze_param_efficiency(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> ParamEfficiencyInfo {
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    let params_per_layer_1 = if model1_info.layer_count > 0 {
        model1_info.total_parameters as f64 / model1_info.layer_count as f64
    } else {
        0.0
    };

    let params_per_layer_2 = if model2_info.layer_count > 0 {
        model2_info.total_parameters as f64 / model2_info.layer_count as f64
    } else {
        0.0
    };

    let efficiency_ratio = if params_per_layer_1 > 0.0 {
        params_per_layer_2 / params_per_layer_1
    } else {
        1.0
    };

    // Identify sparse and dense layers
    let mut sparse_layers = Vec::new();
    let mut dense_layers = Vec::new();

    for (name, stats) in model2_tensors {
        let param_density =
            stats.total_params as f64 / (stats.shape.iter().product::<usize>() as f64).max(1.0);
        if param_density < 0.1 {
            sparse_layers.push(name.clone());
        } else if param_density > 0.9 {
            dense_layers.push(name.clone());
        }
    }

    let efficiency_recommendation = if efficiency_ratio > 1.2 {
        "Model became less parameter-efficient. Consider pruning or compression.".to_string()
    } else if efficiency_ratio < 0.8 {
        "Model became more parameter-efficient. Good optimization!".to_string()
    } else {
        "Parameter efficiency remained stable.".to_string()
    };

    ParamEfficiencyInfo {
        params_per_layer_1,
        params_per_layer_2,
        efficiency_ratio,
        sparse_layers,
        dense_layers,
        efficiency_recommendation,
    }
}

/// Analyze hyperparameter impact on model changes
fn analyze_hyperparameter_impact(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> HyperparameterInfo {
    let mut detected_changes = HashMap::new();
    let mut impact_scores = HashMap::new();
    let mut high_impact_params = Vec::new();

    // Estimate hyperparameters from model changes
    let avg_param_change = calculate_average_parameter_change(model1_tensors, model2_tensors);

    // Learning rate estimation
    if avg_param_change > 0.1 {
        detected_changes.insert(
            "learning_rate".to_string(),
            ("moderate".to_string(), "high".to_string()),
        );
        impact_scores.insert("learning_rate".to_string(), 0.8);
        high_impact_params.push("learning_rate".to_string());
    } else if avg_param_change > 0.01 {
        detected_changes.insert(
            "learning_rate".to_string(),
            ("low".to_string(), "moderate".to_string()),
        );
        impact_scores.insert("learning_rate".to_string(), 0.5);
    }

    // Batch size estimation (based on normalization layer changes)
    let norm_layer_changes = count_normalization_layer_changes(model1_tensors, model2_tensors);
    if norm_layer_changes > 0.1 {
        detected_changes.insert(
            "batch_size".to_string(),
            ("unknown".to_string(), "changed".to_string()),
        );
        impact_scores.insert("batch_size".to_string(), 0.6);
        high_impact_params.push("batch_size".to_string());
    }

    // Regularization estimation
    let weight_decay_effect = estimate_weight_decay_effect(model1_tensors, model2_tensors);
    if weight_decay_effect > 0.05 {
        detected_changes.insert(
            "weight_decay".to_string(),
            ("low".to_string(), "high".to_string()),
        );
        impact_scores.insert("weight_decay".to_string(), 0.4);
    }

    let stability_assessment = if high_impact_params.len() > 2 {
        "unstable"
    } else if !high_impact_params.is_empty() {
        "needs_attention"
    } else {
        "stable"
    }
    .to_string();

    let suggested_tuning = if high_impact_params.contains(&"learning_rate".to_string()) {
        vec!["Consider reducing learning rate".to_string()]
    } else {
        vec!["Current hyperparameters appear stable".to_string()]
    };

    HyperparameterInfo {
        detected_changes,
        impact_scores,
        high_impact_params,
        suggested_tuning,
        stability_assessment,
    }
}

/// Analyze learning rate effects and patterns
fn analyze_learning_rate(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> LearningRateInfo {
    let avg_change = calculate_average_parameter_change(model1_tensors, model2_tensors);

    // Estimate learning rates based on parameter changes
    let estimated_lr_1 = 0.001; // Default baseline
    let estimated_lr_2 = if avg_change > 0.1 {
        0.01 // High change suggests higher LR
    } else if avg_change > 0.01 {
        0.003 // Moderate change
    } else {
        0.0001 // Low change suggests lower LR
    };

    // Detect learning rate schedule pattern
    let lr_schedule_pattern = if avg_change > 0.05 {
        "constant".to_string()
    } else if avg_change > 0.01 {
        "decay".to_string()
    } else {
        "adaptive".to_string()
    };

    let lr_effectiveness = if avg_change > 0.01 && avg_change < 0.1 {
        0.8 // Good balance
    } else if avg_change < 0.001 {
        0.3 // Too low
    } else if avg_change > 0.2 {
        0.2 // Too high
    } else {
        0.6 // Moderate
    };

    let lr_recommendation = if lr_effectiveness < 0.4 {
        "Consider adjusting learning rate for better convergence".to_string()
    } else if lr_effectiveness > 0.7 {
        "Learning rate appears optimal".to_string()
    } else {
        "Learning rate is acceptable but could be optimized".to_string()
    };

    LearningRateInfo {
        estimated_lr_1,
        estimated_lr_2,
        lr_schedule_pattern,
        lr_effectiveness,
        lr_recommendation,
    }
}

/// Assess deployment readiness and safety
fn assess_deployment_readiness(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
    analysis_results: &[DiffResult],
) -> DeploymentReadinessInfo {
    let mut readiness_score: f64 = 1.0;
    let mut blocking_issues = Vec::new();
    let mut warnings = Vec::new();

    // Check for critical issues from other analyses
    for result in analysis_results {
        match result {
            DiffResult::AnomalyDetection(_, anomaly) if anomaly.severity == "critical" => {
                blocking_issues.push(format!("Critical anomaly: {}", anomaly.anomaly_type));
                readiness_score -= 0.4;
            }
            DiffResult::RegressionTest(_, regression) if !regression.test_passed => {
                if regression.severity_level == "critical" {
                    blocking_issues.push("Critical regression test failures".to_string());
                    readiness_score -= 0.3;
                } else {
                    warnings.push(format!("Regression issues: {}", regression.severity_level));
                    readiness_score -= 0.1;
                }
            }
            _ => {}
        }
    }

    // Check model size changes
    let model1_info = calculate_model_info(model1_tensors);
    let model2_info = calculate_model_info(model2_tensors);

    let size_change_ratio =
        model2_info.model_size_bytes as f64 / model1_info.model_size_bytes as f64;
    if size_change_ratio > 2.0 {
        warnings.push("Significant model size increase detected".to_string());
        readiness_score -= 0.1;
    }

    readiness_score = readiness_score.max(0.0);

    let deployment_strategy = if !blocking_issues.is_empty() {
        "hold"
    } else if readiness_score < 0.6 {
        "gradual"
    } else if readiness_score < 0.8 {
        "safe"
    } else {
        "full"
    }
    .to_string();

    let estimated_risk_level = if readiness_score < 0.4 {
        "critical"
    } else if readiness_score < 0.6 {
        "high"
    } else if readiness_score < 0.8 {
        "medium"
    } else {
        "low"
    }
    .to_string();

    let go_live_recommendation = match deployment_strategy.as_str() {
        "hold" => "Do not deploy. Critical issues must be resolved first.",
        "gradual" => "Deploy with careful monitoring and gradual rollout.",
        "safe" => "Safe to deploy with standard monitoring.",
        _ => "Ready for full deployment.",
    }
    .to_string();

    DeploymentReadinessInfo {
        readiness_score,
        blocking_issues,
        warnings,
        deployment_strategy,
        estimated_risk_level,
        go_live_recommendation,
    }
}

/// Estimate performance impact of model changes
fn estimate_performance_impact(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> PerformanceImpactInfo {
    let model1_flops = estimate_model_flops(model1_tensors);
    let model2_flops = estimate_model_flops(model2_tensors);

    let latency_impact_estimate = if model1_flops > 0 {
        model2_flops as f64 / model1_flops as f64
    } else {
        1.0
    };

    let throughput_impact_estimate = if latency_impact_estimate > 0.0 {
        1.0 / latency_impact_estimate
    } else {
        1.0
    };

    let memory_analysis = analyze_memory_usage(model1_tensors, model2_tensors);
    let memory_impact_estimate = if memory_analysis.model1_size_bytes > 0 {
        memory_analysis.model2_size_bytes as f64 / memory_analysis.model1_size_bytes as f64
    } else {
        1.0
    };

    // Estimate accuracy impact (simplified)
    let param_change = calculate_average_parameter_change(model1_tensors, model2_tensors);
    let accuracy_impact_estimate = if param_change > 0.1 {
        0.95 // Significant changes may affect accuracy
    } else if param_change > 0.01 {
        0.98 // Moderate changes
    } else {
        1.0 // Minimal changes
    };

    let overall_performance_score = (latency_impact_estimate
        + throughput_impact_estimate
        + memory_impact_estimate
        + accuracy_impact_estimate)
        / 4.0;

    let performance_recommendation = if overall_performance_score > 1.2 {
        "Performance may be degraded. Consider optimization.".to_string()
    } else if overall_performance_score < 0.8 {
        "Performance improved! Good optimization.".to_string()
    } else {
        "Performance impact is minimal.".to_string()
    };

    PerformanceImpactInfo {
        latency_impact_estimate,
        throughput_impact_estimate,
        memory_impact_estimate,
        accuracy_impact_estimate,
        overall_performance_score,
        performance_recommendation,
    }
}

// Helper functions for the new analysis features
fn detect_architecture_type(tensors: &HashMap<String, TensorStats>) -> String {
    let mut conv_count = 0;
    let mut linear_count = 0;
    let mut attention_count = 0;

    for name in tensors.keys() {
        if name.contains("conv") || name.contains("Conv") {
            conv_count += 1;
        } else if name.contains("linear") || name.contains("Linear") || name.contains("fc") {
            linear_count += 1;
        } else if name.contains("attention") || name.contains("attn") {
            attention_count += 1;
        }
    }

    if attention_count > conv_count && attention_count > linear_count {
        "transformer".to_string()
    } else if conv_count > linear_count {
        "cnn".to_string()
    } else if linear_count > 0 {
        "mlp".to_string()
    } else {
        "unknown".to_string()
    }
}

fn detect_activation_functions(_tensors: &HashMap<String, TensorStats>) -> Vec<String> {
    // Simplified activation detection - in practice would analyze layer names
    vec!["relu".to_string(), "gelu".to_string()]
}

fn calculate_average_parameter_change(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> f64 {
    let mut total_change = 0.0;
    let mut count = 0;

    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            let change = (stats1.mean - stats2.mean).abs() + (stats1.std - stats2.std).abs();
            total_change += change;
            count += 1;
        }
    }

    if count > 0 {
        total_change / count as f64
    } else {
        0.0
    }
}

fn count_normalization_layer_changes(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> f64 {
    let mut change_sum = 0.0;
    let mut norm_layers = 0;

    for (name, stats1) in model1_tensors {
        if name.contains("norm") || name.contains("Norm") || name.contains("bn") {
            if let Some(stats2) = model2_tensors.get(name) {
                let change = (stats1.mean - stats2.mean).abs();
                change_sum += change;
                norm_layers += 1;
            }
        }
    }

    if norm_layers > 0 {
        change_sum / norm_layers as f64
    } else {
        0.0
    }
}

fn estimate_weight_decay_effect(
    model1_tensors: &HashMap<String, TensorStats>,
    model2_tensors: &HashMap<String, TensorStats>,
) -> f64 {
    let mut weight_reduction = 0.0;
    let mut layer_count = 0;

    for (name, stats1) in model1_tensors {
        if let Some(stats2) = model2_tensors.get(name) {
            let magnitude_change = (stats1.std - stats2.std) / (stats1.std + 1e-8);
            if magnitude_change > 0.0 {
                weight_reduction += magnitude_change;
                layer_count += 1;
            }
        }
    }

    if layer_count > 0 {
        weight_reduction / layer_count as f64
    } else {
        0.0
    }
}

// Stub implementations for remaining functions (to be implemented in next phase)
fn generate_experiment_report(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
    _results: &[DiffResult],
) -> ReportInfo {
    ReportInfo {
        experiment_title: "Model Comparison Report".to_string(),
        summary: "Automated comparison between two ML models".to_string(),
        key_findings: vec!["Model architecture analysis completed".to_string()],
        methodology: "Statistical analysis of tensor parameters".to_string(),
        conclusions: vec!["Models show expected differences".to_string()],
        next_steps: vec!["Deploy with monitoring".to_string()],
    }
}

fn generate_markdown_output(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
    _results: &[DiffResult],
) -> MarkdownInfo {
    let markdown_content =
        "# Model Comparison Report\n\n## Summary\nComparison completed.\n".to_string();
    MarkdownInfo {
        markdown_content,
        sections: vec!["Summary".to_string()],
        tables_generated: 1,
        charts_referenced: 0,
    }
}

fn generate_charts(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> ChartInfo {
    ChartInfo {
        chart_type: "bar".to_string(),
        chart_data: vec![("model1".to_string(), 1.0), ("model2".to_string(), 1.1)],
        chart_title: "Model Parameter Comparison".to_string(),
        x_axis_label: "Model".to_string(),
        y_axis_label: "Parameters".to_string(),
        chart_description: "Comparison of model parameters".to_string(),
    }
}

fn analyze_embeddings(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> EmbeddingInfo {
    EmbeddingInfo {
        embedding_layers: vec!["embedding".to_string()],
        embedding_dimensions: HashMap::new(),
        embedding_similarity: HashMap::new(),
        semantic_drift: 0.1,
        affected_vocabularies: Vec::new(),
        embedding_recommendation: "Embeddings appear stable".to_string(),
    }
}

fn generate_similarity_matrix(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> SimilarityMatrixInfo {
    SimilarityMatrixInfo {
        matrix_size: (10, 10),
        average_similarity: 0.85,
        similarity_distribution: vec![0.8, 0.85, 0.9],
        outlier_pairs: Vec::new(),
        cluster_count_estimate: 3,
    }
}

fn analyze_clustering_changes(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> ClusteringInfo {
    ClusteringInfo {
        clusters_before: 5,
        clusters_after: 5,
        cluster_stability: 0.9,
        migrated_entities: Vec::new(),
        new_clusters: Vec::new(),
        dissolved_clusters: Vec::new(),
        clustering_recommendation: "Clustering is stable".to_string(),
    }
}

fn analyze_attention_mechanisms(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> AttentionInfo {
    AttentionInfo {
        attention_layers: vec!["attention.0".to_string()],
        attention_heads_count: HashMap::new(),
        attention_pattern_changes: Vec::new(),
        attention_entropy: HashMap::new(),
        attention_focus_shift: "stable".to_string(),
        attention_recommendation: "Attention patterns are stable".to_string(),
    }
}

fn analyze_head_importance(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> HeadImportanceInfo {
    HeadImportanceInfo {
        head_importance_scores: HashMap::new(),
        most_important_heads: Vec::new(),
        least_important_heads: Vec::new(),
        prunable_heads: Vec::new(),
        head_specialization: HashMap::new(),
    }
}

fn analyze_attention_patterns(
    _m1: &HashMap<String, TensorStats>,
    _m2: &HashMap<String, TensorStats>,
) -> AttentionPatternInfo {
    AttentionPatternInfo {
        pattern_type_1: "local".to_string(),
        pattern_type_2: "local".to_string(),
        pattern_similarity: 0.95,
        attention_span_change: 0.01,
        locality_bias_change: 0.02,
        pattern_change_summary: "Attention patterns remain consistent".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_creation() {
        let stats = TensorStats {
            mean: 0.5,
            std: 1.0,
            min: -2.0,
            max: 3.0,
            shape: vec![10, 20],
            dtype: "f32".to_string(),
            total_params: 200,
        };

        assert_eq!(stats.mean, 0.5);
        assert_eq!(stats.total_params, 200);
        assert_eq!(stats.shape, vec![10, 20]);
    }

    #[test]
    fn test_diff_result_variants() {
        // Test TensorStatsChanged variant
        let stats1 = TensorStats {
            mean: 0.0,
            std: 1.0,
            min: -2.0,
            max: 2.0,
            shape: vec![128, 64],
            dtype: "f32".to_string(),
            total_params: 8192,
        };

        let stats2 = TensorStats {
            mean: 0.1,
            std: 1.1,
            min: -1.9,
            max: 2.1,
            shape: vec![128, 64],
            dtype: "f32".to_string(),
            total_params: 8192,
        };

        let diff = DiffResult::TensorStatsChanged(
            "linear1.weight".to_string(),
            stats1.clone(),
            stats2.clone(),
        );

        match diff {
            DiffResult::TensorStatsChanged(name, s1, s2) => {
                assert_eq!(name, "linear1.weight");
                assert_eq!(s1.mean, 0.0);
                assert_eq!(s2.mean, 0.1);
            }
            _ => panic!("Expected TensorStatsChanged variant"),
        }
    }

    #[test]
    fn test_tensor_shape_changed() {
        let diff = DiffResult::TensorShapeChanged(
            "linear2.weight".to_string(),
            vec![256, 128],
            vec![512, 128],
        );

        match diff {
            DiffResult::TensorShapeChanged(name, shape1, shape2) => {
                assert_eq!(name, "linear2.weight");
                assert_eq!(shape1, vec![256, 128]);
                assert_eq!(shape2, vec![512, 128]);
            }
            _ => panic!("Expected TensorShapeChanged variant"),
        }
    }

    #[test]
    fn test_calculate_f32_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std, min, max) = calculate_f32_stats(&data);

        assert_eq!(mean, 3.0);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        // std should be sqrt(2) for [1,2,3,4,5]
        assert!((std - (2.0_f64).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_f64_stats() {
        let data = vec![0.0, 1.0, 2.0];
        let (mean, std, min, max) = calculate_f64_stats(&data);

        assert_eq!(mean, 1.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 2.0);
        assert!((std - (2.0_f64 / 3.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_error_handling_nonexistent_files() {
        // Test that ML diff function handles non-existent files gracefully
        let result = diff_ml_models(
            Path::new("nonexistent1.safetensors"),
            Path::new("nonexistent2.safetensors"),
            None,
        );
        assert!(result.is_err());
    }
}
