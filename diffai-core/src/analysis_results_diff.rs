/// ML Analysis Results Comparison Module
/// 
/// This module provides functionality to compare ML analysis results using diffx-core,
/// enabling meta-analysis of experiments and analysis history comparison.

use anyhow::Result;
use serde_json::Value;
use crate::{DiffResult, convert_diffx_result};

/// Compare two sets of ML analysis results
/// 
/// This function enables comparing the analysis results from two different
/// experiments, model versions, or analysis runs. Useful for:
/// - Experiment A vs Experiment B comparison
/// - Before/after analysis changes tracking  
/// - MLOps analysis history management
pub fn diff_analysis_results(
    results1: &[DiffResult],
    results2: &[DiffResult], 
) -> Result<Vec<DiffResult>> {
    // Convert analysis results to JSON for comparison
    let json1 = analysis_results_to_json(results1)?;
    let json2 = analysis_results_to_json(results2)?;
    
    // Use diffx-core for structured comparison
    let diff_results = diffx_core::diff(&json1, &json2)
        .into_iter()
        .map(convert_diffx_result)
        .collect();
    
    Ok(diff_results)
}

/// Convert analysis results to structured JSON for diffx-core comparison
fn analysis_results_to_json(results: &[DiffResult]) -> Result<Value> {
    let mut analysis_summary = serde_json::Map::new();
    
    // Categorize and summarize analysis results
    for result in results {
        match result {
            DiffResult::LearningProgress(key, progress) => {
                analysis_summary.insert(
                    format!("learning_progress_{}", key),
                    serde_json::to_value(progress)?
                );
            }
            DiffResult::ConvergenceAnalysis(key, convergence) => {
                analysis_summary.insert(
                    format!("convergence_{}", key), 
                    serde_json::to_value(convergence)?
                );
            }
            DiffResult::AnomalyDetection(key, anomaly) => {
                analysis_summary.insert(
                    format!("anomaly_{}", key),
                    serde_json::to_value(anomaly)?
                );
            }
            DiffResult::MemoryAnalysis(key, memory) => {
                analysis_summary.insert(
                    format!("memory_{}", key),
                    serde_json::to_value(memory)?
                );
            }
            DiffResult::DeploymentReadiness(key, deployment) => {
                analysis_summary.insert(
                    format!("deployment_{}", key),
                    serde_json::to_value(deployment)?
                );
            }
            DiffResult::QuantizationAnalysis(key, quant) => {
                analysis_summary.insert(
                    format!("quantization_{}", key),
                    serde_json::to_value(quant)?
                );
            }
            DiffResult::TransferLearningAnalysis(key, transfer) => {
                analysis_summary.insert(
                    format!("transfer_learning_{}", key),
                    serde_json::to_value(transfer)?
                );
            }
            DiffResult::ExperimentReproducibility(key, experiment) => {
                analysis_summary.insert(
                    format!("reproducibility_{}", key),
                    serde_json::to_value(experiment)?
                );
            }
            DiffResult::EnsembleAnalysis(key, ensemble) => {
                analysis_summary.insert(
                    format!("ensemble_{}", key),
                    serde_json::to_value(ensemble)?
                );
            }
            // Add tensor statistics for completeness
            DiffResult::TensorStatsChanged(key, stats1, stats2) => {
                analysis_summary.insert(
                    format!("tensor_stats_{}", key),
                    serde_json::json!({
                        "before": stats1,
                        "after": stats2,
                        "mean_change": stats2.mean - stats1.mean,
                        "std_change": stats2.std - stats1.std
                    })
                );
            }
            _ => {
                // Include other analysis types as basic summaries
                analysis_summary.insert(
                    format!("analysis_{}", get_result_type_name(result)),
                    serde_json::json!({
                        "type": get_result_type_name(result),
                        "key": get_result_key(result)
                    })
                );
            }
        }
    }
    
    Ok(Value::Object(analysis_summary))
}

/// Get the type name of a DiffResult for categorization
fn get_result_type_name(result: &DiffResult) -> &'static str {
    match result {
        DiffResult::Added(_, _) => "added",
        DiffResult::Removed(_, _) => "removed", 
        DiffResult::Modified(_, _, _) => "modified",
        DiffResult::TypeChanged(_, _, _) => "type_changed",
        DiffResult::TensorShapeChanged(_, _, _) => "tensor_shape_changed",
        DiffResult::TensorStatsChanged(_, _, _) => "tensor_stats_changed",
        DiffResult::TensorAdded(_, _) => "tensor_added",
        DiffResult::TensorRemoved(_, _) => "tensor_removed",
        DiffResult::ModelArchitectureChanged(_, _, _) => "model_architecture_changed",
        DiffResult::LearningProgress(_, _) => "learning_progress",
        DiffResult::ConvergenceAnalysis(_, _) => "convergence_analysis",
        DiffResult::AnomalyDetection(_, _) => "anomaly_detection",
        DiffResult::GradientAnalysis(_, _) => "gradient_analysis",
        DiffResult::MemoryAnalysis(_, _) => "memory_analysis",
        DiffResult::InferenceSpeedAnalysis(_, _) => "inference_speed_analysis",
        DiffResult::RegressionTest(_, _) => "regression_test",
        DiffResult::AlertOnDegradation(_, _) => "alert_on_degradation",
        DiffResult::ReviewFriendly(_, _) => "review_friendly",
        DiffResult::ChangeSummary(_, _) => "change_summary",
        DiffResult::RiskAssessment(_, _) => "risk_assessment",
        DiffResult::ArchitectureComparison(_, _) => "architecture_comparison",
        DiffResult::ParamEfficiencyAnalysis(_, _) => "param_efficiency_analysis",
        DiffResult::HyperparameterImpact(_, _) => "hyperparameter_impact",
        DiffResult::LearningRateAnalysis(_, _) => "learning_rate_analysis",
        DiffResult::DeploymentReadiness(_, _) => "deployment_readiness",
        DiffResult::PerformanceImpactEstimate(_, _) => "performance_impact_estimate",
        DiffResult::GenerateReport(_, _) => "generate_report",
        DiffResult::MarkdownOutput(_, _) => "markdown_output",
        DiffResult::IncludeCharts(_, _) => "include_charts",
        DiffResult::EmbeddingAnalysis(_, _) => "embedding_analysis",
        DiffResult::SimilarityMatrix(_, _) => "similarity_matrix",
        DiffResult::ClusteringChange(_, _) => "clustering_change",
        DiffResult::AttentionAnalysis(_, _) => "attention_analysis",
        DiffResult::HeadImportance(_, _) => "head_importance",
        DiffResult::AttentionPatternDiff(_, _) => "attention_pattern_diff",
        DiffResult::QuantizationAnalysis(_, _) => "quantization_analysis",
        DiffResult::TransferLearningAnalysis(_, _) => "transfer_learning_analysis",
        DiffResult::ExperimentReproducibility(_, _) => "experiment_reproducibility",
        DiffResult::EnsembleAnalysis(_, _) => "ensemble_analysis",
    }
}

/// Get the key from any DiffResult for identification
fn get_result_key(result: &DiffResult) -> &str {
    match result {
        DiffResult::Added(k, _) => k,
        DiffResult::Removed(k, _) => k,
        DiffResult::Modified(k, _, _) => k,
        DiffResult::TypeChanged(k, _, _) => k,
        DiffResult::TensorShapeChanged(k, _, _) => k,
        DiffResult::TensorStatsChanged(k, _, _) => k,
        DiffResult::TensorAdded(k, _) => k,
        DiffResult::TensorRemoved(k, _) => k,
        DiffResult::ModelArchitectureChanged(k, _, _) => k,
        DiffResult::LearningProgress(k, _) => k,
        DiffResult::ConvergenceAnalysis(k, _) => k,
        DiffResult::AnomalyDetection(k, _) => k,
        DiffResult::GradientAnalysis(k, _) => k,
        DiffResult::MemoryAnalysis(k, _) => k,
        DiffResult::InferenceSpeedAnalysis(k, _) => k,
        DiffResult::RegressionTest(k, _) => k,
        DiffResult::AlertOnDegradation(k, _) => k,
        DiffResult::ReviewFriendly(k, _) => k,
        DiffResult::ChangeSummary(k, _) => k,
        DiffResult::RiskAssessment(k, _) => k,
        DiffResult::ArchitectureComparison(k, _) => k,
        DiffResult::ParamEfficiencyAnalysis(k, _) => k,
        DiffResult::HyperparameterImpact(k, _) => k,
        DiffResult::LearningRateAnalysis(k, _) => k,
        DiffResult::DeploymentReadiness(k, _) => k,
        DiffResult::PerformanceImpactEstimate(k, _) => k,
        DiffResult::GenerateReport(k, _) => k,
        DiffResult::MarkdownOutput(k, _) => k,
        DiffResult::IncludeCharts(k, _) => k,
        DiffResult::EmbeddingAnalysis(k, _) => k,
        DiffResult::SimilarityMatrix(k, _) => k,
        DiffResult::ClusteringChange(k, _) => k,
        DiffResult::AttentionAnalysis(k, _) => k,
        DiffResult::HeadImportance(k, _) => k,
        DiffResult::AttentionPatternDiff(k, _) => k,
        DiffResult::QuantizationAnalysis(k, _) => k,
        DiffResult::TransferLearningAnalysis(k, _) => k,
        DiffResult::ExperimentReproducibility(k, _) => k,
        DiffResult::EnsembleAnalysis(k, _) => k,
    }
}

/// Compare HuggingFace config files using diffx-core
pub fn diff_huggingface_configs(config1: &Value, config2: &Value) -> Vec<DiffResult> {
    diffx_core::diff(config1, config2)
        .into_iter()
        .map(convert_diffx_result)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LearningProgressInfo, MemoryAnalysisInfo};

    #[test]
    fn test_analysis_results_comparison() {
        // Create sample analysis results
        let results1 = vec![
            DiffResult::LearningProgress("layer1".to_string(), LearningProgressInfo {
                loss_trend: "improving".to_string(),
                parameter_update_magnitude: 0.5,
                gradient_norm_ratio: 1.2,
                convergence_speed: 0.8,
                training_efficiency: 0.9,
                learning_rate_schedule: "cosine".to_string(),
                momentum_coefficient: 0.9,
                weight_decay_effect: 0.01,
                batch_size_impact: 32,
                optimization_algorithm: "Adam".to_string(),
            }),
            DiffResult::MemoryAnalysis("model".to_string(), MemoryAnalysisInfo {
                memory_delta_bytes: 1024000,
                peak_memory_usage: 2048000,
                memory_efficiency_ratio: 0.85,
                gpu_memory_utilization: 0.75,
                memory_fragmentation_level: 0.1,
                cache_efficiency: 0.9,
                memory_leak_indicators: vec![],
                optimization_opportunities: vec!["reduce_batch_size".to_string()],
                estimated_gpu_memory_mb: 2000.0,
                memory_recommendation: "optimize".to_string(),
            }),
        ];

        let results2 = vec![
            DiffResult::LearningProgress("layer1".to_string(), LearningProgressInfo {
                loss_trend: "stable".to_string(),
                parameter_update_magnitude: 0.3,
                gradient_norm_ratio: 1.0,
                convergence_speed: 0.9,
                training_efficiency: 0.95,
                learning_rate_schedule: "cosine".to_string(),
                momentum_coefficient: 0.9,
                weight_decay_effect: 0.01,
                batch_size_impact: 32,
                optimization_algorithm: "Adam".to_string(),
            }),
        ];

        let diff_result = diff_analysis_results(&results1, &results2);
        assert!(diff_result.is_ok());
        let diff = diff_result.unwrap();
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_huggingface_config_comparison() {
        let config1 = serde_json::json!({
            "model_type": "bert",
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 12
        });

        let config2 = serde_json::json!({
            "model_type": "bert", 
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 12
        });

        let diff = diff_huggingface_configs(&config1, &config2);
        assert!(!diff.is_empty());
        
        // Should detect changes in hidden_size and num_attention_heads
        let has_hidden_size_change = diff.iter().any(|d| match d {
            DiffResult::Modified(key, _, _) => key.contains("hidden_size"),
            _ => false,
        });
        assert!(has_hidden_size_change);
    }
}