use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use colored::*;
use diffai_core::{diff, value_type_name, DiffResult, parse_ini, parse_xml, parse_csv, diff_ml_models, diff_ml_models_enhanced};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::io::{self, Read};
use walkdir::WalkDir;
use regex::Regex;

#[derive(Debug, Deserialize, Default)]
struct Config {
    #[serde(default)]
    output: Option<OutputFormat>,
    #[serde(default)]
    format: Option<Format>,
}

fn load_config() -> Config {
    let config_path = dirs::config_dir()
        .map(|p| p.join("diffx").join("config.toml"))
        .or_else(|| {
            // Fallback for systems without a standard config directory
            Some(PathBuf::from(".diffx.toml"))
        });

    if let Some(path) = config_path {
        if path.exists() {
            match fs::read_to_string(&path) {
                Ok(content) => {
                    match toml::from_str(&content) {
                        Ok(config) => return config,
                        Err(e) => eprintln!("Warning: Could not parse config file {}: {}", path.display(), e),
                    }
                }
                Err(e) => eprintln!("Warning: Could not read config file {}: {}", path.display(), e),
            }
        }
    }
    Config::default()
}


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The first input (file path or directory path, use '-' for stdin)
    #[arg(value_name = "INPUT1")]
    input1: PathBuf,

    /// The second input (file path or directory path, use '-' for stdin)
    #[arg(value_name = "INPUT2")]
    input2: PathBuf,

    /// Input file format
    #[arg(short, long, value_enum)]
    format: Option<Format>,

    /// Output format
    #[arg(short, long, value_enum)]
    output: Option<OutputFormat>,

    /// Compare directories recursively
    #[arg(short, long)]
    recursive: bool,

    /// Filter differences by a specific path (e.g., "config.users\[0\].name")
    #[arg(long)]
    path: Option<String>,

    /// Ignore keys matching a regular expression (e.g., "^id$")
    #[arg(long)]
    ignore_keys_regex: Option<String>,

    /// Tolerance for float comparisons (e.g., "0.001")
    #[arg(long)]
    epsilon: Option<f64>,

    /// Key to use for identifying array elements (e.g., "id")
    #[arg(long)]
    array_id_key: Option<String>,

    /// Show layer-by-layer impact analysis for ML models
    #[arg(long)]
    show_layer_impact: bool,

    /// Enable quantization analysis for ML models
    #[arg(long)]
    quantization_analysis: bool,

    /// Sort differences by change magnitude (ML models only)
    #[arg(long)]
    sort_by_change_magnitude: bool,

    /// Show detailed statistics for ML models
    #[arg(long)]
    stats: bool,

    /// Analyze learning progress between training checkpoints
    #[arg(long)]
    learning_progress: bool,

    /// Perform convergence analysis for training stability
    #[arg(long)]
    convergence_analysis: bool,

    /// Detect training anomalies (gradient explosion, vanishing gradients)
    #[arg(long)]
    anomaly_detection: bool,

    /// Analyze gradient characteristics and stability
    #[arg(long)]
    gradient_analysis: bool,

    /// Analyze memory usage and efficiency between models
    #[arg(long)]
    memory_analysis: bool,

    /// Estimate inference speed and performance characteristics
    #[arg(long)]
    inference_speed_estimate: bool,

    /// Perform automated regression testing
    #[arg(long)]
    regression_test: bool,

    /// Alert on performance degradation beyond thresholds
    #[arg(long)]
    alert_on_degradation: bool,

    /// Generate review-friendly output for human reviewers
    #[arg(long)]
    review_friendly: bool,

    /// Generate detailed change summary
    #[arg(long)]
    change_summary: bool,

    /// Assess deployment risk and readiness
    #[arg(long)]
    risk_assessment: bool,

    /// Compare model architectures and structural differences
    #[arg(long)]
    architecture_comparison: bool,

    /// Analyze parameter efficiency between models
    #[arg(long)]
    param_efficiency_analysis: bool,

    /// Analyze hyperparameter impact on model changes
    #[arg(long)]
    hyperparameter_impact: bool,

    /// Analyze learning rate effects and patterns
    #[arg(long)]
    learning_rate_analysis: bool,

    /// Assess deployment readiness and safety
    #[arg(long)]
    deployment_readiness: bool,

    /// Estimate performance impact of model changes
    #[arg(long)]
    performance_impact_estimate: bool,

    /// Generate comprehensive analysis report
    #[arg(long)]
    generate_report: bool,

    /// Output results in markdown format
    #[arg(long)]
    markdown_output: bool,

    /// Include charts and visualizations in output
    #[arg(long)]
    include_charts: bool,

    /// Analyze embedding layer changes and semantic drift
    #[arg(long)]
    embedding_analysis: bool,

    /// Generate similarity matrix for model comparison
    #[arg(long)]
    similarity_matrix: bool,

    /// Analyze clustering changes in model representations
    #[arg(long)]
    clustering_change: bool,

    /// Analyze attention mechanism patterns (Transformer models)
    #[arg(long)]
    attention_analysis: bool,

    /// Analyze attention head importance and specialization
    #[arg(long)]
    head_importance: bool,

    /// Compare attention patterns between models
    #[arg(long)]
    attention_pattern_diff: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Serialize, Deserialize)]
enum OutputFormat {
    Cli,
    Json,
    Yaml,
    Unified,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug, Serialize, Deserialize)]
enum Format {
    Json,
    Yaml,
    Toml,
    Ini,
    Xml,
    Csv,
    Safetensors,
    Pytorch,
}

fn infer_format_from_path(path: &Path) -> Option<Format> {
    if path.to_str() == Some("-") {
        // Cannot infer format from stdin, user must specify --format
        None
    } else {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext_str| {
                match ext_str.to_lowercase().as_str() {
                    "json" => Some(Format::Json),
                    "yaml" | "yml" => Some(Format::Yaml),
                    "toml" => Some(Format::Toml),
                    "ini" => Some(Format::Ini),
                    "xml" => Some(Format::Xml),
                    "csv" => Some(Format::Csv),
                    "safetensors" => Some(Format::Safetensors),
                    "pt" | "pth" => Some(Format::Pytorch),
                    _ => None,
                }
            })
    }
}

fn read_input(file_path: &Path) -> Result<String> {
    if file_path.to_str() == Some("-") {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer).context("Failed to read from stdin")?;
        Ok(buffer)
    } else {
        fs::read_to_string(file_path).context(format!("Failed to read file: {}", file_path.display()))
    }
}

fn parse_content(content: &str, format: Format) -> Result<Value> {
    match format {
        Format::Json => serde_json::from_str(content).context("Failed to parse JSON"),
        Format::Yaml => serde_yml::from_str(content).context("Failed to parse YAML"),
        Format::Toml => toml::from_str(content).context("Failed to parse TOML"),
        Format::Ini => parse_ini(content).context("Failed to parse INI"),
        Format::Xml => parse_xml(content).context("Failed to parse XML"),
        Format::Csv => parse_csv(content).context("Failed to parse CSV"),
        Format::Safetensors | Format::Pytorch => {
            bail!("ML model formats (safetensors, pytorch) cannot be parsed as text. Use the model comparison feature instead.")
        }
    }
}

fn print_cli_output(mut differences: Vec<DiffResult>, sort_by_magnitude: bool) {
    if differences.is_empty() {
        println!("No differences found.");
        return;
    }

    let get_key = |d: &DiffResult| -> String {
        match d {
            DiffResult::Added(k, _) => k.clone(),
            DiffResult::Removed(k, _) => k.clone(),
            DiffResult::Modified(k, _, _) => k.clone(),
            DiffResult::TypeChanged(k, _, _) => k.clone(),
            DiffResult::TensorShapeChanged(k, _, _) => k.clone(),
            DiffResult::TensorStatsChanged(k, _, _) => k.clone(),
            DiffResult::ModelArchitectureChanged(k, _, _) => k.clone(),
            DiffResult::LearningProgress(k, _) => k.clone(),
            DiffResult::ConvergenceAnalysis(k, _) => k.clone(),
            DiffResult::AnomalyDetection(k, _) => k.clone(),
            DiffResult::GradientAnalysis(k, _) => k.clone(),
            DiffResult::MemoryAnalysis(k, _) => k.clone(),
            DiffResult::InferenceSpeedAnalysis(k, _) => k.clone(),
            DiffResult::RegressionTest(k, _) => k.clone(),
            DiffResult::AlertOnDegradation(k, _) => k.clone(),
            DiffResult::ReviewFriendly(k, _) => k.clone(),
            DiffResult::ChangeSummary(k, _) => k.clone(),
            DiffResult::RiskAssessment(k, _) => k.clone(),
            DiffResult::ArchitectureComparison(k, _) => k.clone(),
            DiffResult::ParamEfficiencyAnalysis(k, _) => k.clone(),
            DiffResult::HyperparameterImpact(k, _) => k.clone(),
            DiffResult::LearningRateAnalysis(k, _) => k.clone(),
            DiffResult::DeploymentReadiness(k, _) => k.clone(),
            DiffResult::PerformanceImpactEstimate(k, _) => k.clone(),
            DiffResult::GenerateReport(k, _) => k.clone(),
            DiffResult::MarkdownOutput(k, _) => k.clone(),
            DiffResult::IncludeCharts(k, _) => k.clone(),
            DiffResult::EmbeddingAnalysis(k, _) => k.clone(),
            DiffResult::SimilarityMatrix(k, _) => k.clone(),
            DiffResult::ClusteringChange(k, _) => k.clone(),
            DiffResult::AttentionAnalysis(k, _) => k.clone(),
            DiffResult::HeadImportance(k, _) => k.clone(),
            DiffResult::AttentionPatternDiff(k, _) => k.clone(),
        }
    };

    let get_change_magnitude = |d: &DiffResult| -> f64 {
        match d {
            DiffResult::TensorStatsChanged(_, stats1, stats2) => {
                // Calculate magnitude of change in tensor statistics
                let mean_change = (stats1.mean - stats2.mean).abs();
                let std_change = (stats1.std - stats2.std).abs();
                mean_change + std_change
            }
            DiffResult::TensorShapeChanged(_, shape1, shape2) => {
                // Calculate magnitude of shape change
                let size1: usize = shape1.iter().product();
                let size2: usize = shape2.iter().product();
                (size1 as f64 - size2 as f64).abs()
            }
            DiffResult::ModelArchitectureChanged(_, info1, info2) => {
                // Calculate magnitude of parameter count change
                (info1.total_parameters as f64 - info2.total_parameters as f64).abs()
            }
            DiffResult::LearningProgress(_, progress) => {
                // Use parameter update magnitude as sort key
                progress.parameter_update_magnitude
            }
            DiffResult::ConvergenceAnalysis(_, convergence) => {
                // Use parameter stability as sort key (inverted - less stable = higher magnitude)
                1.0 - convergence.parameter_stability
            }
            DiffResult::AnomalyDetection(_, anomaly) => {
                // Use severity as magnitude (critical=2.0, warning=1.0, none=0.0)
                match anomaly.severity.as_str() {
                    "critical" => 2.0,
                    "warning" => 1.0,
                    _ => 0.0,
                }
            }
            DiffResult::GradientAnalysis(_, gradient) => {
                // Use gradient norm estimate as magnitude
                gradient.gradient_norm_estimate
            }
            DiffResult::MemoryAnalysis(_, memory) => {
                // Use absolute memory delta as magnitude (in MB)
                (memory.memory_delta_bytes.abs() as f64) / (1024.0 * 1024.0)
            }
            DiffResult::InferenceSpeedAnalysis(_, speed) => {
                // Use speed change ratio distance from 1.0 as magnitude
                (speed.speed_change_ratio - 1.0).abs()
            }
            DiffResult::RegressionTest(_, regression) => {
                // Use performance degradation percentage as magnitude
                regression.performance_degradation
            }
            DiffResult::AlertOnDegradation(_, alert) => {
                // Use threshold exceeded factor as magnitude
                if alert.alert_triggered {
                    alert.threshold_exceeded
                } else {
                    0.0
                }
            }
            DiffResult::ReviewFriendly(_, review) => {
                // Use impact assessment as magnitude
                match review.impact_assessment.as_str() {
                    "critical" => 4.0,
                    "high" => 3.0,
                    "medium" => 2.0,
                    "low" => 1.0,
                    _ => 0.0,
                }
            }
            DiffResult::ChangeSummary(_, summary) => {
                // Use overall change magnitude
                summary.overall_change_magnitude
            }
            DiffResult::RiskAssessment(_, risk) => {
                // Use risk level as magnitude
                match risk.overall_risk_level.as_str() {
                    "critical" => 4.0,
                    "high" => 3.0,
                    "medium" => 2.0,
                    "low" => 1.0,
                    _ => 0.0,
                }
            }
            DiffResult::ArchitectureComparison(_, arch) => {
                // Use layer depth difference as magnitude
                let depth_diff = (arch.layer_depth_comparison.0 as f64 - arch.layer_depth_comparison.1 as f64).abs();
                depth_diff
            }
            DiffResult::ParamEfficiencyAnalysis(_, efficiency) => {
                // Use efficiency ratio distance from 1.0 as magnitude
                (efficiency.efficiency_ratio - 1.0).abs()
            }
            DiffResult::HyperparameterImpact(_, hyper) => {
                // Use maximum impact score as magnitude
                hyper.impact_scores.values().copied().fold(0.0_f64, f64::max)
            }
            DiffResult::LearningRateAnalysis(_, lr) => {
                // Use learning rate effectiveness as magnitude
                lr.lr_effectiveness
            }
            DiffResult::DeploymentReadiness(_, deploy) => {
                // Use readiness score (inverted - lower readiness = higher magnitude)
                1.0 - deploy.readiness_score
            }
            DiffResult::PerformanceImpactEstimate(_, perf) => {
                // Use overall performance score
                perf.overall_performance_score
            }
            DiffResult::GenerateReport(_, _) => 1.0, // Static magnitude for reports
            DiffResult::MarkdownOutput(_, _) => 1.0, // Static magnitude for markdown
            DiffResult::IncludeCharts(_, chart) => {
                // Use data variance as magnitude
                if chart.chart_data.is_empty() {
                    0.0
                } else {
                    let values: Vec<f64> = chart.chart_data.iter().map(|(_, v)| *v).collect();
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                    variance.sqrt()
                }
            }
            DiffResult::EmbeddingAnalysis(_, embed) => {
                // Use semantic drift as magnitude
                embed.semantic_drift
            }
            DiffResult::SimilarityMatrix(_, sim) => {
                // Use average similarity as magnitude
                sim.average_similarity
            }
            DiffResult::ClusteringChange(_, cluster) => {
                // Use cluster stability (inverted) as magnitude
                1.0 - cluster.cluster_stability
            }
            DiffResult::AttentionAnalysis(_, attention) => {
                // Use average attention entropy as magnitude
                if attention.attention_entropy.is_empty() {
                    0.0
                } else {
                    attention.attention_entropy.values().sum::<f64>() / attention.attention_entropy.len() as f64
                }
            }
            DiffResult::HeadImportance(_, head) => {
                // Use maximum head importance as magnitude
                head.head_importance_scores.values()
                    .flat_map(|scores| scores.iter())
                    .copied()
                    .fold(0.0_f64, f64::max)
            }
            DiffResult::AttentionPatternDiff(_, pattern) => {
                // Use pattern similarity (inverted) as magnitude
                1.0 - pattern.pattern_similarity
            }
            _ => 0.0, // Non-ML changes have no magnitude
        }
    };

    if sort_by_magnitude {
        differences.sort_by(|a, b| get_change_magnitude(b).partial_cmp(&get_change_magnitude(a)).unwrap_or(std::cmp::Ordering::Equal));
    } else {
        differences.sort_by(|a, b| get_key(a).cmp(&get_key(b)));
    }

    for diff in &differences {
        let key = get_key(diff);
        // Indent based on the depth of the key
        let depth = key.chars().filter(|&c| c == '.' || c == '[').count();
        let indent = "  ".repeat(depth);

        let diff_str = match diff {
            DiffResult::Added(k, value) => format!("+ {}: {}", k, value).blue(),
            DiffResult::Removed(k, value) => format!("- {}: {}", k, value).yellow(),
            DiffResult::Modified(k, v1, v2) => format!("~ {}: {} -> {}", k, v1, v2).cyan(),
            DiffResult::TypeChanged(k, v1, v2) => {
                format!("! {}: {} ({}) -> {} ({})", k, v1, value_type_name(v1), v2, value_type_name(v2))
                    .magenta()
            }
            DiffResult::TensorShapeChanged(k, shape1, shape2) => {
                format!("⬚ {}: {:?} -> {:?}", k, shape1, shape2).purple()
            }
            DiffResult::TensorStatsChanged(k, stats1, stats2) => {
                format!("📊 {}: mean={:.4}→{:.4}, std={:.4}→{:.4}", 
                    k, stats1.mean, stats2.mean, stats1.std, stats2.std).bright_purple()
            }
            DiffResult::ModelArchitectureChanged(k, info1, info2) => {
                format!("🏗️ {}: params={}→{}, layers={}→{}", 
                    k, info1.total_parameters, info2.total_parameters, 
                    info1.layer_count, info2.layer_count).bright_magenta()
            }
            DiffResult::LearningProgress(k, progress) => {
                format!("📈 {}: trend={}, magnitude={:.4}, speed={:.2}", 
                    k, progress.loss_trend, progress.parameter_update_magnitude, 
                    progress.convergence_speed).bright_green()
            }
            DiffResult::ConvergenceAnalysis(k, convergence) => {
                format!("🎯 {}: status={}, stability={:.4}, action=\"{}\"", 
                    k, convergence.convergence_status, convergence.parameter_stability,
                    convergence.recommended_action).bright_yellow()
            }
            DiffResult::AnomalyDetection(k, anomaly) => {
                let color = match anomaly.severity.as_str() {
                    "critical" => format!("🚨 {}: type={}, severity={}, affected={} layers, action=\"{}\"", 
                        k, anomaly.anomaly_type, anomaly.severity, anomaly.affected_layers.len(),
                        anomaly.recommended_action).bright_red(),
                    "warning" => format!("⚠️ {}: type={}, severity={}, affected={} layers, action=\"{}\"", 
                        k, anomaly.anomaly_type, anomaly.severity, anomaly.affected_layers.len(),
                        anomaly.recommended_action).yellow(),
                    _ => format!("✅ {}: type={}, severity={}, action=\"{}\"", 
                        k, anomaly.anomaly_type, anomaly.severity,
                        anomaly.recommended_action).green(),
                };
                color
            }
            DiffResult::GradientAnalysis(k, gradient) => {
                let color = match gradient.gradient_flow_health.as_str() {
                    "exploding" => format!("💥 {}: flow_health={}, norm={:.6}, problematic={} layers", 
                        k, gradient.gradient_flow_health, gradient.gradient_norm_estimate,
                        gradient.problematic_layers.len()).bright_red(),
                    "dead" | "diminishing" => format!("☠️ {}: flow_health={}, norm={:.6}, problematic={} layers", 
                        k, gradient.gradient_flow_health, gradient.gradient_norm_estimate,
                        gradient.problematic_layers.len()).bright_red(),
                    _ => format!("🌊 {}: flow_health={}, norm={:.6}, ratio={:.4}", 
                        k, gradient.gradient_flow_health, gradient.gradient_norm_estimate,
                        gradient.gradient_ratio).bright_cyan(),
                };
                color
            }
            DiffResult::MemoryAnalysis(k, memory) => {
                let delta_mb = memory.memory_delta_bytes as f64 / (1024.0 * 1024.0);
                let color = if memory.memory_delta_bytes > 100_000_000 {  // > 100MB increase
                    format!("🧠 {}: delta={:+.1}MB, gpu_est={:.1}MB, efficiency={:.6}, \"{}\"", 
                        k, delta_mb, memory.estimated_gpu_memory_mb, memory.memory_efficiency_ratio,
                        memory.memory_recommendation).yellow()
                } else if memory.memory_delta_bytes < -50_000_000 {  // > 50MB decrease
                    format!("🧠 {}: delta={:+.1}MB, gpu_est={:.1}MB, efficiency={:.6}, \"{}\"", 
                        k, delta_mb, memory.estimated_gpu_memory_mb, memory.memory_efficiency_ratio,
                        memory.memory_recommendation).green()
                } else {
                    format!("🧠 {}: delta={:+.1}MB, gpu_est={:.1}MB, efficiency={:.6}, \"{}\"", 
                        k, delta_mb, memory.estimated_gpu_memory_mb, memory.memory_efficiency_ratio,
                        memory.memory_recommendation).bright_blue()
                };
                color
            }
            DiffResult::InferenceSpeedAnalysis(k, speed) => {
                let flops_ratio = if speed.model1_flops_estimate > 0 {
                    speed.model2_flops_estimate as f64 / speed.model1_flops_estimate as f64
                } else {
                    1.0
                };
                let color = if speed.speed_change_ratio > 1.5 {
                    format!("⚡ {}: speed_ratio={:.2}x, flops_ratio={:.2}x, bottlenecks={}, \"{}\"", 
                        k, speed.speed_change_ratio, flops_ratio, speed.bottleneck_layers.len(),
                        speed.inference_recommendation).red()
                } else if speed.speed_change_ratio < 0.7 {
                    format!("⚡ {}: speed_ratio={:.2}x, flops_ratio={:.2}x, bottlenecks={}, \"{}\"", 
                        k, speed.speed_change_ratio, flops_ratio, speed.bottleneck_layers.len(),
                        speed.inference_recommendation).bright_green()
                } else {
                    format!("⚡ {}: speed_ratio={:.2}x, flops_ratio={:.2}x, bottlenecks={}, \"{}\"", 
                        k, speed.speed_change_ratio, flops_ratio, speed.bottleneck_layers.len(),
                        speed.inference_recommendation).bright_cyan()
                };
                color
            }
            DiffResult::RegressionTest(k, regression) => {
                let color = if regression.test_passed {
                    format!("✅ {}: passed={}, degradation={:.1}%, severity={}, \"{}\"", 
                        k, regression.test_passed, regression.performance_degradation,
                        regression.severity_level, regression.recommended_action).green()
                } else {
                    match regression.severity_level.as_str() {
                        "critical" => format!("🚨 {}: passed={}, degradation={:.1}%, severity={}, failed={} checks, \"{}\"", 
                            k, regression.test_passed, regression.performance_degradation,
                            regression.severity_level, regression.failed_checks.len(),
                            regression.recommended_action).bright_red(),
                        "high" => format!("⚠️ {}: passed={}, degradation={:.1}%, severity={}, failed={} checks, \"{}\"", 
                            k, regression.test_passed, regression.performance_degradation,
                            regression.severity_level, regression.failed_checks.len(),
                            regression.recommended_action).red(),
                        _ => format!("⚠️ {}: passed={}, degradation={:.1}%, severity={}, failed={} checks, \"{}\"", 
                            k, regression.test_passed, regression.performance_degradation,
                            regression.severity_level, regression.failed_checks.len(),
                            regression.recommended_action).yellow(),
                    }
                };
                color
            }
            DiffResult::AlertOnDegradation(k, alert) => {
                let color = if alert.alert_triggered {
                    match alert.alert_type.as_str() {
                        "memory" | "performance" => format!("🚨 {}: triggered={}, type={}, threshold_exceeded={:.2}x, \"{}\"", 
                            k, alert.alert_triggered, alert.alert_type, alert.threshold_exceeded,
                            alert.alert_message).bright_red(),
                        "stability" => format!("⚠️ {}: triggered={}, type={}, threshold_exceeded={:.2}x, \"{}\"", 
                            k, alert.alert_triggered, alert.alert_type, alert.threshold_exceeded,
                            alert.alert_message).yellow(),
                        _ => format!("ℹ️ {}: triggered={}, type={}, \"{}\"", 
                            k, alert.alert_triggered, alert.alert_type, alert.alert_message).blue(),
                    }
                } else {
                    format!("✅ {}: triggered={}, \"{}\"", 
                        k, alert.alert_triggered, alert.alert_message).green()
                };
                color
            }
            DiffResult::ReviewFriendly(k, review) => {
                format!("👥 {}: impact={}, approval={}, key_changes={}, \"{}\"", 
                    k, review.impact_assessment, review.approval_recommendation,
                    review.key_changes.len(), review.summary).bright_cyan()
            }
            DiffResult::ChangeSummary(k, summary) => {
                format!("📝 {}: layers_changed={}, magnitude={:.4}, patterns={}, most_changed={}", 
                    k, summary.total_layers_changed, summary.overall_change_magnitude,
                    summary.change_patterns.len(), summary.most_changed_layers.len()).bright_blue()
            }
            DiffResult::RiskAssessment(k, risk) => {
                let color = match risk.overall_risk_level.as_str() {
                    "critical" => format!("🚨 {}: risk={}, readiness={}, factors={}, rollback={}", 
                        k, risk.overall_risk_level, risk.deployment_readiness, 
                        risk.risk_factors.len(), risk.rollback_difficulty).bright_red(),
                    "high" => format!("⚠️ {}: risk={}, readiness={}, factors={}, rollback={}", 
                        k, risk.overall_risk_level, risk.deployment_readiness, 
                        risk.risk_factors.len(), risk.rollback_difficulty).yellow(),
                    _ => format!("✅ {}: risk={}, readiness={}, factors={}, rollback={}", 
                        k, risk.overall_risk_level, risk.deployment_readiness, 
                        risk.risk_factors.len(), risk.rollback_difficulty).green(),
                };
                color
            }
            DiffResult::ArchitectureComparison(k, arch) => {
                format!("🏗️ {}: type1={}, type2={}, depth={}→{}, differences={}, \"{}\"", 
                    k, arch.architecture_type_1, arch.architecture_type_2,
                    arch.layer_depth_comparison.0, arch.layer_depth_comparison.1,
                    arch.structural_differences.len(), arch.comparison_summary).bright_magenta()
            }
            DiffResult::ParamEfficiencyAnalysis(k, efficiency) => {
                format!("⚡ {}: efficiency_ratio={:.4}, params_per_layer={}→{}, sparse={}, dense={}, \"{}\"", 
                    k, efficiency.efficiency_ratio, efficiency.params_per_layer_1, 
                    efficiency.params_per_layer_2, efficiency.sparse_layers.len(),
                    efficiency.dense_layers.len(), efficiency.efficiency_recommendation).bright_yellow()
            }
            DiffResult::HyperparameterImpact(k, hyper) => {
                format!("🎛️ {}: changes={}, high_impact={}, stability={}, max_impact={:.4}", 
                    k, hyper.detected_changes.len(), hyper.high_impact_params.len(),
                    hyper.stability_assessment, 
                    hyper.impact_scores.values().copied().fold(0.0_f64, f64::max)).bright_cyan()
            }
            DiffResult::LearningRateAnalysis(k, lr) => {
                format!("📈 {}: lr={}→{}, pattern={}, effectiveness={:.4}, \"{}\"", 
                    k, lr.estimated_lr_1, lr.estimated_lr_2, lr.lr_schedule_pattern,
                    lr.lr_effectiveness, lr.lr_recommendation).bright_green()
            }
            DiffResult::DeploymentReadiness(k, deploy) => {
                let color = match deploy.deployment_strategy.as_str() {
                    "hold" => format!("🛑 {}: readiness={:.2}, strategy={}, risk={}, blockers={}, \"{}\"", 
                        k, deploy.readiness_score, deploy.deployment_strategy, 
                        deploy.estimated_risk_level, deploy.blocking_issues.len(),
                        deploy.go_live_recommendation).red(),
                    "gradual" => format!("⚠️ {}: readiness={:.2}, strategy={}, risk={}, warnings={}, \"{}\"", 
                        k, deploy.readiness_score, deploy.deployment_strategy, 
                        deploy.estimated_risk_level, deploy.warnings.len(),
                        deploy.go_live_recommendation).yellow(),
                    _ => format!("✅ {}: readiness={:.2}, strategy={}, risk={}, \"{}\"", 
                        k, deploy.readiness_score, deploy.deployment_strategy, 
                        deploy.estimated_risk_level, deploy.go_live_recommendation).green(),
                };
                color
            }
            DiffResult::PerformanceImpactEstimate(k, perf) => {
                format!("🚀 {}: latency={:.2}x, throughput={:.2}x, memory={:.2}x, score={:.4}, \"{}\"", 
                    k, perf.latency_impact_estimate, perf.throughput_impact_estimate,
                    perf.memory_impact_estimate, perf.overall_performance_score,
                    perf.performance_recommendation).bright_purple()
            }
            DiffResult::GenerateReport(k, report) => {
                format!("📄 {}: title=\"{}\", findings={}, conclusions={}, next_steps={}", 
                    k, report.experiment_title, report.key_findings.len(),
                    report.conclusions.len(), report.next_steps.len()).bright_blue()
            }
            DiffResult::MarkdownOutput(k, markdown) => {
                format!("📋 {}: sections={}, tables={}, charts={}, length={} chars", 
                    k, markdown.sections.len(), markdown.tables_generated,
                    markdown.charts_referenced, markdown.markdown_content.len()).blue()
            }
            DiffResult::IncludeCharts(k, chart) => {
                format!("📊 {}: type={}, data_points={}, title=\"{}\", \"{}\"", 
                    k, chart.chart_type, chart.chart_data.len(), 
                    chart.chart_title, chart.chart_description).cyan()
            }
            DiffResult::EmbeddingAnalysis(k, embed) => {
                format!("🧬 {}: layers={}, semantic_drift={:.4}, affected_vocab={}, \"{}\"", 
                    k, embed.embedding_layers.len(), embed.semantic_drift,
                    embed.affected_vocabularies.len(), embed.embedding_recommendation).purple()
            }
            DiffResult::SimilarityMatrix(k, sim) => {
                format!("🔗 {}: matrix_size={}x{}, avg_similarity={:.4}, clusters_est={}, outliers={}", 
                    k, sim.matrix_size.0, sim.matrix_size.1, sim.average_similarity,
                    sim.cluster_count_estimate, sim.outlier_pairs.len()).bright_purple()
            }
            DiffResult::ClusteringChange(k, cluster) => {
                format!("🎯 {}: clusters={}→{}, stability={:.4}, migrated={}, new={}, dissolved={}, \"{}\"", 
                    k, cluster.clusters_before, cluster.clusters_after, cluster.cluster_stability,
                    cluster.migrated_entities.len(), cluster.new_clusters.len(),
                    cluster.dissolved_clusters.len(), cluster.clustering_recommendation).magenta()
            }
            DiffResult::AttentionAnalysis(k, attention) => {
                format!("👁️ {}: layers={}, pattern_changes={}, focus_shift={}, \"{}\"", 
                    k, attention.attention_layers.len(), attention.attention_pattern_changes.len(),
                    attention.attention_focus_shift, attention.attention_recommendation).bright_red()
            }
            DiffResult::HeadImportance(k, head) => {
                format!("🎭 {}: important_heads={}, prunable_heads={}, specializations={}", 
                    k, head.most_important_heads.len(), head.prunable_heads.len(),
                    head.head_specialization.len()).red()
            }
            DiffResult::AttentionPatternDiff(k, pattern) => {
                format!("🔍 {}: pattern={}→{}, similarity={:.4}, span_change={:.4}, \"{}\"", 
                    k, pattern.pattern_type_1, pattern.pattern_type_2, pattern.pattern_similarity,
                    pattern.attention_span_change, pattern.pattern_change_summary).bright_cyan()
            }
        };

        println!("{}{}", indent, diff_str);
    }
}

fn print_json_output(differences: Vec<DiffResult>) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&differences)?);
    Ok(())
}

fn print_yaml_output(differences: Vec<DiffResult>) -> Result<()> {
    // Convert DiffResult to a more standard YAML format
    let yaml_data: Vec<serde_json::Value> = differences.into_iter().map(|diff| {
        match diff {
            DiffResult::Added(key, value) => serde_json::json!({
                "Added": [key, value]
            }),
            DiffResult::Removed(key, value) => serde_json::json!({
                "Removed": [key, value]
            }),
            DiffResult::Modified(key, old_value, new_value) => serde_json::json!({
                "Modified": [key, old_value, new_value]
            }),
            DiffResult::TypeChanged(key, old_value, new_value) => serde_json::json!({
                "TypeChanged": [key, old_value, new_value]
            }),
            DiffResult::TensorShapeChanged(key, shape1, shape2) => serde_json::json!({
                "TensorShapeChanged": [key, shape1, shape2]
            }),
            DiffResult::TensorStatsChanged(key, stats1, stats2) => serde_json::json!({
                "TensorStatsChanged": [key, stats1, stats2]
            }),
            DiffResult::ModelArchitectureChanged(key, info1, info2) => serde_json::json!({
                "ModelArchitectureChanged": [key, info1, info2]
            }),
            DiffResult::LearningProgress(key, progress) => serde_json::json!({
                "LearningProgress": [key, progress]
            }),
            DiffResult::ConvergenceAnalysis(key, convergence) => serde_json::json!({
                "ConvergenceAnalysis": [key, convergence]
            }),
            DiffResult::AnomalyDetection(key, anomaly) => serde_json::json!({
                "AnomalyDetection": [key, anomaly]
            }),
            DiffResult::GradientAnalysis(key, gradient) => serde_json::json!({
                "GradientAnalysis": [key, gradient]
            }),
            DiffResult::MemoryAnalysis(key, memory) => serde_json::json!({
                "MemoryAnalysis": [key, memory]
            }),
            DiffResult::InferenceSpeedAnalysis(key, speed) => serde_json::json!({
                "InferenceSpeedAnalysis": [key, speed]
            }),
            DiffResult::RegressionTest(key, regression) => serde_json::json!({
                "RegressionTest": [key, regression]
            }),
            DiffResult::AlertOnDegradation(key, alert) => serde_json::json!({
                "AlertOnDegradation": [key, alert]
            }),
            DiffResult::ReviewFriendly(key, review) => serde_json::json!({
                "ReviewFriendly": [key, review]
            }),
            DiffResult::ChangeSummary(key, summary) => serde_json::json!({
                "ChangeSummary": [key, summary]
            }),
            DiffResult::RiskAssessment(key, risk) => serde_json::json!({
                "RiskAssessment": [key, risk]
            }),
            DiffResult::ArchitectureComparison(key, arch) => serde_json::json!({
                "ArchitectureComparison": [key, arch]
            }),
            DiffResult::ParamEfficiencyAnalysis(key, efficiency) => serde_json::json!({
                "ParamEfficiencyAnalysis": [key, efficiency]
            }),
            DiffResult::HyperparameterImpact(key, hyper) => serde_json::json!({
                "HyperparameterImpact": [key, hyper]
            }),
            DiffResult::LearningRateAnalysis(key, lr) => serde_json::json!({
                "LearningRateAnalysis": [key, lr]
            }),
            DiffResult::DeploymentReadiness(key, deploy) => serde_json::json!({
                "DeploymentReadiness": [key, deploy]
            }),
            DiffResult::PerformanceImpactEstimate(key, perf) => serde_json::json!({
                "PerformanceImpactEstimate": [key, perf]
            }),
            DiffResult::GenerateReport(key, report) => serde_json::json!({
                "GenerateReport": [key, report]
            }),
            DiffResult::MarkdownOutput(key, markdown) => serde_json::json!({
                "MarkdownOutput": [key, markdown]
            }),
            DiffResult::IncludeCharts(key, chart) => serde_json::json!({
                "IncludeCharts": [key, chart]
            }),
            DiffResult::EmbeddingAnalysis(key, embed) => serde_json::json!({
                "EmbeddingAnalysis": [key, embed]
            }),
            DiffResult::SimilarityMatrix(key, sim) => serde_json::json!({
                "SimilarityMatrix": [key, sim]
            }),
            DiffResult::ClusteringChange(key, cluster) => serde_json::json!({
                "ClusteringChange": [key, cluster]
            }),
            DiffResult::AttentionAnalysis(key, attention) => serde_json::json!({
                "AttentionAnalysis": [key, attention]
            }),
            DiffResult::HeadImportance(key, head) => serde_json::json!({
                "HeadImportance": [key, head]
            }),
            DiffResult::AttentionPatternDiff(key, pattern) => serde_json::json!({
                "AttentionPatternDiff": [key, pattern]
            }),
        }
    }).collect();
    
    println!("{}", serde_yml::to_string(&yaml_data)?);
    Ok(())
}

fn print_unified_output(v1: &Value, v2: &Value) -> Result<()> {
    let content1_pretty = serde_json::to_string_pretty(v1)?;
    let content2_pretty = serde_json::to_string_pretty(v2)?;

    let diff = similar::TextDiff::from_lines(&content1_pretty, &content2_pretty);

    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            similar::ChangeTag::Delete => "-",
            similar::ChangeTag::Insert => "+",
            similar::ChangeTag::Equal => " ",
        };
        print!("{}{}", sign, change);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_config();

    let output_format = args.output.or(config.output).unwrap_or(OutputFormat::Cli);
    let input_format_from_config = config.format;

    let ignore_keys_regex = if let Some(regex_str) = &args.ignore_keys_regex {
        Some(Regex::new(regex_str).context("Invalid regex for --ignore-keys-regex")?)
    } else {
        None
    };

    let epsilon = args.epsilon;
    let array_id_key = args.array_id_key.as_deref();

    // Handle directory comparison
    if args.recursive {
        if !args.input1.is_dir() || !args.input2.is_dir() {
            bail!("Both inputs must be directories for recursive comparison.");
        }
        compare_directories(&args.input1, &args.input2, args.format.or(input_format_from_config), output_format, args.path, ignore_keys_regex.as_ref(), epsilon, array_id_key)?;
        return Ok(());
    }

    // Handle single file/stdin comparison
    let input_format = if let Some(fmt) = args.format {
        fmt
    } else if let Some(fmt) = input_format_from_config {
        fmt
    } else {
        infer_format_from_path(&args.input1)
            .or_else(|| infer_format_from_path(&args.input2))
            .context("Could not infer format from file extensions. Please specify --format or configure in diffx.toml.")?
    };

    let mut differences = match input_format {
        Format::Safetensors | Format::Pytorch => {
            // Check if any ML-specific options are enabled
            if args.show_layer_impact || args.quantization_analysis || args.stats || 
               args.learning_progress || args.convergence_analysis || 
               args.anomaly_detection || args.gradient_analysis ||
               args.memory_analysis || args.inference_speed_estimate ||
               args.regression_test || args.alert_on_degradation ||
               args.review_friendly || args.change_summary || args.risk_assessment ||
               args.architecture_comparison || args.param_efficiency_analysis ||
               args.hyperparameter_impact || args.learning_rate_analysis ||
               args.deployment_readiness || args.performance_impact_estimate ||
               args.generate_report || args.markdown_output || args.include_charts ||
               args.embedding_analysis || args.similarity_matrix || args.clustering_change ||
               args.attention_analysis || args.head_importance || args.attention_pattern_diff {
                // Use enhanced ML analysis
                diff_ml_models_enhanced(&args.input1, &args.input2, epsilon, 
                                       args.show_layer_impact, args.quantization_analysis, 
                                       args.stats, args.learning_progress, args.convergence_analysis,
                                       args.anomaly_detection, args.gradient_analysis,
                                       args.memory_analysis, args.inference_speed_estimate,
                                       args.regression_test, args.alert_on_degradation,
                                       args.review_friendly, args.change_summary, args.risk_assessment,
                                       args.architecture_comparison, args.param_efficiency_analysis,
                                       args.hyperparameter_impact, args.learning_rate_analysis,
                                       args.deployment_readiness, args.performance_impact_estimate,
                                       args.generate_report, args.markdown_output, args.include_charts,
                                       args.embedding_analysis, args.similarity_matrix, args.clustering_change,
                                       args.attention_analysis, args.head_importance, args.attention_pattern_diff)?
            } else {
                // Use basic ML comparison for backward compatibility
                diff_ml_models(&args.input1, &args.input2, epsilon)?
            }
        }
        _ => {
            // Handle regular structured data files
            let content1 = read_input(&args.input1)?;
            let content2 = read_input(&args.input2)?;
            let v1: Value = parse_content(&content1, input_format)?;
            let v2: Value = parse_content(&content2, input_format)?;
            diff(&v1, &v2, ignore_keys_regex.as_ref(), epsilon, array_id_key)
        }
    };

    if let Some(filter_path) = args.path {
        differences.retain(|d| {
            let key = match d {
                DiffResult::Added(k, _) => k,
                DiffResult::Removed(k, _) => k,
                DiffResult::Modified(k, _, _) => k,
                DiffResult::TypeChanged(k, _, _) => k,
                DiffResult::TensorShapeChanged(k, _, _) => k,
                DiffResult::TensorStatsChanged(k, _, _) => k,
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
            };
            key.starts_with(&filter_path)
        });
    }

    match output_format {
        OutputFormat::Cli => print_cli_output(differences, args.sort_by_change_magnitude),
        OutputFormat::Json => print_json_output(differences)?,
        OutputFormat::Yaml => print_yaml_output(differences)?,
        OutputFormat::Unified => {
            match input_format {
                Format::Safetensors | Format::Pytorch => {
                    bail!("Unified output format is not supported for ML model files")
                }
                _ => {
                    let content1 = read_input(&args.input1)?;
                    let content2 = read_input(&args.input2)?;
                    let v1: Value = parse_content(&content1, input_format)?;
                    let v2: Value = parse_content(&content2, input_format)?;
                    print_unified_output(&v1, &v2)?
                }
            }
        }
    }

    Ok(())
}

fn compare_directories(
    dir1: &Path,
    dir2: &Path,
    format_option: Option<Format>,
    output: OutputFormat,
    filter_path: Option<String>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
) -> Result<()> {
    let mut files1: HashMap<PathBuf, PathBuf> = HashMap::new();
    for entry in WalkDir::new(dir1).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            let relative_path = path.strip_prefix(dir1)?.to_path_buf();
            files1.insert(relative_path, path.to_path_buf());
        }
    }

    let mut files2: HashMap<PathBuf, PathBuf> = HashMap::new();
    for entry in WalkDir::new(dir2).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            let relative_path = path.strip_prefix(dir2)?.to_path_buf();
            files2.insert(relative_path, path.to_path_buf());
        }
    }

    let mut all_relative_paths: std::collections::HashSet<PathBuf> = files1.keys().cloned().collect();
    all_relative_paths.extend(files2.keys().cloned());

    let mut compared_files = 0;

    for relative_path in &all_relative_paths {
        let path1_option = files1.get(relative_path.as_path());
        let path2_option = files2.get(relative_path.as_path());

        match (path1_option, path2_option) {
            (Some(path1), Some(path2)) => {
                println!("
--- Comparing {} ---", relative_path.display());
                let content1 = read_input(path1)?;
                let content2 = read_input(path2)?;

                let input_format = if let Some(fmt) = format_option {
                    fmt
                } else {
                    infer_format_from_path(path1)
                        .or_else(|| infer_format_from_path(path2))
                        .context(format!("Could not infer format for {}. Please specify --format or configure in diffx.toml.", relative_path.display()))?
                };

                let v1: Value = parse_content(&content1, input_format)?;
                let v2: Value = parse_content(&content2, input_format)?;

                let mut differences = diff(&v1, &v2, ignore_keys_regex, epsilon, array_id_key);

                if let Some(filter_path_str) = &filter_path {
                    differences.retain(|d| {
                        let key = match d {
                            DiffResult::Added(k, _) => k,
                            DiffResult::Removed(k, _) => k,
                            DiffResult::Modified(k, _, _) => k,
                            DiffResult::TypeChanged(k, _, _) => k,
                            DiffResult::TensorShapeChanged(k, _, _) => k,
                            DiffResult::TensorStatsChanged(k, _, _) => k,
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
                        };
                        key.starts_with(filter_path_str)
                    });
                }

                match output {
                    OutputFormat::Cli => print_cli_output(differences, false), // No magnitude sort for directory comparison
                    OutputFormat::Json => print_json_output(differences)?,
                    OutputFormat::Yaml => print_yaml_output(differences)?,
                    OutputFormat::Unified => print_unified_output(&v1, &v2)?,
                }
                compared_files += 1;
            },
            (Some(_), None) => {
                println!("
--- Only in {}: {} ---", dir1.display(), relative_path.display());
            },
            (None, Some(_)) => {
                println!("
--- Only in {}: {} ---", dir2.display(), relative_path.display());
            },
            (None, None) => { /* Should not happen */ }
        }
    }

    if compared_files == 0 && all_relative_paths.is_empty() {
        println!("No comparable files found in directories.");
    }

    Ok(())
}
