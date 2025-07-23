#![allow(clippy::uninlined_format_args)]

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use colored::*;
use diffai_core::{diff, diff_ml_models_enhanced, parse_csv, parse_ini, parse_xml, DiffResult};
use diffx_core::value_type_name;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;
use walkdir::WalkDir;

/// Color helper functions to support --no-color option
mod color_utils {
    use colored::*;

    pub fn apply_color<T: Colorize>(text: T, no_color: bool) -> ColoredString {
        if no_color {
            text.clear()
        } else {
            text.normal()
        }
    }

    pub fn blue(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.blue()
        }
    }

    pub fn yellow(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.yellow()
        }
    }

    pub fn cyan(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.cyan()
        }
    }

    pub fn magenta(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.magenta()
        }
    }

    pub fn green(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.green()
        }
    }

    pub fn red(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.red()
        }
    }

    pub fn bright_red(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_red()
        }
    }

    pub fn bright_green(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_green()
        }
    }

    pub fn bright_blue(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_blue()
        }
    }

    pub fn bright_cyan(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_cyan()
        }
    }

    pub fn bright_yellow(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_yellow()
        }
    }

    pub fn bright_magenta(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_magenta()
        }
    }

    pub fn bright_purple(text: &str, no_color: bool) -> ColoredString {
        if no_color {
            text.normal()
        } else {
            text.bright_purple()
        }
    }
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

    /// Show verbose processing information
    #[arg(short, long)]
    verbose: bool,

    /// Disable colored output
    #[arg(long)]
    no_color: bool,
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
    Numpy,
    Npz,
    Matlab,
}

fn infer_format_from_path(path: &Path) -> Option<Format> {
    if path.to_str() == Some("-") {
        // Cannot infer format from stdin, user must specify --format
        None
    } else {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext_str| match ext_str.to_lowercase().as_str() {
                "json" => Some(Format::Json),
                "yaml" | "yml" => Some(Format::Yaml),
                "toml" => Some(Format::Toml),
                "ini" => Some(Format::Ini),
                "xml" => Some(Format::Xml),
                "csv" => Some(Format::Csv),
                "safetensors" => Some(Format::Safetensors),
                "pt" | "pth" => Some(Format::Pytorch),
                "npy" => Some(Format::Numpy),
                "npz" => Some(Format::Npz),
                "mat" => Some(Format::Matlab),
                _ => None,
            })
    }
}

fn read_input(file_path: &Path) -> Result<String> {
    if file_path.to_str() == Some("-") {
        let mut buffer = String::new();
        io::stdin()
            .read_to_string(&mut buffer)
            .context("Failed to read from stdin")?;
        Ok(buffer)
    } else {
        fs::read_to_string(file_path)
            .context(format!("Failed to read file: {}", file_path.display()))
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
        Format::Safetensors | Format::Pytorch | Format::Numpy | Format::Npz | Format::Matlab => {
            bail!("ML/Scientific data formats (safetensors, pytorch, numpy, npz) cannot be parsed as text. Use the model/array comparison feature instead.")
        }
    }
}

fn print_cli_output(mut differences: Vec<DiffResult>, sort_by_magnitude: bool, no_color: bool) {
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
            DiffResult::TensorAdded(k, _) => k.clone(),
            DiffResult::TensorRemoved(k, _) => k.clone(),
            DiffResult::QuantizationAnalysis(k, _) => k.clone(),
            DiffResult::TransferLearningAnalysis(k, _) => k.clone(),
            DiffResult::ExperimentReproducibility(k, _) => k.clone(),
            DiffResult::EnsembleAnalysis(k, _) => k.clone(),
            DiffResult::HyperparameterComparison(k, _) => k.clone(),
            DiffResult::LearningCurveAnalysis(k, _) => k.clone(),
            DiffResult::StatisticalSignificance(k, _) => k.clone(),
            DiffResult::NumpyArrayChanged(k, _, _) => k.clone(),
            DiffResult::NumpyArrayAdded(k, _) => k.clone(),
            DiffResult::NumpyArrayRemoved(k, _) => k.clone(),
            DiffResult::MatlabArrayChanged(k, _, _) => k.clone(),
            DiffResult::MatlabArrayAdded(k, _) => k.clone(),
            DiffResult::MatlabArrayRemoved(k, _) => k.clone(),
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
            DiffResult::NumpyArrayChanged(_, stats1, stats2) => {
                // Calculate magnitude of change in NumPy array statistics
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
                (info1.total_params as f64 - info2.total_params as f64).abs()
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
                (arch.layer_depth_comparison.0 as f64 - arch.layer_depth_comparison.1 as f64).abs()
            }
            DiffResult::ParamEfficiencyAnalysis(_, efficiency) => {
                // Use efficiency ratio distance from 1.0 as magnitude
                (efficiency.efficiency_ratio - 1.0).abs()
            }
            DiffResult::HyperparameterImpact(_, hyper) => {
                // Use maximum hyperparameter sensitivity as magnitude
                hyper
                    .hyperparameter_sensitivity
                    .values()
                    .copied()
                    .fold(0.0_f64, f64::max)
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
                // Use impact confidence as magnitude
                perf.impact_confidence
            }
            DiffResult::GenerateReport(_, _) => 1.0, // Static magnitude for reports
            DiffResult::MarkdownOutput(_, _) => 1.0, // Static magnitude for markdown
            DiffResult::IncludeCharts(_, chart) => {
                // Use data points count as magnitude
                chart.data_points as f64 / 100.0
            }
            DiffResult::EmbeddingAnalysis(_, embed) => {
                // Use semantic drift as magnitude
                embed.semantic_drift
            }
            DiffResult::SimilarityMatrix(_, sim) => {
                // Use average similarity as magnitude
                sim.clustering_coefficient
            }
            DiffResult::ClusteringChange(_, cluster) => {
                // Use cluster stability (inverted) as magnitude
                1.0 - cluster.cluster_stability
            }
            DiffResult::AttentionAnalysis(_, attention) => {
                // Use attention entropy as magnitude
                attention.attention_entropy
            }
            DiffResult::HeadImportance(_, head) => {
                // Use maximum head importance as magnitude
                head.head_rankings
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(0.0_f64, f64::max)
            }
            DiffResult::AttentionPatternDiff(_, pattern) => {
                // Use pattern similarity (inverted) as magnitude
                1.0 - pattern.pattern_similarity
            }
            DiffResult::TensorAdded(_, stats) => {
                // Use tensor parameter count as magnitude (normalized)
                (stats.total_params as f64).log10().max(0.0)
            }
            DiffResult::TensorRemoved(_, stats) => {
                // Use tensor parameter count as magnitude (normalized)
                (stats.total_params as f64).log10().max(0.0)
            }
            DiffResult::QuantizationAnalysis(_, quant) => {
                // Use compression ratio as magnitude
                quant.compression_ratio
            }
            DiffResult::TransferLearningAnalysis(_, transfer) => {
                // Use parameter update ratio as magnitude
                transfer.parameter_update_ratio
            }
            DiffResult::ExperimentReproducibility(_, experiment) => {
                // Use inverse of reproducibility score as magnitude (higher = more concerning)
                1.0 - experiment.reproducibility_score
            }
            DiffResult::EnsembleAnalysis(_, ensemble) => {
                // Use diversity score as magnitude
                ensemble.diversity_score
            }
            DiffResult::HyperparameterComparison(_, hyper) => {
                // Use convergence impact as magnitude
                hyper.convergence_impact
            }
            DiffResult::LearningCurveAnalysis(_, curve) => {
                // Use overfitting risk as magnitude
                curve.overfitting_risk
            }
            DiffResult::StatisticalSignificance(_, stats) => {
                // Use effect size as magnitude
                stats.effect_size
            }
            _ => 0.0, // Non-ML changes have no magnitude
        }
    };

    if sort_by_magnitude {
        differences.sort_by(|a, b| {
            get_change_magnitude(b)
                .partial_cmp(&get_change_magnitude(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        differences.sort_by_key(&get_key);
    }

    for diff in &differences {
        let key = get_key(diff);
        // Indent based on the depth of the key
        let depth = key.chars().filter(|&c| c == '.' || c == '[').count();
        let indent = "  ".repeat(depth);

        let diff_str = match diff {
            DiffResult::Added(k, value) => color_utils::blue(&format!("+ {}: {}", k, value), no_color),
            DiffResult::Removed(k, value) => color_utils::yellow(&format!("- {}: {}", k, value), no_color),
            DiffResult::Modified(k, v1, v2) => color_utils::cyan(&format!("~ {}: {} -> {}", k, v1, v2), no_color),
            DiffResult::TypeChanged(k, v1, v2) => {
                color_utils::magenta(&format!("! {}: {} ({}) -> {} ({})", k, v1, value_type_name(v1), v2, value_type_name(v2)), no_color)
            }
            DiffResult::TensorShapeChanged(k, shape1, shape2) => {
                color_utils::magenta(&format!("! {}: {:?} -> {:?} (shape)", k, shape1, shape2), no_color)
            }
            DiffResult::TensorStatsChanged(k, stats1, stats2) => {
                color_utils::cyan(&format!("~ {}: mean={:.4}->{:.4}, std={:.4}->{:.4}",
                    k, stats1.mean, stats2.mean, stats1.std, stats2.std), no_color)
            }
            DiffResult::NumpyArrayChanged(k, stats1, stats2) => {
                color_utils::cyan(&format!("~ {}: shape={:?}, mean={:.4}->{:.4}, std={:.4}->{:.4}, dtype={}",
                    k, stats1.shape, stats1.mean, stats2.mean, stats1.std, stats2.std, stats1.dtype), no_color)
            }
            DiffResult::NumpyArrayAdded(k, stats) => {
                color_utils::green(&format!("+ {}: shape={:?}, dtype={}, elements={}, size={}MB",
                    k, stats.shape, stats.dtype, stats.total_elements,
                    stats.memory_size_bytes as f64 / 1024.0 / 1024.0), no_color)
            }
            DiffResult::NumpyArrayRemoved(k, stats) => {
                color_utils::red(&format!("- {}: shape={:?}, dtype={}, elements={}, size={}MB",
                    k, stats.shape, stats.dtype, stats.total_elements,
                    stats.memory_size_bytes as f64 / 1024.0 / 1024.0), no_color)
            }
            DiffResult::MatlabArrayChanged(k, stats1, stats2) => {
                let complex_info = if stats1.is_complex || stats2.is_complex { " (complex)" } else { "" };
                format!("~ {}: var={}, shape={:?}, mean={:.4}->{:.4}, std={:.4}->{:.4}, dtype={}{}",
                    k, stats1.variable_name, stats1.shape, stats1.mean, stats2.mean,
                    stats1.std, stats2.std, stats1.dtype, complex_info).cyan()
            }
            DiffResult::MatlabArrayAdded(k, stats) => {
                let complex_info = if stats.is_complex { " (complex)" } else { "" };
                format!("+ {}: var={}, shape={:?}, dtype={}, elements={}, size={}MB{}",
                    k, stats.variable_name, stats.shape, stats.dtype, stats.total_elements,
                    stats.memory_size_bytes as f64 / 1024.0 / 1024.0, complex_info).green()
            }
            DiffResult::MatlabArrayRemoved(k, stats) => {
                let complex_info = if stats.is_complex { " (complex)" } else { "" };
                format!("- {}: var={}, shape={:?}, dtype={}, elements={}, size={}MB{}",
                    k, stats.variable_name, stats.shape, stats.dtype, stats.total_elements,
                    stats.memory_size_bytes as f64 / 1024.0 / 1024.0, complex_info).red()
            }
            DiffResult::ModelArchitectureChanged(k, info1, info2) => {
                format!("! {}: params={}->{}, layers={}->{} (architecture)",
                    k, info1.total_params, info2.total_params,
                    info1.layer_count, info2.layer_count).magenta()
            }
            DiffResult::LearningProgress(k, progress) => {
                format!("+ {}: trend={}, magnitude={:.4}, speed={:.2} (learning_progress)",
                    k, progress.loss_trend, progress.parameter_update_magnitude,
                    progress.convergence_speed).blue()
            }
            DiffResult::ConvergenceAnalysis(k, convergence) => {
                format!("+ {}: status={}, stability={:.4} (convergence)",
                    k, convergence.convergence_status, convergence.parameter_stability).blue()
            }
            DiffResult::AnomalyDetection(k, anomaly) => {
                let regression_test_result = if anomaly.severity == "none" || anomaly.severity == "low" {
                    "passed"
                } else {
                    "required"
                };
                let color = match anomaly.severity.as_str() {
                    "critical" => format!("[CRITICAL] {}: type={}, severity={}, affected={} layers, action=\"{}\", regression_test={}",
                        k, anomaly.anomaly_type, anomaly.severity, anomaly.affected_layers.len(),
                        anomaly.recommended_action, regression_test_result).bright_red(),
                    "warning" => format!("[WARNING] {}: type={}, severity={}, affected={} layers, action=\"{}\", regression_test={}",
                        k, anomaly.anomaly_type, anomaly.severity, anomaly.affected_layers.len(),
                        anomaly.recommended_action, regression_test_result).yellow(),
                    _ => format!("{}: type={}, severity={}, action=\"{}\", regression_test={}",
                        k, anomaly.anomaly_type, anomaly.severity,
                        anomaly.recommended_action, regression_test_result).green(),
                };
                color
            }
            DiffResult::GradientAnalysis(k, gradient) => {
                let color = match gradient.gradient_flow_health.as_str() {
                    "exploding" => format!("{}: flow_health={}, norm={:.6}, problematic={} layers",
                        k, gradient.gradient_flow_health, gradient.gradient_norm_estimate,
                        gradient.problematic_layers.len()).bright_red(),
                    "dead" | "diminishing" => format!("{}: flow_health={}, norm={:.6}, problematic={} layers",
                        k, gradient.gradient_flow_health, gradient.gradient_norm_estimate,
                        gradient.problematic_layers.len()).bright_red(),
                    _ => format!("{}: flow_health={}, norm={:.6}, ratio={:.4}",
                        k, gradient.gradient_flow_health, gradient.gradient_norm_estimate,
                        gradient.gradient_ratio).bright_cyan(),
                };
                color
            }
            DiffResult::MemoryAnalysis(k, memory) => {
                let delta_mb = memory.memory_delta_bytes as f64 / (1024.0 * 1024.0);
                let color = if memory.memory_delta_bytes > 100_000_000 {  // > 100MB increase
                    format!("{}: delta={:+.1}MB, gpu_est={:.1}MB, efficiency={:.6}, review_friendly=\"{}\", \"{}\"",
                        k, delta_mb, memory.estimated_gpu_memory_mb, memory.memory_efficiency_ratio,
                        memory.memory_recommendation, memory.memory_recommendation).yellow()
                } else if memory.memory_delta_bytes < -50_000_000 {  // > 50MB decrease
                    format!("{}: delta={:+.1}MB, gpu_est={:.1}MB, efficiency={:.6}, review_friendly=\"{}\", \"{}\"",
                        k, delta_mb, memory.estimated_gpu_memory_mb, memory.memory_efficiency_ratio,
                        memory.memory_recommendation, memory.memory_recommendation).green()
                } else {
                    format!("{}: delta={:+.1}MB, gpu_est={:.1}MB, efficiency={:.6}, review_friendly=\"{}\", \"{}\"",
                        k, delta_mb, memory.estimated_gpu_memory_mb, memory.memory_efficiency_ratio,
                        memory.memory_recommendation, memory.memory_recommendation).bright_blue()
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
                    format!("{}: speed_ratio={:.2}x, flops_ratio={:.2}x, bottlenecks={}, \"{}\"",
                        k, speed.speed_change_ratio, flops_ratio, speed.bottleneck_layers.len(),
                        speed.inference_recommendation).red()
                } else if speed.speed_change_ratio < 0.7 {
                    format!("{}: speed_ratio={:.2}x, flops_ratio={:.2}x, bottlenecks={}, \"{}\"",
                        k, speed.speed_change_ratio, flops_ratio, speed.bottleneck_layers.len(),
                        speed.inference_recommendation).bright_green()
                } else {
                    format!("{}: speed_ratio={:.2}x, flops_ratio={:.2}x, bottlenecks={}, \"{}\"",
                        k, speed.speed_change_ratio, flops_ratio, speed.bottleneck_layers.len(),
                        speed.inference_recommendation).bright_cyan()
                };
                color
            }
            DiffResult::RegressionTest(k, regression) => {
                let color = if regression.test_passed {
                    format!("{}: passed={}, degradation={:.1}%, severity={}, \"{}\"",
                        k, regression.test_passed, regression.performance_degradation,
                        regression.severity_level, regression.recommended_action).green()
                } else {
                    match regression.severity_level.as_str() {
                        "critical" => format!("[CRITICAL] {}: passed={}, degradation={:.1}%, severity={}, failed={} checks, \"{}\"",
                            k, regression.test_passed, regression.performance_degradation,
                            regression.severity_level, regression.failed_checks.len(),
                            regression.recommended_action).bright_red(),
                        "high" => format!("[HIGH] {}: passed={}, degradation={:.1}%, severity={}, failed={} checks, \"{}\"",
                            k, regression.test_passed, regression.performance_degradation,
                            regression.severity_level, regression.failed_checks.len(),
                            regression.recommended_action).red(),
                        _ => format!("{}: passed={}, degradation={:.1}%, severity={}, failed={} checks, \"{}\"",
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
                        "memory" | "performance" => format!("[ALERT] {}: triggered={}, type={}, threshold_exceeded={:.2}x, \"{}\"",
                            k, alert.alert_triggered, alert.alert_type, alert.threshold_exceeded,
                            alert.alert_message).bright_red(),
                        "stability" => format!("[WARNING] {}: triggered={}, type={}, threshold_exceeded={:.2}x, \"{}\"",
                            k, alert.alert_triggered, alert.alert_type, alert.threshold_exceeded,
                            alert.alert_message).yellow(),
                        _ => format!("{}: triggered={}, type={}, \"{}\"",
                            k, alert.alert_triggered, alert.alert_type, alert.alert_message).blue(),
                    }
                } else {
                    format!("{}: triggered={}, \"{}\"",
                        k, alert.alert_triggered, alert.alert_message).green()
                };
                color
            }
            DiffResult::ReviewFriendly(k, review) => {
                format!("{}: impact={}, approval={}, key_changes={}, \"{}\"",
                    k, review.impact_assessment, review.approval_recommendation,
                    review.key_changes.len(), review.summary).bright_cyan()
            }
            DiffResult::ChangeSummary(k, summary) => {
                format!("{}: layers_changed={}, magnitude={:.4}, patterns={}, most_changed={}",
                    k, summary.total_layers_changed, summary.overall_change_magnitude,
                    summary.change_patterns.len(), summary.most_changed_layers.len()).bright_blue()
            }
            DiffResult::RiskAssessment(k, risk) => {
                let color = match risk.overall_risk_level.as_str() {
                    "critical" => format!("[CRITICAL] ◦ {}: risk={}, readiness={}, factors={}, rollback={}",
                        k, risk.overall_risk_level, risk.deployment_readiness,
                        risk.risk_factors.len(), risk.rollback_difficulty).bright_red(),
                    "high" => format!("[HIGH] ◦ {}: risk={}, readiness={}, factors={}, rollback={}",
                        k, risk.overall_risk_level, risk.deployment_readiness,
                        risk.risk_factors.len(), risk.rollback_difficulty).yellow(),
                    _ => format!("◦ {}: risk={}, readiness={}, factors={}, rollback={}",
                        k, risk.overall_risk_level, risk.deployment_readiness,
                        risk.risk_factors.len(), risk.rollback_difficulty).green(),
                };
                color
            }
            DiffResult::ArchitectureComparison(k, arch) => {
                format!("{}: type1={}, type2={}, depth={}→{}, differences={}, deployment_readiness={}, \"{}\"",
                    k, arch.architecture_type_1, arch.architecture_type_2,
                    arch.layer_depth_comparison.0, arch.layer_depth_comparison.1,
                    arch.architectural_differences.len(), arch.deployment_readiness, arch.recommendation).bright_magenta()
            }
            DiffResult::ParamEfficiencyAnalysis(k, efficiency) => {
                format!("◦ {}: efficiency_ratio={:.4}, utilization={:.2}, pruning_potential={:.2}, category={}, bottlenecks={}, \"{}\"",
                    k, efficiency.efficiency_ratio, efficiency.parameter_utilization,
                    efficiency.pruning_potential, efficiency.efficiency_category,
                    efficiency.efficiency_bottlenecks.len(), efficiency.model_scaling_recommendation).bright_yellow()
            }
            DiffResult::HyperparameterImpact(k, hyper) => {
                format!("◦ {}: lr_impact={:.4}, batch_impact={:.4}, convergence={:.4}, performance={:.4}",
                    k, hyper.learning_rate_impact, hyper.batch_size_impact,
                    hyper.convergence_impact, hyper.performance_prediction).bright_cyan()
            }
            DiffResult::LearningRateAnalysis(k, lr) => {
                format!("◦ {}: current_lr={:.6}, schedule={}, effectiveness={:.4}, stability_impact={:.4}, \"{}\"",
                    k, lr.current_lr, lr.lr_schedule_type,
                    lr.lr_effectiveness, lr.stability_impact, lr.schedule_optimization).bright_green()
            }
            DiffResult::DeploymentReadiness(k, deploy) => {
                let color = match deploy.deployment_strategy.as_str() {
                    "hold" => format!("[HOLD] ◦ {}: readiness={:.2}, strategy={}, risk={}, blockers={}, rollback={}",
                        k, deploy.readiness_score, deploy.deployment_strategy,
                        deploy.risk_level, deploy.deployment_blockers.len(),
                        deploy.rollback_plan_quality).red(),
                    "gradual" => format!("[GRADUAL] ◦ {}: readiness={:.2}, strategy={}, risk={}, prerequisites={}, scalability={}",
                        k, deploy.readiness_score, deploy.deployment_strategy,
                        deploy.risk_level, deploy.prerequisites.len(),
                        deploy.scalability_assessment).yellow(),
                    _ => format!("◦ {}: readiness={:.2}, strategy={}, risk={}, timeline={}",
                        k, deploy.readiness_score, deploy.deployment_strategy,
                        deploy.risk_level, deploy.deployment_timeline).green(),
                };
                color
            }
            DiffResult::PerformanceImpactEstimate(k, perf) => {
                format!("◦ {}: latency_change={:.2}%, throughput_change={:.2}%, memory_change={:.2}%, category={}, confidence={:.4}",
                    k, perf.latency_change_estimate, perf.throughput_change_estimate,
                    perf.memory_usage_change, perf.performance_category,
                    perf.impact_confidence).bright_purple()
            }
            DiffResult::GenerateReport(k, report) => {
                format!("◦ {}: type=\"{}\", findings={}, recommendations={}, confidence={:.2}",
                    k, report.report_type, report.key_findings.len(),
                    report.recommendations.len(), report.confidence_level).bright_blue()
            }
            DiffResult::MarkdownOutput(k, markdown) => {
                format!("◦ {}: sections={}, tables={}, charts={}, length={} chars",
                    k, markdown.sections.len(), markdown.tables.len(),
                    markdown.charts.len(), markdown.markdown_content.len()).blue()
            }
            DiffResult::IncludeCharts(k, chart) => {
                format!("◦ {}: types={}, data_points={}, library={}, complexity={}",
                    k, chart.chart_types.len(), chart.data_points,
                    chart.chart_library, chart.chart_complexity).cyan()
            }
            DiffResult::EmbeddingAnalysis(k, embed) => {
                format!("◦ {}: dim_change={}→{}, semantic_drift={:.4}, similarity_preservation={:.4}, clustering_stability={:.4}",
                    k, embed.embedding_dimension_change.0, embed.embedding_dimension_change.1,
                    embed.semantic_drift, embed.similarity_preservation, embed.clustering_stability).purple()
            }
            DiffResult::SimilarityMatrix(k, sim) => {
                format!("◦ {}: matrix_dims={}x{}, clustering_coeff={:.4}, sparsity={:.4}, outliers={}, metric={}",
                    k, sim.matrix_dimensions.0, sim.matrix_dimensions.1, sim.clustering_coefficient,
                    sim.matrix_sparsity, sim.outlier_detection.len(), sim.distance_metric).bright_purple()
            }
            DiffResult::ClusteringChange(k, cluster) => {
                format!("◦ {}: clusters={}→{}, stability={:.4}, migrated={}, new={}, dissolved={}, \"{}\"",
                    k, cluster.cluster_count_change.0, cluster.cluster_count_change.1, cluster.cluster_stability,
                    cluster.cluster_count_change.0, cluster.cluster_count_change.1,
                    cluster.optimal_cluster_count, cluster.clustering_recommendation).magenta()
            }
            DiffResult::AttentionAnalysis(k, attention) => {
                format!("◦ {}: layers={}, pattern_changes={}, focus_shift={}, \"{}\"",
                    k, attention.attention_head_count, attention.attention_pattern_changes.len(),
                    attention.pattern_consistency, attention.pattern_interpretability).bright_red()
            }
            DiffResult::HeadImportance(k, head) => {
                format!("◦ {}: important_heads={}, prunable_heads={}, specializations={}",
                    k, head.critical_heads.len(), head.prunable_heads.len(),
                    head.head_rankings.len()).red()
            }
            DiffResult::AttentionPatternDiff(k, pattern) => {
                format!("◦ {}: pattern={}→{}, similarity={:.4}, span_change={:.4}, \"{}\"",
                    k, pattern.pattern_evolution, pattern.attention_shift_analysis, pattern.pattern_similarity,
                    pattern.attention_focus_changes.len(), pattern.pattern_recommendation).bright_cyan()
            }
            DiffResult::TensorAdded(k, stats) => {
                format!("+ {}: shape={:?}, dtype={}, params={}",
                    k, stats.shape, stats.dtype, stats.total_params).green()
            }
            DiffResult::TensorRemoved(k, stats) => {
                format!("- {}: shape={:?}, dtype={}, params={}",
                    k, stats.shape, stats.dtype, stats.total_params).red()
            }
            DiffResult::QuantizationAnalysis(k, quant) => {
                format!("◦ {}: compression={:.1}%, speedup={:.1}x, precision_loss={:.1}%, suitability={} (quantization)",
                    k, quant.compression_ratio * 100.0, quant.estimated_speedup,
                    quant.precision_loss_estimate * 100.0, quant.deployment_suitability).bright_blue()
            }
            DiffResult::TransferLearningAnalysis(k, transfer) => {
                format!("◦ {}: frozen={}/{}, updated_params={:.1}%, adaptation={}, efficiency={:.2} (transfer_learning)",
                    k, transfer.frozen_layers, transfer.updated_layers,
                    transfer.parameter_update_ratio * 100.0, transfer.domain_adaptation_strength,
                    transfer.transfer_efficiency_score).green()
            }
            DiffResult::ExperimentReproducibility(k, experiment) => {
                let color = match experiment.reproducibility_score {
                    x if x > 0.8 => format!("◦ {}: score={:.2}, critical_changes={}, determinism={} (reproducibility)",
                        k, experiment.reproducibility_score, experiment.critical_changes.len(),
                        experiment.reproducibility_score).green(),
                    x if x > 0.5 => format!("◦ {}: score={:.2}, critical_changes={}, determinism={} (reproducibility)",
                        k, experiment.reproducibility_score, experiment.critical_changes.len(),
                        experiment.reproducibility_score).yellow(),
                    _ => format!("◦ {}: score={:.2}, critical_changes={}, determinism={} (reproducibility)",
                        k, experiment.reproducibility_score, experiment.critical_changes.len(),
                        experiment.reproducibility_score).red(),
                };
                color
            }
            DiffResult::EnsembleAnalysis(k, ensemble) => {
                format!("◦ {}: models={}, diversity={:.2}, efficiency={:.2}x, redundancy={} (ensemble)",
                    k, ensemble.model_count, ensemble.diversity_score,
                    ensemble.ensemble_efficiency, ensemble.optimal_subset.len()).magenta()
            }
            DiffResult::HyperparameterComparison(k, hyper) => {
                let color = match hyper.risk_assessment.as_str() {
                    "high" => format!("◦ {}: changed={}, convergence_impact={:.2}, risk={}, \"{}\" (hyperparameters)",
                        k, hyper.changed_parameters.len(), hyper.convergence_impact,
                        hyper.risk_assessment, hyper.recommendation).red(),
                    "medium" => format!("◦ {}: changed={}, convergence_impact={:.2}, risk={}, \"{}\" (hyperparameters)",
                        k, hyper.changed_parameters.len(), hyper.convergence_impact,
                        hyper.risk_assessment, hyper.recommendation).yellow(),
                    _ => format!("◦ {}: changed={}, convergence_impact={:.2}, risk={}, \"{}\" (hyperparameters)",
                        k, hyper.changed_parameters.len(), hyper.convergence_impact,
                        hyper.risk_assessment, hyper.recommendation).green(),
                };
                color
            }
            DiffResult::LearningCurveAnalysis(k, curve) => {
                let color = match curve.trend_analysis.as_str() {
                    "overfitting" => format!("◦ {}: trend={}, convergence={:?}, overfitting_risk={:.2}, efficiency={:.2} (learning_curve)",
                        k, curve.trend_analysis, curve.convergence_point,
                        curve.overfitting_risk, curve.learning_efficiency).red(),
                    "plateauing" => format!("◦ {}: trend={}, convergence={:?}, overfitting_risk={:.2}, efficiency={:.2} (learning_curve)",
                        k, curve.trend_analysis, curve.convergence_point,
                        curve.overfitting_risk, curve.learning_efficiency).yellow(),
                    _ => format!("◦ {}: trend={}, convergence={:?}, overfitting_risk={:.2}, efficiency={:.2} (learning_curve)",
                        k, curve.trend_analysis, curve.convergence_point,
                        curve.overfitting_risk, curve.learning_efficiency).green(),
                };
                color
            }
            DiffResult::StatisticalSignificance(k, stats) => {
                let color = match stats.significance_level.as_str() {
                    "significant" => format!("◦ {}: metric={}, p_value={:.4}, effect_size={:.2}, power={:.2}, \"{}\" (statistical)",
                        k, stats.metric_name, stats.p_value, stats.effect_size,
                        stats.statistical_power, stats.recommendation).green(),
                    "marginal" => format!("◦ {}: metric={}, p_value={:.4}, effect_size={:.2}, power={:.2}, \"{}\" (statistical)",
                        k, stats.metric_name, stats.p_value, stats.effect_size,
                        stats.statistical_power, stats.recommendation).yellow(),
                    _ => format!("◦ {}: metric={}, p_value={:.4}, effect_size={:.2}, power={:.2}, \"{}\" (statistical)",
                        k, stats.metric_name, stats.p_value, stats.effect_size,
                        stats.statistical_power, stats.recommendation).red(),
                };
                color
            }
        };

        println!("{}{}", indent, diff_str);
    }

    // Generate ML-specific recommendations
    generate_ml_recommendations(&differences);
}

fn print_json_output(differences: Vec<DiffResult>) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(&differences)?);
    Ok(())
}

fn print_yaml_output(differences: Vec<DiffResult>) -> Result<()> {
    // Convert DiffResult to a more standard YAML format
    let yaml_data: Vec<serde_json::Value> = differences
        .into_iter()
        .map(|diff| match diff {
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
            DiffResult::TensorAdded(key, stats) => serde_json::json!({
                "TensorAdded": [key, stats]
            }),
            DiffResult::TensorRemoved(key, stats) => serde_json::json!({
                "TensorRemoved": [key, stats]
            }),
            DiffResult::NumpyArrayChanged(key, stats1, stats2) => serde_json::json!({
                "NumpyArrayChanged": [key, stats1, stats2]
            }),
            DiffResult::NumpyArrayAdded(key, stats) => serde_json::json!({
                "NumpyArrayAdded": [key, stats]
            }),
            DiffResult::NumpyArrayRemoved(key, stats) => serde_json::json!({
                "NumpyArrayRemoved": [key, stats]
            }),
            DiffResult::MatlabArrayChanged(key, stats1, stats2) => serde_json::json!({
                "MatlabArrayChanged": [key, stats1, stats2]
            }),
            DiffResult::MatlabArrayAdded(key, stats) => serde_json::json!({
                "MatlabArrayAdded": [key, stats]
            }),
            DiffResult::MatlabArrayRemoved(key, stats) => serde_json::json!({
                "MatlabArrayRemoved": [key, stats]
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
            DiffResult::QuantizationAnalysis(key, quant) => serde_json::json!({
                "QuantizationAnalysis": [key, quant]
            }),
            DiffResult::TransferLearningAnalysis(key, transfer) => serde_json::json!({
                "TransferLearningAnalysis": [key, transfer]
            }),
            DiffResult::ExperimentReproducibility(key, experiment) => serde_json::json!({
                "ExperimentReproducibility": [key, experiment]
            }),
            DiffResult::EnsembleAnalysis(key, ensemble) => serde_json::json!({
                "EnsembleAnalysis": [key, ensemble]
            }),
            DiffResult::HyperparameterComparison(key, hyper) => serde_json::json!({
                "HyperparameterComparison": [key, hyper]
            }),
            DiffResult::LearningCurveAnalysis(key, curve) => serde_json::json!({
                "LearningCurveAnalysis": [key, curve]
            }),
            DiffResult::StatisticalSignificance(key, stats) => serde_json::json!({
                "StatisticalSignificance": [key, stats]
            }),
        })
        .collect();

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

    let output_format = args.output.unwrap_or(OutputFormat::Cli);
    let input_format_from_config = None;

    // Verbose configuration information (will be shown after format detection)
    let show_verbose_config = args.verbose;

    let ignore_keys_regex = if let Some(regex_str) = &args.ignore_keys_regex {
        Some(Regex::new(regex_str).context("Invalid regex for --ignore-keys-regex")?)
    } else {
        None
    };

    let epsilon = args.epsilon;
    let array_id_key = args.array_id_key.as_deref();

    // Start timing measurement for verbose mode
    let start_time = if args.verbose {
        Some(Instant::now())
    } else {
        None
    };

    // Handle directory comparison (automatically detect directories)
    if args.input1.is_dir() || args.input2.is_dir() {
        if !args.input1.is_dir() || !args.input2.is_dir() {
            bail!("Both inputs must be directories when comparing directories.");
        }
        compare_directories(
            &args.input1,
            &args.input2,
            args.format.or(input_format_from_config),
            output_format,
            args.path,
            ignore_keys_regex.as_ref(),
            epsilon,
            array_id_key,
            args.verbose,
            args.recursive, // Pass recursive flag to control depth
            args.no_color,
        )?;
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
            .context("Could not infer format from file extensions. Please specify --format.")?
    };

    // Verbose file analysis information
    if args.verbose {
        eprintln!();
        eprintln!("File analysis:");
        eprintln!("  Input 1: {}", args.input1.display());
        eprintln!("  Input 2: {}", args.input2.display());
        eprintln!("  Detected format: {:?}", input_format);

        // File sizes
        if let Ok(metadata1) = fs::metadata(&args.input1) {
            eprintln!("  File 1 size: {} bytes", metadata1.len());
        }
        if let Ok(metadata2) = fs::metadata(&args.input2) {
            eprintln!("  File 2 size: {} bytes", metadata2.len());
        }
    }

    // Show verbose configuration after format detection
    if show_verbose_config {
        eprintln!("=== diffai verbose mode enabled ===");
        eprintln!("Configuration:");
        eprintln!("  Input format: {:?}", input_format);
        eprintln!("  Output format: {:?}", output_format);
        eprintln!("  Recursive mode: {}", args.recursive);

        // ML analysis features (always enabled for ML models)
        match input_format {
            Format::Safetensors | Format::Pytorch => {
                eprintln!("  ML analysis: Full analysis enabled (all 28 features)");
            }
            Format::Numpy | Format::Npz | Format::Matlab => {
                eprintln!("  Scientific data analysis: Array statistics enabled");
            }
            _ => {
                eprintln!("  Analysis mode: Structured data comparison");
            }
        }

        // Advanced options
        if let Some(epsilon) = args.epsilon {
            eprintln!("  Epsilon tolerance: {}", epsilon);
        }
        if let Some(regex) = &args.ignore_keys_regex {
            eprintln!("  Ignore keys regex: {}", regex);
        }
        if let Some(path) = &args.path {
            eprintln!("  Path filter: {}", path);
        }
    }

    let mut differences = match input_format {
        Format::Numpy | Format::Npz => {
            // Handle NumPy scientific array comparison
            diffai_core::diff_numpy_files(&args.input1, &args.input2)?
        }
        Format::Matlab => {
            // Handle MATLAB .mat file comparison
            diffai_core::diff_matlab_files(&args.input1, &args.input2)?
        }
        Format::Safetensors | Format::Pytorch => {
            // Always use enhanced ML analysis with all features enabled
            diff_ml_models_enhanced(
                &args.input1,
                &args.input2,
                true, // learning_progress
                true, // convergence_analysis
                true, // anomaly_detection
                true, // gradient_analysis
                true, // memory_analysis
                true, // inference_speed_estimate
                true, // regression_test
                true, // alert_on_degradation
                true, // review_friendly
                true, // change_summary
                true, // risk_assessment
                true, // architecture_comparison
                true, // param_efficiency_analysis
                true, // hyperparameter_impact
                true, // learning_rate_analysis
                true, // deployment_readiness
                true, // performance_impact_estimate
                true, // generate_report
                true, // markdown_output
                true, // include_charts
                true, // embedding_analysis
                true, // similarity_matrix
                true, // clustering_change
                true, // attention_analysis
                true, // head_importance
                true, // attention_pattern_diff
                true, // hyperparameter_comparison
                true, // learning_curve_analysis
                true, // statistical_significance
                true, // quantization_analysis
                true, // transfer_learning_analysis
                true, // experiment_reproducibility
                true, // ensemble_analysis
            )?
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
                DiffResult::TensorAdded(k, _) => k,
                DiffResult::TensorRemoved(k, _) => k,
                DiffResult::QuantizationAnalysis(k, _) => k,
                DiffResult::TransferLearningAnalysis(k, _) => k,
                DiffResult::ExperimentReproducibility(k, _) => k,
                DiffResult::EnsembleAnalysis(k, _) => k,
                DiffResult::HyperparameterComparison(k, _) => k,
                DiffResult::LearningCurveAnalysis(k, _) => k,
                DiffResult::StatisticalSignificance(k, _) => k,
                DiffResult::NumpyArrayChanged(k, _, _) => k,
                DiffResult::NumpyArrayAdded(k, _) => k,
                DiffResult::NumpyArrayRemoved(k, _) => k,
                DiffResult::MatlabArrayChanged(k, _, _) => k,
                DiffResult::MatlabArrayAdded(k, _) => k,
                DiffResult::MatlabArrayRemoved(k, _) => k,
            };
            key.starts_with(&filter_path)
        });
    }

    // Verbose processing results
    if args.verbose {
        eprintln!();
        eprintln!("Processing results:");
        if let Some(start) = start_time {
            let elapsed = start.elapsed();
            if elapsed.as_millis() > 0 {
                eprintln!(
                    "  Total processing time: {:.3}ms",
                    elapsed.as_secs_f64() * 1000.0
                );
            } else if elapsed.as_micros() > 0 {
                eprintln!(
                    "  Total processing time: {:.3}µs",
                    elapsed.as_micros() as f64
                );
            } else {
                eprintln!("  Total processing time: {}ns", elapsed.as_nanos());
            }
        }
        eprintln!("  Differences found: {}", differences.len());
        match input_format {
            Format::Safetensors
            | Format::Pytorch
            | Format::Numpy
            | Format::Npz
            | Format::Matlab => {
                eprintln!("  ML/Scientific data analysis completed");
            }
            _ => {
                eprintln!("  Format-specific analysis: {:?}", input_format);
            }
        }
        eprintln!();
    }

    match output_format {
        OutputFormat::Cli => print_cli_output(differences, false, args.no_color),
        OutputFormat::Json => print_json_output(differences)?,
        OutputFormat::Yaml => print_yaml_output(differences)?,
        OutputFormat::Unified => match input_format {
            Format::Safetensors
            | Format::Pytorch
            | Format::Numpy
            | Format::Npz
            | Format::Matlab => {
                bail!("Unified output format is not supported for ML/Scientific data files")
            }
            _ => {
                let content1 = read_input(&args.input1)?;
                let content2 = read_input(&args.input2)?;
                let v1: Value = parse_content(&content1, input_format)?;
                let v2: Value = parse_content(&content2, input_format)?;
                print_unified_output(&v1, &v2)?
            }
        },
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn compare_directories(
    dir1: &Path,
    dir2: &Path,
    format_option: Option<Format>,
    output: OutputFormat,
    filter_path: Option<String>,
    ignore_keys_regex: Option<&Regex>,
    epsilon: Option<f64>,
    array_id_key: Option<&str>,
    _verbose: bool,
    recursive: bool,
    no_color: bool,
) -> Result<()> {
    let mut files1: HashMap<PathBuf, PathBuf> = HashMap::new();

    // Configure WalkDir based on recursive flag (like standard diff)
    let walker1 = if recursive {
        WalkDir::new(dir1) // Recursive traversal
    } else {
        WalkDir::new(dir1).max_depth(1) // Only direct children
    };

    for entry in walker1.into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            let relative_path = path.strip_prefix(dir1)?.to_path_buf();
            files1.insert(relative_path, path.to_path_buf());
        }
    }

    let mut files2: HashMap<PathBuf, PathBuf> = HashMap::new();

    // Configure WalkDir for dir2 with same recursive setting
    let walker2 = if recursive {
        WalkDir::new(dir2) // Recursive traversal
    } else {
        WalkDir::new(dir2).max_depth(1) // Only direct children
    };

    for entry in walker2.into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() {
            let relative_path = path.strip_prefix(dir2)?.to_path_buf();
            files2.insert(relative_path, path.to_path_buf());
        }
    }

    let mut all_relative_paths: std::collections::HashSet<PathBuf> =
        files1.keys().cloned().collect();
    all_relative_paths.extend(files2.keys().cloned());

    let mut compared_files = 0;

    for relative_path in &all_relative_paths {
        let path1_option = files1.get(relative_path.as_path());
        let path2_option = files2.get(relative_path.as_path());

        match (path1_option, path2_option) {
            (Some(path1), Some(path2)) => {
                println!(
                    "
--- Comparing {} ---",
                    relative_path.display()
                );
                let content1 = read_input(path1)?;
                let content2 = read_input(path2)?;

                let input_format = if let Some(fmt) = format_option {
                    fmt
                } else {
                    infer_format_from_path(path1)
                        .or_else(|| infer_format_from_path(path2))
                        .context(format!(
                            "Could not infer format for {}. Please specify --format.",
                            relative_path.display()
                        ))?
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
                            DiffResult::TensorAdded(k, _) => k,
                            DiffResult::TensorRemoved(k, _) => k,
                            DiffResult::QuantizationAnalysis(k, _) => k,
                            DiffResult::TransferLearningAnalysis(k, _) => k,
                            DiffResult::ExperimentReproducibility(k, _) => k,
                            DiffResult::EnsembleAnalysis(k, _) => k,
                            DiffResult::HyperparameterComparison(k, _) => k,
                            DiffResult::LearningCurveAnalysis(k, _) => k,
                            DiffResult::StatisticalSignificance(k, _) => k,
                            DiffResult::NumpyArrayChanged(k, _, _) => k,
                            DiffResult::NumpyArrayAdded(k, _) => k,
                            DiffResult::NumpyArrayRemoved(k, _) => k,
                            DiffResult::MatlabArrayChanged(k, _, _) => k,
                            DiffResult::MatlabArrayAdded(k, _) => k,
                            DiffResult::MatlabArrayRemoved(k, _) => k,
                        };
                        key.starts_with(filter_path_str)
                    });
                }

                match output {
                    OutputFormat::Cli => print_cli_output(differences, false, no_color), // No magnitude sort for directory comparison
                    OutputFormat::Json => print_json_output(differences)?,
                    OutputFormat::Yaml => print_yaml_output(differences)?,
                    OutputFormat::Unified => print_unified_output(&v1, &v2)?,
                }
                compared_files += 1;
            }
            (Some(_), None) => {
                println!(
                    "
--- Only in {}: {} ---",
                    dir1.display(),
                    relative_path.display()
                );
            }
            (None, Some(_)) => {
                println!(
                    "
--- Only in {}: {} ---",
                    dir2.display(),
                    relative_path.display()
                );
            }
            (None, None) => { /* Should not happen */ }
        }
    }

    if compared_files == 0 && all_relative_paths.is_empty() {
        println!("No comparable files found in directories.");
    }

    Ok(())
}

/// Generate ML-specific recommendations based on analysis results
/// Using 11-axis × 3-level priority matrix with predefined natural English messages
fn generate_ml_recommendations(differences: &[DiffResult]) {
    let mut recommendations = Vec::new();
    let mut has_critical = false;
    let mut has_warning = false;

    for diff in differences {
        match diff {
            // 1. 精度・性能の劣化 (Performance Degradation)
            DiffResult::RegressionTest(_, regression) => {
                if !regression.test_passed {
                    if regression.performance_degradation > 10.0
                        && regression.severity_level == "critical"
                    {
                        has_critical = true;
                        recommendations.push(format!("Model performance severely degraded by {:.1}%. Stop deployment and investigate root cause.", regression.performance_degradation));
                    } else if regression.performance_degradation > 5.0 {
                        has_warning = true;
                        recommendations.push(format!("Performance regression detected ({:.1}% drop). Run comprehensive validation before proceeding.", regression.performance_degradation));
                    } else if regression.performance_degradation > 2.0 {
                        recommendations.push(format!("Minor performance change ({:.1}%). Monitor metrics and validate on holdout set.", regression.performance_degradation));
                    }
                }
            }
            DiffResult::InferenceSpeedAnalysis(_, speed) => {
                if speed.speed_change_ratio > 3.0 {
                    has_critical = true;
                    recommendations.push(format!("Inference speed critically degraded ({:.1}x slower). Identify and fix bottlenecks immediately.", speed.speed_change_ratio));
                } else if speed.speed_change_ratio > 1.5 {
                    has_warning = true;
                    recommendations.push(format!("Inference latency increased significantly ({:.1}x slower). Profile model and optimize hot paths.", speed.speed_change_ratio));
                } else if speed.speed_change_ratio > 1.2 {
                    recommendations.push(format!("Inference speed moderately affected ({:.1}x slower). Consider optimization opportunities.", speed.speed_change_ratio));
                }
            }
            DiffResult::MemoryAnalysis(_, memory) => {
                let delta_mb = memory.memory_delta_bytes as f64 / (1024.0 * 1024.0);
                if delta_mb > 1000.0 {
                    has_critical = true;
                    recommendations.push(format!("Memory usage increased critically (+{:.0}MB). Risk of GPU memory exhaustion.", delta_mb));
                } else if delta_mb > 500.0 {
                    has_warning = true;
                    recommendations.push(format!("Significant memory increase (+{:.0}MB). Consider model quantization or pruning.", delta_mb));
                } else if delta_mb > 200.0 {
                    recommendations.push(format!("Memory usage increased moderately (+{:.0}MB). Monitor resource consumption.", delta_mb));
                }
            }

            // 2. 過学習・汎化性能 (Overfitting & Generalization)
            DiffResult::LearningCurveAnalysis(_, curve) => {
                if curve.overfitting_risk > 0.9 {
                    has_critical = true;
                    recommendations.push(format!("Severe overfitting detected ({:.0}% risk). Stop training immediately and add regularization.", curve.overfitting_risk * 100.0));
                } else if curve.overfitting_risk > 0.7 {
                    has_warning = true;
                    recommendations.push(format!("High overfitting risk ({:.0}%). Implement early stopping or increase regularization.", curve.overfitting_risk * 100.0));
                } else if curve.overfitting_risk > 0.5 {
                    recommendations.push(format!(
                        "Moderate overfitting risk ({:.0}%). Monitor validation metrics closely.",
                        curve.overfitting_risk * 100.0
                    ));
                }
            }

            // 3. 再現性・実験管理 (Reproducibility & Experiment Management)
            DiffResult::ExperimentReproducibility(_, experiment) => {
                if experiment.reproducibility_score < 0.5 {
                    has_critical = true;
                    recommendations.push(format!("Reproducibility critically low ({:.0}%). Fix random seeds and enable deterministic operations.", experiment.reproducibility_score * 100.0));
                } else if experiment.reproducibility_score < 0.8 {
                    has_warning = true;
                    recommendations.push(format!("Low reproducibility ({:.0}%). Set consistent random seeds across all components.", experiment.reproducibility_score * 100.0));
                } else if experiment.reproducibility_score < 0.95 {
                    recommendations.push(format!("Reproducibility could be improved ({:.0}%). Review stochastic operations and configurations.", experiment.reproducibility_score * 100.0));
                }
            }

            // 4. 本番デプロイのリスク (Production Deployment Risk)
            DiffResult::DeploymentReadiness(_, deploy) => {
                if deploy.deployment_strategy == "hold" {
                    has_critical = true;
                    recommendations.push("Production deployment blocked. Address critical blockers before proceeding.".to_string());
                } else if deploy.readiness_score < 0.6 {
                    has_warning = true;
                    recommendations.push(format!("Low deployment readiness ({:.0}%). Review prerequisites and testing requirements.", deploy.readiness_score * 100.0));
                } else if deploy.readiness_score < 0.8 {
                    recommendations.push(format!("Deployment readiness improving ({:.0}%). Continue monitoring progress toward production.", deploy.readiness_score * 100.0));
                }
            }
            DiffResult::RiskAssessment(_, risk) => {
                if risk.overall_risk_level == "critical" {
                    has_critical = true;
                    recommendations.push("Critical deployment risk identified. Implement staged rollout with comprehensive monitoring.".to_string());
                } else if risk.overall_risk_level == "high" {
                    has_warning = true;
                    recommendations.push("High deployment risk detected. Enhanced monitoring and rollback procedures required.".to_string());
                }
            }

            // 5. データドリフト・分布シフト (Data Drift & Distribution Shift)
            DiffResult::EmbeddingAnalysis(_, embed) => {
                if embed.semantic_drift > 0.8 {
                    has_critical = true;
                    recommendations.push(format!("Severe semantic drift detected ({:.0}%). Model may be obsolete for current data.", embed.semantic_drift * 100.0));
                } else if embed.semantic_drift > 0.4 {
                    has_warning = true;
                    recommendations.push(format!("Significant semantic drift ({:.0}%). Retrain model with recent data samples.", embed.semantic_drift * 100.0));
                } else if embed.semantic_drift > 0.2 {
                    recommendations.push(format!("Moderate semantic drift ({:.0}%). Monitor data quality and distribution changes.", embed.semantic_drift * 100.0));
                }
            }

            // 6. 計算効率・コスト (Computational Efficiency & Cost)
            DiffResult::ParamEfficiencyAnalysis(_, efficiency) => {
                if efficiency.efficiency_ratio < 0.3 {
                    has_critical = true;
                    recommendations.push(format!("Extremely low parameter efficiency ({:.0}%). Consider fundamental model redesign.", efficiency.efficiency_ratio * 100.0));
                } else if efficiency.efficiency_ratio < 0.6 {
                    has_warning = true;
                    recommendations.push(format!("Low parameter efficiency ({:.0}%). Apply pruning, distillation, or architecture optimization.", efficiency.efficiency_ratio * 100.0));
                } else if efficiency.efficiency_ratio < 0.8 {
                    recommendations.push(format!("Parameter efficiency could be improved ({:.0}%). Consider optimization techniques.", efficiency.efficiency_ratio * 100.0));
                }
            }

            // 7. モデルの解釈性 (Model Interpretability)
            DiffResult::AttentionAnalysis(_, attention) => {
                if attention.pattern_consistency < 0.4 {
                    has_critical = true;
                    recommendations.push(format!("Attention patterns highly unstable ({:.0}% consistency). Model interpretability severely compromised.", attention.pattern_consistency * 100.0));
                } else if attention.pattern_consistency < 0.7 {
                    has_warning = true;
                    recommendations.push(format!("Attention patterns unstable ({:.0}% consistency). May impact model interpretability and trust.", attention.pattern_consistency * 100.0));
                } else if attention.pattern_consistency < 0.85 {
                    recommendations.push(format!("Attention patterns somewhat inconsistent ({:.0}% consistency). Monitor stability for interpretability needs.", attention.pattern_consistency * 100.0));
                }
            }

            // 8. 互換性・統合 (Compatibility & Integration)
            DiffResult::QuantizationAnalysis(_, quant) => {
                if quant.precision_loss_estimate > 10.0 {
                    has_critical = true;
                    recommendations.push(format!("Quantization precision loss too high ({:.1}%). Deployment will fail accuracy requirements.", quant.precision_loss_estimate));
                } else if quant.precision_loss_estimate > 3.0 {
                    has_warning = true;
                    recommendations.push(format!("High precision loss in quantization ({:.1}%). May affect edge deployment performance.", quant.precision_loss_estimate));
                } else if quant.precision_loss_estimate > 1.0 {
                    recommendations.push(format!("Moderate precision loss ({:.1}%). Test edge device compatibility thoroughly.", quant.precision_loss_estimate));
                }
            }

            // 9. セキュリティ・プライバシー (Security & Privacy)
            DiffResult::ModelArchitectureChanged(_, info1, info2) => {
                let param_growth = (info2.total_params as f64 / info1.total_params as f64) - 1.0;
                if param_growth > 0.5 {
                    has_warning = true;
                    recommendations.push(format!("Large model size increase ({:.0}%). Audit for potential data memorization risks.", param_growth * 100.0));
                } else if param_growth > 0.2 {
                    recommendations.push(format!("Model size increased ({:.0}%). Monitor for data leakage patterns in outputs.", param_growth * 100.0));
                }
            }

            // 10. MLOpsワークフロー (MLOps Workflow)
            DiffResult::ConvergenceAnalysis(_, convergence) => {
                if convergence.convergence_status == "diverging" {
                    has_critical = true;
                    recommendations.push("Training diverging. Stop immediately and fix hyperparameters or learning rate.".to_string());
                } else if convergence.convergence_status == "not_converged" {
                    has_warning = true;
                    recommendations.push("Model not converged. Extend training duration or adjust learning rate schedule.".to_string());
                } else if convergence.parameter_stability < 0.8 {
                    recommendations.push(format!("Parameter stability low ({:.0}%). Monitor training progress and consider early stopping.", convergence.parameter_stability * 100.0));
                }
            }

            // 11. ファインチューニング特有 (Fine-tuning Specific)
            DiffResult::TransferLearningAnalysis(_, transfer) => {
                if transfer.parameter_update_ratio > 0.8 {
                    has_critical = true;
                    recommendations.push(format!(
                        "Most parameters updated ({:.0}%). High risk of catastrophic forgetting.",
                        transfer.parameter_update_ratio * 100.0
                    ));
                } else if transfer.parameter_update_ratio > 0.5 {
                    has_warning = true;
                    recommendations.push(format!("Many parameters updated ({:.0}%). Monitor base model performance degradation.", transfer.parameter_update_ratio * 100.0));
                } else if transfer.parameter_update_ratio > 0.3 {
                    recommendations.push(format!("Moderate parameter updates ({:.0}%). Validate transfer learning effectiveness.", transfer.parameter_update_ratio * 100.0));
                }
            }

            // Critical system-level issues
            DiffResult::GradientAnalysis(_, gradient) => {
                match gradient.gradient_flow_health.as_str() {
                    "exploding" => {
                        has_critical = true;
                        recommendations.push("Gradient explosion detected. Reduce learning rate immediately or apply gradient clipping.".to_string());
                    }
                    "dead" | "diminishing" => {
                        has_critical = true;
                        recommendations.push("Dead or diminishing gradients detected. Adjust architecture, initialization, or activation functions.".to_string());
                    }
                    _ => {}
                }
            }
            DiffResult::AnomalyDetection(_, anomaly) => {
                if anomaly.severity == "critical" {
                    has_critical = true;
                    recommendations.push(format!("Critical anomalies detected in {} layers. Run full regression test suite before deployment.", anomaly.affected_layers.len()));
                } else if anomaly.severity == "warning" {
                    has_warning = true;
                    recommendations.push(format!(
                        "Anomaly detected: {}. Recommended action: {}",
                        anomaly.anomaly_type, anomaly.recommended_action
                    ));
                }
            }
            _ => {}
        }
    }

    if !recommendations.is_empty() {
        println!();
        if has_critical {
            println!("{}", "[CRITICAL]".bright_red());
        } else if has_warning {
            println!("{}", "[WARNING]".yellow());
        } else {
            println!("{}", "[RECOMMENDATIONS]".bright_cyan());
        }

        for (i, rec) in recommendations.iter().enumerate() {
            if i < 5 {
                // Limit to top 5 recommendations
                println!("• {}", rec);
            }
        }
    }
}
