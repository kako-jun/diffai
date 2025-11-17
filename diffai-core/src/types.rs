use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// diffx-core統合
pub use diffx_core::OutputFormat as BaseOutputFormat;

// ============================================================================
// UNIFIED API - Core Types
// ============================================================================

// diffx-coreのDiffResultを拡張してAI/ML用機能を追加
#[derive(Debug, PartialEq, Serialize)]
pub enum DiffResult {
    // 基本型はdiffx-coreから継承
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, Value, Value),
    // AI/ML専用拡張型
    TensorShapeChanged(String, Vec<usize>, Vec<usize>),
    TensorStatsChanged(String, TensorStats, TensorStats), // path, old_stats, new_stats
    TensorDataChanged(String, f64, f64), // path, old_mean, new_mean
    ModelArchitectureChanged(String, String, String), // path, old_arch, new_arch
    WeightSignificantChange(String, f64), // path, change_magnitude
    ActivationFunctionChanged(String, String, String), // path, old_fn, new_fn
    LearningRateChanged(String, f64, f64), // path, old_lr, new_lr
    OptimizerChanged(String, String, String), // path, old_opt, new_opt
    LossChange(String, f64, f64),        // path, old_loss, new_loss
    AccuracyChange(String, f64, f64),    // path, old_acc, new_acc
    ModelVersionChanged(String, String, String), // path, old_version, new_version
}

// diffx-coreのDiffResultとの変換関数
impl From<diffx_core::DiffResult> for DiffResult {
    fn from(result: diffx_core::DiffResult) -> Self {
        match result {
            diffx_core::DiffResult::Added(path, value) => DiffResult::Added(path, value),
            diffx_core::DiffResult::Removed(path, value) => DiffResult::Removed(path, value),
            diffx_core::DiffResult::Modified(path, old, new) => DiffResult::Modified(path, old, new),
            diffx_core::DiffResult::TypeChanged(path, old, new) => DiffResult::TypeChanged(path, old, new),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub element_count: usize,
}

impl TensorStats {
    pub fn new(data: &[f64], shape: Vec<usize>, dtype: String) -> Self {
        let element_count = data.len();
        if element_count == 0 {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                shape,
                dtype,
                element_count: 0,
            };
        }

        let mean = data.iter().sum::<f64>() / element_count as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / element_count as f64;
        let std = variance.sqrt();
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            mean,
            std,
            min,
            max,
            shape,
            dtype,
            element_count,
        }
    }
}

// diffx-coreのOutputFormatを拡張してdiffai用にカスタマイズ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    #[serde(rename = "diffai")]
    #[default]
    Diffai,
    #[serde(rename = "json")]
    Json,
    #[serde(rename = "yaml")]
    Yaml,
}

impl OutputFormat {
    pub fn value_variants() -> &'static [Self] {
        &[Self::Diffai, Self::Json, Self::Yaml]
    }

    // diffx-coreのOutputFormatとの変換
    pub fn to_base_format(&self) -> BaseOutputFormat {
        match self {
            Self::Diffai => BaseOutputFormat::Diffx, // diffaiをdiffxとして扱う
            Self::Json => BaseOutputFormat::Json,
            Self::Yaml => BaseOutputFormat::Yaml,
        }
    }

    pub fn from_base_format(base: BaseOutputFormat) -> Self {
        match base {
            BaseOutputFormat::Diffx => Self::Diffai,
            BaseOutputFormat::Json => Self::Json,
            BaseOutputFormat::Yaml => Self::Yaml,
        }
    }

    pub fn parse_format(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "diffai" => Ok(Self::Diffai),
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            _ => Err(anyhow!("Invalid output format: {}", s)),
        }
    }
}

// File format types
#[derive(Debug, Clone, Copy)]
pub enum FileFormat {
    PyTorch,
    Safetensors,
    NumPy,
    Matlab,
}

/// Re-export FileFormat as DiffFormat for compatibility
pub type DiffFormat = FileFormat;

// lawkitパターン：DiffaiSpecificOptionsは削除、ML分析は自動実行

#[derive(Debug, Clone, Default)]
pub struct DiffOptions {
    // Core comparison options
    pub epsilon: Option<f64>,
    pub array_id_key: Option<String>,
    pub ignore_keys_regex: Option<Regex>,
    pub path_filter: Option<String>,

    // Output control
    pub output_format: Option<OutputFormat>,

    // lawkitパターン：メモリ最適化は常に有効、必要に応じて自動調整
    // show_unchanged, show_types, batch_sizeなどは削除してデフォルト動作
}
