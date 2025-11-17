use anyhow::Result;
use diffx_core::format_diff_output;

use crate::types::{DiffResult, OutputFormat};

// ============================================================================
// UTILITY FUNCTIONS - FOR INTERNAL USE ONLY
// ============================================================================
// These functions are public only for CLI and language bindings.
// External users should use the main diff() function.

/// Format output to string - redirects to format_diff_results for DiffResult
pub fn format_output(results: &[DiffResult], format: OutputFormat) -> Result<String> {
    format_diff_results(results, format)
}

/// DiffResult専用のフォーマット関数（diffx-coreと統合）
pub fn format_diff_results(results: &[DiffResult], format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Json => {
            // JSON形式では全てを統一された配列として出力
            Ok(serde_json::to_string_pretty(results)?)
        }
        OutputFormat::Yaml => {
            // YAML形式でも全てを統一して出力
            Ok(serde_yaml::to_string(results)?)
        }
        OutputFormat::Diffai => {
            // Diffai形式では基本型をdiffx-coreでフォーマット、ML型は手動フォーマット
            let base_results: Vec<diffx_core::DiffResult> = results.iter().filter_map(|r| {
                match r {
                    DiffResult::Added(path, value) => Some(diffx_core::DiffResult::Added(path.clone(), value.clone())),
                    DiffResult::Removed(path, value) => Some(diffx_core::DiffResult::Removed(path.clone(), value.clone())),
                    DiffResult::Modified(path, old, new) => Some(diffx_core::DiffResult::Modified(path.clone(), old.clone(), new.clone())),
                    DiffResult::TypeChanged(path, old, new) => Some(diffx_core::DiffResult::TypeChanged(path.clone(), old.clone(), new.clone())),
                    _ => None
                }
            }).collect();

            let mut output = String::new();

            // 基本型があればdiffx-coreでフォーマット
            if !base_results.is_empty() {
                let base_format = format.to_base_format();
                let formatted = format_diff_output(&base_results, base_format, None)?;
                output.push_str(&formatted);
            }

            // AI/ML専用型を追加
            let ml_results: Vec<&DiffResult> = results.iter().filter(|r| {
                matches!(r,
                    DiffResult::TensorShapeChanged(_, _, _) |
                    DiffResult::TensorStatsChanged(_, _, _) |
                    DiffResult::TensorDataChanged(_, _, _) |
                    DiffResult::ModelArchitectureChanged(_, _, _) |
                    DiffResult::WeightSignificantChange(_, _) |
                    DiffResult::ActivationFunctionChanged(_, _, _) |
                    DiffResult::LearningRateChanged(_, _, _) |
                    DiffResult::OptimizerChanged(_, _, _) |
                    DiffResult::LossChange(_, _, _) |
                    DiffResult::AccuracyChange(_, _, _) |
                    DiffResult::ModelVersionChanged(_, _, _)
                )
            }).collect();

            if !ml_results.is_empty() {
                if !output.is_empty() && !output.ends_with('\n') {
                    output.push('\n');
                }
                output.push('\n');

                for result in &ml_results {
                    match result {
                        DiffResult::ModelArchitectureChanged(path, old, new) => {
                            output.push_str(&format!("  ~ {}: {} -> {}\n", path, old, new));
                        }
                        DiffResult::TensorShapeChanged(path, old_shape, new_shape) => {
                            output.push_str(&format!("  ~ {} shape: {:?} -> {:?}\n", path, old_shape, new_shape));
                        }
                        DiffResult::TensorStatsChanged(path, old_stats, new_stats) => {
                            output.push_str(&format!("  ~ {} stats: mean {:.3} -> {:.3}\n", path, old_stats.mean, new_stats.mean));
                        }
                        _ => {
                            // その他のML型もサポート
                            output.push_str(&format!("  ~ ML analysis: {}\n", serde_json::to_string(result)?));
                        }
                    }
                }
            }

            Ok(output)
        }
    }
}
