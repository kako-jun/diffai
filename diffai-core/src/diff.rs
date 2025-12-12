use anyhow::{anyhow, Result};
use diffx_core::{diff as base_diff, DiffOptions as BaseDiffOptions};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::ml_analysis::{
    analyze_activation_pattern_analysis, analyze_attention_patterns,
    analyze_batch_normalization_analysis, analyze_convergence_patterns, analyze_ensemble_patterns,
    analyze_gradient_patterns, analyze_learning_rate_changes, analyze_memory_usage_changes,
    analyze_model_architecture_changes, analyze_model_complexity_assessment,
    analyze_quantization_patterns, analyze_regularization_impact,
    analyze_weight_distribution_analysis,
};
use crate::parsers::{detect_format_from_path, parse_file_by_format};
use crate::types::{DiffOptions, DiffResult, TensorStats};

// ============================================================================
// UNIFIED API - Main Function
// ============================================================================

/// Unified diff function for diffai (path-based entry point)
///
/// This is the main entry point that handles both files and directories automatically.
/// - File vs File: Regular file comparison
/// - Directory vs Directory: Recursive directory comparison  
/// - File vs Directory: Returns error
pub fn diff_paths(
    old_path: &str,
    new_path: &str,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>> {
    let path1 = Path::new(old_path);
    let path2 = Path::new(new_path);

    match (path1.is_dir(), path2.is_dir()) {
        (true, true) => diff_directories(path1, path2, options),
        (false, false) => diff_files(path1, path2, options),
        (true, false) => Err(anyhow!(
            "Cannot compare directory '{}' with file '{}'",
            old_path,
            new_path
        )),
        (false, true) => Err(anyhow!(
            "Cannot compare file '{}' with directory '{}'",
            old_path,
            new_path
        )),
    }
}

/// Unified diff function for diffai (Value-based)
///
/// This function operates on pre-parsed JSON values.
/// For file/directory operations, use diff_paths() instead.
pub fn diff(old: &Value, new: &Value, options: Option<&DiffOptions>) -> Result<Vec<DiffResult>> {
    let default_options = DiffOptions::default();
    let opts = options.unwrap_or(&default_options);

    // diffx-coreの基本diff機能を活用してコード重複を削減
    let base_opts = convert_to_base_options(opts);
    let base_results = base_diff(old, new, Some(&base_opts))?;

    // diffx-coreの結果をdiffai形式に変換
    let mut results: Vec<DiffResult> = base_results.into_iter().map(|r| r.into()).collect();

    // AI/ML分析が有効な場合のみ追加処理を実行
    if should_analyze_ml_features(old, new, opts) {
        analyze_ml_features(old, new, &mut results, opts)?;
    }

    Ok(results)
}

// DiffOptionsをdiffx-coreのDiffOptionsに変換
fn convert_to_base_options(opts: &DiffOptions) -> BaseDiffOptions {
    BaseDiffOptions {
        epsilon: opts.epsilon,
        array_id_key: opts.array_id_key.clone(),
        ignore_keys_regex: opts.ignore_keys_regex.clone(),
        path_filter: opts.path_filter.clone(),
        recursive: None,
        output_format: opts.output_format.map(|f| f.to_base_format()),
        diffx_options: None,
    }
}

// AI/ML分析が必要かどうかを判定
fn should_analyze_ml_features(old: &Value, new: &Value, _opts: &DiffOptions) -> bool {
    // lawkitパターン：MLファイル形式なら常に分析実行
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old, new) {
        // PyTorchファイル構造のキーが含まれている場合
        let pytorch_keys = [
            "binary_size",
            "file_size",
            "detected_components",
            "estimated_layers",
            "structure_fingerprint",
            "pickle_protocol",
            "state_dict",
            "model",
            "optimizer",
            "scheduler",
            "epoch",
            "loss",
            "accuracy",
        ];
        for key in &pytorch_keys {
            if old_obj.contains_key(*key) || new_obj.contains_key(*key) {
                return true;
            }
        }

        // SafeTensorsファイル構造のキーが含まれている場合
        let safetensors_keys = ["tensors"];
        for key in &safetensors_keys {
            if old_obj.contains_key(*key) || new_obj.contains_key(*key) {
                return true;
            }
        }

        // テンソル関連のキーが含まれている場合（直接のテンソル名）
        let tensor_keys = [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ];
        for (key, _) in old_obj.iter().chain(new_obj.iter()) {
            for tensor_key in &tensor_keys {
                if key.contains(tensor_key) {
                    return true;
                }
            }
        }

        // テンソル階層構造の検出 (tensors.layer.weight パターン)
        for (key, _) in old_obj.iter().chain(new_obj.iter()) {
            if key.starts_with("tensors.") || key.contains(".weight") || key.contains(".bias") {
                return true;
            }
        }
    }

    // ML関連のファイルは基本的に分析対象とする
    true
}

// ML特徴分析を実行する統合関数
fn analyze_ml_features(
    old: &Value,
    new: &Value,
    results: &mut Vec<DiffResult>,
    _options: &DiffOptions,
) -> Result<()> {
    // lawkitパターン：ML分析は常に実行（ファイル形式に応じて自動判定）
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old, new) {
        // state_dictなどのテンソル変更を分析
        for (key, old_val) in old_obj {
            if let Some(new_val) = new_obj.get(key) {
                if is_tensor_like(old_val) && is_tensor_like(new_val) {
                    analyze_tensor_changes(key, old_val, new_val, results);
                }
            }
        }

        // すべてのML分析を自動実行（lawkitパターン：ユーザー設定より規約を優先）
        analyze_model_architecture_changes(old, new, results);
        analyze_learning_rate_changes(old, new, results);
        analyze_convergence_patterns(old, new, results);
        analyze_memory_usage_changes(old, new, results);
        analyze_ensemble_patterns(old, new, results);
        analyze_quantization_patterns(old, new, results);
        analyze_attention_patterns(old, new, results);
        analyze_gradient_patterns(old, new, results);

        // Additional ML analysis features
        analyze_batch_normalization_analysis(old, new, results);
        analyze_regularization_impact(old, new, results);
        analyze_activation_pattern_analysis(old, new, results);
        analyze_weight_distribution_analysis(old, new, results);
        analyze_model_complexity_assessment(old, new, results);
    }

    Ok(())
}

fn diff_files(
    path1: &Path,
    path2: &Path,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>> {
    // Detect formats based on file extensions
    let format1 = detect_format_from_path(path1)?;
    let format2 = detect_format_from_path(path2)?;

    // Ensure both files have the same format
    if std::mem::discriminant(&format1) != std::mem::discriminant(&format2) {
        return Err(anyhow!(
            "Cannot compare files with different formats: {:?} vs {:?}",
            format1,
            format2
        ));
    }

    // Parse files based on detected formats
    let value1 = parse_file_by_format(path1, format1)?;
    let value2 = parse_file_by_format(path2, format2)?;

    // Use existing diff implementation
    diff(&value1, &value2, options)
}

fn diff_directories(
    dir1: &Path,
    dir2: &Path,
    options: Option<&DiffOptions>,
) -> Result<Vec<DiffResult>> {
    let mut results = Vec::new();

    // Get all files in both directories recursively
    let files1 = get_all_files_recursive(dir1)?;
    let files2 = get_all_files_recursive(dir2)?;

    // Create maps for easier lookup (relative path -> absolute path)
    let files1_map: HashMap<String, &Path> = files1
        .iter()
        .filter_map(|path| {
            path.strip_prefix(dir1)
                .ok()
                .map(|rel| (rel.to_string_lossy().to_string(), path.as_path()))
        })
        .collect();

    let files2_map: HashMap<String, &Path> = files2
        .iter()
        .filter_map(|path| {
            path.strip_prefix(dir2)
                .ok()
                .map(|rel| (rel.to_string_lossy().to_string(), path.as_path()))
        })
        .collect();

    // Find files that exist in dir1 but not in dir2 (removed)
    for (rel_path, abs_path1) in &files1_map {
        if !files2_map.contains_key(rel_path) {
            if let Ok(format) = detect_format_from_path(abs_path1) {
                if let Ok(value) = parse_file_by_format(abs_path1, format) {
                    results.push(DiffResult::Removed(rel_path.clone(), value));
                }
            }
        }
    }

    // Find files that exist in dir2 but not in dir1 (added)
    for (rel_path, abs_path2) in &files2_map {
        if !files1_map.contains_key(rel_path) {
            if let Ok(format) = detect_format_from_path(abs_path2) {
                if let Ok(value) = parse_file_by_format(abs_path2, format) {
                    results.push(DiffResult::Added(rel_path.clone(), value));
                }
            }
        }
    }

    // Find files that exist in both directories (compare contents)
    for (rel_path, abs_path1) in &files1_map {
        if let Some(abs_path2) = files2_map.get(rel_path) {
            match diff_files(abs_path1, abs_path2, options) {
                Ok(mut file_results) => {
                    // Prefix all paths with the relative path
                    for result in &mut file_results {
                        match result {
                            DiffResult::Added(path, _) => *path = format!("{rel_path}/{path}"),
                            DiffResult::Removed(path, _) => *path = format!("{rel_path}/{path}"),
                            DiffResult::Modified(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::TypeChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            // AI/ML specific result types
                            DiffResult::TensorShapeChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::TensorStatsChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::TensorDataChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::ModelArchitectureChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::WeightSignificantChange(path, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::ActivationFunctionChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::LearningRateChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::OptimizerChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::LossChange(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::AccuracyChange(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                            DiffResult::ModelVersionChanged(path, _, _) => {
                                *path = format!("{rel_path}/{path}")
                            }
                        }
                    }
                    results.extend(file_results);
                }
                Err(_) => {
                    // If file comparison fails, skip this file
                    continue;
                }
            }
        }
    }

    Ok(results)
}

fn get_all_files_recursive(dir: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();

    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                files.extend(get_all_files_recursive(&path)?);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }

    Ok(files)
}

// Helper function to detect tensor-like data structures
fn is_tensor_like(value: &Value) -> bool {
    if let Value::Object(obj) = value {
        // Check for common tensor-like properties
        let has_shape =
            obj.contains_key("shape") || obj.contains_key("dims") || obj.contains_key("size");
        let has_data =
            obj.contains_key("data") || obj.contains_key("values") || obj.contains_key("tensor");
        let has_dtype = obj.contains_key("dtype")
            || obj.contains_key("type")
            || obj.contains_key("element_type");

        // Consider it tensor-like if it has at least shape and data, or if it has common ML keys
        has_shape && (has_data || has_dtype) ||
        // Also check for PyTorch/Safetensors/NumPy-specific keys
        obj.contains_key("weight") || obj.contains_key("bias") ||
        obj.contains_key("mean") || obj.contains_key("std") ||
        obj.contains_key("min") || obj.contains_key("max")
    } else {
        false
    }
}

// AI/ML specific analysis functions
fn analyze_tensor_changes(
    path: &str,
    old_tensor: &Value,
    new_tensor: &Value,
    results: &mut Vec<DiffResult>,
) {
    // Try to extract tensor data and compute statistics
    if let (Some(old_data), Some(new_data)) = (
        extract_tensor_data(old_tensor),
        extract_tensor_data(new_tensor),
    ) {
        let old_shape = extract_tensor_shape(old_tensor).unwrap_or_default();
        let new_shape = extract_tensor_shape(new_tensor).unwrap_or_default();
        let dtype = extract_tensor_dtype(old_tensor).unwrap_or_else(|| "f32".to_string());

        // Check for shape changes first
        if old_shape != new_shape {
            results.push(DiffResult::TensorShapeChanged(
                path.to_string(),
                old_shape,
                new_shape,
            ));
            return;
        }

        // Compute comprehensive statistics
        let old_stats = TensorStats::new(&old_data, old_shape.clone(), dtype.clone());
        let new_stats = TensorStats::new(&new_data, new_shape, dtype);

        // Check if statistics changed significantly
        if stats_changed_significantly(&old_stats, &new_stats) {
            results.push(DiffResult::TensorStatsChanged(
                path.to_string(),
                old_stats,
                new_stats,
            ));
        } else {
            // Fall back to simple data change
            results.push(DiffResult::TensorDataChanged(
                path.to_string(),
                old_stats.mean,
                new_stats.mean,
            ));
        }
    }
}

pub fn extract_tensor_data(tensor: &Value) -> Option<Vec<f64>> {
    match tensor {
        // Direct array format (NumPy, simple tensors)
        Value::Array(arr) => {
            let mut data = Vec::new();
            extract_numbers_from_nested_array(arr, &mut data);
            if !data.is_empty() {
                Some(data)
            } else {
                None
            }
        }

        // Structured tensor format (PyTorch/Safetensors)
        Value::Object(obj) => {
            // Check for various data field names
            let data_fields = ["data", "values", "tensor", "_data", "storage"];
            for field in &data_fields {
                if let Some(data_value) = obj.get(*field) {
                    if let Some(extracted) = extract_tensor_data(data_value) {
                        return Some(extracted);
                    }
                }
            }

            // Check for base64 encoded binary data (Safetensors)
            if let Some(data_str) = obj.get("data").and_then(|v| v.as_str()) {
                if let Ok(decoded) = base64_decode_tensor_data(data_str) {
                    return Some(decoded);
                }
            }

            // Check for hex encoded binary data
            if let Some(data_str) = obj.get("hex_data").and_then(|v| v.as_str()) {
                if let Ok(decoded) = hex_decode_tensor_data(data_str) {
                    return Some(decoded);
                }
            }

            // For PyTorch state_dict format, extract actual tensor values
            if obj.contains_key("requires_grad") || obj.contains_key("grad_fn") {
                // This is likely a PyTorch tensor object
                if let Some(Value::Array(shape)) = obj.get("shape") {
                    if let Some(flattened) = extract_flattened_tensor_values(obj, shape) {
                        return Some(flattened);
                    }
                }
            }

            None
        }

        // Single numerical value
        Value::Number(num) => {
            if let Some(f) = num.as_f64() {
                Some(vec![f])
            } else {
                None
            }
        }

        _ => None,
    }
}

// Recursively extract numbers from nested arrays (handles multi-dimensional tensors)
fn extract_numbers_from_nested_array(arr: &[Value], result: &mut Vec<f64>) {
    for item in arr {
        match item {
            Value::Number(num) => {
                if let Some(f) = num.as_f64() {
                    result.push(f);
                }
            }
            Value::Array(nested_arr) => {
                extract_numbers_from_nested_array(nested_arr, result);
            }
            _ => {}
        }
    }
}

// Decode base64 encoded tensor data (common in Safetensors format)
fn base64_decode_tensor_data(_data_str: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // This would typically use a base64 decoder and binary format parser
    // For now, return error to indicate unsupported format
    Err("Base64 tensor decoding not yet implemented".into())
}

// Decode hex encoded tensor data
fn hex_decode_tensor_data(_data_str: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    // This would typically parse hex string and convert to float values
    Err("Hex tensor decoding not yet implemented".into())
}

// Extract flattened tensor values from PyTorch tensor object
fn extract_flattened_tensor_values(
    obj: &serde_json::Map<String, Value>,
    shape: &[Value],
) -> Option<Vec<f64>> {
    // Calculate total elements from shape
    let total_elements: usize = shape
        .iter()
        .filter_map(|v| v.as_u64())
        .map(|n| n as usize)
        .product();

    if total_elements == 0 {
        return None;
    }

    // Look for various ways tensor data might be stored
    let storage_fields = ["_storage", "storage", "_data"];
    for field in &storage_fields {
        if let Some(storage_value) = obj.get(*field) {
            if let Some(data) = extract_tensor_data(storage_value) {
                // Limit to expected number of elements
                let limited_data: Vec<f64> = data.into_iter().take(total_elements).collect();
                if !limited_data.is_empty() {
                    return Some(limited_data);
                }
            }
        }
    }

    None
}

pub fn extract_tensor_shape(tensor: &Value) -> Option<Vec<usize>> {
    // Extract shape information from tensor metadata
    tensor.get("shape").and_then(|s| s.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect()
    })
}

fn extract_tensor_dtype(tensor: &Value) -> Option<String> {
    // Extract data type from tensor metadata
    tensor
        .get("dtype")
        .and_then(|dt| dt.as_str())
        .map(|s| s.to_string())
}

fn stats_changed_significantly(old_stats: &TensorStats, new_stats: &TensorStats) -> bool {
    let mean_change = (old_stats.mean - new_stats.mean).abs() / old_stats.mean.abs().max(1e-8);
    let std_change = (old_stats.std - new_stats.std).abs() / old_stats.std.abs().max(1e-8);

    // Consider significant if relative change > 1%
    mean_change > 0.01 || std_change > 0.01
}
