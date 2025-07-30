use anyhow::{anyhow, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// AI/ML dependencies
use matfile::MatFile;
use safetensors::SafeTensors;
use std::fs::File;
use std::io::Read;

// ============================================================================
// UNIFIED API - Core Types
// ============================================================================

#[derive(Debug, PartialEq, Serialize)]
pub enum DiffResult {
    Added(String, Value),
    Removed(String, Value),
    Modified(String, Value, Value),
    TypeChanged(String, Value, Value),
    // AI/ML specific diff results
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

    pub fn parse_format(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "diffai" => Ok(Self::Diffai),
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            _ => Err(anyhow!("Invalid output format: {}", s)),
        }
    }
}

/// Re-export FileFormat as DiffFormat for compatibility
pub type DiffFormat = FileFormat;

#[derive(Debug, Clone, Default)]
pub struct DiffaiSpecificOptions {
    pub ml_analysis_enabled: Option<bool>,
    pub tensor_comparison_mode: Option<String>, // "shape", "data", "both"
    pub model_format: Option<String>,           // "pytorch", "safetensors", "onnx", etc.
    pub scientific_precision: Option<bool>,
    pub weight_threshold: Option<f64>, // significance threshold for weight changes
    pub activation_analysis: Option<bool>,
    pub learning_rate_tracking: Option<bool>,
    pub optimizer_comparison: Option<bool>,
    pub loss_tracking: Option<bool>,
    pub accuracy_tracking: Option<bool>,
    pub model_version_check: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct DiffOptions {
    // Core comparison options
    pub epsilon: Option<f64>,
    pub array_id_key: Option<String>,
    pub ignore_keys_regex: Option<Regex>,
    pub path_filter: Option<String>,

    // Output control
    pub output_format: Option<OutputFormat>,
    pub show_unchanged: Option<bool>,
    pub show_types: Option<bool>,

    // Memory optimization
    pub use_memory_optimization: Option<bool>,
    pub batch_size: Option<usize>,

    // diffai-specific options
    pub diffai_options: Option<DiffaiSpecificOptions>,
}

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

    // Apply memory optimization if requested
    if opts.use_memory_optimization.unwrap_or(false) {
        diff_optimized_implementation(old, new, opts)
    } else {
        diff_standard_implementation(old, new, opts)
    }
}

fn diff_standard_implementation(
    old: &Value,
    new: &Value,
    options: &DiffOptions,
) -> Result<Vec<DiffResult>> {
    let mut results = Vec::new();
    diff_recursive(old, new, "", &mut results, options);
    Ok(results)
}

fn diff_optimized_implementation(
    old: &Value,
    new: &Value,
    options: &DiffOptions,
) -> Result<Vec<DiffResult>> {
    // Check memory limits
    if would_exceed_memory_limit(old, new) {
        return Err(anyhow!("Input too large for memory optimization"));
    }

    diff_standard_implementation(old, new, options)
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

#[derive(Debug, Clone, Copy)]
enum FileFormat {
    PyTorch,
    Safetensors,
    NumPy,
    Matlab,
}

pub fn detect_format_from_path(path: &Path) -> Result<FileFormat> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("pt") | Some("pth") => Ok(FileFormat::PyTorch),
        Some("safetensors") => Ok(FileFormat::Safetensors),
        Some("npy") | Some("npz") => Ok(FileFormat::NumPy),
        Some("mat") => Ok(FileFormat::Matlab),
        _ => {
            let ext = path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or("unknown");
            Err(anyhow!(
                "Unsupported file format: '{}'. diffai only supports AI/ML file formats: .pt, .pth, .safetensors, .npy, .npz, .mat. For general structured data formats, please use diffx.",
                ext
            ))
        }
    }
}

pub fn parse_file_by_format(path: &Path, format: FileFormat) -> Result<Value> {
    match format {
        FileFormat::PyTorch => parse_pytorch_model(path),
        FileFormat::Safetensors => parse_safetensors_model(path),
        FileFormat::NumPy => parse_numpy_file(path),
        FileFormat::Matlab => parse_matlab_file(path),
    }
}

fn diff_recursive(
    old: &Value,
    new: &Value,
    path: &str,
    results: &mut Vec<DiffResult>,
    options: &DiffOptions,
) {
    // Apply path filter if specified
    if let Some(filter) = &options.path_filter {
        if !path.contains(filter) {
            return;
        }
    }

    match (old, new) {
        (Value::Object(old_obj), Value::Object(new_obj)) => {
            // Check if this might be tensor data and we have ML analysis enabled
            if let Some(diffai_opts) = &options.diffai_options {
                if diffai_opts.ml_analysis_enabled.unwrap_or(true) {
                    // For root-level model comparison, analyze architecture
                    if path.is_empty() || path == "model" {
                        analyze_model_architecture_changes(old, new, results);
                        analyze_weight_significant_changes(old, new, results, diffai_opts);
                        analyze_memory_usage_changes(old, new, results);
                        analyze_learning_rate_changes(old, new, results, diffai_opts);
                        analyze_convergence_patterns(old, new, results);
                        analyze_gradient_patterns(old, new, results);
                        analyze_attention_patterns(old, new, results);
                        analyze_ensemble_patterns(old, new, results);
                        analyze_quantization_patterns(old, new, results);
                    }
                    
                    // Check if this looks like tensor data (has shape, data, etc.)
                    if is_tensor_like(old) && is_tensor_like(new) {
                        analyze_tensor_changes(path, old, new, results);
                        return; // Don't do standard object diff for tensor data
                    }
                }
            }
            diff_objects(old_obj, new_obj, path, results, options);
        }
        (Value::Array(old_arr), Value::Array(new_arr)) => {
            diff_arrays(old_arr, new_arr, path, results, options);
        }
        (Value::Number(old_num), Value::Number(new_num)) => {
            if let Some(epsilon) = options.epsilon {
                let old_f = old_num.as_f64().unwrap_or(0.0);
                let new_f = new_num.as_f64().unwrap_or(0.0);
                if (old_f - new_f).abs() > epsilon {
                    // Check for AI/ML specific number changes
                    if let Some(diffai_opts) = &options.diffai_options {
                        check_ml_number_changes(path, old_f, new_f, diffai_opts, results);
                    } else {
                        results.push(DiffResult::Modified(
                            path.to_string(),
                            old.clone(),
                            new.clone(),
                        ));
                    }
                }
            } else if old != new {
                results.push(DiffResult::Modified(
                    path.to_string(),
                    old.clone(),
                    new.clone(),
                ));
            }
        }
        _ => {
            if old != new {
                if old.type_name() != new.type_name() {
                    results.push(DiffResult::TypeChanged(
                        path.to_string(),
                        old.clone(),
                        new.clone(),
                    ));
                } else {
                    results.push(DiffResult::Modified(
                        path.to_string(),
                        old.clone(),
                        new.clone(),
                    ));
                }
            }
        }
    }
}

fn diff_objects(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    path: &str,
    results: &mut Vec<DiffResult>,
    options: &DiffOptions,
) {
    // Handle ignore_keys_regex
    let should_ignore_key = |key: &str| -> bool {
        if let Some(regex) = &options.ignore_keys_regex {
            regex.is_match(key)
        } else {
            false
        }
    };

    // Check for removed keys
    for (key, old_value) in old_obj {
        if should_ignore_key(key) {
            continue;
        }

        let new_path = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };

        if !new_obj.contains_key(key) {
            results.push(DiffResult::Removed(new_path, old_value.clone()));
        }
    }

    // Check for added and modified keys
    for (key, new_value) in new_obj {
        if should_ignore_key(key) {
            continue;
        }

        let new_path = if path.is_empty() {
            key.clone()
        } else {
            format!("{path}.{key}")
        };

        match old_obj.get(key) {
            None => {
                results.push(DiffResult::Added(new_path, new_value.clone()));
            }
            Some(old_value) => {
                diff_recursive(old_value, new_value, &new_path, results, options);
            }
        }
    }
}

fn diff_arrays(
    old_arr: &[Value],
    new_arr: &[Value],
    path: &str,
    results: &mut Vec<DiffResult>,
    options: &DiffOptions,
) {
    if let Some(id_key) = &options.array_id_key {
        diff_arrays_with_id(old_arr, new_arr, path, results, options, id_key);
    } else {
        diff_arrays_by_index(old_arr, new_arr, path, results, options);
    }
}

fn diff_arrays_with_id(
    old_arr: &[Value],
    new_arr: &[Value],
    path: &str,
    results: &mut Vec<DiffResult>,
    options: &DiffOptions,
    id_key: &str,
) {
    let mut old_by_id: HashMap<String, &Value> = HashMap::new();
    let mut new_by_id: HashMap<String, &Value> = HashMap::new();

    // Index by ID
    for item in old_arr {
        if let Some(id) = item.get(id_key).and_then(|v| v.as_str()) {
            old_by_id.insert(id.to_string(), item);
        }
    }

    for item in new_arr {
        if let Some(id) = item.get(id_key).and_then(|v| v.as_str()) {
            new_by_id.insert(id.to_string(), item);
        }
    }

    // Find removed items
    for (id, old_item) in &old_by_id {
        if !new_by_id.contains_key(id) {
            let item_path = format!("{path}[{id_key}={id}]");
            results.push(DiffResult::Removed(item_path, (*old_item).clone()));
        }
    }

    // Find added and modified items
    for (id, new_item) in &new_by_id {
        let item_path = format!("{path}[{id_key}={id}]");

        match old_by_id.get(id) {
            None => {
                results.push(DiffResult::Added(item_path, (*new_item).clone()));
            }
            Some(old_item) => {
                diff_recursive(old_item, new_item, &item_path, results, options);
            }
        }
    }
}

fn diff_arrays_by_index(
    old_arr: &[Value],
    new_arr: &[Value],
    path: &str,
    results: &mut Vec<DiffResult>,
    options: &DiffOptions,
) {
    let max_len = old_arr.len().max(new_arr.len());

    for i in 0..max_len {
        let item_path = format!("{path}[{i}]");

        match (old_arr.get(i), new_arr.get(i)) {
            (Some(old_item), Some(new_item)) => {
                diff_recursive(old_item, new_item, &item_path, results, options);
            }
            (Some(old_item), None) => {
                results.push(DiffResult::Removed(item_path, old_item.clone()));
            }
            (None, Some(new_item)) => {
                results.push(DiffResult::Added(item_path, new_item.clone()));
            }
            (None, None) => unreachable!(),
        }
    }
}

// Helper function to detect tensor-like data structures
fn is_tensor_like(value: &Value) -> bool {
    if let Value::Object(obj) = value {
        // Check for common tensor-like properties
        let has_shape = obj.contains_key("shape") || obj.contains_key("dims") || obj.contains_key("size");
        let has_data = obj.contains_key("data") || obj.contains_key("values") || obj.contains_key("tensor");
        let has_dtype = obj.contains_key("dtype") || obj.contains_key("type") || obj.contains_key("element_type");
        
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
    if let (Some(old_data), Some(new_data)) = (extract_tensor_data(old_tensor), extract_tensor_data(new_tensor)) {
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

// Model Architecture Analysis - standard feature for PyTorch/Safetensors
fn analyze_model_architecture_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    let old_arch = extract_model_architecture(old_model);
    let new_arch = extract_model_architecture(new_model);
    
    if old_arch != new_arch {
        results.push(DiffResult::ModelArchitectureChanged(
            "model".to_string(),
            old_arch,
            new_arch,
        ));
    }
}

fn extract_model_architecture(model: &Value) -> String {
    if let Value::Object(obj) = model {
        let mut architecture_info = Vec::new();
        let mut layer_count = 0;
        let mut total_params = 0;
        let mut layer_types = std::collections::HashSet::new();
        
        // Analyze model structure
        for (key, value) in obj {
            if key.contains("weight") || key.contains("bias") {
                layer_count += 1;
                
                // Extract layer type from key (e.g., "conv1.weight" -> "conv")
                if let Some(layer_type) = extract_layer_type(key) {
                    layer_types.insert(layer_type);
                }
                
                // Count parameters
                if let Some(shape) = extract_tensor_shape(value) {
                    let param_count: usize = shape.iter().product();
                    total_params += param_count;
                }
            }
        }
        
        architecture_info.push(format!("layers: {}", layer_count));
        architecture_info.push(format!("parameters: {}", total_params));
        if !layer_types.is_empty() {
            let mut types: Vec<_> = layer_types.into_iter().collect();
            types.sort();
            architecture_info.push(format!("types: [{}]", types.join(", ")));
        }
        
        format!("{{{}}}", architecture_info.join(", "))
    } else {
        "unknown".to_string()
    }
}

fn extract_layer_type(key: &str) -> Option<String> {
    // Extract layer type from parameter names
    // e.g., "features.0.weight" -> "conv", "classifier.weight" -> "linear"
    if key.contains("conv") {
        Some("conv".to_string())
    } else if key.contains("linear") || key.contains("fc") || key.contains("classifier") {
        Some("linear".to_string())
    } else if key.contains("norm") || key.contains("bn") {
        Some("norm".to_string())
    } else if key.contains("attention") || key.contains("attn") {
        Some("attention".to_string())
    } else if key.contains("embedding") || key.contains("embed") {
        Some("embedding".to_string())
    } else {
        // Generic layer type based on position
        let parts: Vec<&str> = key.split('.').collect();
        if parts.len() > 1 {
            Some(parts[0].to_string())
        } else {
            None
        }
    }
}

// Weight Significant Change Analysis - standard feature for PyTorch/Safetensors
fn analyze_weight_significant_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
    diffai_opts: &DiffaiSpecificOptions,
) {
    let threshold = diffai_opts.weight_threshold.unwrap_or(0.01);
    
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Analyze weight parameters specifically
        let mut significant_changes = Vec::new();
        
        for (key, old_value) in old_obj {
            if key.contains("weight") || key.contains("bias") {
                if let Some(new_value) = new_obj.get(key) {
                    // Extract numerical values for comparison
                    let change_magnitude = calculate_weight_change_magnitude(old_value, new_value);
                    
                    if change_magnitude > threshold {
                        let layer_type = extract_layer_type(key).unwrap_or_else(|| "unknown".to_string());
                        significant_changes.push(format!("{}: {:.4}", layer_type, change_magnitude));
                        
                        results.push(DiffResult::WeightSignificantChange(
                            key.clone(),
                            change_magnitude,
                        ));
                    }
                }
            }
        }
        
        // If we found multiple significant changes, also provide a summary
        if significant_changes.len() > 1 {
            results.push(DiffResult::WeightSignificantChange(
                "model_summary".to_string(),
                significant_changes.len() as f64,
            ));
        }
    }
}

// Calculate the magnitude of change between two weight values
fn calculate_weight_change_magnitude(old_value: &Value, new_value: &Value) -> f64 {
    match (old_value, new_value) {
        (Value::Number(old_num), Value::Number(new_num)) => {
            let old_f = old_num.as_f64().unwrap_or(0.0);
            let new_f = new_num.as_f64().unwrap_or(0.0);
            (old_f - new_f).abs()
        }
        (Value::Array(old_arr), Value::Array(new_arr)) => {
            // For tensor arrays, calculate RMS difference
            let mut sum_sq_diff = 0.0;
            let mut count = 0;
            
            for (old_elem, new_elem) in old_arr.iter().zip(new_arr.iter()) {
                let elem_diff = calculate_weight_change_magnitude(old_elem, new_elem);
                sum_sq_diff += elem_diff * elem_diff;
                count += 1;
            }
            
            if count > 0 {
                (sum_sq_diff / count as f64).sqrt()
            } else {
                0.0
            }
        }
        (Value::Object(old_obj), Value::Object(new_obj)) => {
            // For structured tensors (e.g., with shape info), extract data and compare
            if let (Some(old_data), Some(new_data)) = (
                extract_tensor_data_for_weight_analysis(old_obj),
                extract_tensor_data_for_weight_analysis(new_obj),
            ) {
                if old_data.len() == new_data.len() {
                    let sum_sq_diff: f64 = old_data
                        .iter()
                        .zip(new_data.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (sum_sq_diff / old_data.len() as f64).sqrt()
                } else {
                    // Different sizes = significant change
                    1.0
                }
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

// Extract numerical data from tensor objects for weight analysis
fn extract_tensor_data_for_weight_analysis(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
    // Try different keys where tensor data might be stored
    let data_keys = ["data", "values", "tensor", "weight", "bias"];
    
    for key in &data_keys {
        if let Some(data_value) = obj.get(*key) {
            if let Value::Array(_arr) = data_value {
                let mut result = Vec::new();
                extract_numbers_recursive(data_value, &mut result);
                if !result.is_empty() {
                    return Some(result);
                }
            }
        }
    }
    
    None
}

// Recursively extract numbers from nested arrays
fn extract_numbers_recursive(value: &Value, result: &mut Vec<f64>) {
    match value {
        Value::Number(num) => {
            if let Some(f) = num.as_f64() {
                result.push(f);
            }
        }
        Value::Array(arr) => {
            for elem in arr {
                extract_numbers_recursive(elem, result);
            }
        }
        _ => {}
    }
}

// Memory Analysis - standard feature for all ML formats
fn analyze_memory_usage_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    let old_memory = calculate_model_memory_usage(old_model);
    let new_memory = calculate_model_memory_usage(new_model);
    
    if old_memory != new_memory {
        // Create a comprehensive memory analysis result
        let memory_change = new_memory as f64 - old_memory as f64;
        let memory_change_percent = if old_memory > 0 {
            (memory_change / old_memory as f64) * 100.0
        } else {
            0.0
        };
        
        // Use ModelArchitectureChanged variant for memory analysis
        let memory_analysis = format!(
            "memory: {} → {} bytes ({:+.1}%)",
            old_memory, new_memory, memory_change_percent
        );
        
        results.push(DiffResult::ModelArchitectureChanged(
            "memory_analysis".to_string(),
            format!("memory_usage: {} bytes", old_memory),
            format!("memory_usage: {} bytes", new_memory),
        ));
        
        // Add detailed breakdown if significant change
        if memory_change.abs() > 1024.0 { // More than 1KB change
            let breakdown = create_memory_breakdown(old_model, new_model);
            if !breakdown.is_empty() {
                results.push(DiffResult::ModelArchitectureChanged(
                    "memory_breakdown".to_string(),
                    "previous".to_string(),
                    breakdown,
                ));
            }
        }
    }
}

// Calculate estimated memory usage of a model
fn calculate_model_memory_usage(model: &Value) -> usize {
    match model {
        Value::Object(obj) => {
            let mut total_memory = 0;
            
            // Base object overhead
            total_memory += std::mem::size_of::<serde_json::Map<String, Value>>();
            
            for (key, value) in obj {
                // Key memory
                total_memory += key.len();
                
                // Value memory
                total_memory += calculate_value_memory(value);
            }
            
            total_memory
        }
        _ => calculate_value_memory(model),
    }
}

// Calculate memory usage of a single Value
fn calculate_value_memory(value: &Value) -> usize {
    match value {
        Value::Null => std::mem::size_of::<Value>(),
        Value::Bool(_) => std::mem::size_of::<bool>(),
        Value::Number(_) => std::mem::size_of::<f64>(), // Assume f64
        Value::String(s) => s.len() + std::mem::size_of::<String>(),
        Value::Array(arr) => {
            let mut size = std::mem::size_of::<Vec<Value>>();
            for elem in arr {
                size += calculate_value_memory(elem);
            }
            size
        }
        Value::Object(obj) => {
            let mut size = std::mem::size_of::<serde_json::Map<String, Value>>();
            for (key, val) in obj {
                size += key.len() + calculate_value_memory(val);
            }
            size
        }
    }
}

// Create a detailed memory breakdown
fn create_memory_breakdown(old_model: &Value, new_model: &Value) -> String {
    let mut breakdown = Vec::new();
    
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Analyze tensor memory usage
        let old_tensor_memory = calculate_tensor_memory(old_obj);
        let new_tensor_memory = calculate_tensor_memory(new_obj);
        
        if old_tensor_memory != new_tensor_memory {
            let change = new_tensor_memory as i64 - old_tensor_memory as i64;
            breakdown.push(format!(
                "tensors: {:+} bytes ({} → {})",
                change, old_tensor_memory, new_tensor_memory
            ));
        }
        
        // Analyze metadata memory
        let old_meta_memory = calculate_metadata_memory(old_obj);
        let new_meta_memory = calculate_metadata_memory(new_obj);
        
        if old_meta_memory != new_meta_memory {
            let change = new_meta_memory as i64 - old_meta_memory as i64;
            breakdown.push(format!(
                "metadata: {:+} bytes ({} → {})",
                change, old_meta_memory, new_meta_memory
            ));
        }
    }
    
    breakdown.join(", ")
}

// Calculate memory used by tensor data
fn calculate_tensor_memory(obj: &serde_json::Map<String, Value>) -> usize {
    let mut tensor_memory = 0;
    
    for (key, value) in obj {
        if key.contains("weight") || key.contains("bias") || key.contains("data") {
            // Estimate tensor memory based on shape and dtype
            if let Value::Object(tensor_obj) = value {
                if let Some(shape_value) = tensor_obj.get("shape") {
                    if let Value::Array(shape_arr) = shape_value {
                        let element_count: usize = shape_arr
                            .iter()
                            .filter_map(|v| v.as_u64())
                            .map(|x| x as usize)
                            .product();
                        
                        // Assume 4 bytes per element (float32)
                        let dtype_size = if let Some(dtype) = tensor_obj.get("dtype") {
                            estimate_dtype_size(dtype)
                        } else {
                            4
                        };
                        
                        tensor_memory += element_count * dtype_size;
                    }
                }
            } else {
                // For non-structured tensors, use value memory
                tensor_memory += calculate_value_memory(value);
            }
        }
    }
    
    tensor_memory
}

// Calculate memory used by metadata
fn calculate_metadata_memory(obj: &serde_json::Map<String, Value>) -> usize {
    let mut meta_memory = 0;
    
    for (key, value) in obj {
        if !key.contains("weight") && !key.contains("bias") && !key.contains("data") {
            meta_memory += key.len() + calculate_value_memory(value);
        }
    }
    
    meta_memory
}

// Estimate bytes per element based on dtype
fn estimate_dtype_size(dtype: &Value) -> usize {
    if let Value::String(dtype_str) = dtype {
        match dtype_str.to_lowercase().as_str() {
            s if s.contains("float64") || s.contains("f64") => 8,
            s if s.contains("float32") || s.contains("f32") => 4,
            s if s.contains("float16") || s.contains("f16") => 2,
            s if s.contains("int64") || s.contains("i64") => 8,
            s if s.contains("int32") || s.contains("i32") => 4,
            s if s.contains("int16") || s.contains("i16") => 2,
            s if s.contains("int8") || s.contains("i8") => 1,
            s if s.contains("uint64") || s.contains("u64") => 8,
            s if s.contains("uint32") || s.contains("u32") => 4,
            s if s.contains("uint16") || s.contains("u16") => 2,
            s if s.contains("uint8") || s.contains("u8") => 1,
            s if s.contains("bool") => 1,
            _ => 4, // Default to 4 bytes (float32)
        }
    } else {
        4 // Default
    }
}

// Learning Rate Change Analysis - standard feature for PyTorch/Safetensors
fn analyze_learning_rate_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
    diffai_opts: &DiffaiSpecificOptions,
) {
    if !diffai_opts.learning_rate_tracking.unwrap_or(true) {
        return;
    }
    
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Look for learning rate information in various locations
        let lr_keys = [
            "learning_rate", "lr", "initial_lr", "base_lr", 
            "current_lr", "lr_scheduler", "optimizer_lr"
        ];
        
        let mut lr_changes = Vec::new();
        
        for lr_key in &lr_keys {
            if let (Some(old_lr), Some(new_lr)) = (old_obj.get(*lr_key), new_obj.get(*lr_key)) {
                let change_info = analyze_learning_rate_value_change(old_lr, new_lr, lr_key);
                if !change_info.is_empty() {
                    lr_changes.extend(change_info);
                }
            }
        }
        
        // Look for optimizer state with learning rate information
        if let (Some(old_opt), Some(new_opt)) = (old_obj.get("optimizer"), new_obj.get("optimizer")) {
            let optimizer_changes = analyze_optimizer_learning_rates(old_opt, new_opt);
            lr_changes.extend(optimizer_changes);
        }
        
        // Look for scheduler state
        if let (Some(old_sched), Some(new_sched)) = (old_obj.get("scheduler"), new_obj.get("scheduler")) {
            let scheduler_changes = analyze_scheduler_learning_rates(old_sched, new_sched);
            lr_changes.extend(scheduler_changes);
        }
        
        // Check if we found explicit learning rate changes
        let found_explicit_lr = !lr_changes.is_empty();
        
        // Add all detected learning rate changes
        for (path, old_lr, new_lr) in lr_changes {
            results.push(DiffResult::LearningRateChanged(path, old_lr, new_lr));
        }
        
        // If no explicit learning rate found but we detect training metadata, report that
        if !found_explicit_lr && has_training_metadata(old_obj, new_obj) {
            // Try to extract implicit learning rate from training information
            if let Some((implicit_old, implicit_new)) = extract_implicit_learning_rate(old_obj, new_obj) {
                results.push(DiffResult::LearningRateChanged(
                    "implicit_lr".to_string(),
                    implicit_old,
                    implicit_new,
                ));
            }
        }
    }
}

// Analyze learning rate changes for a specific value
fn analyze_learning_rate_value_change(old_val: &Value, new_val: &Value, key: &str) -> Vec<(String, f64, f64)> {
    let mut changes = Vec::new();
    
    match (old_val, new_val) {
        (Value::Number(old_num), Value::Number(new_num)) => {
            let old_f = old_num.as_f64().unwrap_or(0.0);
            let new_f = new_num.as_f64().unwrap_or(0.0);
            if old_f != new_f {
                changes.push((key.to_string(), old_f, new_f));
            }
        }
        (Value::Array(old_arr), Value::Array(new_arr)) => {
            // Handle per-parameter group learning rates
            for (i, (old_item, new_item)) in old_arr.iter().zip(new_arr.iter()).enumerate() {
                if let (Value::Number(old_num), Value::Number(new_num)) = (old_item, new_item) {
                    let old_f = old_num.as_f64().unwrap_or(0.0);
                    let new_f = new_num.as_f64().unwrap_or(0.0);
                    if old_f != new_f {
                        changes.push((format!("{}[{}]", key, i), old_f, new_f));
                    }
                }
            }
        }
        (Value::Object(old_obj), Value::Object(new_obj)) => {
            // Handle structured learning rate objects
            for (sub_key, old_sub_val) in old_obj {
                if let Some(new_sub_val) = new_obj.get(sub_key) {
                    let sub_changes = analyze_learning_rate_value_change(
                        old_sub_val, 
                        new_sub_val, 
                        &format!("{}.{}", key, sub_key)
                    );
                    changes.extend(sub_changes);
                }
            }
        }
        _ => {}
    }
    
    changes
}

// Analyze optimizer state for learning rate changes
fn analyze_optimizer_learning_rates(old_opt: &Value, new_opt: &Value) -> Vec<(String, f64, f64)> {
    let mut changes = Vec::new();
    
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_opt, new_opt) {
        // Look for param_groups (common in PyTorch optimizers)
        if let (Some(old_groups), Some(new_groups)) = (old_obj.get("param_groups"), new_obj.get("param_groups")) {
            if let (Value::Array(old_arr), Value::Array(new_arr)) = (old_groups, new_groups) {
                for (i, (old_group, new_group)) in old_arr.iter().zip(new_arr.iter()).enumerate() {
                    if let (Value::Object(old_g), Value::Object(new_g)) = (old_group, new_group) {
                        if let (Some(old_lr), Some(new_lr)) = (old_g.get("lr"), new_g.get("lr")) {
                            if let (Value::Number(old_num), Value::Number(new_num)) = (old_lr, new_lr) {
                                let old_f = old_num.as_f64().unwrap_or(0.0);
                                let new_f = new_num.as_f64().unwrap_or(0.0);
                                if old_f != new_f {
                                    changes.push((format!("optimizer.param_groups[{}].lr", i), old_f, new_f));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Look for direct lr field in optimizer
        if let (Some(old_lr), Some(new_lr)) = (old_obj.get("lr"), new_obj.get("lr")) {
            let lr_changes = analyze_learning_rate_value_change(old_lr, new_lr, "optimizer.lr");
            changes.extend(lr_changes);
        }
    }
    
    changes
}

// Analyze scheduler state for learning rate changes
fn analyze_scheduler_learning_rates(old_sched: &Value, new_sched: &Value) -> Vec<(String, f64, f64)> {
    let mut changes = Vec::new();
    
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_sched, new_sched) {
        // Common scheduler fields
        let scheduler_lr_keys = ["base_lrs", "last_lr", "_last_lr", "current_lr"];
        
        for key in &scheduler_lr_keys {
            if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
                let lr_changes = analyze_learning_rate_value_change(old_val, new_val, &format!("scheduler.{}", key));
                changes.extend(lr_changes);
            }
        }
    }
    
    changes
}

// Check if models have training metadata
fn has_training_metadata(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> bool {
    let training_keys = ["epoch", "step", "iteration", "optimizer", "scheduler", "loss", "metrics"];
    
    for key in &training_keys {
        if old_obj.contains_key(*key) || new_obj.contains_key(*key) {
            return true;
        }
    }
    
    false
}

// Extract implicit learning rate from training information
fn extract_implicit_learning_rate(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(f64, f64)> {
    // Try to infer learning rate from epoch progression and loss changes
    if let (Some(old_epoch), Some(new_epoch)) = (old_obj.get("epoch"), new_obj.get("epoch")) {
        if let (Some(old_loss), Some(new_loss)) = (old_obj.get("loss"), new_obj.get("loss")) {
            if let (Value::Number(old_e), Value::Number(new_e), Value::Number(old_l), Value::Number(new_l)) = 
                (old_epoch, new_epoch, old_loss, new_loss) {
                
                let epoch_diff = new_e.as_f64().unwrap_or(0.0) - old_e.as_f64().unwrap_or(0.0);
                let loss_diff = old_l.as_f64().unwrap_or(0.0) - new_l.as_f64().unwrap_or(0.0); // Improvement is positive
                
                if epoch_diff > 0.0 && loss_diff.abs() > 0.0001 {
                    // Simple heuristic: learning rate proportional to loss improvement rate
                    let implicit_old_lr = loss_diff / epoch_diff * 0.01; // Scale factor
                    let implicit_new_lr = implicit_old_lr * 0.95; // Assume typical decay
                    return Some((implicit_old_lr.abs(), implicit_new_lr.abs()));
                }
            }
        }
    }
    
    None
}

// Convergence Analysis - standard feature for PyTorch/Safetensors
fn analyze_convergence_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Analyze loss convergence patterns
        let loss_convergence = analyze_loss_convergence(old_obj, new_obj);
        if let Some(convergence_info) = loss_convergence {
            results.push(DiffResult::ModelArchitectureChanged(
                "convergence_analysis".to_string(),
                convergence_info.0,
                convergence_info.1,
            ));
        }
        
        // Analyze training stability from multiple metrics
        let stability_analysis = analyze_training_stability(old_obj, new_obj);
        if let Some(stability_info) = stability_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "training_stability".to_string(),
                stability_info.0,
                stability_info.1,
            ));
        }
        
        // Analyze epoch progression patterns
        let epoch_analysis = analyze_epoch_progression(old_obj, new_obj);
        if let Some(epoch_info) = epoch_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "epoch_progression".to_string(),
                epoch_info.0,
                epoch_info.1,
            ));
        }
    }
}

// Analyze loss convergence patterns between checkpoints
fn analyze_loss_convergence(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    // Look for loss values and calculate convergence metrics
    let old_loss = extract_loss_value(old_obj)?;
    let new_loss = extract_loss_value(new_obj)?;
    
    let loss_change = new_loss - old_loss;
    let loss_change_percent = if old_loss != 0.0 {
        (loss_change / old_loss) * 100.0
    } else {
        0.0
    };
    
    // Determine convergence status
    let convergence_status = if loss_change < -0.001 {
        "improving"
    } else if loss_change > 0.001 {
        "diverging"
    } else {
        "stable"
    };
    
    // Look for loss history if available
    let trend_analysis = if let (Some(old_history), Some(new_history)) = 
        (extract_loss_history(old_obj), extract_loss_history(new_obj)) {
        analyze_loss_trend(&old_history, &new_history)
    } else {
        format!("single_point_change: {:.6}", loss_change)
    };
    
    let old_info = format!("loss: {:.6}, status: evaluating", old_loss);
    let new_info = format!("loss: {:.6}, status: {}, trend: {}, change: {:.2}%", 
                          new_loss, convergence_status, trend_analysis, loss_change_percent);
    
    Some((old_info, new_info))
}

// Extract loss value from model checkpoint
fn extract_loss_value(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Try various common loss field names
    let loss_keys = ["loss", "train_loss", "training_loss", "val_loss", "validation_loss", 
                     "total_loss", "current_loss", "best_loss"];
    
    for key in &loss_keys {
        if let Some(loss_val) = obj.get(*key) {
            if let Value::Number(num) = loss_val {
                return num.as_f64();
            }
        }
    }
    
    // Look in nested structures
    if let Some(metrics) = obj.get("metrics") {
        if let Value::Object(metrics_obj) = metrics {
            for key in &loss_keys {
                if let Some(loss_val) = metrics_obj.get(*key) {
                    if let Value::Number(num) = loss_val {
                        return num.as_f64();
                    }
                }
            }
        }
    }
    
    None
}

// Extract loss history for trend analysis
fn extract_loss_history(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
    let history_keys = ["loss_history", "train_losses", "validation_losses", "loss_curve"];
    
    for key in &history_keys {
        if let Some(history_val) = obj.get(*key) {
            if let Value::Array(history_arr) = history_val {
                let mut losses = Vec::new();
                for item in history_arr {
                    if let Value::Number(num) = item {
                        if let Some(loss) = num.as_f64() {
                            losses.push(loss);
                        }
                    }
                }
                if !losses.is_empty() {
                    return Some(losses);
                }
            }
        }
    }
    
    None
}

// Analyze loss trend from historical data
fn analyze_loss_trend(old_history: &[f64], new_history: &[f64]) -> String {
    if old_history.is_empty() || new_history.is_empty() {
        return "insufficient_data".to_string();
    }
    
    // Calculate trend slope for recent history
    let old_trend = calculate_trend_slope(&old_history[old_history.len().saturating_sub(5)..]);
    let new_trend = calculate_trend_slope(&new_history[new_history.len().saturating_sub(5)..]);
    
    let trend_change = new_trend - old_trend;
    
    if trend_change < -0.01 {
        "accelerating_improvement".to_string()
    } else if trend_change > 0.01 {
        "slowing_improvement".to_string()
    } else if new_trend < -0.001 {
        "steady_improvement".to_string()
    } else if new_trend > 0.001 {
        "deteriorating".to_string()
    } else {
        "plateauing".to_string()
    }
}

// Calculate trend slope using simple linear regression
fn calculate_trend_slope(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    
    let n = values.len() as f64;
    let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
    let y_sum: f64 = values.iter().sum();
    let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let x_sq_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
    
    let denominator = n * x_sq_sum - x_sum * x_sum;
    if denominator.abs() < 1e-10 {
        return 0.0;
    }
    
    (n * xy_sum - x_sum * y_sum) / denominator
}

// Analyze training stability from various metrics
fn analyze_training_stability(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let mut stability_factors = Vec::new();
    
    // Check gradient norms if available
    if let (Some(old_grad), Some(new_grad)) = (
        extract_gradient_norm(old_obj), 
        extract_gradient_norm(new_obj)
    ) {
        let grad_change = (new_grad / old_grad - 1.0) * 100.0;
        let grad_stability = if grad_change.abs() < 10.0 {
            "stable"
        } else if grad_change.abs() < 50.0 {
            "moderate_variation"
        } else {
            "high_variation"
        };
        stability_factors.push(format!("gradient_norm: {}", grad_stability));
    }
    
    // Check learning rate stability
    if let (Some(old_lr), Some(new_lr)) = (
        extract_current_learning_rate(old_obj),
        extract_current_learning_rate(new_obj)
    ) {
        let lr_ratio = new_lr / old_lr;
        let lr_stability = if (lr_ratio - 1.0).abs() < 0.1 {
            "stable"
        } else if lr_ratio < 1.0 {
            "decreasing"
        } else {
            "increasing"
        };
        stability_factors.push(format!("learning_rate: {}", lr_stability));
    }
    
    // Check parameter magnitude changes
    if let (Some(old_params), Some(new_params)) = (
        estimate_parameter_magnitude(old_obj),
        estimate_parameter_magnitude(new_obj)
    ) {
        let param_change = ((new_params / old_params - 1.0) * 100.0).abs();
        let param_stability = if param_change < 1.0 {
            "stable"
        } else if param_change < 5.0 {
            "mild_change"
        } else {
            "significant_change"
        };
        stability_factors.push(format!("parameters: {}", param_stability));
    }
    
    if stability_factors.is_empty() {
        return None;
    }
    
    let old_info = "evaluating".to_string();
    let new_info = stability_factors.join(", ");
    
    Some((old_info, new_info))
}

// Analyze epoch progression patterns
fn analyze_epoch_progression(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_epoch = extract_epoch_info(old_obj)?;
    let new_epoch = extract_epoch_info(new_obj)?;
    
    if new_epoch <= old_epoch {
        return None; // No progression or regression
    }
    
    let epoch_diff = new_epoch - old_epoch;
    let progression_rate = if epoch_diff == 1.0 {
        "normal"
    } else if epoch_diff < 1.0 {
        "fractional"
    } else {
        "skipped_epochs"
    };
    
    let old_info = format!("epoch: {}", old_epoch);
    let new_info = format!("epoch: {}, progression: {} ({:+.1})", new_epoch, progression_rate, epoch_diff);
    
    Some((old_info, new_info))
}

// Helper functions for convergence analysis
fn extract_gradient_norm(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let grad_keys = ["grad_norm", "gradient_norm", "total_grad_norm"];
    for key in &grad_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

fn extract_current_learning_rate(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let lr_keys = ["lr", "learning_rate", "current_lr"];
    for key in &lr_keys {
        if let Some(Value::Number(num)) = obj.get(*key) {
            return num.as_f64();
        }
    }
    None
}

fn estimate_parameter_magnitude(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Simple heuristic based on detected weights
    let mut total_magnitude = 0.0;
    let mut count = 0;
    
    for (key, value) in obj {
        if key.contains("weight") || key.contains("bias") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    total_magnitude += val.abs();
                    count += 1;
                }
            }
        }
    }
    
    if count > 0 {
        Some(total_magnitude / count as f64)
    } else {
        None
    }
}

fn extract_epoch_info(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    if let Some(Value::Number(num)) = obj.get("epoch") {
        return num.as_f64();
    }
    None
}

// ============================================================================
// A4-3: GRADIENT ANALYSIS - Medium Priority ML Feature
// ============================================================================

// A4-3: GradientAnalysis - Gradient patterns and optimization behavior analysis
fn analyze_gradient_patterns(
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

// Extract gradient statistics from model data
fn extract_gradient_statistics(obj: &serde_json::Map<String, Value>) -> Option<GradientStatistics> {
    let mut total_norm = None;
    let mut max_gradient = None;
    let mut variance = None;
    let mut sparsity = None;
    let mut outlier_count = None;
    
    // Look for gradient statistics in various locations
    let grad_keys = [
        "grad_norm", "gradient_norm", "total_grad_norm",
        "max_grad", "gradient_max", "grad_variance",
        "grad_sparsity", "gradient_outliers"
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
    
    // If explicit gradient stats not found, estimate from weights
    if total_norm.is_none() {
        total_norm = estimate_gradient_norm_from_weights(obj);
    }
    
    if max_gradient.is_none() {
        max_gradient = estimate_max_gradient_from_weights(obj);
    }
    
    // Calculate sparsity and outliers if data available
    if sparsity.is_none() {
        sparsity = estimate_gradient_sparsity(obj);
    }
    
    if outlier_count.is_none() {
        outlier_count = count_gradient_outliers(obj);
    }
    
    Some(GradientStatistics {
        total_norm,
        max_gradient,
        variance,
        sparsity,
        outlier_count,
    })
}

// Extract gradient flow information
fn extract_gradient_flow_info(obj: &serde_json::Map<String, Value>) -> Option<GradientFlowInfo> {
    let mut vanishing_layers = 0;
    let mut exploding_layers = 0;
    let mut flow_balance = None;
    
    // Count layers with potential gradient problems
    for (key, value) in obj {
        if key.contains("grad") || key.contains("gradient") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    // Vanishing gradient threshold
                    if val.abs() < 1e-6 {
                        vanishing_layers += 1;
                    }
                    // Exploding gradient threshold
                    else if val.abs() > 10.0 {
                        exploding_layers += 1;
                    }
                }
            }
        }
    }
    
    // Estimate flow balance
    flow_balance = estimate_gradient_flow_balance(obj);
    
    Some(GradientFlowInfo {
        vanishing_layers,
        exploding_layers,
        flow_balance,
    })
}

// Helper functions for gradient estimation
fn estimate_gradient_norm_from_weights(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut sum_squares = 0.0;
    let mut count = 0;
    
    for (key, value) in obj {
        if key.contains("weight") || key.contains("bias") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    sum_squares += val * val;
                    count += 1;
                }
            }
        }
    }
    
    if count > 0 {
        Some(sum_squares.sqrt())
    } else {
        None
    }
}

fn estimate_max_gradient_from_weights(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut max_val: f64 = 0.0;
    let mut found = false;
    
    for (key, value) in obj {
        if key.contains("weight") || key.contains("bias") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    max_val = max_val.max(val.abs());
                    found = true;
                }
            }
        }
    }
    
    if found { Some(max_val) } else { None }
}

fn estimate_gradient_sparsity(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut near_zero_count = 0;
    let mut total_count = 0;
    let threshold = 1e-8;
    
    for (key, value) in obj {
        if key.contains("grad") || key.contains("gradient") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    if val.abs() < threshold {
                        near_zero_count += 1;
                    }
                    total_count += 1;
                }
            }
        }
    }
    
    if total_count > 0 {
        Some(near_zero_count as f64 / total_count as f64)
    } else {
        None
    }
}

fn count_gradient_outliers(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    let mut outliers = 0;
    let outlier_threshold = 3.0; // 3 standard deviations
    
    // Simplified outlier detection
    for (key, value) in obj {
        if key.contains("grad") || key.contains("gradient") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    if val.abs() > outlier_threshold {
                        outliers += 1;
                    }
                }
            }
        }
    }
    
    Some(outliers)
}

fn estimate_gradient_flow_balance(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Simplified gradient flow balance estimation
    // In a real implementation, this would analyze layer-wise gradient magnitudes
    let mut forward_strength = 0.0;
    let mut backward_strength = 0.0;
    let mut count = 0;
    
    for (key, value) in obj {
        if key.contains("weight") {
            if let Value::Number(num) = value {
                if let Some(val) = num.as_f64() {
                    if key.contains("0") || key.contains("first") {
                        forward_strength += val.abs();
                    } else {
                        backward_strength += val.abs();
                    }
                    count += 1;
                }
            }
        }
    }
    
    if count > 0 && backward_strength > 0.0 {
        Some(forward_strength / backward_strength)
    } else {
        None
    }
}

// ============================================================================
// A5-1: ATTENTION ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-1: AttentionAnalysis - Transformer attention mechanism analysis
fn analyze_attention_patterns(
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
        if let Some((old_dist, new_dist)) = analyze_attention_weight_distributions(old_obj, new_obj) {
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
fn analyze_attention_heads(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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
            head_analysis.push(format!(
                "head_dim: {} -> {}",
                old_dim, new_dim
            ));
        }
    }
    
    // Compare attention patterns per head
    if old_heads.head_patterns != new_heads.head_patterns {
        let pattern_changes = compare_attention_patterns(&old_heads.head_patterns, &new_heads.head_patterns);
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
fn analyze_attention_weight_distributions(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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
            new_sparsity * 100.0, sparsity_change * 100.0, sparsity_trend
        ));
    }
    
    // Compare attention entropy
    if let (Some(old_entropy), Some(new_entropy)) = (old_dist.entropy, new_dist.entropy) {
        let entropy_change = ((new_entropy / old_entropy - 1.0) * 100.0);
        let entropy_trend = if entropy_change.abs() < 5.0 {
            "stable"
        } else if entropy_change > 0.0 {
            "more_diverse"
        } else {
            "more_focused"
        };
        distribution_analysis.push(format!(
            "entropy: {:.3} ({:+.1}%, {})",
            new_entropy, entropy_change, entropy_trend
        ));
    }
    
    // Compare attention peak concentration
    if let (Some(old_peak), Some(new_peak)) = (old_dist.peak_concentration, new_dist.peak_concentration) {
        let peak_change = new_peak - old_peak;
        distribution_analysis.push(format!(
            "peak_concentration: {:.3} ({:+.3})",
            new_peak, peak_change
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
fn analyze_multihead_attention(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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
    if let (Some(old_ratio), Some(new_ratio)) = (old_mha.self_attention_ratio, new_mha.self_attention_ratio) {
        let ratio_change = new_ratio - old_ratio;
        if ratio_change.abs() > 0.05 {
            mha_analysis.push(format!(
                "self_attention_ratio: {:.2} ({:+.2})",
                new_ratio, ratio_change
            ));
        }
    }
    
    // Compare attention dropout
    if let (Some(old_dropout), Some(new_dropout)) = (old_mha.attention_dropout, new_mha.attention_dropout) {
        if (old_dropout - new_dropout).abs() > 0.001 {
            mha_analysis.push(format!(
                "dropout: {:.3} -> {:.3}",
                old_dropout, new_dropout
            ));
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
            if key.contains("weight") || key.contains("query") || key.contains("key") || key.contains("value") {
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
fn extract_attention_weight_distribution(obj: &serde_json::Map<String, Value>) -> Option<AttentionWeightDistribution> {
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
                peak_concentration = data.iter().map(|x| x.abs()).fold(0.0f64, |a, b| a.max(b)).into();
                
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
fn extract_multihead_attention_info(obj: &serde_json::Map<String, Value>) -> Option<MultiHeadAttentionInfo> {
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
    } else if obj.keys().any(|k| k.contains("sinusoidal") || k.contains("sin_pos")) {
        position_encoding = "sinusoidal".to_string();
    } else if obj.keys().any(|k| k.contains("relative") || k.contains("rel_pos")) {
        position_encoding = "relative".to_string();
    }
    
    // Estimate self-attention ratio
    let self_attn_count = obj.keys().filter(|k| k.contains("self_attn") || k.contains("self_attention")).count();
    let cross_attn_count = obj.keys().filter(|k| k.contains("cross_attn") || k.contains("cross_attention")).count();
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
        changes.push(format!("+{}", pattern));
    }
    
    // Find removed patterns
    for pattern in old_set.difference(&new_set) {
        changes.push(format!("-{}", pattern));
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

// ============================================================================
// A5-2: ENSEMBLE ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-2: EnsembleAnalysis - Multiple model combination and ensemble method analysis
fn analyze_ensemble_patterns(
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

// ============================================================================
// A5-3: QUANTIZATION ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-3: QuantizationAnalysis - Model quantization and precision analysis
fn analyze_quantization_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Quantization precision analysis
        if let Some((old_prec, new_prec)) = analyze_quantization_precision(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "quantization_precision".to_string(),
                old_prec,
                new_prec,
            ));
        }
        
        // Quantization method analysis
        if let Some((old_method, new_method)) = analyze_quantization_methods(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "quantization_methods".to_string(),
                old_method,
                new_method,
            ));
        }
        
        // Quantization impact analysis
        if let Some((old_impact, new_impact)) = analyze_quantization_impact(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "quantization_impact".to_string(),
                old_impact,
                new_impact,
            ));
        }
    }
}

// Analyze quantization precision changes
fn analyze_quantization_precision(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_quant = extract_quantization_info(old_obj)?;
    let new_quant = extract_quantization_info(new_obj)?;
    
    let mut precision_analysis = Vec::new();
    
    // Compare bit width
    if old_quant.bit_width != new_quant.bit_width {
        precision_analysis.push(format!(
            "bit_width: {} -> {}",
            old_quant.bit_width, new_quant.bit_width
        ));
    }
    
    // Compare data types
    if old_quant.data_type != new_quant.data_type {
        precision_analysis.push(format!(
            "data_type: {} -> {}",
            old_quant.data_type, new_quant.data_type
        ));
    }
    
    // Compare quantization coverage
    if old_quant.quantized_layers != new_quant.quantized_layers {
        let coverage_change = new_quant.quantized_layers as i32 - old_quant.quantized_layers as i32;
        precision_analysis.push(format!(
            "quantized_layers: {} ({:+})",
            new_quant.quantized_layers, coverage_change
        ));
    }
    
    // Compare mixed precision usage
    if old_quant.mixed_precision != new_quant.mixed_precision {
        precision_analysis.push(format!(
            "mixed_precision: {} -> {}",
            old_quant.mixed_precision, new_quant.mixed_precision
        ));
    }
    
    if precision_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "{}bit {}, layers: {}, mixed: {}",
        old_quant.bit_width,
        old_quant.data_type,
        old_quant.quantized_layers,
        old_quant.mixed_precision
    );
    let new_info = precision_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Analyze quantization method changes
fn analyze_quantization_methods(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_methods = extract_quantization_methods(old_obj)?;
    let new_methods = extract_quantization_methods(new_obj)?;
    
    let mut method_analysis = Vec::new();
    
    // Compare quantization strategy
    if old_methods.strategy != new_methods.strategy {
        method_analysis.push(format!(
            "strategy: {} -> {}",
            old_methods.strategy, new_methods.strategy
        ));
    }
    
    // Compare calibration method
    if old_methods.calibration_method != new_methods.calibration_method {
        method_analysis.push(format!(
            "calibration: {} -> {}",
            old_methods.calibration_method, new_methods.calibration_method
        ));
    }
    
    // Compare symmetric vs asymmetric quantization
    if old_methods.symmetric != new_methods.symmetric {
        method_analysis.push(format!(
            "symmetric: {} -> {}",
            old_methods.symmetric, new_methods.symmetric
        ));
    }
    
    // Compare per-channel vs per-tensor quantization
    if old_methods.per_channel != new_methods.per_channel {
        method_analysis.push(format!(
            "per_channel: {} -> {}",
            old_methods.per_channel, new_methods.per_channel
        ));
    }
    
    if method_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "strategy: {}, calibration: {}, symmetric: {}, per_channel: {}",
        old_methods.strategy,
        old_methods.calibration_method,
        old_methods.symmetric,
        old_methods.per_channel
    );
    let new_info = method_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Analyze quantization impact on model
fn analyze_quantization_impact(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_impact = extract_quantization_impact(old_obj)?;
    let new_impact = extract_quantization_impact(new_obj)?;
    
    let mut impact_analysis = Vec::new();
    
    // Compare model size reduction
    if let (Some(old_size), Some(new_size)) = (old_impact.size_reduction, new_impact.size_reduction) {
        let size_change = new_size - old_size;
        if size_change.abs() > 0.01 {
            impact_analysis.push(format!(
                "size_reduction: {:.1}% ({:+.1}%)",
                new_size * 100.0, size_change * 100.0
            ));
        }
    }
    
    // Compare accuracy impact
    if let (Some(old_acc), Some(new_acc)) = (old_impact.accuracy_impact, new_impact.accuracy_impact) {
        let acc_change = new_acc - old_acc;
        if acc_change.abs() > 0.001 {
            let impact_trend = if acc_change > 0.0 {
                "degraded"
            } else {
                "improved"
            };
            impact_analysis.push(format!(
                "accuracy_impact: {:.3} ({:+.3}, {})",
                new_acc, acc_change, impact_trend
            ));
        }
    }
    
    // Compare speed improvement
    if let (Some(old_speed), Some(new_speed)) = (old_impact.speed_improvement, new_impact.speed_improvement) {
        let speed_change = new_speed - old_speed;
        if speed_change.abs() > 0.01 {
            impact_analysis.push(format!(
                "speed_improvement: {:.1}x ({:+.1}x)",
                new_speed, speed_change
            ));
        }
    }
    
    // Compare memory efficiency
    if let (Some(old_mem), Some(new_mem)) = (old_impact.memory_efficiency, new_impact.memory_efficiency) {
        let mem_change = new_mem - old_mem;
        if mem_change.abs() > 0.01 {
            impact_analysis.push(format!(
                "memory_efficiency: {:.1}% ({:+.1}%)",
                new_mem * 100.0, mem_change * 100.0
            ));
        }
    }
    
    if impact_analysis.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "size: {:.1}%, acc_impact: {:.3}, speed: {:.1}x, mem: {:.1}%",
        old_impact.size_reduction.unwrap_or(0.0) * 100.0,
        old_impact.accuracy_impact.unwrap_or(0.0),
        old_impact.speed_improvement.unwrap_or(1.0),
        old_impact.memory_efficiency.unwrap_or(0.0) * 100.0
    );
    let new_info = impact_analysis.join(", ");
    
    Some((old_info, new_info))
}

// Helper structures for quantization analysis
#[derive(Debug)]
struct QuantizationInfo {
    bit_width: u8,
    data_type: String,
    quantized_layers: usize,
    mixed_precision: bool,
}

#[derive(Debug)]
struct QuantizationMethods {
    strategy: String,
    calibration_method: String,
    symmetric: bool,
    per_channel: bool,
}

#[derive(Debug)]
struct QuantizationImpact {
    size_reduction: Option<f64>,
    accuracy_impact: Option<f64>,
    speed_improvement: Option<f64>,
    memory_efficiency: Option<f64>,
}

// Extract quantization information
fn extract_quantization_info(obj: &serde_json::Map<String, Value>) -> Option<QuantizationInfo> {
    let mut bit_width = 32u8; // Default FP32
    let mut data_type = "float32".to_string();
    let mut quantized_layers = 0;
    let mut mixed_precision = false;
    
    // Look for quantization-specific keys
    for (key, value) in obj {
        if key.contains("quant") || key.contains("precision") || key.contains("bit") {
            // Extract bit width
            if key.contains("bit") || key.contains("width") {
                if let Value::Number(bits) = value {
                    if let Some(bits_val) = bits.as_u64() {
                        bit_width = bits_val as u8;
                    }
                }
            }
            
            // Extract data type
            if key.contains("dtype") || key.contains("type") {
                if let Value::String(dtype) = value {
                    data_type = dtype.clone();
                }
            }
            
            // Count quantized layers
            if key.contains("layer") && key.contains("quant") {
                quantized_layers += 1;
            }
            
            // Check for mixed precision
            if key.contains("mixed") || key.contains("amp") {
                mixed_precision = true;
            }
        }
        
        // Infer from data types in tensors
        if key.contains("weight") || key.contains("bias") {
            if let Value::Object(tensor_obj) = value {
                if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                    data_type = dtype.clone();
                    
                    // Infer bit width from dtype
                    bit_width = match dtype.as_str() {
                        "int8" | "uint8" => 8,
                        "int16" | "uint16" | "float16" | "half" => 16,
                        "int32" | "uint32" | "float32" => 32,
                        "int64" | "uint64" | "float64" => 64,
                        _ => bit_width,
                    };
                    
                    if dtype.contains("int") && bit_width < 32 {
                        quantized_layers += 1;
                    }
                }
            }
        }
    }
    
    // Check for mixed precision indicators
    let has_fp16 = obj.values().any(|v| {
        if let Value::Object(tensor_obj) = v {
            if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                return dtype.contains("16") || dtype.contains("half");
            }
        }
        false
    });
    
    let has_fp32 = obj.values().any(|v| {
        if let Value::Object(tensor_obj) = v {
            if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                return dtype.contains("32") || dtype.contains("float");
            }
        }
        false
    });
    
    if has_fp16 && has_fp32 {
        mixed_precision = true;
    }
    
    Some(QuantizationInfo {
        bit_width,
        data_type,
        quantized_layers,
        mixed_precision,
    })
}

// Extract quantization methods
fn extract_quantization_methods(obj: &serde_json::Map<String, Value>) -> Option<QuantizationMethods> {
    let mut strategy = "post_training".to_string();
    let mut calibration_method = "minmax".to_string();
    let mut symmetric = true;
    let mut per_channel = false;
    
    // Look for quantization method indicators
    for (key, value) in obj {
        if key.contains("quant") {
            if key.contains("strategy") || key.contains("method") {
                if let Value::String(strat) = value {
                    strategy = strat.clone();
                }
            } else if key.contains("calibration") {
                if let Value::String(calib) = value {
                    calibration_method = calib.clone();
                }
            } else if key.contains("symmetric") {
                if let Value::Bool(sym) = value {
                    symmetric = *sym;
                }
            } else if key.contains("per_channel") || key.contains("channel_wise") {
                if let Value::Bool(per_ch) = value {
                    per_channel = *per_ch;
                } else {
                    per_channel = true; // If key exists, assume true
                }
            }
        }
    }
    
    // Infer strategy from model structure
    if obj.contains_key("quantization_aware_training") || obj.contains_key("qat") {
        strategy = "quantization_aware_training".to_string();
    } else if obj.contains_key("dynamic_quantization") {
        strategy = "dynamic".to_string();
    } else if obj.contains_key("static_quantization") {
        strategy = "static".to_string();
    }
    
    // Infer calibration method
    if obj.contains_key("entropy_calibration") || obj.contains_key("kl_divergence") {
        calibration_method = "entropy".to_string();
    } else if obj.contains_key("percentile_calibration") {
        calibration_method = "percentile".to_string();
    }
    
    Some(QuantizationMethods {
        strategy,
        calibration_method,
        symmetric,
        per_channel,
    })
}

// Extract quantization impact metrics
fn extract_quantization_impact(obj: &serde_json::Map<String, Value>) -> Option<QuantizationImpact> {
    let mut size_reduction = None;
    let mut accuracy_impact = None;
    let mut speed_improvement = None;
    let mut memory_efficiency = None;
    
    // Look for performance metrics
    for (key, value) in obj {
        if key.contains("size") && key.contains("reduction") {
            if let Value::Number(reduction) = value {
                size_reduction = reduction.as_f64();
            }
        } else if key.contains("accuracy") && (key.contains("drop") || key.contains("impact") || key.contains("loss")) {
            if let Value::Number(acc_impact) = value {
                accuracy_impact = acc_impact.as_f64();
            }
        } else if key.contains("speed") && (key.contains("up") || key.contains("improvement") || key.contains("gain")) {
            if let Value::Number(speed) = value {
                speed_improvement = speed.as_f64();
            }
        } else if key.contains("memory") && (key.contains("efficiency") || key.contains("reduction")) {
            if let Value::Number(mem_eff) = value {
                memory_efficiency = mem_eff.as_f64();
            }
        }
    }
    
    // Estimate metrics if not explicitly provided
    if size_reduction.is_none() {
        // Estimate based on bit width
        let quantized_count = obj.iter()
            .filter(|(key, value)| {
                key.contains("weight") && {
                    if let Value::Object(tensor_obj) = value {
                        if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                            return dtype.contains("int8") || dtype.contains("int16");
                        }
                    }
                    false
                }
            })
            .count();
        
        let total_count = obj.iter()
            .filter(|(key, _)| key.contains("weight"))
            .count();
        
        if total_count > 0 {
            let quantized_ratio = quantized_count as f64 / total_count as f64;
            // Rough estimate: int8 gives ~75% size reduction, int16 gives ~50%
            size_reduction = Some(quantized_ratio * 0.6); // Conservative estimate
        }
    }
    
    // Provide default speed improvement for quantized models
    if speed_improvement.is_none() && size_reduction.is_some() {
        let reduction = size_reduction.unwrap();
        if reduction > 0.0 {
            speed_improvement = Some(1.0 + reduction); // Modest speed improvement
        }
    }
    
    Some(QuantizationImpact {
        size_reduction,
        accuracy_impact,
        speed_improvement,
        memory_efficiency,
    })
}

fn check_ml_number_changes(
    path: &str,
    old_val: f64,
    new_val: f64,
    diffai_opts: &DiffaiSpecificOptions,
    results: &mut Vec<DiffResult>,
) {
    let change_magnitude = (new_val - old_val).abs();

    // Check for learning rate changes
    if diffai_opts.learning_rate_tracking.unwrap_or(true) && path.contains("learning_rate") {
        results.push(DiffResult::LearningRateChanged(
            path.to_string(),
            old_val,
            new_val,
        ));
        return;
    }

    // Check for loss changes
    if diffai_opts.loss_tracking.unwrap_or(true)
        && (path.contains("loss") || path.contains("cost"))
    {
        results.push(DiffResult::LossChange(path.to_string(), old_val, new_val));
        return;
    }

    // Check for accuracy changes
    if diffai_opts.accuracy_tracking.unwrap_or(true)
        && (path.contains("accuracy") || path.contains("acc"))
    {
        results.push(DiffResult::AccuracyChange(
            path.to_string(),
            old_val,
            new_val,
        ));
        return;
    }

    // Check for significant weight changes (default threshold 0.01)
    let threshold = diffai_opts.weight_threshold.unwrap_or(0.01);
    if change_magnitude > threshold && (path.contains("weight") || path.contains("bias")) {
        results.push(DiffResult::WeightSignificantChange(
            path.to_string(),
            change_magnitude,
        ));
        return;
    }

    // Default to regular modification
    results.push(DiffResult::TensorDataChanged(
        path.to_string(),
        old_val,
        new_val,
    ));
}

// Helper functions for tensor analysis
fn extract_tensor_data(tensor: &Value) -> Option<Vec<f64>> {
    // Extract numerical data from tensor representation
    // This is a simplified implementation - real implementation would handle PyTorch/Safetensors formats
    match tensor {
        Value::Array(arr) => {
            let mut data = Vec::new();
            for item in arr {
                if let Value::Number(num) = item {
                    if let Some(f) = num.as_f64() {
                        data.push(f);
                    }
                }
            }
            if !data.is_empty() { Some(data) } else { None }
        }
        _ => None,
    }
}

fn extract_tensor_shape(tensor: &Value) -> Option<Vec<usize>> {
    // Extract shape information from tensor metadata
    tensor.get("shape")
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect()
        })
}

fn extract_tensor_dtype(tensor: &Value) -> Option<String> {
    // Extract data type from tensor metadata
    tensor.get("dtype")
        .and_then(|dt| dt.as_str())
        .map(|s| s.to_string())
}

fn stats_changed_significantly(old_stats: &TensorStats, new_stats: &TensorStats) -> bool {
    let mean_change = (old_stats.mean - new_stats.mean).abs() / old_stats.mean.abs().max(1e-8);
    let std_change = (old_stats.std - new_stats.std).abs() / old_stats.std.abs().max(1e-8);
    
    // Consider significant if relative change > 1%
    mean_change > 0.01 || std_change > 0.01
}

// ============================================================================
// PARSER FUNCTIONS - FOR INTERNAL USE ONLY
// ============================================================================
// These functions are public only for CLI and language bindings.
// External users should use the main diff() function with file reading.

// ============================================================================
// AI/ML SPECIFIC PARSERS
// ============================================================================

/// Parse PyTorch model file - FOR INTERNAL USE ONLY (diffai-specific)
pub fn parse_pytorch_model(file_path: &Path) -> Result<Value> {
    // Parse PyTorch model file and convert to JSON representation
    let file = File::open(file_path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // Try to extract basic model structure information from the binary data
    // This is a heuristic approach since full pickle parsing is complex
    let mut result = serde_json::Map::new();
    result.insert(
        "model_type".to_string(),
        Value::String("pytorch".to_string()),
    );
    result.insert("file_size".to_string(), Value::Number(buffer.len().into()));
    result.insert("format".to_string(), Value::String("pickle".to_string()));
    
    // Extract model structure information through heuristic analysis
    let model_info = extract_pytorch_model_info(&buffer);
    for (key, value) in model_info {
        result.insert(key, value);
    }

    Ok(Value::Object(result))
}

// Extract basic model information from PyTorch binary data using heuristics
fn extract_pytorch_model_info(buffer: &[u8]) -> serde_json::Map<String, Value> {
    let mut info = serde_json::Map::new();
    
    // First, try binary analysis by looking for specific byte patterns
    let mut weight_count = 0;
    let mut bias_count = 0;
    let mut layer_count = 0;
    
    // Search for common PyTorch string patterns in binary data
    // Look for null-terminated strings that match layer names
    let searchable_content = String::from_utf8_lossy(buffer);
    
    // Count weight and bias parameters more accurately
    weight_count = searchable_content.matches("weight").count();
    bias_count = searchable_content.matches("bias").count();
    
    // Look for layer-specific patterns
    let conv_count = searchable_content.matches("conv").count();
    let linear_count = searchable_content.matches("linear").count() + searchable_content.matches("fc.").count();
    let bn_count = searchable_content.matches("bn").count() + searchable_content.matches("batch_norm").count();
    
    // Build layer information
    let mut detected_layers = Vec::new();
    if conv_count > 0 {
        detected_layers.push(format!("convolution: {}", conv_count));
    }
    if linear_count > 0 {
        detected_layers.push(format!("linear: {}", linear_count));
    }
    if bn_count > 0 {
        detected_layers.push(format!("batch_norm: {}", bn_count));
    }
    if weight_count > 0 {
        detected_layers.push(format!("weight_params: {}", weight_count));
    }
    if bias_count > 0 {
        detected_layers.push(format!("bias_params: {}", bias_count));
    }
    
    if !detected_layers.is_empty() {
        info.insert("detected_components".to_string(), 
                   Value::String(detected_layers.join(", ")));
    }
    
    // Estimate model complexity based on parameter count
    layer_count = weight_count.max(bias_count / 2); // rough estimation
    if layer_count > 0 {
        info.insert("estimated_layers".to_string(), 
                   Value::Number(layer_count.into()));
    }
    
    // Look for model architecture signatures
    let architectures = [
        ("resnet", "ResNet"),
        ("vgg", "VGG"), 
        ("densenet", "DenseNet"),
        ("mobilenet", "MobileNet"),
        ("efficientnet", "EfficientNet"),
        ("transformer", "Transformer"),
        ("bert", "BERT"),
        ("gpt", "GPT"),
    ];
    
    for (pattern, arch_name) in &architectures {
        if searchable_content.to_lowercase().contains(pattern) {
            info.insert("detected_architecture".to_string(),
                       Value::String(arch_name.to_string()));
            break;
        }
    }
    
    // Look for optimizer state information (for training checkpoints)
    if searchable_content.contains("optimizer") {
        info.insert("has_optimizer_state".to_string(), Value::Bool(true));
    }
    if searchable_content.contains("epoch") {
        info.insert("has_training_metadata".to_string(), Value::Bool(true));
    }
    if searchable_content.contains("lr") || searchable_content.contains("learning_rate") {
        info.insert("has_learning_rate".to_string(), Value::Bool(true));
    }
    
    // Add binary-level analysis
    info.insert("binary_size".to_string(), Value::Number(buffer.len().into()));
    
    // Detect pickle protocol version 
    if buffer.len() > 2 {
        let protocol_byte = buffer[1];
        if protocol_byte <= 5 {
            info.insert("pickle_protocol".to_string(), 
                       Value::Number(protocol_byte.into()));
        }
    }
    
    // Calculate a simple hash for model structure comparison
    let structure_hash = calculate_simple_hash(&searchable_content);
    info.insert("structure_fingerprint".to_string(), 
               Value::String(format!("{:x}", structure_hash)));
    
    info
}

// Simple hash calculation for model structure fingerprinting
fn calculate_simple_hash(content: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    // Hash only the structure-relevant parts to detect architecture changes
    let structure_parts: Vec<&str> = content
        .matches(|c: char| c.is_alphanumeric() || c == '.')
        .take(1000) // limit to prevent performance issues
        .collect();
    structure_parts.hash(&mut hasher);
    hasher.finish()
}

/// Parse SafeTensors model file - FOR INTERNAL USE ONLY (diffai-specific)
pub fn parse_safetensors_model(file_path: &Path) -> Result<Value> {
    let buffer = std::fs::read(file_path)?;
    let safetensors = SafeTensors::deserialize(&buffer)?;

    let mut result = serde_json::Map::new();
    let mut tensors = serde_json::Map::new();

    for tensor_name in safetensors.names() {
        let tensor_view = safetensors.tensor(tensor_name)?;
        let mut tensor_info = serde_json::Map::new();

        tensor_info.insert(
            "shape".to_string(),
            Value::Array(
                tensor_view
                    .shape()
                    .iter()
                    .map(|&s| Value::Number(s.into()))
                    .collect(),
            ),
        );
        tensor_info.insert(
            "dtype".to_string(),
            Value::String(format!("{:?}", tensor_view.dtype())),
        );

        tensors.insert(tensor_name.to_string(), Value::Object(tensor_info));
    }

    result.insert(
        "model_type".to_string(),
        Value::String("safetensors".to_string()),
    );
    result.insert("tensors".to_string(), Value::Object(tensors));

    Ok(Value::Object(result))
}

/// Parse NumPy file - FOR INTERNAL USE ONLY (diffai-specific)
pub fn parse_numpy_file(path: &Path) -> Result<Value> {
    // Simplified numpy file parsing
    let mut result = serde_json::Map::new();
    result.insert("model_type".to_string(), Value::String("numpy".to_string()));
    result.insert(
        "file_path".to_string(),
        Value::String(path.to_string_lossy().to_string()),
    );

    Ok(Value::Object(result))
}

/// Parse MATLAB file - FOR INTERNAL USE ONLY (diffai-specific)
pub fn parse_matlab_file(path: &Path) -> Result<Value> {
    let file = File::open(path)?;
    let _mat_file = MatFile::parse(file)?;

    let mut result = serde_json::Map::new();
    let arrays = serde_json::Map::new();

    // Simplified MATLAB file parsing - would need proper implementation
    result.insert(
        "model_type".to_string(),
        Value::String("matlab".to_string()),
    );
    result.insert(
        "file_path".to_string(),
        Value::String(path.to_string_lossy().to_string()),
    );
    result.insert("arrays".to_string(), Value::Object(arrays));

    Ok(Value::Object(result))
}



// ============================================================================
// UTILITY FUNCTIONS - FOR INTERNAL USE ONLY
// ============================================================================
// These functions are public only for CLI and language bindings.
// External users should use the main diff() function.

/// Get type name of a JSON value - FOR INTERNAL USE ONLY
pub fn value_type_name(value: &Value) -> &str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Estimate memory usage of a JSON value - FOR INTERNAL USE ONLY
pub fn estimate_memory_usage(value: &Value) -> usize {
    match value {
        Value::Null => 8,
        Value::Bool(_) => 8,
        Value::Number(_) => 16,
        Value::String(s) => s.len() + 24,
        Value::Array(arr) => arr.iter().map(estimate_memory_usage).sum::<usize>() + 24,
        Value::Object(obj) => {
            obj.iter()
                .map(|(k, v)| k.len() + estimate_memory_usage(v))
                .sum::<usize>()
                + 24
        }
    }
}

/// Check if values would exceed memory limit - FOR INTERNAL USE ONLY
pub fn would_exceed_memory_limit(v1: &Value, v2: &Value) -> bool {
    const MAX_MEMORY_MB: usize = 100;
    const BYTES_PER_MB: usize = 1024 * 1024;

    let total_size = estimate_memory_usage(v1) + estimate_memory_usage(v2);
    total_size > MAX_MEMORY_MB * BYTES_PER_MB
}

/// Format output to string - FOR INTERNAL USE ONLY
pub fn format_output<T: Serialize>(results: &[T], format: OutputFormat) -> Result<String> {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(results)
            .map_err(|e| anyhow!("JSON serialization error: {}", e)),
        OutputFormat::Yaml => {
            serde_yaml::to_string(results).map_err(|e| anyhow!("YAML serialization error: {}", e))
        }
        OutputFormat::Diffai => {
            let mut output = String::new();
            for result in results {
                let json = serde_json::to_string(result)?;
                output.push_str(&json);
                output.push('\n');
            }
            Ok(output)
        }
    }
}

// ============================================================================
// TRAITS
// ============================================================================

trait ValueTypeExt {
    fn type_name(&self) -> &str;
}

impl ValueTypeExt for Value {
    fn type_name(&self) -> &str {
        value_type_name(self)
    }
}
