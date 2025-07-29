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

fn detect_format_from_path(path: &Path) -> Result<FileFormat> {
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

fn parse_file_by_format(path: &Path, format: FileFormat) -> Result<Value> {
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
