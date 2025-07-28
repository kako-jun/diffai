use anyhow::{anyhow, Result};
use csv::ReaderBuilder;
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
    // Read file contents
    let content1 = fs::read_to_string(path1)?;
    let content2 = fs::read_to_string(path2)?;

    // Detect formats based on file extensions
    let format1 = detect_format_from_path(path1);
    let format2 = detect_format_from_path(path2);

    // Parse content based on detected formats
    let value1 = parse_content_by_format(&content1, format1)?;
    let value2 = parse_content_by_format(&content2, format2)?;

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
            let content = fs::read_to_string(abs_path1).unwrap_or_default();
            if let Ok(value) = parse_content_by_format(&content, detect_format_from_path(abs_path1))
            {
                results.push(DiffResult::Removed(rel_path.clone(), value));
            }
        }
    }

    // Find files that exist in dir2 but not in dir1 (added)
    for (rel_path, abs_path2) in &files2_map {
        if !files1_map.contains_key(rel_path) {
            let content = fs::read_to_string(abs_path2).unwrap_or_default();
            if let Ok(value) = parse_content_by_format(&content, detect_format_from_path(abs_path2))
            {
                results.push(DiffResult::Added(rel_path.clone(), value));
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
    Json,
    Yaml,
    Csv,
    Toml,
    Ini,
    Xml,
}

fn detect_format_from_path(path: &Path) -> FileFormat {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => FileFormat::Json,
        Some("yaml") | Some("yml") => FileFormat::Yaml,
        Some("csv") => FileFormat::Csv,
        Some("toml") => FileFormat::Toml,
        Some("ini") | Some("cfg") => FileFormat::Ini,
        Some("xml") => FileFormat::Xml,
        _ => FileFormat::Json, // Default fallback
    }
}

fn parse_content_by_format(content: &str, format: FileFormat) -> Result<Value> {
    match format {
        FileFormat::Json => parse_json(content),
        FileFormat::Yaml => parse_yaml(content),
        FileFormat::Csv => parse_csv(content),
        FileFormat::Toml => parse_toml(content),
        FileFormat::Ini => parse_ini(content),
        FileFormat::Xml => parse_xml(content),
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

// AI/ML specific analysis functions
fn check_ml_number_changes(
    path: &str,
    old_val: f64,
    new_val: f64,
    diffai_opts: &DiffaiSpecificOptions,
    results: &mut Vec<DiffResult>,
) {
    let change_magnitude = (new_val - old_val).abs();

    // Check for learning rate changes
    if diffai_opts.learning_rate_tracking.unwrap_or(false) && path.contains("learning_rate") {
        results.push(DiffResult::LearningRateChanged(
            path.to_string(),
            old_val,
            new_val,
        ));
        return;
    }

    // Check for loss changes
    if diffai_opts.loss_tracking.unwrap_or(false)
        && (path.contains("loss") || path.contains("cost"))
    {
        results.push(DiffResult::LossChange(path.to_string(), old_val, new_val));
        return;
    }

    // Check for accuracy changes
    if diffai_opts.accuracy_tracking.unwrap_or(false)
        && (path.contains("accuracy") || path.contains("acc"))
    {
        results.push(DiffResult::AccuracyChange(
            path.to_string(),
            old_val,
            new_val,
        ));
        return;
    }

    // Check for significant weight changes
    if let Some(threshold) = diffai_opts.weight_threshold {
        if change_magnitude > threshold && (path.contains("weight") || path.contains("bias")) {
            results.push(DiffResult::WeightSignificantChange(
                path.to_string(),
                change_magnitude,
            ));
            return;
        }
    }

    // Default to regular modification
    results.push(DiffResult::TensorDataChanged(
        path.to_string(),
        old_val,
        new_val,
    ));
}

// ============================================================================
// PARSER FUNCTIONS - FOR INTERNAL USE ONLY
// ============================================================================
// These functions are public only for CLI and language bindings.
// External users should use the main diff() function with file reading.

/// Parse JSON content - FOR INTERNAL USE ONLY
/// External users should read files themselves and use diff() function
pub fn parse_json(content: &str) -> Result<Value> {
    serde_json::from_str(content).map_err(|e| anyhow!("JSON parse error: {}", e))
}

/// Parse CSV content - FOR INTERNAL USE ONLY
pub fn parse_csv(content: &str) -> Result<Value> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    let headers = reader.headers()?.clone();
    let mut records = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut map = serde_json::Map::new();

        for (i, field) in record.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                map.insert(header.to_string(), Value::String(field.to_string()));
            }
        }

        records.push(Value::Object(map));
    }

    Ok(Value::Array(records))
}

/// Parse YAML content - FOR INTERNAL USE ONLY
pub fn parse_yaml(content: &str) -> Result<Value> {
    serde_yaml::from_str(content).map_err(|e| anyhow!("YAML parse error: {}", e))
}

/// Parse TOML content - FOR INTERNAL USE ONLY
pub fn parse_toml(content: &str) -> Result<Value> {
    let toml_value: toml::Value = content.parse()?;
    toml_to_json_value(toml_value)
}

fn toml_to_json_value(toml_val: toml::Value) -> Result<Value> {
    match toml_val {
        toml::Value::String(s) => Ok(Value::String(s)),
        toml::Value::Integer(i) => Ok(Value::Number(i.into())),
        toml::Value::Float(f) => Ok(Value::Number(
            serde_json::Number::from_f64(f).ok_or_else(|| anyhow!("Invalid float"))?,
        )),
        toml::Value::Boolean(b) => Ok(Value::Bool(b)),
        toml::Value::Array(arr) => {
            let mut json_arr = Vec::new();
            for item in arr {
                json_arr.push(toml_to_json_value(item)?);
            }
            Ok(Value::Array(json_arr))
        }
        toml::Value::Table(table) => {
            let mut json_obj = serde_json::Map::new();
            for (key, value) in table {
                json_obj.insert(key, toml_to_json_value(value)?);
            }
            Ok(Value::Object(json_obj))
        }
        toml::Value::Datetime(dt) => Ok(Value::String(dt.to_string())),
    }
}

/// Parse INI content - FOR INTERNAL USE ONLY
pub fn parse_ini(content: &str) -> Result<Value> {
    let mut result = serde_json::Map::new();
    let mut current_section = String::new();

    for line in content.lines() {
        let line = line.trim();

        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            current_section = line[1..line.len() - 1].to_string();
            result.insert(
                current_section.clone(),
                Value::Object(serde_json::Map::new()),
            );
        } else if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim().to_string();
            let value = line[eq_pos + 1..].trim().to_string();

            if current_section.is_empty() {
                result.insert(key, Value::String(value));
            } else if let Some(Value::Object(section)) = result.get_mut(&current_section) {
                section.insert(key, Value::String(value));
            }
        }
    }

    Ok(Value::Object(result))
}

/// Parse XML content - FOR INTERNAL USE ONLY
pub fn parse_xml(content: &str) -> Result<Value> {
    // Simple XML parser - for production use, consider using quick-xml
    // This is a simplified implementation
    Ok(Value::String(format!(
        "XML parsing not fully implemented: {}",
        content.len()
    )))
}

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

    // This is a simplified version - real implementation would parse the pickle format
    let mut result = serde_json::Map::new();
    result.insert(
        "model_type".to_string(),
        Value::String("pytorch".to_string()),
    );
    result.insert("file_size".to_string(), Value::Number(buffer.len().into()));
    result.insert("format".to_string(), Value::String("pickle".to_string()));

    Ok(Value::Object(result))
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
