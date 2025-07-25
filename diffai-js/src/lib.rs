use napi::bindgen_prelude::*;
use napi_derive::napi;
use diffai_core::{diff as core_diff, DiffOptions, DiffaiSpecificOptions, OutputFormat, DiffResult};
use regex::Regex;

#[napi(object)]
pub struct JsDiffOptions {
    /// Numerical comparison tolerance
    pub epsilon: Option<f64>,
    
    /// Key to use for array element identification
    pub array_id_key: Option<String>,
    
    /// Regex pattern for keys to ignore
    pub ignore_keys_regex: Option<String>,
    
    /// Only show differences in paths containing this string
    pub path_filter: Option<String>,
    
    /// Output format
    pub output_format: Option<String>,
    
    /// Show unchanged values as well
    pub show_unchanged: Option<bool>,
    
    /// Show type information in output
    pub show_types: Option<bool>,
    
    /// Enable memory optimization for large files
    pub use_memory_optimization: Option<bool>,
    
    /// Batch size for memory optimization
    pub batch_size: Option<u32>,
    
    // diffai-specific options
    /// Enable ML analysis features
    pub ml_analysis_enabled: Option<bool>,
    
    /// Tensor comparison mode: "shape", "data", "both"
    pub tensor_comparison_mode: Option<String>,
    
    /// Model format: "pytorch", "safetensors", "onnx", etc.
    pub model_format: Option<String>,
    
    /// Enable scientific precision mode
    pub scientific_precision: Option<bool>,
    
    /// Threshold for detecting significant weight changes
    pub weight_threshold: Option<f64>,
    
    /// Enable activation function analysis
    pub activation_analysis: Option<bool>,
    
    /// Enable learning rate tracking
    pub learning_rate_tracking: Option<bool>,
    
    /// Enable optimizer comparison
    pub optimizer_comparison: Option<bool>,
    
    /// Enable loss tracking
    pub loss_tracking: Option<bool>,
    
    /// Enable accuracy tracking
    pub accuracy_tracking: Option<bool>,
    
    /// Enable model version checking
    pub model_version_check: Option<bool>,
}

#[napi(object)]
pub struct JsDiffResult {
    /// Type of difference ('Added', 'Removed', 'Modified', 'TypeChanged', etc.)
    pub diff_type: String,
    
    /// Path to the changed element
    pub path: String,
    
    /// Old value (for Modified/TypeChanged)
    pub old_value: Option<serde_json::Value>,
    
    /// New value (for Modified/TypeChanged/Added)
    pub new_value: Option<serde_json::Value>,
    
    /// Value (for Removed)
    pub value: Option<serde_json::Value>,
    
    /// Specific data for AI/ML results
    pub old_shape: Option<Vec<u32>>,
    pub new_shape: Option<Vec<u32>>,
    pub old_norm: Option<f64>,
    pub new_norm: Option<f64>,
    pub old_description: Option<String>,
    pub new_description: Option<String>,
    pub magnitude: Option<f64>,
    pub old_function: Option<String>,
    pub new_function: Option<String>,
    pub old_learning_rate: Option<f64>,
    pub new_learning_rate: Option<f64>,
    pub old_optimizer: Option<String>,
    pub new_optimizer: Option<String>,
    pub old_loss: Option<f64>,
    pub new_loss: Option<f64>,
    pub old_accuracy: Option<f64>,
    pub new_accuracy: Option<f64>,
    pub old_version: Option<String>,
    pub new_version: Option<String>,
}

/// Unified diff function for JavaScript/Node.js with AI/ML capabilities
/// 
/// Compare two JavaScript objects or values and return differences with AI/ML specific analysis.
/// 
/// # Arguments
/// 
/// * `old` - The old value (JavaScript object, array, or primitive)
/// * `new` - The new value (JavaScript object, array, or primitive)
/// * `options` - Optional configuration object
/// 
/// # Returns
/// 
/// Array of difference objects with AI/ML specific difference types
/// 
/// # Example
/// 
/// ```javascript
/// const { diff } = require('diffai-js');
/// 
/// const old = { model: { layers: [{ type: "dense", units: 128 }] } };
/// const new = { model: { layers: [{ type: "dense", units: 256 }] } };
/// const result = diff(old, new);
/// console.log(result); // [{ type: 'Modified', path: 'model.layers[0].units', oldValue: 128, newValue: 256 }]
/// ```
#[napi]
pub fn diff(old: serde_json::Value, new: serde_json::Value, options: Option<JsDiffOptions>) -> Result<Vec<JsDiffResult>> {
    // Convert options
    let rust_options = options.map(|opts| build_diff_options(opts)).transpose()?;
    
    // Perform diff
    let results = core_diff(&old, &new, rust_options.as_ref())
        .map_err(|e| Error::new(Status::GenericFailure, format!("Diff error: {}", e)))?;
    
    // Convert results to JavaScript objects
    let js_results = results.into_iter()
        .map(|result| convert_diff_result(result))
        .collect::<Result<Vec<_>>>()?;
    
    Ok(js_results)
}

// Helper functions

fn build_diff_options(js_options: JsDiffOptions) -> Result<DiffOptions> {
    let mut options = DiffOptions::default();
    
    // Core options
    if let Some(epsilon) = js_options.epsilon {
        options.epsilon = Some(epsilon);
    }
    
    if let Some(array_id_key) = js_options.array_id_key {
        options.array_id_key = Some(array_id_key);
    }
    
    if let Some(ignore_keys_regex) = js_options.ignore_keys_regex {
        let regex = Regex::new(&ignore_keys_regex)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid regex: {}", e)))?;
        options.ignore_keys_regex = Some(regex);
    }
    
    if let Some(path_filter) = js_options.path_filter {
        options.path_filter = Some(path_filter);
    }
    
    if let Some(output_format) = js_options.output_format {
        let format = OutputFormat::from_str(&output_format)
            .map_err(|e| Error::new(Status::InvalidArg, format!("Invalid output format: {}", e)))?;
        options.output_format = Some(format);
    }
    
    if let Some(show_unchanged) = js_options.show_unchanged {
        options.show_unchanged = Some(show_unchanged);
    }
    
    if let Some(show_types) = js_options.show_types {
        options.show_types = Some(show_types);
    }
    
    if let Some(use_memory_optimization) = js_options.use_memory_optimization {
        options.use_memory_optimization = Some(use_memory_optimization);
    }
    
    if let Some(batch_size) = js_options.batch_size {
        options.batch_size = Some(batch_size as usize);
    }
    
    // diffai-specific options
    let mut diffai_options = DiffaiSpecificOptions::default();
    let mut has_diffai_options = false;
    
    if let Some(ml_analysis_enabled) = js_options.ml_analysis_enabled {
        diffai_options.ml_analysis_enabled = Some(ml_analysis_enabled);
        has_diffai_options = true;
    }
    
    if let Some(tensor_comparison_mode) = js_options.tensor_comparison_mode {
        diffai_options.tensor_comparison_mode = Some(tensor_comparison_mode);
        has_diffai_options = true;
    }
    
    if let Some(model_format) = js_options.model_format {
        diffai_options.model_format = Some(model_format);
        has_diffai_options = true;
    }
    
    if let Some(scientific_precision) = js_options.scientific_precision {
        diffai_options.scientific_precision = Some(scientific_precision);
        has_diffai_options = true;
    }
    
    if let Some(weight_threshold) = js_options.weight_threshold {
        diffai_options.weight_threshold = Some(weight_threshold);
        has_diffai_options = true;
    }
    
    if let Some(activation_analysis) = js_options.activation_analysis {
        diffai_options.activation_analysis = Some(activation_analysis);
        has_diffai_options = true;
    }
    
    if let Some(learning_rate_tracking) = js_options.learning_rate_tracking {
        diffai_options.learning_rate_tracking = Some(learning_rate_tracking);
        has_diffai_options = true;
    }
    
    if let Some(optimizer_comparison) = js_options.optimizer_comparison {
        diffai_options.optimizer_comparison = Some(optimizer_comparison);
        has_diffai_options = true;
    }
    
    if let Some(loss_tracking) = js_options.loss_tracking {
        diffai_options.loss_tracking = Some(loss_tracking);
        has_diffai_options = true;
    }
    
    if let Some(accuracy_tracking) = js_options.accuracy_tracking {
        diffai_options.accuracy_tracking = Some(accuracy_tracking);
        has_diffai_options = true;
    }
    
    if let Some(model_version_check) = js_options.model_version_check {
        diffai_options.model_version_check = Some(model_version_check);
        has_diffai_options = true;
    }
    
    if has_diffai_options {
        options.diffai_options = Some(diffai_options);
    }
    
    Ok(options)
}

fn convert_diff_result(result: DiffResult) -> Result<JsDiffResult> {
    let mut js_result = JsDiffResult {
        diff_type: String::new(),
        path: String::new(),
        old_value: None,
        new_value: None,
        value: None,
        old_shape: None,
        new_shape: None,
        old_norm: None,
        new_norm: None,
        old_description: None,
        new_description: None,
        magnitude: None,
        old_function: None,
        new_function: None,
        old_learning_rate: None,
        new_learning_rate: None,
        old_optimizer: None,
        new_optimizer: None,
        old_loss: None,
        new_loss: None,
        old_accuracy: None,
        new_accuracy: None,
        old_version: None,
        new_version: None,
    };
    
    match result {
        DiffResult::Added(path, value) => {
            js_result.diff_type = "Added".to_string();
            js_result.path = path;
            js_result.new_value = Some(value);
        }
        DiffResult::Removed(path, value) => {
            js_result.diff_type = "Removed".to_string();
            js_result.path = path;
            js_result.value = Some(value);
        }
        DiffResult::Modified(path, old_val, new_val) => {
            js_result.diff_type = "Modified".to_string();
            js_result.path = path;
            js_result.old_value = Some(old_val);
            js_result.new_value = Some(new_val);
        }
        DiffResult::TypeChanged(path, old_val, new_val) => {
            js_result.diff_type = "TypeChanged".to_string();
            js_result.path = path;
            js_result.old_value = Some(old_val);
            js_result.new_value = Some(new_val);
        }
        // AI/ML specific diff results
        DiffResult::TensorShapeChanged(path, old_shape, new_shape) => {
            js_result.diff_type = "TensorShapeChanged".to_string();
            js_result.path = path;
            js_result.old_shape = Some(old_shape.into_iter().map(|x| x as u32).collect());
            js_result.new_shape = Some(new_shape.into_iter().map(|x| x as u32).collect());
        }
        DiffResult::TensorDataChanged(path, old_norm, new_norm) => {
            js_result.diff_type = "TensorDataChanged".to_string();
            js_result.path = path;
            js_result.old_norm = Some(old_norm);
            js_result.new_norm = Some(new_norm);
        }
        DiffResult::ModelArchitectureChanged(path, old_desc, new_desc) => {
            js_result.diff_type = "ModelArchitectureChanged".to_string();
            js_result.path = path;
            js_result.old_description = Some(old_desc);
            js_result.new_description = Some(new_desc);
        }
        DiffResult::WeightSignificantChange(path, magnitude) => {
            js_result.diff_type = "WeightSignificantChange".to_string();
            js_result.path = path;
            js_result.magnitude = Some(magnitude);
        }
        DiffResult::ActivationFunctionChanged(path, old_fn, new_fn) => {
            js_result.diff_type = "ActivationFunctionChanged".to_string();
            js_result.path = path;
            js_result.old_function = Some(old_fn);
            js_result.new_function = Some(new_fn);
        }
        DiffResult::LearningRateChanged(path, old_lr, new_lr) => {
            js_result.diff_type = "LearningRateChanged".to_string();
            js_result.path = path;
            js_result.old_learning_rate = Some(old_lr);
            js_result.new_learning_rate = Some(new_lr);
        }
        DiffResult::OptimizerChanged(path, old_opt, new_opt) => {
            js_result.diff_type = "OptimizerChanged".to_string();
            js_result.path = path;
            js_result.old_optimizer = Some(old_opt);
            js_result.new_optimizer = Some(new_opt);
        }
        DiffResult::LossChange(path, old_loss, new_loss) => {
            js_result.diff_type = "LossChange".to_string();
            js_result.path = path;
            js_result.old_loss = Some(old_loss);
            js_result.new_loss = Some(new_loss);
        }
        DiffResult::AccuracyChange(path, old_acc, new_acc) => {
            js_result.diff_type = "AccuracyChange".to_string();
            js_result.path = path;
            js_result.old_accuracy = Some(old_acc);
            js_result.new_accuracy = Some(new_acc);
        }
        DiffResult::ModelVersionChanged(path, old_version, new_version) => {
            js_result.diff_type = "ModelVersionChanged".to_string();
            js_result.path = path;
            js_result.old_version = Some(old_version);
            js_result.new_version = Some(new_version);
        }
    }
    
    Ok(js_result)
}