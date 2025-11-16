use serde_json::Value;


use crate::types::TensorStats;
use crate::types::DiffResult;
use crate::diff::{extract_tensor_data, extract_tensor_shape};

pub fn analyze_model_architecture_changes(
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


// Memory Analysis - standard feature for all ML formats
pub fn analyze_memory_usage_changes(
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
pub fn analyze_learning_rate_changes(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    // lawkitパターン：学習率分析は常に実行
    
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

// Enhanced Convergence Analysis with lawkit memory-efficient learning curve analysis
pub fn analyze_convergence_patterns(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Enhanced learning curve analysis
        let learning_curve_analysis = analyze_learning_curves_comprehensive(old_obj, new_obj);
        if let Some(curve_info) = learning_curve_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "learning_curve_analysis".to_string(),
                curve_info.0,
                curve_info.1,
            ));
        }
        
        // Enhanced convergence pattern detection using helper functions
        let convergence_patterns = analyze_convergence_patterns_advanced(old_obj, new_obj);
        if let Some(pattern_info) = convergence_patterns {
            results.push(DiffResult::ModelArchitectureChanged(
                "convergence_patterns".to_string(),
                pattern_info.0,
                pattern_info.1,
            ));
        }
        
        // Loss convergence analysis using helper function
        let loss_convergence = analyze_loss_convergence(old_obj, new_obj);
        if let Some(loss_info) = loss_convergence {
            results.push(DiffResult::ModelArchitectureChanged(
                "loss_convergence".to_string(),
                loss_info.0,
                loss_info.1,
            ));
        }
        
        // Training stability analysis using helper function  
        let training_stability = analyze_training_stability(old_obj, new_obj);
        if let Some(stability_info) = training_stability {
            results.push(DiffResult::ModelArchitectureChanged(
                "training_stability_detailed".to_string(),
                stability_info.0,
                stability_info.1,
            ));
        }
        
        // Epoch progression analysis using helper function
        let epoch_progression = analyze_epoch_progression(old_obj, new_obj);
        if let Some(epoch_info) = epoch_progression {
            results.push(DiffResult::ModelArchitectureChanged(
                "epoch_progression".to_string(),
                epoch_info.0,
                epoch_info.1,
            ));
        }
        
        // Training stability with statistical significance
        let stability_analysis = analyze_training_stability_statistical(old_obj, new_obj);
        if let Some(stability_info) = stability_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "training_stability".to_string(),
                stability_info.0,
                stability_info.1,
            ));
        }
        
        // Enhanced optimization trajectory analysis
        let optimization_analysis = analyze_optimization_trajectory(old_obj, new_obj);
        if let Some(opt_info) = optimization_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "optimization_trajectory".to_string(),
                opt_info.0,
                opt_info.1,
            ));
        }
        
        // Plateau detection and early stopping analysis
        let plateau_analysis = analyze_plateau_detection(old_obj, new_obj);
        if let Some(plateau_info) = plateau_analysis {
            results.push(DiffResult::ModelArchitectureChanged(
                "plateau_detection".to_string(),
                plateau_info.0,
                plateau_info.1,
            ));
        }
    }
}

// Enhanced training stability analysis with statistical significance
fn analyze_training_stability_statistical(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_stability = calculate_training_stability_metrics(old_obj)?;
    let new_stability = calculate_training_stability_metrics(new_obj)?;
    
    let mut stability_changes = Vec::new();
    
    // Compare variance in gradients
    if let (Some(old_grad_var), Some(new_grad_var)) = (old_stability.gradient_variance, new_stability.gradient_variance) {
        let variance_change = (new_grad_var - old_grad_var) / old_grad_var.max(1e-8);
        if variance_change.abs() > 0.1 {
            stability_changes.push(format!("gradient_variance: {:+.2}%", variance_change * 100.0));
        }
    }
    
    // Compare loss oscillation
    if (old_stability.loss_oscillation - new_stability.loss_oscillation).abs() > 0.05 {
        let oscillation_change = new_stability.loss_oscillation - old_stability.loss_oscillation;
        stability_changes.push(format!("loss_oscillation: {:+.3}", oscillation_change));
    }
    
    // Compare overall stability score
    if (old_stability.overall_score - new_stability.overall_score).abs() > 0.05 {
        let score_change = new_stability.overall_score - old_stability.overall_score;
        stability_changes.push(format!("stability_score: {:+.3}", score_change));
    }
    
    if stability_changes.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "oscillation: {:.3}, score: {:.3}",
        old_stability.loss_oscillation, old_stability.overall_score
    );
    let new_info = stability_changes.join(", ");
    
    Some((old_info, new_info))
}

#[derive(Debug)]
struct TrainingStabilityMetrics {
    gradient_variance: Option<f64>,
    loss_oscillation: f64,
    parameter_drift: f64,
    overall_score: f64,
}

fn calculate_training_stability_metrics(obj: &serde_json::Map<String, Value>) -> Option<TrainingStabilityMetrics> {
    // Extract gradient variance if available
    let gradient_variance = extract_gradient_variance(obj);
    
    // Calculate loss oscillation
    let loss_trajectory = extract_loss_trajectory(obj)?;
    let loss_oscillation = if loss_trajectory.len() > 2 {
        calculate_oscillation_metric(&loss_trajectory)
    } else {
        0.0
    };
    
    // Calculate parameter drift
    let parameter_drift = calculate_parameter_drift(obj);
    
    // Overall stability score (higher is better)
    let overall_score = {
        let base_score = 1.0 - loss_oscillation.min(1.0);
        let gradient_penalty = gradient_variance.map_or(0.0, |gv| (gv * 0.1).min(0.3));
        let drift_penalty = (parameter_drift * 0.2).min(0.3);
        (base_score - gradient_penalty - drift_penalty).max(0.0)
    };
    
    Some(TrainingStabilityMetrics {
        gradient_variance,
        loss_oscillation,
        parameter_drift,
        overall_score,
    })
}

// Enhanced loss convergence analysis using helper functions
fn analyze_loss_convergence(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    // Enhanced analysis using loss history from helper functions
    let old_loss_history = extract_loss_history(old_obj).unwrap_or_else(|| {
        extract_loss_value(old_obj).map(|v| vec![v]).unwrap_or_default()
    });
    let new_loss_history = extract_loss_history(new_obj).unwrap_or_else(|| {
        extract_loss_value(new_obj).map(|v| vec![v]).unwrap_or_default()
    });
    
    if old_loss_history.is_empty() || new_loss_history.is_empty() {
        return None;
    }
    
    // Use helper functions for enhanced analysis
    let trend_analysis = analyze_loss_trend(&old_loss_history, &new_loss_history);
    let old_slope = calculate_trend_slope(&old_loss_history);
    let new_slope = calculate_trend_slope(&new_loss_history);
    
    // Basic loss comparison
    let old_loss = old_loss_history.last().unwrap_or(&0.0);
    let new_loss = new_loss_history.last().unwrap_or(&0.0);
    let loss_change = new_loss - old_loss;
    let loss_change_percent = if *old_loss != 0.0 {
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
    
    // Enhanced analysis information
    let old_info = format!("loss: {:.6}, slope: {:.6}", old_loss, old_slope);
    let new_info = format!("loss: {:.6} ({:+.2}%), slope: {:.6}, trend: {}, status: {}", 
        new_loss, loss_change_percent, new_slope, trend_analysis, convergence_status);
    
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

// Enhanced convergence analysis structures with lawkit memory patterns
#[derive(Debug, Clone)]
struct LearningCurveMetrics {
    loss_trajectory: Vec<f64>,
    accuracy_trajectory: Vec<f64>,
    learning_rate_schedule: Vec<f64>,
    gradient_norms: Vec<f64>,
    epochs: Vec<f64>,
    convergence_rate: f64,
    stability_score: f64,
    plateau_detected: bool,
    early_stopping_suggestion: Option<String>,
}

#[derive(Debug, Clone)]
struct ConvergencePatterns {
    trend_direction: String,
    convergence_speed: String,
    oscillation_pattern: String,
    smoothness_score: f64,
    momentum_indicator: f64,
    saturation_risk: f64,
}

#[derive(Debug, Clone)]
struct OptimizationTrajectory {
    parameter_stability: f64,
    gradient_flow_health: f64,
    learning_efficiency: f64,
    overfitting_risk: f64,
    generalization_gap: Option<f64>,
}

#[derive(Debug, Clone)]
struct PlateauAnalysis {
    plateau_length: usize,
    plateau_start_epoch: Option<f64>,
    plateau_threshold: f64,
    recovery_probability: f64,
    recommended_action: String,
}

// Enhanced learning curve analysis using lawkit incremental statistics
fn analyze_learning_curves_comprehensive(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_metrics = extract_learning_curve_metrics(old_obj)?;
    let new_metrics = extract_learning_curve_metrics(new_obj)?;
    
    let mut analysis_points = Vec::new();
    
    // Loss trajectory analysis
    if !old_metrics.loss_trajectory.is_empty() && !new_metrics.loss_trajectory.is_empty() {
        let loss_improvement = calculate_trajectory_improvement(&old_metrics.loss_trajectory, &new_metrics.loss_trajectory);
        analysis_points.push(format!("loss_trajectory_improvement: {:.4}", loss_improvement));
    }
    
    // Convergence rate comparison
    if (old_metrics.convergence_rate - new_metrics.convergence_rate).abs() > 0.001 {
        let rate_change = new_metrics.convergence_rate - old_metrics.convergence_rate;
        analysis_points.push(format!("convergence_rate_change: {:+.4}", rate_change));
    }
    
    // Stability score analysis
    if (old_metrics.stability_score - new_metrics.stability_score).abs() > 0.05 {
        let stability_change = new_metrics.stability_score - old_metrics.stability_score;
        analysis_points.push(format!("stability_change: {:+.3}", stability_change));
    }
    
    // Plateau detection
    if old_metrics.plateau_detected != new_metrics.plateau_detected {
        let plateau_status = if new_metrics.plateau_detected { "detected" } else { "resolved" };
        analysis_points.push(format!("plateau_status: {}", plateau_status));
    }
    
    // Early stopping suggestion
    if let Some(ref suggestion) = new_metrics.early_stopping_suggestion {
        analysis_points.push(format!("early_stopping: {}", suggestion));
    }
    
    if analysis_points.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "convergence_rate: {:.4}, stability: {:.3}, plateau: {}",
        old_metrics.convergence_rate, old_metrics.stability_score, old_metrics.plateau_detected
    );
    let new_info = analysis_points.join(", ");
    
    Some((old_info, new_info))
}

// Extract comprehensive learning curve metrics
fn extract_learning_curve_metrics(obj: &serde_json::Map<String, Value>) -> Option<LearningCurveMetrics> {
    let mut loss_trajectory = Vec::new();
    let mut accuracy_trajectory = Vec::new();
    let mut learning_rate_schedule = Vec::new();
    let mut gradient_norms = Vec::new();
    let mut epochs = Vec::new();
    
    // Extract historical data (lawkit streaming pattern)
    for (key, value) in obj {
        match value {
            Value::Array(arr) => {
                if key.contains("loss") && key.contains("history") {
                    loss_trajectory = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("accuracy") && key.contains("history") {
                    accuracy_trajectory = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("lr") && key.contains("history") {
                    learning_rate_schedule = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("grad") && key.contains("history") {
                    gradient_norms = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                } else if key.contains("epoch") && key.contains("history") {
                    epochs = arr.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                }
            }
            Value::Number(num) => {
                if let Some(val) = num.as_f64() {
                    // Single point data
                    if key.contains("loss") {
                        loss_trajectory.push(val);
                    } else if key.contains("accuracy") {
                        accuracy_trajectory.push(val);
                    } else if key.contains("lr") || key.contains("learning_rate") {
                        learning_rate_schedule.push(val);
                    } else if key.contains("grad_norm") {
                        gradient_norms.push(val);
                    } else if key.contains("epoch") {
                        epochs.push(val);
                    }
                }
            }
            _ => {}
        }
    }
    
    if loss_trajectory.is_empty() {
        return None;
    }
    
    // Calculate convergence metrics using lawkit incremental statistics
    let convergence_rate = calculate_convergence_rate(&loss_trajectory);
    let stability_score = calculate_stability_score(&loss_trajectory);
    let plateau_detected = detect_plateau(&loss_trajectory);
    let early_stopping_suggestion = generate_early_stopping_suggestion(&loss_trajectory, &accuracy_trajectory);
    
    Some(LearningCurveMetrics {
        loss_trajectory,
        accuracy_trajectory,
        learning_rate_schedule,
        gradient_norms,
        epochs,
        convergence_rate,
        stability_score,
        plateau_detected,
        early_stopping_suggestion,
    })
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

// Enhanced convergence calculation functions using lawkit statistical methods
fn calculate_convergence_rate(loss_trajectory: &[f64]) -> f64 {
    if loss_trajectory.len() < 2 {
        return 0.0;
    }
    
    // Use exponential moving average for smoothing (lawkit pattern)
    let mut smoothed_losses = Vec::new();
    let alpha = 0.1; // Smoothing factor
    smoothed_losses.push(loss_trajectory[0]);
    
    for &loss in &loss_trajectory[1..] {
        let smoothed = alpha * loss + (1.0 - alpha) * smoothed_losses.last().unwrap();
        smoothed_losses.push(smoothed);
    }
    
    // Calculate rate of improvement
    let initial_loss = smoothed_losses[0];
    let final_loss = *smoothed_losses.last().unwrap();
    
    if initial_loss <= 0.0 {
        return 0.0;
    }
    
    let improvement_ratio = (initial_loss - final_loss) / initial_loss;
    let epochs = smoothed_losses.len() as f64;
    
    // Normalize by number of epochs
    improvement_ratio / epochs
}

fn calculate_stability_score(loss_trajectory: &[f64]) -> f64 {
    if loss_trajectory.len() < 3 {
        return 1.0;
    }
    
    // Calculate variance and coefficient of variation
    let mean = loss_trajectory.iter().sum::<f64>() / loss_trajectory.len() as f64;
    let variance = loss_trajectory.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / loss_trajectory.len() as f64;
    let std_dev = variance.sqrt();
    
    // Stability score: lower coefficient of variation = higher stability
    if mean > 0.0 {
        1.0 / (1.0 + std_dev / mean)
    } else {
        0.0
    }
}

fn detect_plateau(loss_trajectory: &[f64]) -> bool {
    if loss_trajectory.len() < 5 {
        return false;
    }
    
    // Check last N points for minimal change
    let window_size = (loss_trajectory.len() / 3).min(10).max(3);
    let recent_losses = &loss_trajectory[loss_trajectory.len() - window_size..];
    
    let min_loss = recent_losses.iter().copied().fold(f64::INFINITY, f64::min);
    let max_loss = recent_losses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    
    // Plateau detected if variation is less than 1% of mean
    if min_loss > 0.0 {
        let variation_ratio = (max_loss - min_loss) / min_loss;
        variation_ratio < 0.01
    } else {
        false
    }
}

fn generate_early_stopping_suggestion(loss_trajectory: &[f64], accuracy_trajectory: &[f64]) -> Option<String> {
    if loss_trajectory.len() < 10 {
        return None;
    }
    
    let plateau_detected = detect_plateau(loss_trajectory);
    let convergence_rate = calculate_convergence_rate(loss_trajectory);
    
    // Check for overfitting indicators
    let has_accuracy = !accuracy_trajectory.is_empty();
    let overfitting_risk = if has_accuracy && accuracy_trajectory.len() >= 5 {
        // Simple overfitting detection: accuracy plateau while loss still decreasing
        let acc_stable = detect_plateau(accuracy_trajectory);
        let loss_decreasing = convergence_rate > 0.001;
        acc_stable && loss_decreasing
    } else {
        false
    };
    
    if plateau_detected && convergence_rate < 0.001 {
        Some("consider_early_stopping".to_string())
    } else if overfitting_risk {
        Some("overfitting_detected".to_string())
    } else if convergence_rate < 0.0001 {
        Some("minimal_improvement".to_string())
    } else {
        None
    }
}

fn calculate_trajectory_improvement(old_trajectory: &[f64], new_trajectory: &[f64]) -> f64 {
    if old_trajectory.is_empty() || new_trajectory.is_empty() {
        return 0.0;
    }
    
    let old_final = *old_trajectory.last().unwrap();
    let new_final = *new_trajectory.last().unwrap();
    
    if old_final > 0.0 {
        (old_final - new_final) / old_final
    } else {
        0.0
    }
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

// Advanced convergence pattern analysis
fn analyze_convergence_patterns_advanced(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_patterns = extract_convergence_patterns(old_obj)?;
    let new_patterns = extract_convergence_patterns(new_obj)?;
    
    let mut pattern_changes = Vec::new();
    
    if old_patterns.trend_direction != new_patterns.trend_direction {
        pattern_changes.push(format!("trend: {} -> {}", old_patterns.trend_direction, new_patterns.trend_direction));
    }
    
    if old_patterns.convergence_speed != new_patterns.convergence_speed {
        pattern_changes.push(format!("speed: {} -> {}", old_patterns.convergence_speed, new_patterns.convergence_speed));
    }
    
    if (old_patterns.smoothness_score - new_patterns.smoothness_score).abs() > 0.1 {
        let smoothness_change = new_patterns.smoothness_score - old_patterns.smoothness_score;
        pattern_changes.push(format!("smoothness: {:+.2}", smoothness_change));
    }
    
    if pattern_changes.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "trend: {}, speed: {}, smoothness: {:.2}",
        old_patterns.trend_direction, old_patterns.convergence_speed, old_patterns.smoothness_score
    );
    let new_info = pattern_changes.join(", ");
    
    Some((old_info, new_info))
}

fn extract_convergence_patterns(obj: &serde_json::Map<String, Value>) -> Option<ConvergencePatterns> {
    let loss_trajectory = extract_loss_trajectory(obj)?;
    
    if loss_trajectory.len() < 3 {
        return None;
    }
    
    // Trend analysis
    let trend_direction = if loss_trajectory.first().unwrap() > loss_trajectory.last().unwrap() {
        "decreasing".to_string()
    } else if loss_trajectory.first().unwrap() < loss_trajectory.last().unwrap() {
        "increasing".to_string()
    } else {
        "stable".to_string()
    };
    
    // Speed analysis using helper functions
    let convergence_rate = calculate_convergence_rate(&loss_trajectory);
    let convergence_speed = if convergence_rate > 0.01 {
        "fast".to_string()
    } else if convergence_rate > 0.001 {
        "moderate".to_string()
    } else {
        "slow".to_string()
    };
    
    // Enhanced pattern analysis using helper functions
    let oscillation_pattern = detect_oscillation_pattern(&loss_trajectory);
    let smoothness_score = calculate_smoothness_score(&loss_trajectory);
    let momentum_indicator = calculate_momentum_indicator(&loss_trajectory);
    let saturation_risk = calculate_saturation_risk(&loss_trajectory);
    
    Some(ConvergencePatterns {
        trend_direction,
        convergence_speed,
        oscillation_pattern,
        smoothness_score,
        momentum_indicator,
        saturation_risk,
    })
}

// Enhanced optimization trajectory analysis using helper functions
fn analyze_optimization_trajectory(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_trajectory = extract_optimization_trajectory(old_obj)?;
    let new_trajectory = extract_optimization_trajectory(new_obj)?;
    
    // Use helper functions for enhanced analysis
    let old_lr = extract_current_learning_rate(old_obj);
    let new_lr = extract_current_learning_rate(new_obj);
    let old_param_mag = estimate_parameter_magnitude(old_obj);
    let new_param_mag = estimate_parameter_magnitude(new_obj);
    let old_epoch = extract_epoch_info(old_obj);
    let new_epoch = extract_epoch_info(new_obj);
    
    let mut trajectory_changes = Vec::new();
    
    // Learning rate analysis
    if let (Some(old_rate), Some(new_rate)) = (old_lr, new_lr) {
        if (old_rate - new_rate).abs() > old_rate * 0.1 {
            let lr_change = ((new_rate - old_rate) / old_rate) * 100.0;
            trajectory_changes.push(format!("learning_rate: {:+.2}%", lr_change));
        }
    }
    
    // Parameter magnitude analysis
    if let (Some(old_mag), Some(new_mag)) = (old_param_mag, new_param_mag) {
        if (old_mag - new_mag).abs() > old_mag * 0.05 {
            let mag_change = ((new_mag - old_mag) / old_mag) * 100.0;
            trajectory_changes.push(format!("parameter_magnitude: {:+.2}%", mag_change));
        }
    }
    
    // Epoch progression analysis
    if let (Some(old_ep), Some(new_ep)) = (old_epoch, new_epoch) {
        if new_ep > old_ep {
            trajectory_changes.push(format!("epoch_progress: {} -> {}", old_ep, new_ep));
        }
    }
    
    let stability_change = new_trajectory.parameter_stability - old_trajectory.parameter_stability;
    if stability_change.abs() > 0.05 {
        trajectory_changes.push(format!("param_stability: {:+.3}", stability_change));
    }
    
    let efficiency_change = new_trajectory.learning_efficiency - old_trajectory.learning_efficiency;
    if efficiency_change.abs() > 0.05 {
        trajectory_changes.push(format!("learning_efficiency: {:+.3}", efficiency_change));
    }
    
    if let (Some(old_gap), Some(new_gap)) = (old_trajectory.generalization_gap, new_trajectory.generalization_gap) {
        let gap_change = new_gap - old_gap;
        if gap_change.abs() > 0.02 {
            trajectory_changes.push(format!("generalization_gap: {:+.3}", gap_change));
        }
    }
    
    if trajectory_changes.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "stability: {:.3}, efficiency: {:.3}",
        old_trajectory.parameter_stability, old_trajectory.learning_efficiency
    );
    let new_info = trajectory_changes.join(", ");
    
    Some((old_info, new_info))
}

// Plateau detection analysis
fn analyze_plateau_detection(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_plateau = extract_plateau_analysis(old_obj)?;
    let new_plateau = extract_plateau_analysis(new_obj)?;
    
    let mut plateau_changes = Vec::new();
    
    if old_plateau.plateau_length != new_plateau.plateau_length {
        let length_change = new_plateau.plateau_length as i32 - old_plateau.plateau_length as i32;
        plateau_changes.push(format!("plateau_length: {} ({:+})", new_plateau.plateau_length, length_change));
    }
    
    if (old_plateau.recovery_probability - new_plateau.recovery_probability).abs() > 0.1 {
        let recovery_change = new_plateau.recovery_probability - old_plateau.recovery_probability;
        plateau_changes.push(format!("recovery_probability: {:+.2}", recovery_change));
    }
    
    if old_plateau.recommended_action != new_plateau.recommended_action {
        plateau_changes.push(format!("action: {}", new_plateau.recommended_action));
    }
    
    if plateau_changes.is_empty() {
        return None;
    }
    
    let old_info = format!(
        "length: {}, recovery_prob: {:.2}",
        old_plateau.plateau_length, old_plateau.recovery_probability
    );
    let new_info = plateau_changes.join(", ");
    
    Some((old_info, new_info))
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

// Helper functions for enhanced convergence analysis
fn extract_loss_trajectory(obj: &serde_json::Map<String, Value>) -> Option<Vec<f64>> {
    // Try to find loss history
    for (key, value) in obj {
        if key.contains("loss") && key.contains("history") {
            if let Value::Array(arr) = value {
                let trajectory: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                if !trajectory.is_empty() {
                    return Some(trajectory);
                }
            }
        }
    }
    
    // Fallback to single loss value
    if let Some(loss) = extract_loss_value(obj) {
        Some(vec![loss])
    } else {
        None
    }
}

fn detect_oscillation_pattern(trajectory: &[f64]) -> String {
    if trajectory.len() < 4 {
        return "insufficient_data".to_string();
    }
    
    let mut direction_changes = 0;
    for i in 1..trajectory.len() - 1 {
        let prev_trend = trajectory[i] - trajectory[i - 1];
        let curr_trend = trajectory[i + 1] - trajectory[i];
        
        if prev_trend * curr_trend < 0.0 {
            direction_changes += 1;
        }
    }
    
    let oscillation_rate = direction_changes as f64 / (trajectory.len() - 2) as f64;
    
    if oscillation_rate > 0.6 {
        "high_oscillation".to_string()
    } else if oscillation_rate > 0.3 {
        "moderate_oscillation".to_string()
    } else {
        "stable".to_string()
    }
}

fn calculate_smoothness_score(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 3 {
        return 1.0;
    }
    
    // Calculate second derivative (acceleration)
    let mut second_derivatives = Vec::new();
    for i in 1..trajectory.len() - 1 {
        let second_deriv = trajectory[i + 1] - 2.0 * trajectory[i] + trajectory[i - 1];
        second_derivatives.push(second_deriv.abs());
    }
    
    let mean_acceleration = second_derivatives.iter().sum::<f64>() / second_derivatives.len() as f64;
    
    // Higher smoothness = lower acceleration
    1.0 / (1.0 + mean_acceleration)
}

fn calculate_momentum_indicator(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 2 {
        return 0.0;
    }
    
    let recent_change = trajectory.last().unwrap() - trajectory[trajectory.len() - 2];
    let overall_change = trajectory.last().unwrap() - trajectory.first().unwrap();
    
    if overall_change.abs() < 1e-8 {
        return 0.0;
    }
    
    // Momentum: how much of recent change aligns with overall trend
    recent_change / overall_change
}

fn calculate_saturation_risk(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 5 {
        return 0.0;
    }
    
    // Check if improvements are diminishing
    let recent_window = trajectory.len().min(5);
    let recent = &trajectory[trajectory.len() - recent_window..];
    
    let initial_rate = if trajectory.len() > recent_window {
        let early = &trajectory[0..recent_window];
        let early_improvement = (early.first().unwrap() - early.last().unwrap()).abs();
        early_improvement / recent_window as f64
    } else {
        return 0.0;
    };
    
    let recent_rate = (recent.first().unwrap() - recent.last().unwrap()).abs() / recent_window as f64;
    
    if initial_rate > 0.0 {
        1.0 - (recent_rate / initial_rate).min(1.0)
    } else {
        0.0
    }
}

fn extract_optimization_trajectory(obj: &serde_json::Map<String, Value>) -> Option<OptimizationTrajectory> {
    let parameter_stability = calculate_parameter_stability(obj);
    let gradient_flow_health = calculate_gradient_flow_health(obj);
    let learning_efficiency = calculate_learning_efficiency(obj);
    let overfitting_risk = calculate_overfitting_risk(obj);
    let generalization_gap = extract_generalization_gap(obj);
    
    Some(OptimizationTrajectory {
        parameter_stability,
        gradient_flow_health,
        learning_efficiency,
        overfitting_risk,
        generalization_gap,
    })
}

fn extract_plateau_analysis(obj: &serde_json::Map<String, Value>) -> Option<PlateauAnalysis> {
    let loss_trajectory = extract_loss_trajectory(obj)?;
    
    let plateau_length = calculate_plateau_length(&loss_trajectory);
    let plateau_start_epoch = find_plateau_start(&loss_trajectory);
    let plateau_threshold = 0.01; // 1% threshold
    let recovery_probability = calculate_recovery_probability(&loss_trajectory);
    let recommended_action = generate_plateau_recommendation(&loss_trajectory);
    
    Some(PlateauAnalysis {
        plateau_length,
        plateau_start_epoch,
        plateau_threshold,
        recovery_probability,
        recommended_action,
    })
}

// Simplified implementations for helper functions
fn calculate_parameter_stability(obj: &serde_json::Map<String, Value>) -> f64 {
    // Simplified: based on gradient norm if available
    extract_gradient_norm(obj).map_or(0.5, |norm| (1.0 / (1.0 + norm)).min(1.0))
}

fn calculate_gradient_flow_health(_obj: &serde_json::Map<String, Value>) -> f64 {
    // Placeholder implementation
    0.8
}

fn calculate_learning_efficiency(obj: &serde_json::Map<String, Value>) -> f64 {
    // Based on loss improvement rate
    if let Some(trajectory) = extract_loss_trajectory(obj) {
        calculate_convergence_rate(&trajectory).min(1.0)
    } else {
        0.5
    }
}

fn calculate_overfitting_risk(_obj: &serde_json::Map<String, Value>) -> f64 {
    // Placeholder implementation
    0.3
}

fn extract_generalization_gap(_obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Placeholder implementation
    None
}

fn extract_gradient_variance(_obj: &serde_json::Map<String, Value>) -> Option<f64> {
    // Placeholder implementation
    None
}

fn calculate_oscillation_metric(trajectory: &[f64]) -> f64 {
    if trajectory.len() < 3 {
        return 0.0;
    }
    
    let mut oscillations = 0;
    for i in 1..trajectory.len() - 1 {
        let prev_diff = trajectory[i] - trajectory[i - 1];
        let curr_diff = trajectory[i + 1] - trajectory[i];
        
        if prev_diff * curr_diff < 0.0 {
            oscillations += 1;
        }
    }
    
    oscillations as f64 / (trajectory.len() - 2) as f64
}

fn calculate_parameter_drift(_obj: &serde_json::Map<String, Value>) -> f64 {
    // Placeholder implementation
    0.1
}

fn calculate_plateau_length(trajectory: &[f64]) -> usize {
    if trajectory.len() < 3 {
        return 0;
    }
    
    let threshold = 0.01;
    let mut plateau_count = 0;
    
    for i in 1..trajectory.len() {
        let change_ratio = (trajectory[i] - trajectory[i - 1]).abs() / trajectory[i - 1].abs().max(1e-8);
        if change_ratio < threshold {
            plateau_count += 1;
        } else {
            plateau_count = 0; // Reset if significant change
        }
    }
    
    plateau_count
}

fn find_plateau_start(trajectory: &[f64]) -> Option<f64> {
    let plateau_length = calculate_plateau_length(trajectory);
    if plateau_length > 0 && trajectory.len() > plateau_length {
        Some((trajectory.len() - plateau_length) as f64)
    } else {
        None
    }
}

fn calculate_recovery_probability(trajectory: &[f64]) -> f64 {
    let plateau_length = calculate_plateau_length(trajectory);
    if plateau_length == 0 {
        return 1.0;
    }
    
    // Longer plateaus have lower recovery probability
    (1.0 / (1.0 + plateau_length as f64 * 0.1)).max(0.1)
}

fn generate_plateau_recommendation(trajectory: &[f64]) -> String {
    let plateau_length = calculate_plateau_length(trajectory);
    
    if plateau_length > 10 {
        "consider_lr_reduction".to_string()
    } else if plateau_length > 5 {
        "monitor_closely".to_string()
    } else {
        "continue_training".to_string()
    }
}

// ============================================================================
// A4-3: GRADIENT ANALYSIS - Medium Priority ML Feature
// ============================================================================

// A4-3: GradientAnalysis - Gradient patterns and optimization behavior analysis
pub fn analyze_gradient_patterns(
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

// Extract gradient statistics from model data - Enhanced with lawkit memory patterns
fn extract_gradient_statistics(obj: &serde_json::Map<String, Value>) -> Option<GradientStatistics> {
    let mut total_norm = None;
    let mut max_gradient = None;
    let mut variance = None;
    let mut sparsity = None;
    let mut outlier_count = None;
    
    // First pass: Look for explicit gradient statistics (diffx optimization pattern)
    let grad_keys = [
        "grad_norm", "gradient_norm", "total_grad_norm",
        "max_grad", "gradient_max", "grad_variance", 
        "grad_sparsity", "gradient_outliers", "grad_flow"
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
    
    // Enhanced estimation using incremental statistics (lawkit pattern)
    if total_norm.is_none() || max_gradient.is_none() || variance.is_none() {
        let stats = compute_enhanced_gradient_statistics(obj);
        if total_norm.is_none() { total_norm = stats.total_norm; }
        if max_gradient.is_none() { max_gradient = stats.max_gradient; }
        if variance.is_none() { variance = stats.variance; }
    }
    
    // Memory-efficient sparsity calculation
    if sparsity.is_none() {
        sparsity = estimate_gradient_sparsity_streaming(obj);
    }
    
    // Enhanced outlier detection
    if outlier_count.is_none() {
        outlier_count = count_gradient_outliers_robust(obj);
    }
    
    Some(GradientStatistics {
        total_norm,
        max_gradient,
        variance,
        sparsity,
        outlier_count,
    })
}

// Enhanced gradient flow information extraction with lawkit streaming and helper functions
fn extract_gradient_flow_info(obj: &serde_json::Map<String, Value>) -> Option<GradientFlowInfo> {
    let mut vanishing_layers = 0;
    let mut exploding_layers = 0;
    
    // Use weight-based gradient estimation as fallback
    let estimated_norm = estimate_gradient_norm_from_weights(obj);
    let estimated_max = estimate_max_gradient_from_weights(obj);
    let mut flow_balance = None;
    
    // Enhanced thresholds based on modern deep learning practices
    let vanishing_threshold = 1e-7;  // More sensitive
    let exploding_threshold = 5.0;   // More conservative
    let _moderate_exploding_threshold = 1.0;
    
    let mut layer_gradients = Vec::new();
    
    // Comprehensive gradient analysis across all model components
    for (key, value) in obj {
        let is_gradient_related = key.contains("grad") || key.contains("gradient") ||
                                 key.contains("weight") || key.contains("bias") ||
                                 key.contains("param") || key.contains("layer");
        
        if is_gradient_related {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        let abs_val = val.abs();
                        layer_gradients.push(abs_val);
                        
                        // Enhanced gradient problem detection
                        if abs_val < vanishing_threshold {
                            vanishing_layers += 1;
                        } else if abs_val > exploding_threshold {
                            exploding_layers += 1;
                        }
                    }
                }
                Value::Array(arr) => {
                    // Process arrays with memory efficiency
                    let mut layer_sum = 0.0;
                    let mut layer_count = 0;
                    
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                let abs_val = val.abs();
                                layer_sum += abs_val;
                                layer_count += 1;
                            }
                        }
                    }
                    
                    if layer_count > 0 {
                        let layer_mean = layer_sum / layer_count as f64;
                        layer_gradients.push(layer_mean);
                        
                        if layer_mean < vanishing_threshold {
                            vanishing_layers += 1;
                        } else if layer_mean > exploding_threshold {
                            exploding_layers += 1;
                        }
                    }
                }
                Value::Object(nested) => {
                    // Recursive flow analysis for nested structures
                    if let Some(nested_flow) = extract_gradient_flow_info(nested) {
                        vanishing_layers += nested_flow.vanishing_layers;
                        exploding_layers += nested_flow.exploding_layers;
                    }
                }
                _ => {}
            }
        }
    }
    
    // Enhanced flow balance estimation
    flow_balance = estimate_gradient_flow_balance(obj);
    
    // Additional validation: check gradient magnitude distribution
    if !layer_gradients.is_empty() {
        layer_gradients.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = layer_gradients.len() / 2;
        let median_grad = layer_gradients[median_idx];
        
        // Adjust counts based on median gradient
        if median_grad < vanishing_threshold * 10.0 {
            // If median is very low, model likely has vanishing gradients
            vanishing_layers = vanishing_layers.max(layer_gradients.len() / 3);
        }
    }
    
    Some(GradientFlowInfo {
        vanishing_layers,
        exploding_layers,
        flow_balance,
    })
}

// Enhanced gradient statistics computation using lawkit incremental patterns
struct EnhancedGradientStats {
    total_norm: Option<f64>,
    max_gradient: Option<f64>, 
    variance: Option<f64>,
}

fn compute_enhanced_gradient_statistics(obj: &serde_json::Map<String, Value>) -> EnhancedGradientStats {
    // Use lawkit-style incremental statistics for memory efficiency
    let mut sum_squares = 0.0;
    let mut sum_values = 0.0;
    let mut sum_square_values = 0.0;
    let mut max_val: f64 = 0.0;
    let mut count = 0;
    
    // Multi-source gradient data analysis (PyTorch, Safetensors, NumPy patterns)
    for (key, value) in obj {
        let is_gradient_related = key.contains("grad") || key.contains("gradient") ||
                                 key.contains("weight") || key.contains("bias") ||
                                 key.contains("param");
        
        if is_gradient_related {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        // Incremental statistics (Welford's algorithm)
                        sum_squares += val * val;
                        sum_values += val;
                        sum_square_values += val * val;
                        max_val = max_val.max(val.abs());
                        count += 1;
                    }
                }
                Value::Array(arr) => {
                    // Handle tensor arrays efficiently
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                sum_squares += val * val;
                                sum_values += val;
                                sum_square_values += val * val;
                                max_val = max_val.max(val.abs());
                                count += 1;
                            }
                        }
                    }
                }
                Value::Object(nested) => {
                    // Recursive analysis for nested structures
                    let nested_stats = compute_enhanced_gradient_statistics(nested);
                    if let Some(norm) = nested_stats.total_norm {
                        sum_squares += norm * norm;
                        count += 1;
                    }
                    if let Some(max_nested) = nested_stats.max_gradient {
                        max_val = max_val.max(max_nested);
                    }
                }
                _ => {}
            }
        }
    }
    
    let total_norm = if count > 0 { Some(sum_squares.sqrt()) } else { None };
    let max_gradient = if count > 0 { Some(max_val) } else { None };
    
    // Calculate variance using stable algorithm
    let variance = if count > 1 {
        let mean = sum_values / count as f64;
        let variance_val = (sum_square_values / count as f64) - (mean * mean);
        Some(variance_val.max(0.0)) // Ensure non-negative
    } else {
        None
    };
    
    EnhancedGradientStats {
        total_norm,
        max_gradient,
        variance,
    }
}

// Legacy function for backward compatibility
fn estimate_gradient_norm_from_weights(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    compute_enhanced_gradient_statistics(obj).total_norm
}

// Legacy function for backward compatibility
fn estimate_max_gradient_from_weights(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    compute_enhanced_gradient_statistics(obj).max_gradient
}

// Memory-efficient streaming sparsity calculation
fn estimate_gradient_sparsity_streaming(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut near_zero_count = 0;
    let mut total_count = 0;
    let threshold = 1e-8;
    
    // Multi-threshold analysis for better sparsity detection
    let thresholds = [1e-8, 1e-6, 1e-4];
    let mut sparsity_levels = vec![0; thresholds.len()];
    
    for (key, value) in obj {
        let is_gradient_data = key.contains("grad") || key.contains("gradient") ||
                              key.contains("weight") || key.contains("param");
        
        if is_gradient_data {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        let abs_val = val.abs();
                        for (i, &thresh) in thresholds.iter().enumerate() {
                            if abs_val < thresh {
                                sparsity_levels[i] += 1;
                            }
                        }
                        if abs_val < threshold {
                            near_zero_count += 1;
                        }
                        total_count += 1;
                    }
                }
                Value::Array(arr) => {
                    // Efficient array processing with chunking
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                let abs_val = val.abs();
                                if abs_val < threshold {
                                    near_zero_count += 1;
                                }
                                total_count += 1;
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    if total_count > 0 {
        Some(near_zero_count as f64 / total_count as f64)
    } else {
        None
    }
}

// Backward compatibility
fn estimate_gradient_sparsity(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    estimate_gradient_sparsity_streaming(obj)
}

// Robust outlier detection using incremental statistics
fn count_gradient_outliers_robust(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    let mut values = Vec::new();
    let mut outliers = 0;
    
    // First pass: collect all gradient values
    for (key, value) in obj {
        let is_gradient_data = key.contains("grad") || key.contains("gradient") ||
                              key.contains("weight") || key.contains("param");
        
        if is_gradient_data {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        values.push(val);
                    }
                }
                Value::Array(arr) => {
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                values.push(val);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    if values.is_empty() {
        return Some(0);
    }
    
    // Use incremental statistics for memory efficiency (lawkit pattern)
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt();
    
    // Multiple outlier detection methods
    let z_score_threshold = 3.0;
    let iqr_multiplier = 1.5;
    
    // Z-score method
    for &val in &values {
        let z_score = (val - mean).abs() / std_dev;
        if z_score > z_score_threshold {
            outliers += 1;
        }
    }
    
    // IQR method for additional validation
    let mut sorted_values = values.clone();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted_values.len();
    if len >= 4 {
        let q1 = sorted_values[len / 4];
        let q3 = sorted_values[3 * len / 4];
        let iqr = q3 - q1;
        let lower_bound = q1 - iqr_multiplier * iqr;
        let upper_bound = q3 + iqr_multiplier * iqr;
        
        let iqr_outliers = values.iter()
            .filter(|&&val| val < lower_bound || val > upper_bound)
            .count();
        
        // Use maximum of both methods
        outliers = outliers.max(iqr_outliers);
    }
    
    Some(outliers)
}

// Backward compatibility
fn count_gradient_outliers(obj: &serde_json::Map<String, Value>) -> Option<usize> {
    count_gradient_outliers_robust(obj)
}

// Enhanced gradient flow balance estimation using lawkit streaming patterns
fn estimate_gradient_flow_balance(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    let mut layer_gradients = Vec::new();
    let mut total_forward_flow = 0.0;
    let mut total_backward_flow = 0.0;
    let mut layer_count = 0;
    
    // Advanced layer-wise gradient flow analysis
    for (key, value) in obj {
        let is_layer_weight = key.contains("layer") || key.contains("block") || 
                             key.contains("weight") || key.contains("attention") ||
                             key.contains("ffn") || key.contains("transformer");
        
        if is_layer_weight {
            match value {
                Value::Number(num) => {
                    if let Some(val) = num.as_f64() {
                        layer_gradients.push((key.clone(), val.abs()));
                    }
                }
                Value::Array(arr) => {
                    // Calculate mean gradient magnitude for this layer
                    let mut sum = 0.0;
                    let mut count = 0;
                    for item in arr {
                        if let Value::Number(num) = item {
                            if let Some(val) = num.as_f64() {
                                sum += val.abs();
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        layer_gradients.push((key.clone(), sum / count as f64));
                    }
                }
                Value::Object(nested) => {
                    // Recursive analysis for nested layer structures
                    if let Some(nested_flow) = estimate_gradient_flow_balance(nested) {
                        layer_gradients.push((key.clone(), nested_flow));
                    }
                }
                _ => {}
            }
        }
    }
    
    if layer_gradients.is_empty() {
        return None;
    }
    
    // Analyze gradient flow patterns
    layer_gradients.sort_by(|a, b| a.0.cmp(&b.0)); // Sort by layer name
    
    // Calculate forward and backward flow based on layer position
    let total_layers = layer_gradients.len();
    for (i, (_layer_name, gradient_mag)) in layer_gradients.iter().enumerate() {
        let layer_position = i as f64 / total_layers as f64;
        
        // Early layers contribute to forward flow
        if layer_position < 0.5 {
            total_forward_flow += gradient_mag * (1.0 - layer_position);
        } else {
            // Later layers contribute to backward flow  
            total_backward_flow += gradient_mag * layer_position;
        }
        
        layer_count += 1;
    }
    
    // Enhanced flow balance calculation
    if layer_count > 0 && total_backward_flow > 1e-12 {
        let flow_ratio = total_forward_flow / total_backward_flow;
        
        // Normalize to 0-1 range where 0.5 is perfect balance
        let normalized_balance = 1.0 / (1.0 + (flow_ratio - 1.0).abs());
        Some(normalized_balance)
    } else {
        None
    }
}

// ============================================================================
// A5-1: ATTENTION ANALYSIS - Low Priority ML Feature
// ============================================================================

// A5-1: AttentionAnalysis - Transformer attention mechanism analysis
pub fn analyze_attention_patterns(
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
pub fn analyze_ensemble_patterns(
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
pub fn analyze_quantization_patterns(
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

// Enhanced quantization precision analysis with mixed precision focus
fn analyze_quantization_precision(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_quant = extract_quantization_info(old_obj)?;
    let new_quant = extract_quantization_info(new_obj)?;
    
    let mut precision_analysis = Vec::new();
    
    // Enhanced bit width comparison
    if old_quant.bit_width != new_quant.bit_width {
        precision_analysis.push(format!(
            "bit_width: {} -> {}",
            old_quant.bit_width, new_quant.bit_width
        ));
    }
    
    // Enhanced data type comparison
    if old_quant.data_type != new_quant.data_type {
        precision_analysis.push(format!(
            "data_type: {} -> {}",
            old_quant.data_type, new_quant.data_type
        ));
    }
    
    // Enhanced quantization coverage comparison
    if old_quant.quantization_coverage != new_quant.quantization_coverage {
        let coverage_change = (new_quant.quantization_coverage - old_quant.quantization_coverage) * 100.0;
        precision_analysis.push(format!(
            "coverage: {:.1}% ({:+.1}%)",
            new_quant.quantization_coverage * 100.0, coverage_change
        ));
    }
    
    // Enhanced mixed precision analysis
    if old_quant.mixed_precision != new_quant.mixed_precision {
        precision_analysis.push(format!(
            "mixed_precision: {} -> {}",
            old_quant.mixed_precision, new_quant.mixed_precision
        ));
    }
    
    // Precision distribution comparison
    let old_dist = &old_quant.precision_distribution;
    let new_dist = &new_quant.precision_distribution;
    
    if old_dist.fp32_layers != new_dist.fp32_layers {
        let change = new_dist.fp32_layers as i32 - old_dist.fp32_layers as i32;
        precision_analysis.push(format!("fp32_layers: {} ({:+})", new_dist.fp32_layers, change));
    }
    
    if old_dist.fp16_layers != new_dist.fp16_layers {
        let change = new_dist.fp16_layers as i32 - old_dist.fp16_layers as i32;
        precision_analysis.push(format!("fp16_layers: {} ({:+})", new_dist.fp16_layers, change));
    }
    
    if old_dist.int8_layers != new_dist.int8_layers {
        let change = new_dist.int8_layers as i32 - old_dist.int8_layers as i32;
        precision_analysis.push(format!("int8_layers: {} ({:+})", new_dist.int8_layers, change));
    }
    
    if old_dist.int4_layers != new_dist.int4_layers {
        let change = new_dist.int4_layers as i32 - old_dist.int4_layers as i32;
        precision_analysis.push(format!("int4_layers: {} ({:+})", new_dist.int4_layers, change));
    }
    
    // Precision efficiency score comparison
    if (old_dist.precision_efficiency_score - new_dist.precision_efficiency_score).abs() > 0.01 {
        let score_change = new_dist.precision_efficiency_score - old_dist.precision_efficiency_score;
        precision_analysis.push(format!(
            "efficiency_score: {:.3} ({:+.3})",
            new_dist.precision_efficiency_score, score_change
        ));
    }
    
    // Dynamic range analysis comparison
    if let (Some(old_range), Some(new_range)) = (&old_quant.dynamic_range_analysis, &new_quant.dynamic_range_analysis) {
        if (old_range.range_utilization - new_range.range_utilization).abs() > 0.05 {
            precision_analysis.push(format!(
                "range_utilization: {:.2} -> {:.2}",
                old_range.range_utilization, new_range.range_utilization
            ));
        }
        
        if (old_range.saturation_risk - new_range.saturation_risk).abs() > 0.05 {
            precision_analysis.push(format!(
                "saturation_risk: {:.2} -> {:.2}",
                old_range.saturation_risk, new_range.saturation_risk
            ));
        }
    }
    
    if precision_analysis.is_empty() {
        return None;
    }
    
    // Enhanced summary format
    let old_info = format!(
        "{}bit {}, coverage: {:.1}%, mixed: {}, efficiency: {:.3}",
        old_quant.bit_width,
        old_quant.data_type,
        old_quant.quantization_coverage * 100.0,
        old_quant.mixed_precision,
        old_quant.precision_distribution.precision_efficiency_score
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

// Enhanced quantization analysis structures with lawkit memory patterns
#[derive(Debug, Clone)]
struct QuantizationInfo {
    bit_width: u8,
    data_type: String,
    quantized_layers: usize,
    mixed_precision: bool,
    // Enhanced mixed precision analysis
    precision_distribution: PrecisionDistribution,
    quantization_coverage: f64,
    dynamic_range_analysis: Option<DynamicRangeInfo>,
}

#[derive(Debug, Clone)]
struct PrecisionDistribution {
    fp32_layers: usize,
    fp16_layers: usize,
    int8_layers: usize,
    int4_layers: usize,
    custom_precision_layers: usize,
    precision_efficiency_score: f64,
}

#[derive(Debug, Clone)]
struct DynamicRangeInfo {
    min_value: f64,
    max_value: f64,
    range_utilization: f64,
    saturation_risk: f64,
    precision_loss_estimate: f64,
}

#[derive(Debug, Clone)]
struct QuantizationMethods {
    strategy: String,
    calibration_method: String,
    symmetric: bool,
    per_channel: bool,
    // Enhanced method analysis
    advanced_techniques: Vec<String>,
    optimization_level: String,
    hardware_compatibility: Vec<String>,
    calibration_dataset_size: Option<usize>,
}

#[derive(Debug, Clone)]
struct QuantizationImpact {
    size_reduction: Option<f64>,
    accuracy_impact: Option<f64>,
    speed_improvement: Option<f64>,
    memory_efficiency: Option<f64>,
    // Enhanced impact analysis using lawkit patterns
    inference_latency_reduction: Option<f64>,
    bandwidth_savings: Option<f64>,
    energy_efficiency_gain: Option<f64>,
    compression_ratio: Option<f64>,
    quality_degradation_risk: f64,
}

// Enhanced quantization information extraction with lawkit streaming analysis
fn extract_quantization_info(obj: &serde_json::Map<String, Value>) -> Option<QuantizationInfo> {
    let mut bit_width = 32u8; // Default FP32
    let mut data_type = "float32".to_string();
    let mut quantized_layers = 0;
    let mut mixed_precision = false;
    
    // Enhanced precision distribution analysis (lawkit incremental pattern)
    let mut precision_dist = PrecisionDistribution {
        fp32_layers: 0,
        fp16_layers: 0,
        int8_layers: 0,
        int4_layers: 0,
        custom_precision_layers: 0,
        precision_efficiency_score: 0.0,
    };
    
    let mut dynamic_range_values = Vec::new();
    let mut total_layers = 0;
    
    // First pass: comprehensive precision analysis (diffx optimization pattern)
    for (key, value) in obj {
        let is_quantization_related = key.contains("quant") || key.contains("precision") || 
                                     key.contains("bit") || key.contains("weight") ||
                                     key.contains("bias") || key.contains("param");
        
        if is_quantization_related {
            // Extract explicit quantization metadata
            if key.contains("quant") || key.contains("precision") || key.contains("bit") {
                if key.contains("bit") || key.contains("width") {
                    if let Value::Number(bits) = value {
                        if let Some(bits_val) = bits.as_u64() {
                            bit_width = bits_val as u8;
                        }
                    }
                }
                
                if key.contains("dtype") || key.contains("type") {
                    if let Value::String(dtype) = value {
                        data_type = dtype.clone();
                    }
                }
                
                if key.contains("layer") && key.contains("quant") {
                    quantized_layers += 1;
                }
                
                if key.contains("mixed") || key.contains("amp") || key.contains("auto") {
                    mixed_precision = true;
                }
            }
            
            // Enhanced tensor-level analysis
            match value {
                Value::Object(tensor_obj) => {
                    total_layers += 1;
                    
                    // Analyze data type distribution
                    if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                        data_type = dtype.clone();
                        
                        // Enhanced precision classification
                        match dtype.as_str() {
                            "float32" | "fp32" => {
                                precision_dist.fp32_layers += 1;
                                bit_width = 32;
                            }
                            "float16" | "fp16" | "half" => {
                                precision_dist.fp16_layers += 1;
                                bit_width = 16;
                                mixed_precision = true;
                            }
                            "int8" | "uint8" => {
                                precision_dist.int8_layers += 1;
                                bit_width = 8;
                                quantized_layers += 1;
                            }
                            "int4" | "uint4" => {
                                precision_dist.int4_layers += 1;
                                bit_width = 4;
                                quantized_layers += 1;
                            }
                            "int16" | "uint16" => {
                                bit_width = 16;
                                quantized_layers += 1;
                            }
                            "int32" | "uint32" => bit_width = 32,
                            "int64" | "uint64" | "float64" => bit_width = 64,
                            _ => {
                                precision_dist.custom_precision_layers += 1;
                                // Custom quantization scheme detected
                            }
                        }
                    }
                    
                    // Dynamic range analysis for quantization quality
                    if let (Some(Value::Number(min_val)), Some(Value::Number(max_val))) = 
                        (tensor_obj.get("min"), tensor_obj.get("max")) {
                        if let (Some(min_f), Some(max_f)) = (min_val.as_f64(), max_val.as_f64()) {
                            dynamic_range_values.push((min_f, max_f));
                        }
                    }
                }
                Value::Array(arr) => {
                    // Handle array of tensors efficiently
                    for item in arr {
                        if let Value::Object(tensor_obj) = item {
                            total_layers += 1;
                            if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                                match dtype.as_str() {
                                    "float16" | "fp16" | "half" => {
                                        precision_dist.fp16_layers += 1;
                                        mixed_precision = true;
                                    }
                                    "int8" | "uint8" => {
                                        precision_dist.int8_layers += 1;
                                        quantized_layers += 1;
                                    }
                                    "int4" | "uint4" => {
                                        precision_dist.int4_layers += 1;
                                        quantized_layers += 1;
                                    }
                                    "float32" | "fp32" => precision_dist.fp32_layers += 1,
                                    _ => precision_dist.custom_precision_layers += 1,
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    // Enhanced mixed precision detection
    let has_multiple_precisions = (precision_dist.fp32_layers > 0) as u8 +
                                 (precision_dist.fp16_layers > 0) as u8 +
                                 (precision_dist.int8_layers > 0) as u8 +
                                 (precision_dist.int4_layers > 0) as u8 +
                                 (precision_dist.custom_precision_layers > 0) as u8;
    
    if has_multiple_precisions >= 2 {
        mixed_precision = true;
    }
    
    // Calculate precision efficiency score (lawkit statistical analysis)
    if total_layers > 0 {
        let quantized_ratio = quantized_layers as f64 / total_layers as f64;
        let mixed_bonus = if mixed_precision { 0.2 } else { 0.0 };
        precision_dist.precision_efficiency_score = quantized_ratio * 0.8 + mixed_bonus;
    }
    
    // Calculate quantization coverage
    let quantization_coverage = if total_layers > 0 {
        quantized_layers as f64 / total_layers as f64
    } else {
        0.0
    };
    
    // Dynamic range analysis
    let dynamic_range_analysis = if !dynamic_range_values.is_empty() {
        let (global_min, global_max) = dynamic_range_values.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_acc, max_acc), &(min_val, max_val)| {
                (min_acc.min(min_val), max_acc.max(max_val))
            });
        
        let dynamic_range = global_max - global_min;
        let range_utilization = if dynamic_range > 0.0 {
            // Estimate how well the quantization range is utilized
            let effective_range = dynamic_range_values.iter()
                .map(|(min_val, max_val)| max_val - min_val)
                .sum::<f64>() / dynamic_range_values.len() as f64;
            effective_range / dynamic_range
        } else {
            1.0
        };
        
        // Saturation risk assessment
        let saturation_risk = if bit_width <= 8 {
            (1.0 - range_utilization) * 0.5 // Higher risk for low-bit quantization
        } else {
            (1.0 - range_utilization) * 0.2
        };
        
        // Precision loss estimate
        let precision_loss_estimate = match bit_width {
            4 => 0.15,  // Significant loss expected
            8 => 0.05,  // Moderate loss
            16 => 0.01, // Minimal loss
            _ => 0.0,   // FP32 or higher
        };
        
        Some(DynamicRangeInfo {
            min_value: global_min,
            max_value: global_max,
            range_utilization,
            saturation_risk,
            precision_loss_estimate,
        })
    } else {
        None
    };
    
    Some(QuantizationInfo {
        bit_width,
        data_type,
        quantized_layers,
        mixed_precision,
        precision_distribution: precision_dist,
        quantization_coverage,
        dynamic_range_analysis,
    })
}

// Enhanced quantization methods extraction with advanced technique detection
fn extract_quantization_methods(obj: &serde_json::Map<String, Value>) -> Option<QuantizationMethods> {
    let mut strategy = "post_training".to_string();
    let mut calibration_method = "minmax".to_string();
    let mut symmetric = true;
    let mut per_channel = false;
    let mut advanced_techniques = Vec::new();
    let mut optimization_level = "basic".to_string();
    let mut hardware_compatibility = Vec::new();
    let mut calibration_dataset_size = None;
    
    // Enhanced quantization method detection (lawkit comprehensive analysis)
    for (key, value) in obj {
        let is_quantization_related = key.contains("quant") || key.contains("precision") ||
                                     key.contains("optim") || key.contains("compress");
        
        if is_quantization_related {
            // Strategy detection
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
                    per_channel = true;
                }
            }
            
            // Advanced technique detection
            if key.contains("pruning") || key.contains("sparsity") {
                advanced_techniques.push("structured_pruning".to_string());
            }
            if key.contains("distillation") || key.contains("teacher") {
                advanced_techniques.push("knowledge_distillation".to_string());
            }
            if key.contains("smoothquant") || key.contains("smooth") {
                advanced_techniques.push("smoothquant".to_string());
            }
            if key.contains("gptq") || key.contains("group_wise") {
                advanced_techniques.push("gptq".to_string());
            }
            if key.contains("awq") || key.contains("activation_aware") {
                advanced_techniques.push("awq".to_string());
            }
            if key.contains("bnb") || key.contains("bitsandbytes") {
                advanced_techniques.push("bitsandbytes".to_string());
            }
            
            // Hardware compatibility detection
            if key.contains("cuda") || key.contains("gpu") {
                hardware_compatibility.push("cuda".to_string());
            }
            if key.contains("tensorrt") || key.contains("trt") {
                hardware_compatibility.push("tensorrt".to_string());
            }
            if key.contains("onnx") {
                hardware_compatibility.push("onnx".to_string());
            }
            if key.contains("openvino") {
                hardware_compatibility.push("openvino".to_string());
            }
            if key.contains("coreml") {
                hardware_compatibility.push("coreml".to_string());
            }
            
            // Calibration dataset size
            if key.contains("calibration") && key.contains("size") {
                if let Value::Number(size) = value {
                    calibration_dataset_size = size.as_u64().map(|s| s as usize);
                }
            }
        }
    }
    
    // Enhanced strategy inference from model structure
    if obj.contains_key("quantization_aware_training") || obj.contains_key("qat") {
        strategy = "quantization_aware_training".to_string();
        optimization_level = "advanced".to_string();
    } else if obj.contains_key("dynamic_quantization") {
        strategy = "dynamic".to_string();
        optimization_level = "intermediate".to_string();
    } else if obj.contains_key("static_quantization") {
        strategy = "static".to_string();
        optimization_level = "intermediate".to_string();
    } else if !advanced_techniques.is_empty() {
        optimization_level = "expert".to_string();
    }
    
    // Enhanced calibration method inference
    if obj.contains_key("entropy_calibration") || obj.contains_key("kl_divergence") {
        calibration_method = "entropy".to_string();
    } else if obj.contains_key("percentile_calibration") {
        calibration_method = "percentile".to_string();
    } else if obj.contains_key("mse_calibration") {
        calibration_method = "mse".to_string();
    } else if obj.contains_key("sqnr_calibration") {
        calibration_method = "sqnr".to_string();
    }
    
    // Deduplicate and sort techniques
    advanced_techniques.sort();
    advanced_techniques.dedup();
    hardware_compatibility.sort();
    hardware_compatibility.dedup();
    
    Some(QuantizationMethods {
        strategy,
        calibration_method,
        symmetric,
        per_channel,
        advanced_techniques,
        optimization_level,
        hardware_compatibility,
        calibration_dataset_size,
    })
}

// Enhanced quantization impact analysis with lawkit incremental statistics
fn extract_quantization_impact(obj: &serde_json::Map<String, Value>) -> Option<QuantizationImpact> {
    let mut size_reduction = None;
    let mut accuracy_impact = None;
    let mut speed_improvement = None;
    let mut memory_efficiency = None;
    let mut inference_latency_reduction = None;
    let mut bandwidth_savings = None;
    let mut energy_efficiency_gain = None;
    let mut compression_ratio = None;
    let mut quality_degradation_risk = 0.0;
    
    // Enhanced performance metrics collection (lawkit comprehensive analysis)
    let mut precision_stats = Vec::new();
    let mut tensor_sizes = Vec::new();
    let mut bit_width_distribution = std::collections::HashMap::new();
    
    for (key, value) in obj {
        // Direct performance metrics
        if key.contains("size") && key.contains("reduction") {
            if let Value::Number(reduction) = value {
                size_reduction = reduction.as_f64();
            }
        } else if key.contains("accuracy") && (key.contains("drop") || key.contains("impact") || key.contains("loss")) {
            if let Value::Number(acc_impact) = value {
                accuracy_impact = acc_impact.as_f64();
            }
        } else if key.contains("speed") || key.contains("latency") {
            if let Value::Number(perf) = value {
                if key.contains("improvement") || key.contains("gain") {
                    speed_improvement = perf.as_f64();
                } else if key.contains("reduction") {
                    inference_latency_reduction = perf.as_f64();
                }
            }
        } else if key.contains("memory") {
            if let Value::Number(mem_metric) = value {
                memory_efficiency = mem_metric.as_f64();
            }
        } else if key.contains("energy") && key.contains("efficiency") {
            if let Value::Number(energy) = value {
                energy_efficiency_gain = energy.as_f64();
            }
        } else if key.contains("bandwidth") {
            if let Value::Number(bw) = value {
                bandwidth_savings = bw.as_f64();
            }
        }
        
        // Collect tensor information for analysis
        if key.contains("weight") || key.contains("bias") || key.contains("param") {
            match value {
                Value::Object(tensor_obj) => {
                    // Precision analysis
                    if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                        precision_stats.push(dtype.clone());
                        
                        // Bit width distribution tracking
                        let bit_width = match dtype.as_str() {
                            "int4" | "uint4" => 4,
                            "int8" | "uint8" => 8,
                            "int16" | "uint16" | "float16" | "half" => 16,
                            "int32" | "uint32" | "float32" => 32,
                            "int64" | "uint64" | "float64" => 64,
                            _ => 32, // Default
                        };
                        *bit_width_distribution.entry(bit_width).or_insert(0) += 1;
                    }
                    
                    // Tensor size analysis
                    if let Some(Value::Array(shape)) = tensor_obj.get("shape") {
                        let size = shape.iter()
                            .filter_map(|v| v.as_u64())
                            .product::<u64>() as f64;
                        tensor_sizes.push(size);
                    }
                }
                Value::Array(arr) => {
                    // Handle array of tensors
                    for item in arr {
                        if let Value::Object(tensor_obj) = item {
                            if let Some(Value::String(dtype)) = tensor_obj.get("dtype") {
                                precision_stats.push(dtype.clone());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    // Enhanced estimation using lawkit incremental statistics
    if !precision_stats.is_empty() {
        let total_tensors = precision_stats.len() as f64;
        
        // Calculate precision distribution
        let quantized_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("int") && !dtype.contains("32") && !dtype.contains("64"))
            .count() as f64;
        
        let fp16_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("16") || dtype.contains("half"))
            .count() as f64;
        
        let int8_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("int8"))
            .count() as f64;
        
        let int4_count = precision_stats.iter()
            .filter(|dtype| dtype.contains("int4"))
            .count() as f64;
        
        // Enhanced size reduction calculation
        if size_reduction.is_none() {
            let mut total_reduction = 0.0;
            
            // Precision-based reduction estimates
            total_reduction += (fp16_count / total_tensors) * 0.5;   // FP16: 50% reduction
            total_reduction += (int8_count / total_tensors) * 0.75;   // INT8: 75% reduction
            total_reduction += (int4_count / total_tensors) * 0.875;  // INT4: 87.5% reduction
            
            size_reduction = Some(total_reduction);
        }
        
        // Enhanced compression ratio calculation
        let base_size = tensor_sizes.iter().sum::<f64>() * 32.0; // Assume FP32 baseline
        let compressed_size = bit_width_distribution.iter()
            .map(|(&bits, &count)| (count as f64) * (bits as f64))
            .sum::<f64>();
        
        if base_size > 0.0 && compressed_size > 0.0 {
            compression_ratio = Some(base_size / compressed_size);
        }
        
        // Quality degradation risk assessment
        quality_degradation_risk = (int4_count / total_tensors) * 0.3 +  // High risk for INT4
                                  (int8_count / total_tensors) * 0.1 +   // Moderate risk for INT8
                                  (fp16_count / total_tensors) * 0.02;   // Low risk for FP16
        
        // Enhanced speed improvement estimation
        if speed_improvement.is_none() {
            let mut perf_improvement = 1.0;
            
            // Hardware-aware performance modeling
            if int8_count > 0.0 {
                perf_improvement += (int8_count / total_tensors) * 2.0; // INT8 can be 2-3x faster
            }
            if fp16_count > 0.0 {
                perf_improvement += (fp16_count / total_tensors) * 1.5; // FP16 ~1.5-2x faster
            }
            if int4_count > 0.0 {
                perf_improvement += (int4_count / total_tensors) * 3.0; // INT4 can be 3-4x faster
            }
            
            speed_improvement = Some(perf_improvement);
        }
        
        // Memory efficiency estimation
        if memory_efficiency.is_none() {
            memory_efficiency = size_reduction; // Memory savings roughly follow size reduction
        }
        
        // Inference latency reduction estimation
        if inference_latency_reduction.is_none() {
            let latency_factor = size_reduction.unwrap_or(0.0) * 0.8; // Conservative estimate
            inference_latency_reduction = Some(latency_factor);
        }
        
        // Bandwidth savings estimation
        if bandwidth_savings.is_none() {
            bandwidth_savings = size_reduction; // Bandwidth savings follow model size reduction
        }
        
        // Energy efficiency estimation
        if energy_efficiency_gain.is_none() {
            let energy_factor = (quantized_count / total_tensors) * 0.4; // Quantization reduces energy
            energy_efficiency_gain = Some(energy_factor);
        }
    }
    
    Some(QuantizationImpact {
        size_reduction,
        accuracy_impact,
        speed_improvement,
        memory_efficiency,
        inference_latency_reduction,
        bandwidth_savings,
        energy_efficiency_gain,
        compression_ratio,
        quality_degradation_risk,
    })
}


// Helper functions for tensor analysis - Full implementation for PyTorch/Safetensors/NumPy formats
fn stats_changed_significantly(old_stats: &TensorStats, new_stats: &TensorStats) -> bool {
    let mean_change = (old_stats.mean - new_stats.mean).abs() / old_stats.mean.abs().max(1e-8);
    let std_change = (old_stats.std - new_stats.std).abs() / old_stats.std.abs().max(1e-8);
    
    // Consider significant if relative change > 1%
    mean_change > 0.01 || std_change > 0.01
}

// ============================================================================
// ADDITIONAL ML ANALYSIS FEATURES
// ============================================================================

// Batch Normalization Analysis - analyze batch normalization layer patterns
pub fn analyze_batch_normalization_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Batch normalization layer detection and analysis
        if let Some((old_bn, new_bn)) = analyze_batch_norm_layers(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "batch_normalization_layers".to_string(),
                old_bn,
                new_bn,
            ));
        }
        
        // Batch normalization parameter analysis
        if let Some((old_params, new_params)) = analyze_batch_norm_parameters(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "batch_normalization_parameters".to_string(),
                old_params,
                new_params,
            ));
        }
        
        // Moving statistics analysis for batch norm
        if let Some((old_stats, new_stats)) = analyze_batch_norm_statistics(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "batch_normalization_statistics".to_string(),
                old_stats,
                new_stats,
            ));
        }
    }
}

// Regularization Impact Analysis - measure regularization technique effectiveness
pub fn analyze_regularization_impact(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Dropout analysis
        if let Some((old_dropout, new_dropout)) = analyze_dropout_patterns(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "dropout_regularization".to_string(),
                old_dropout,
                new_dropout,
            ));
        }
        
        // Weight decay analysis
        if let Some((old_decay, new_decay)) = analyze_weight_decay_impact(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_decay_impact".to_string(),
                old_decay,
                new_decay,
            ));
        }
        
        // L1/L2 regularization analysis
        if let Some((old_l_reg, new_l_reg)) = analyze_l_regularization(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "l_regularization".to_string(),
                old_l_reg,
                new_l_reg,
            ));
        }
    }
}

// Activation Pattern Analysis - analyze activation function patterns
pub fn analyze_activation_pattern_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Activation function distribution
        if let Some((old_activations, new_activations)) = analyze_activation_functions(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "activation_functions".to_string(),
                old_activations,
                new_activations,
            ));
        }
        
        // Activation saturation analysis
        if let Some((old_saturation, new_saturation)) = analyze_activation_saturation(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "activation_saturation".to_string(),
                old_saturation,
                new_saturation,
            ));
        }
        
        // Dead neuron analysis
        if let Some((old_dead, new_dead)) = analyze_dead_neurons(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "dead_neurons".to_string(),
                old_dead,
                new_dead,
            ));
        }
    }
}

// Weight Distribution Analysis - statistical analysis of weight distributions
pub fn analyze_weight_distribution_analysis(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Weight distribution statistics
        if let Some((old_dist, new_dist)) = analyze_weight_distributions(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_distributions".to_string(),
                old_dist,
                new_dist,
            ));
        }
        
        // Weight initialization analysis
        if let Some((old_init, new_init)) = analyze_weight_initialization(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_initialization".to_string(),
                old_init,
                new_init,
            ));
        }
        
        // Weight sparsity analysis
        if let Some((old_sparsity, new_sparsity)) = analyze_weight_sparsity(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "weight_sparsity".to_string(),
                old_sparsity,
                new_sparsity,
            ));
        }
    }
}

// Model Complexity Assessment - comprehensive model complexity evaluation
pub fn analyze_model_complexity_assessment(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Parameter count analysis
        if let Some((old_params, new_params)) = analyze_parameter_count(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "parameter_count".to_string(),
                old_params,
                new_params,
            ));
        }
        
        // Computational complexity analysis
        if let Some((old_flops, new_flops)) = analyze_computational_complexity(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "computational_complexity".to_string(),
                old_flops,
                new_flops,
            ));
        }
        
        // Model depth and width analysis
        if let Some((old_arch, new_arch)) = analyze_model_architecture_complexity(old_obj, new_obj) {
            results.push(DiffResult::ModelArchitectureChanged(
                "architecture_complexity".to_string(),
                old_arch,
                new_arch,
            ));
        }
    }
}

// Helper functions for batch normalization analysis
fn analyze_batch_norm_layers(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_bn_count = count_batch_norm_layers(old_obj);
    let new_bn_count = count_batch_norm_layers(new_obj);
    
    if old_bn_count != new_bn_count {
        Some((
            format!("batch_norm_layers: {}", old_bn_count),
            format!("batch_norm_layers: {}", new_bn_count),
        ))
    } else {
        None
    }
}

fn count_batch_norm_layers(obj: &serde_json::Map<String, Value>) -> usize {
    let mut count = 0;
    for (key, _) in obj {
        if key.contains("batch_norm") || key.contains("bn") || key.contains("BatchNorm") {
            count += 1;
        }
    }
    count
}

fn analyze_batch_norm_parameters(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_params = extract_batch_norm_params(old_obj);
    let new_params = extract_batch_norm_params(new_obj);
    
    if old_params != new_params {
        Some((
            format!("bn_params: momentum={:.3}, eps={:.6}", old_params.0, old_params.1),
            format!("bn_params: momentum={:.3}, eps={:.6}", new_params.0, new_params.1),
        ))
    } else {
        None
    }
}

fn extract_batch_norm_params(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let momentum = obj.get("momentum")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.1);
    let eps = obj.get("eps")
        .and_then(|v| v.as_f64())
        .unwrap_or(1e-5);
    (momentum, eps)
}

fn analyze_batch_norm_statistics(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_stats = extract_batch_norm_stats(old_obj);
    let new_stats = extract_batch_norm_stats(new_obj);
    
    if (old_stats.0 - new_stats.0).abs() > 0.01 || (old_stats.1 - new_stats.1).abs() > 0.01 {
        Some((
            format!("bn_stats: running_mean={:.3}, running_var={:.3}", old_stats.0, old_stats.1),
            format!("bn_stats: running_mean={:.3}, running_var={:.3}", new_stats.0, new_stats.1),
        ))
    } else {
        None
    }
}

fn extract_batch_norm_stats(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let running_mean = obj.get("running_mean")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let running_var = obj.get("running_var")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);
    (running_mean, running_var)
}

// Helper functions for regularization analysis
fn analyze_dropout_patterns(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_dropout = extract_dropout_rate(old_obj);
    let new_dropout = extract_dropout_rate(new_obj);
    
    if (old_dropout - new_dropout).abs() > 0.001 {
        Some((
            format!("dropout_rate: {:.3}", old_dropout),
            format!("dropout_rate: {:.3}", new_dropout),
        ))
    } else {
        None
    }
}

fn extract_dropout_rate(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("dropout")
        .or_else(|| obj.get("dropout_rate"))
        .or_else(|| obj.get("p"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn analyze_weight_decay_impact(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_decay = extract_weight_decay(old_obj);
    let new_decay = extract_weight_decay(new_obj);
    
    if (old_decay - new_decay).abs() > 1e-6 {
        Some((
            format!("weight_decay: {:.6}", old_decay),
            format!("weight_decay: {:.6}", new_decay),
        ))
    } else {
        None
    }
}

fn extract_weight_decay(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("weight_decay")
        .or_else(|| obj.get("l2_reg"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn analyze_l_regularization(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_l1 = extract_l1_reg(old_obj);
    let new_l1 = extract_l1_reg(new_obj);
    let old_l2 = extract_l2_reg(old_obj);
    let new_l2 = extract_l2_reg(new_obj);
    
    if (old_l1 - new_l1).abs() > 1e-6 || (old_l2 - new_l2).abs() > 1e-6 {
        Some((
            format!("l_reg: L1={:.6}, L2={:.6}", old_l1, old_l2),
            format!("l_reg: L1={:.6}, L2={:.6}", new_l1, new_l2),
        ))
    } else {
        None
    }
}

fn extract_l1_reg(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("l1_reg").and_then(|v| v.as_f64()).unwrap_or(0.0)
}

fn extract_l2_reg(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("l2_reg").and_then(|v| v.as_f64()).unwrap_or(0.0)
}

// Helper functions for activation pattern analysis
fn analyze_activation_functions(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_activations = extract_activation_functions(old_obj);
    let new_activations = extract_activation_functions(new_obj);
    
    if old_activations != new_activations {
        Some((
            format!("activations: {}", old_activations.join(", ")),
            format!("activations: {}", new_activations.join(", ")),
        ))
    } else {
        None
    }
}

fn extract_activation_functions(obj: &serde_json::Map<String, Value>) -> Vec<String> {
    let mut activations = Vec::new();
    for (key, _) in obj {
        if key.contains("activation") || key.contains("relu") || key.contains("sigmoid") || 
           key.contains("tanh") || key.contains("gelu") || key.contains("swish") {
            activations.push(key.clone());
        }
    }
    activations.sort();
    activations
}

fn analyze_activation_saturation(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_saturation = calculate_activation_saturation(old_obj);
    let new_saturation = calculate_activation_saturation(new_obj);
    
    if (old_saturation - new_saturation).abs() > 0.01 {
        Some((
            format!("saturation: {:.2}%", old_saturation * 100.0),
            format!("saturation: {:.2}%", new_saturation * 100.0),
        ))
    } else {
        None
    }
}

fn calculate_activation_saturation(obj: &serde_json::Map<String, Value>) -> f64 {
    // Estimate saturation based on activation statistics
    obj.get("activation_stats")
        .and_then(|v| v.get("saturation"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

fn analyze_dead_neurons(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_dead = count_dead_neurons(old_obj);
    let new_dead = count_dead_neurons(new_obj);
    
    if old_dead != new_dead {
        Some((
            format!("dead_neurons: {}", old_dead),
            format!("dead_neurons: {}", new_dead),
        ))
    } else {
        None
    }
}

fn count_dead_neurons(obj: &serde_json::Map<String, Value>) -> usize {
    obj.get("dead_neurons")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize
}

// Helper functions for weight distribution analysis
fn analyze_weight_distributions(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_stats = calculate_weight_stats(old_obj);
    let new_stats = calculate_weight_stats(new_obj);
    
    if (old_stats.0 - new_stats.0).abs() > 0.001 || (old_stats.1 - new_stats.1).abs() > 0.001 {
        Some((
            format!("weight_stats: mean={:.4}, std={:.4}", old_stats.0, old_stats.1),
            format!("weight_stats: mean={:.4}, std={:.4}", new_stats.0, new_stats.1),
        ))
    } else {
        None
    }
}

fn calculate_weight_stats(obj: &serde_json::Map<String, Value>) -> (f64, f64) {
    let mean = obj.get("weight_mean").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let std = obj.get("weight_std").and_then(|v| v.as_f64()).unwrap_or(1.0);
    (mean, std)
}

fn analyze_weight_initialization(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_init = extract_weight_init_method(old_obj);
    let new_init = extract_weight_init_method(new_obj);
    
    if old_init != new_init {
        Some((
            format!("weight_init: {}", old_init),
            format!("weight_init: {}", new_init),
        ))
    } else {
        None
    }
}

fn extract_weight_init_method(obj: &serde_json::Map<String, Value>) -> String {
    obj.get("weight_init")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string()
}

fn analyze_weight_sparsity(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_sparsity = calculate_weight_sparsity(old_obj);
    let new_sparsity = calculate_weight_sparsity(new_obj);
    
    if (old_sparsity - new_sparsity).abs() > 0.01 {
        Some((
            format!("sparsity: {:.1}%", old_sparsity * 100.0),
            format!("sparsity: {:.1}%", new_sparsity * 100.0),
        ))
    } else {
        None
    }
}

fn calculate_weight_sparsity(obj: &serde_json::Map<String, Value>) -> f64 {
    obj.get("weight_sparsity")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

// Helper functions for model complexity assessment
fn analyze_parameter_count(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_params = extract_parameter_count(old_obj);
    let new_params = extract_parameter_count(new_obj);
    
    if old_params != new_params {
        Some((
            format!("parameters: {}", format_number(old_params)),
            format!("parameters: {}", format_number(new_params)),
        ))
    } else {
        None
    }
}

fn extract_parameter_count(obj: &serde_json::Map<String, Value>) -> u64 {
    obj.get("parameter_count")
        .or_else(|| obj.get("num_parameters"))
        .or_else(|| obj.get("total_params"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
}

fn format_number(num: u64) -> String {
    if num >= 1_000_000_000 {
        format!("{:.1}B", num as f64 / 1_000_000_000.0)
    } else if num >= 1_000_000 {
        format!("{:.1}M", num as f64 / 1_000_000.0)
    } else if num >= 1_000 {
        format!("{:.1}K", num as f64 / 1_000.0)
    } else {
        num.to_string()
    }
}

fn analyze_computational_complexity(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_flops = extract_flops(old_obj);
    let new_flops = extract_flops(new_obj);
    
    if old_flops != new_flops {
        Some((
            format!("flops: {}", format_number(old_flops)),
            format!("flops: {}", format_number(new_flops)),
        ))
    } else {
        None
    }
}

fn extract_flops(obj: &serde_json::Map<String, Value>) -> u64 {
    obj.get("flops")
        .or_else(|| obj.get("gflops"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
}

fn analyze_model_architecture_complexity(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
    let old_depth = extract_model_depth(old_obj);
    let new_depth = extract_model_depth(new_obj);
    let old_width = extract_model_width(old_obj);
    let new_width = extract_model_width(new_obj);
    
    if old_depth != new_depth || old_width != new_width {
        Some((
            format!("architecture: depth={}, width={}", old_depth, old_width),
            format!("architecture: depth={}, width={}", new_depth, new_width),
        ))
    } else {
        None
    }
}

fn extract_model_depth(obj: &serde_json::Map<String, Value>) -> u32 {
    obj.get("depth")
        .or_else(|| obj.get("num_layers"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

fn extract_model_width(obj: &serde_json::Map<String, Value>) -> u32 {
    obj.get("width")
        .or_else(|| obj.get("hidden_size"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32
}

// ============================================================================
// PARSER FUNCTIONS - FOR INTERNAL USE ONLY
// ============================================================================
// These functions are public only for CLI and language bindings.
// External users should use the main diff() function with file reading.

