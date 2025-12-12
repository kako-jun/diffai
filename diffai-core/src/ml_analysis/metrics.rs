use serde_json::Value;

use crate::types::DiffResult;

/// Analyze loss and accuracy changes in training metrics
pub fn analyze_training_metrics(
    old_model: &Value,
    new_model: &Value,
    results: &mut Vec<DiffResult>,
) {
    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_model, new_model) {
        // Analyze loss changes
        analyze_loss_changes(old_obj, new_obj, results);

        // Analyze accuracy changes
        analyze_accuracy_changes(old_obj, new_obj, results);

        // Analyze version changes
        analyze_version_changes(old_obj, new_obj, results);

        // Analyze optimizer changes
        analyze_optimizer_changes(old_obj, new_obj, results);
    }
}

/// Detect loss value changes
fn analyze_loss_changes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
) {
    // Common keys for loss values
    let loss_keys = [
        "loss",
        "train_loss",
        "val_loss",
        "validation_loss",
        "test_loss",
        "total_loss",
        "best_loss",
    ];

    for key in &loss_keys {
        if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
            if let (Some(old_f), Some(new_f)) = (old_val.as_f64(), new_val.as_f64()) {
                if (old_f - new_f).abs() > 1e-10 {
                    results.push(DiffResult::LossChange(key.to_string(), old_f, new_f));
                }
            }
        }
    }

    // Check nested structures (training_metrics, metrics, etc.)
    let nested_keys = ["training_metrics", "metrics", "history", "logs"];
    for nested_key in &nested_keys {
        if let (Some(Value::Object(old_nested)), Some(Value::Object(new_nested))) =
            (old_obj.get(*nested_key), new_obj.get(*nested_key))
        {
            for key in &loss_keys {
                if let (Some(old_val), Some(new_val)) = (old_nested.get(*key), new_nested.get(*key))
                {
                    if let (Some(old_f), Some(new_f)) = (old_val.as_f64(), new_val.as_f64()) {
                        if (old_f - new_f).abs() > 1e-10 {
                            results.push(DiffResult::LossChange(
                                format!("{nested_key}.{key}"),
                                old_f,
                                new_f,
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Detect accuracy value changes
fn analyze_accuracy_changes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
) {
    // Common keys for accuracy values
    let accuracy_keys = [
        "accuracy",
        "acc",
        "train_accuracy",
        "val_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "top1_accuracy",
        "top5_accuracy",
        "best_accuracy",
    ];

    for key in &accuracy_keys {
        if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
            if let (Some(old_f), Some(new_f)) = (old_val.as_f64(), new_val.as_f64()) {
                if (old_f - new_f).abs() > 1e-10 {
                    results.push(DiffResult::AccuracyChange(key.to_string(), old_f, new_f));
                }
            }
        }
    }

    // Check nested structures
    let nested_keys = ["training_metrics", "metrics", "history", "logs"];
    for nested_key in &nested_keys {
        if let (Some(Value::Object(old_nested)), Some(Value::Object(new_nested))) =
            (old_obj.get(*nested_key), new_obj.get(*nested_key))
        {
            for key in &accuracy_keys {
                if let (Some(old_val), Some(new_val)) = (old_nested.get(*key), new_nested.get(*key))
                {
                    if let (Some(old_f), Some(new_f)) = (old_val.as_f64(), new_val.as_f64()) {
                        if (old_f - new_f).abs() > 1e-10 {
                            results.push(DiffResult::AccuracyChange(
                                format!("{nested_key}.{key}"),
                                old_f,
                                new_f,
                            ));
                        }
                    }
                }
            }
        }
    }
}

/// Detect model/framework version changes
fn analyze_version_changes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
) {
    // Common keys for version information
    let version_keys = [
        "version",
        "model_version",
        "pytorch_version",
        "torch_version",
        "tensorflow_version",
        "framework_version",
    ];

    for key in &version_keys {
        if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
            let old_str = value_to_version_string(old_val);
            let new_str = value_to_version_string(new_val);
            if old_str != new_str {
                results.push(DiffResult::ModelVersionChanged(
                    key.to_string(),
                    old_str,
                    new_str,
                ));
            }
        }
    }

    // Check __metadata__ (Safetensors format)
    if let (Some(Value::Object(old_meta)), Some(Value::Object(new_meta))) =
        (old_obj.get("__metadata__"), new_obj.get("__metadata__"))
    {
        for key in &version_keys {
            if let (Some(old_val), Some(new_val)) = (old_meta.get(*key), new_meta.get(*key)) {
                let old_str = value_to_version_string(old_val);
                let new_str = value_to_version_string(new_val);
                if old_str != new_str {
                    results.push(DiffResult::ModelVersionChanged(
                        format!("__metadata__.{key}"),
                        old_str,
                        new_str,
                    ));
                }
            }
        }
    }

    // Check model_metadata
    if let (Some(Value::Object(old_meta)), Some(Value::Object(new_meta))) =
        (old_obj.get("model_metadata"), new_obj.get("model_metadata"))
    {
        for key in &version_keys {
            if let (Some(old_val), Some(new_val)) = (old_meta.get(*key), new_meta.get(*key)) {
                let old_str = value_to_version_string(old_val);
                let new_str = value_to_version_string(new_val);
                if old_str != new_str {
                    results.push(DiffResult::ModelVersionChanged(
                        format!("model_metadata.{key}"),
                        old_str,
                        new_str,
                    ));
                }
            }
        }
    }
}

fn value_to_version_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        _ => val.to_string(),
    }
}

/// Detect optimizer type changes
fn analyze_optimizer_changes(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
    results: &mut Vec<DiffResult>,
) {
    // Common keys for optimizer type
    let optimizer_type_keys = ["optimizer_type", "optimizer_name", "optimizer", "opt_type"];

    for key in &optimizer_type_keys {
        if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
            let old_str = value_to_string(old_val);
            let new_str = value_to_string(new_val);
            if old_str != new_str && !old_str.is_empty() && !new_str.is_empty() {
                results.push(DiffResult::OptimizerChanged(
                    key.to_string(),
                    old_str,
                    new_str,
                ));
            }
        }
    }

    // Check nested optimizer config
    let nested_keys = ["training_config", "config", "hyperparameters"];
    for nested_key in &nested_keys {
        if let (Some(Value::Object(old_nested)), Some(Value::Object(new_nested))) =
            (old_obj.get(*nested_key), new_obj.get(*nested_key))
        {
            for key in &optimizer_type_keys {
                if let (Some(old_val), Some(new_val)) = (old_nested.get(*key), new_nested.get(*key))
                {
                    let old_str = value_to_string(old_val);
                    let new_str = value_to_string(new_val);
                    if old_str != new_str && !old_str.is_empty() && !new_str.is_empty() {
                        results.push(DiffResult::OptimizerChanged(
                            format!("{nested_key}.{key}"),
                            old_str,
                            new_str,
                        ));
                    }
                }
            }
        }
    }
}

fn value_to_string(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Object(_) => String::new(), // Don't convert objects
        _ => val.to_string(),
    }
}
