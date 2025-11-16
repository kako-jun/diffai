use serde_json::Value;

use crate::types::DiffResult;

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
            "learning_rate",
            "lr",
            "initial_lr",
            "base_lr",
            "current_lr",
            "lr_scheduler",
            "optimizer_lr",
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
        if let (Some(old_opt), Some(new_opt)) =
            (old_obj.get("optimizer"), new_obj.get("optimizer"))
        {
            let optimizer_changes = analyze_optimizer_learning_rates(old_opt, new_opt);
            lr_changes.extend(optimizer_changes);
        }

        // Look for scheduler state
        if let (Some(old_sched), Some(new_sched)) =
            (old_obj.get("scheduler"), new_obj.get("scheduler"))
        {
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
            if let Some((implicit_old, implicit_new)) =
                extract_implicit_learning_rate(old_obj, new_obj)
            {
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
pub(crate) fn analyze_learning_rate_value_change(
    old_val: &Value,
    new_val: &Value,
    key: &str,
) -> Vec<(String, f64, f64)> {
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
                        &format!("{}.{}", key, sub_key),
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
pub(crate) fn analyze_optimizer_learning_rates(
    old_opt: &Value,
    new_opt: &Value,
) -> Vec<(String, f64, f64)> {
    let mut changes = Vec::new();

    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_opt, new_opt) {
        // Look for param_groups (common in PyTorch optimizers)
        if let (Some(old_groups), Some(new_groups)) =
            (old_obj.get("param_groups"), new_obj.get("param_groups"))
        {
            if let (Value::Array(old_arr), Value::Array(new_arr)) = (old_groups, new_groups) {
                for (i, (old_group, new_group)) in old_arr.iter().zip(new_arr.iter()).enumerate() {
                    if let (Value::Object(old_g), Value::Object(new_g)) = (old_group, new_group) {
                        if let (Some(old_lr), Some(new_lr)) = (old_g.get("lr"), new_g.get("lr")) {
                            if let (Value::Number(old_num), Value::Number(new_num)) =
                                (old_lr, new_lr)
                            {
                                let old_f = old_num.as_f64().unwrap_or(0.0);
                                let new_f = new_num.as_f64().unwrap_or(0.0);
                                if old_f != new_f {
                                    changes.push((
                                        format!("optimizer.param_groups[{}].lr", i),
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

        // Look for direct lr field in optimizer
        if let (Some(old_lr), Some(new_lr)) = (old_obj.get("lr"), new_obj.get("lr")) {
            let lr_changes = analyze_learning_rate_value_change(old_lr, new_lr, "optimizer.lr");
            changes.extend(lr_changes);
        }
    }

    changes
}

// Analyze scheduler state for learning rate changes
pub(crate) fn analyze_scheduler_learning_rates(
    old_sched: &Value,
    new_sched: &Value,
) -> Vec<(String, f64, f64)> {
    let mut changes = Vec::new();

    if let (Value::Object(old_obj), Value::Object(new_obj)) = (old_sched, new_sched) {
        // Common scheduler fields
        let scheduler_lr_keys = ["base_lrs", "last_lr", "_last_lr", "current_lr"];

        for key in &scheduler_lr_keys {
            if let (Some(old_val), Some(new_val)) = (old_obj.get(*key), new_obj.get(*key)) {
                let lr_changes = analyze_learning_rate_value_change(
                    old_val,
                    new_val,
                    &format!("scheduler.{}", key),
                );
                changes.extend(lr_changes);
            }
        }
    }

    changes
}

// Check if models have training metadata
pub(crate) fn has_training_metadata(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> bool {
    let training_keys = [
        "epoch",
        "step",
        "iteration",
        "optimizer",
        "scheduler",
        "loss",
        "metrics",
    ];

    for key in &training_keys {
        if old_obj.contains_key(*key) || new_obj.contains_key(*key) {
            return true;
        }
    }

    false
}

// Extract implicit learning rate from training information
pub(crate) fn extract_implicit_learning_rate(
    old_obj: &serde_json::Map<String, Value>,
    new_obj: &serde_json::Map<String, Value>,
) -> Option<(f64, f64)> {
    // Try to infer learning rate from epoch progression and loss changes
    if let (Some(old_epoch), Some(new_epoch)) = (old_obj.get("epoch"), new_obj.get("epoch")) {
        if let (Some(old_loss), Some(new_loss)) = (old_obj.get("loss"), new_obj.get("loss")) {
            if let (
                Value::Number(old_e),
                Value::Number(new_e),
                Value::Number(old_l),
                Value::Number(new_l),
            ) = (old_epoch, new_epoch, old_loss, new_loss)
            {
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
