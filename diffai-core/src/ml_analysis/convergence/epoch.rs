use serde_json::Value;

/// Analyze epoch progression patterns
pub(crate) fn analyze_epoch_progression(old_obj: &serde_json::Map<String, Value>, new_obj: &serde_json::Map<String, Value>) -> Option<(String, String)> {
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

/// Extract epoch information from model checkpoint
pub(crate) fn extract_epoch_info(obj: &serde_json::Map<String, Value>) -> Option<f64> {
    if let Some(Value::Number(num)) = obj.get("epoch") {
        return num.as_f64();
    }
    None
}
