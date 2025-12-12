use serde_json::Value;

use crate::types::DiffResult;

// Submodules
mod impact;
mod methods;
mod precision;
mod types;

// Internal use only - no re-exports needed as types are used within submodules
use impact::analyze_quantization_impact;
use methods::analyze_quantization_methods;
use precision::analyze_quantization_precision;

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
