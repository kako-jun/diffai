use serde_json::Value;

use crate::types::DiffResult;

// Module declarations
mod distributions;
mod flow;
mod magnitudes;
mod statistics;
mod types;

// Re-export the main public function's dependencies
use distributions::analyze_gradient_distributions;
use flow::analyze_gradient_flow;
use magnitudes::analyze_gradient_magnitudes;

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
