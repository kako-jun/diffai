use serde_json::Value;

use crate::types::DiffResult;

// Module declarations
mod epoch;
mod learning_curves;
mod loss;
mod optimization;
mod patterns;
mod plateau;
mod stability;

// Re-export the main public function
pub use self::analyze_convergence_patterns_impl as analyze_convergence_patterns;

// Internal module imports
use epoch::analyze_epoch_progression;
use learning_curves::analyze_learning_curves_comprehensive;
use loss::analyze_loss_convergence;
use optimization::analyze_optimization_trajectory;
use patterns::analyze_convergence_patterns_advanced;
use plateau::analyze_plateau_detection;
use stability::{analyze_training_stability, analyze_training_stability_statistical};

/// Main convergence pattern analysis function
pub fn analyze_convergence_patterns_impl(
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
