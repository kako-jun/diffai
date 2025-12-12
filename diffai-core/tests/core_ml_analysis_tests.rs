use diffai_core::*;
use serde_json::json;

#[path = "fixtures.rs"]
mod fixtures;
use fixtures::ml_generators;

// ============================================================================
// ML ANALYSIS FEATURES - COMPREHENSIVE TESTS
// ============================================================================

/// Test TensorStatsChanged detection with real tensor data
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_tensor_stats_changed_detailed() {
    let old = json!({
        "model_state_dict": {
            "conv1.weight": {
                "shape": [32, 3, 3, 3],
                "dtype": "float32",
                "statistics": {
                    "mean": 0.01,
                    "std": 0.1,
                    "min": -0.2,
                    "max": 0.3
                }
            }
        }
    });

    let new = json!({
        "model_state_dict": {
            "conv1.weight": {
                "shape": [32, 3, 3, 3],
                "dtype": "float32",
                "statistics": {
                    "mean": 0.02,  // Changed
                    "std": 0.12,   // Changed
                    "min": -0.25,  // Changed
                    "max": 0.35    // Changed
                }
            }
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let tensor_stats_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::TensorStatsChanged(_, _, _)))
        .count();

    assert!(
        tensor_stats_changes > 0,
        "Should detect tensor statistics changes"
    );

    // Verify specific tensor stats change
    let stats_change = results
        .iter()
        .find(|r| matches!(r, DiffResult::TensorStatsChanged(path, _, _) if path.contains("conv1.weight")));

    assert!(
        stats_change.is_some(),
        "Should find specific tensor stats change"
    );
}

/// Test ModelArchitectureChanged detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_model_architecture_changed_detailed() {
    let old = json!({
        "net": {
            "architectures": ["ResNet"],
            "num_layers": 18,
            "model_type": "vision_model"
        }
    });

    let new = json!({
        "net": {
            "architectures": ["EfficientNet"],  // Changed architecture
            "num_layers": 25,                   // Changed layer count
            "model_type": "vision_model"
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let architecture_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::ModelArchitectureChanged(_, _, _)))
        .count();

    assert!(
        architecture_changes > 0,
        "Should detect model architecture changes"
    );
}

/// Test WeightSignificantChange with threshold
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_weight_significant_change_threshold() {
    let old = json!({
        "parameters": {
            "layer1.weight": 0.1,
            "layer2.weight": 0.05,
            "layer3.weight": 0.2
        }
    });

    let new = json!({
        "parameters": {
            "layer1.weight": 0.15,  // 0.05 change - borderline
            "layer2.weight": 0.051, // 0.001 change - insignificant
            "layer3.weight": 0.3    // 0.1 change - significant
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let significant_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::WeightSignificantChange(_, magnitude) if *magnitude >= 0.05))
        .count();

    assert!(
        significant_changes >= 1,
        "Should detect at least one significant weight change"
    );
}

/// Test LearningRateChanged detection with various formats
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_learning_rate_changed_formats() {
    let test_cases = vec![
        // Standard optimizer format
        (
            json!({"optimizer": {"learning_rate": 0.001}}),
            json!({"optimizer": {"learning_rate": 0.01}}),
        ),
        // Short form lr
        (
            json!({"optimizer": {"lr": 0.001}}),
            json!({"optimizer": {"lr": 0.01}}),
        ),
        // Scheduler format
        (
            json!({"lr_scheduler": {"base_lr": 0.001}}),
            json!({"lr_scheduler": {"base_lr": 0.01}}),
        ),
    ];

    for (old, new) in test_cases {
        let results = diff(&old, &new, None).unwrap();

        let lr_changes = results
            .iter()
            .filter(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)))
            .count();

        assert!(
            lr_changes > 0,
            "Should detect learning rate changes in format: {old:?}"
        );
    }
}

/// Test OptimizerChanged detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_optimizer_changed_detection() {
    let old = json!({
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0,
                "amsgrad": false
            }]
        },
        "optimizer_type": "Adam"
    });

    let new = json!({
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{
                "lr": 0.01,
                "momentum": 0.9,
                "dampening": 0,
                "weight_decay": 0,
                "nesterov": false
            }]
        },
        "optimizer_type": "SGD"
    });

    let results = diff(&old, &new, None).unwrap();

    let optimizer_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::OptimizerChanged(_, _, _)))
        .count();

    assert!(
        optimizer_changes > 0,
        "Should detect optimizer type changes"
    );
}

/// Test LossChange detection in training metrics
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_loss_change_detection() {
    let old = json!({
        "training_metrics": {
            "loss": 1.25,
            "val_loss": 1.35
        }
    });

    let new = json!({
        "training_metrics": {
            "loss": 0.85,     // Improved
            "val_loss": 0.95  // Improved
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let loss_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::LossChange(_, _, _)))
        .count();

    assert!(loss_changes > 0, "Should detect loss changes");
}

/// Test AccuracyChange detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_accuracy_change_detection() {
    let old = json!({
        "metrics": {
            "accuracy": 0.85,
            "val_accuracy": 0.82,
            "test_acc": 0.80
        }
    });

    let new = json!({
        "metrics": {
            "accuracy": 0.92,     // Improved
            "val_accuracy": 0.89, // Improved
            "test_acc": 0.87      // Improved
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let accuracy_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::AccuracyChange(_, _, _)))
        .count();

    assert!(accuracy_changes > 0, "Should detect accuracy changes");
}

/// Test ModelVersionChanged detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_model_version_changed_detection() {
    let old = json!({
        "model_metadata": {
            "version": "1.0.0",
            "pytorch_version": "1.11.0"
        }
    });

    let new = json!({
        "model_metadata": {
            "version": "2.0.0",     // Changed
            "pytorch_version": "1.12.0"  // Changed
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let version_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::ModelVersionChanged(_, _, _)))
        .count();

    assert!(version_changes > 0, "Should detect model version changes");
}

/// Test ActivationFunctionChanged detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_activation_function_changed_detection() {
    let old = json!({
        "model_config": {
            "hidden_act": "relu",
            "output_activation": "softmax"
        }
    });

    let new = json!({
        "model_config": {
            "hidden_act": "gelu",     // Changed
            "output_activation": "softmax"
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let activation_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::ActivationFunctionChanged(_, _, _)))
        .count();

    assert!(
        activation_changes > 0,
        "Should detect activation function changes"
    );
}

/// Test TensorShapeChanged detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_tensor_shape_changed_detection() {
    let old = json!({
        "tensors": {
            "embedding.weight": {
                "shape": [30000, 768],
                "dtype": "float32"
            }
        }
    });

    let new = json!({
        "tensors": {
            "embedding.weight": {
                "shape": [30000, 1024],  // Changed dimension
                "dtype": "float32"
            }
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let shape_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::TensorShapeChanged(_, _, _)))
        .count();

    assert!(shape_changes > 0, "Should detect tensor shape changes");
}

/// Test TensorDataChanged detection
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_tensor_data_changed_detection() {
    let old = json!({
        "layer_data": {
            "conv1": {
                "mean": 0.01,
                "data_summary": "normalized"
            }
        }
    });

    let new = json!({
        "layer_data": {
            "conv1": {
                "mean": 0.05,  // Changed mean
                "data_summary": "normalized"
            }
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let data_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::TensorDataChanged(_, _, _)))
        .count();

    assert!(data_changes > 0, "Should detect tensor data changes");
}

// ============================================================================
// ADVANCED ML ANALYSIS INTEGRATION TESTS
// ============================================================================

/// Test comprehensive ML analysis on complex model
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_comprehensive_ml_analysis() {
    let old_model = json!({
        "model_state_dict": {
            "embedding.weight": {"shape": [50000, 768], "mean": 0.01},
            "encoder.layer.0.attention.self.query.weight": {"shape": [768, 768], "mean": 0.02}
        },
        "optimizer_state_dict": {
            "param_groups": [{"lr": 0.001}]
        },
        "training_info": {
            "loss": 2.5,
            "accuracy": 0.85,
            "epoch": 10
        },
        "model_config": {
            "architecture": "BERT",
            "hidden_act": "relu",
            "version": "1.0"
        }
    });

    let new_model = json!({
        "model_state_dict": {
            "embedding.weight": {"shape": [50000, 1024], "mean": 0.015},  // Shape and data changed
            "encoder.layer.0.attention.self.query.weight": {"shape": [1024, 1024], "mean": 0.025}  // Shape and data changed
        },
        "optimizer_state_dict": {
            "param_groups": [{"lr": 0.01}]  // LR changed
        },
        "training_info": {
            "loss": 1.2,      // Loss improved
            "accuracy": 0.92, // Accuracy improved
            "epoch": 15
        },
        "model_config": {
            "architecture": "BERT-Large",  // Architecture changed
            "hidden_act": "gelu",          // Activation changed
            "version": "2.0"               // Version changed
        }
    });

    let results = diff(&old_model, &new_model, None).unwrap();

    // Should detect multiple ML-specific changes
    let ml_change_types: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            DiffResult::TensorShapeChanged(_, _, _) => "tensor_shape",
            DiffResult::TensorDataChanged(_, _, _) => "tensor_data",
            DiffResult::LearningRateChanged(_, _, _) => "learning_rate",
            DiffResult::LossChange(_, _, _) => "loss",
            DiffResult::AccuracyChange(_, _, _) => "accuracy",
            DiffResult::ModelArchitectureChanged(_, _, _) => "architecture",
            DiffResult::ActivationFunctionChanged(_, _, _) => "activation",
            DiffResult::ModelVersionChanged(_, _, _) => "version",
            _ => "other",
        })
        .collect();

    assert!(
        ml_change_types.len() >= 5,
        "Should detect multiple ML-specific change types: {ml_change_types:?}"
    );
}

/// Test ML analysis with PyTorch checkpoint format
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_pytorch_checkpoint_analysis() {
    let old_checkpoint = json!({
        "epoch": 10,
        "model_state_dict": {
            "conv1.weight": {"shape": [64, 3, 7, 7], "mean": 0.01}
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 0.001, "momentum": 0.9}]
        },
        "loss": 0.5,
        "best_acc": 0.85
    });

    let new_checkpoint = json!({
        "epoch": 15,
        "model_state_dict": {
            "conv1.weight": {"shape": [64, 3, 7, 7], "mean": 0.02}  // Weight changed
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 0.0001, "momentum": 0.9}]  // LR changed
        },
        "loss": 0.3,    // Loss improved
        "best_acc": 0.92  // Accuracy improved
    });

    let results = diff(&old_checkpoint, &new_checkpoint, None).unwrap();

    // Should detect training progress
    let has_lr_change = results
        .iter()
        .any(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));
    let has_loss_change = results
        .iter()
        .any(|r| matches!(r, DiffResult::LossChange(_, _, _)));
    let has_weight_change = results
        .iter()
        .any(|r| matches!(r, DiffResult::WeightSignificantChange(_, _)));

    assert!(has_lr_change, "Should detect learning rate changes");
    assert!(has_loss_change, "Should detect loss changes");
    assert!(has_weight_change, "Should detect weight changes");
}

/// Test ML analysis with safetensors metadata
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_safetensors_metadata_analysis() {
    let old_safetensors = json!({
        "__metadata__": {
            "format": "pt",
            "version": "0.3.1"
        },
        "model.embed_tokens.weight": {"dtype": "F32", "shape": [32000, 4096]},
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "F32", "shape": [4096, 4096]}
    });

    let new_safetensors = json!({
        "__metadata__": {
            "format": "pt",
            "version": "0.4.0"  // Version changed
        },
        "model.embed_tokens.weight": {"dtype": "F16", "shape": [32000, 4096]},  // Dtype changed
        "model.layers.0.self_attn.q_proj.weight": {"dtype": "F16", "shape": [4096, 4096]}  // Dtype changed
    });

    let results = diff(&old_safetensors, &new_safetensors, None).unwrap();

    let has_version_change = results
        .iter()
        .any(|r| matches!(r, DiffResult::ModelVersionChanged(_, _, _)));
    let has_dtype_changes = results
        .iter()
        .any(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("dtype")));

    assert!(has_version_change, "Should detect version changes");
    assert!(has_dtype_changes, "Should detect dtype changes");
}

// ============================================================================
// EDGE CASES AND ERROR HANDLING
// ============================================================================

/// Test ML analysis with missing expected fields
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_ml_analysis_missing_fields() {
    let old = json!({
        "some_field": "value"
    });

    let new = json!({
        "some_field": "new_value",
        "optimizer": {"lr": 0.01}  // Added optimizer
    });

    let results = diff(&old, &new, None).unwrap();

    // Should handle gracefully without panicking
    assert!(!results.is_empty());
}

/// Test ML analysis with malformed data
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_ml_analysis_malformed_data() {
    let old = json!({
        "optimizer": {
            "lr": "not_a_number"  // Invalid type
        }
    });

    let new = json!({
        "optimizer": {
            "lr": 0.01
        }
    });

    let results = diff(&old, &new, None).unwrap();

    // Should detect as type change, not learning rate change
    let has_type_change = results
        .iter()
        .any(|r| matches!(r, DiffResult::TypeChanged(_, _, _)));
    assert!(
        has_type_change,
        "Should detect type changes for malformed data"
    );
}

/// Test performance with large ML model data
#[test]
#[ignore = "ML analysis integration needs refinement"]
fn test_large_ml_model_performance() {
    // Generate large model with many layers
    let large_model_old =
        ml_generators::generate_model_weights(vec![2048, 1024, 512, 256, 128, 64, 10]);
    let large_model_new =
        ml_generators::generate_model_weights(vec![2048, 1024, 512, 256, 128, 64, 10]);

    // Add training info to make it more realistic
    let mut old_with_training = large_model_old;
    let mut new_with_training = large_model_new;

    old_with_training["training"] = json!({"loss": 2.5, "accuracy": 0.85});
    new_with_training["training"] = json!({"loss": 1.2, "accuracy": 0.92});

    let start = std::time::Instant::now();
    let results = diff(&old_with_training, &new_with_training, None).unwrap();
    let duration = start.elapsed();

    assert!(!results.is_empty());
    assert!(
        duration.as_secs() < 10,
        "Large ML model analysis should complete within 10 seconds"
    );
}
