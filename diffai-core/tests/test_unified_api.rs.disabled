use diffai_core::*;
use serde_json::json;

#[path = "fixtures.rs"]
mod fixtures;
use fixtures::{ml_generators, TestFixtures};

// ============================================================================
// UNIFIED API TESTS - Core Functionality
// ============================================================================

#[test]
fn test_diff_basic_modification() {
    let old = json!({"name": "Alice", "age": 30});
    let new = json!({"name": "Alice", "age": 31});

    let results = diff(&old, &new, None).unwrap();

    assert_eq!(results.len(), 1);
    match &results[0] {
        DiffResult::Modified(path, old_val, new_val) => {
            assert_eq!(path, "age");
            assert_eq!(old_val, &json!(30));
            assert_eq!(new_val, &json!(31));
        }
        _ => panic!("Expected Modified result"),
    }
}

#[test]
fn test_diff_ai_ml_specific_results() {
    let old = json!({"learning_rate": 0.001, "accuracy": 0.85});
    let new = json!({"learning_rate": 0.01, "accuracy": 0.92});

    let diffai_options = DiffaiSpecificOptions {
        learning_rate_tracking: Some(true),
        accuracy_tracking: Some(true),
        ..Default::default()
    };

    let options = DiffOptions {
        diffai_options: Some(diffai_options),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    assert_eq!(results.len(), 2);

    // Check for learning rate change
    let lr_result = results
        .iter()
        .find(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));
    assert!(lr_result.is_some());

    // Check for accuracy change
    let acc_result = results
        .iter()
        .find(|r| matches!(r, DiffResult::AccuracyChange(_, _, _)));
    assert!(acc_result.is_some());
}

#[test]
fn test_diff_weight_threshold() {
    let old = json!({"weights": {"layer1": 0.1, "layer2": 0.05}});
    let new = json!({"weights": {"layer1": 0.2, "layer2": 0.051}});

    let diffai_options = DiffaiSpecificOptions {
        weight_threshold: Some(0.05), // Only changes > 0.05 are significant
        ..Default::default()
    };

    let options = DiffOptions {
        diffai_options: Some(diffai_options),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    // Should only detect layer1 as significant change (0.1 difference > 0.05 threshold)
    assert_eq!(results.len(), 1);
    match &results[0] {
        DiffResult::WeightSignificantChange(path, magnitude) => {
            assert!(path.contains("layer1"));
            assert_eq!(*magnitude, 0.1);
        }
        _ => panic!("Expected WeightSignificantChange result"),
    }
}

// ============================================================================
// AI/ML SPECIFIC TESTS - PyTorch Models
// ============================================================================

#[test]
fn test_pytorch_model_comparison() {
    let old = TestFixtures::pytorch_model_old();
    let new = TestFixtures::pytorch_model_new();

    let diffai_options = DiffaiSpecificOptions {
        ml_analysis_enabled: Some(true),
        learning_rate_tracking: Some(true),
        optimizer_comparison: Some(true),
        loss_tracking: Some(true),
        accuracy_tracking: Some(true),
        ..Default::default()
    };

    let options = DiffOptions {
        diffai_options: Some(diffai_options),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    assert!(!results.is_empty());

    // Should detect optimizer change (Adam -> SGD)
    let optimizer_changes = results
        .iter()
        .filter(
            |r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("optimizer.type")),
        )
        .count();
    assert!(optimizer_changes > 0);

    // Should detect learning rate change
    let lr_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)))
        .count();
    assert!(lr_changes > 0);
}

#[test]
fn test_pytorch_layer_weight_changes() {
    let old = json!({
        "layers": {
            "conv1": {
                "weights": {
                    "mean": 0.01,
                    "std": 0.1
                }
            },
            "fc": {
                "weights": {
                    "mean": 0.0,
                    "std": 0.05
                }
            }
        }
    });

    let new = json!({
        "layers": {
            "conv1": {
                "weights": {
                    "mean": 0.015,  // Small change
                    "std": 0.12     // Small change
                }
            },
            "fc": {
                "weights": {
                    "mean": 0.1,    // Large change
                    "std": 0.15     // Large change
                }
            }
        }
    });

    let diffai_options = DiffaiSpecificOptions {
        weight_threshold: Some(0.05),
        ..Default::default()
    };

    let options = DiffOptions {
        diffai_options: Some(diffai_options),
        epsilon: Some(0.001),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    // Should detect significant changes in fc layer
    let significant_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::WeightSignificantChange(_, _)))
        .count();

    assert!(significant_changes > 0);
}

// ============================================================================
// AI/ML SPECIFIC TESTS - SafeTensors Models
// ============================================================================

#[test]
fn test_safetensors_model_comparison() {
    let old = TestFixtures::safetensors_model_old();
    let new = TestFixtures::safetensors_model_new();

    let options = DiffOptions {
        diffai_options: Some(DiffaiSpecificOptions {
            tensor_comparison_mode: Some("both".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    assert!(!results.is_empty());

    // Should detect tensor shape changes
    let shape_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("shape")))
        .count();
    assert!(shape_changes > 0);

    // Should detect dtype changes
    let dtype_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("dtype")))
        .count();
    assert!(dtype_changes > 0);

    // Should detect new tensors
    let added_tensors = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Added(path, _) if path.contains("new_layer")))
        .count();
    assert!(added_tensors > 0);
}

#[test]
fn test_safetensors_metadata_comparison() {
    let old = TestFixtures::safetensors_model_old();
    let new = TestFixtures::safetensors_model_new();

    let results = diff(&old, &new, None).unwrap();

    // Should detect metadata changes
    let metadata_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("metadata")))
        .count();
    assert!(metadata_changes > 0);

    // Check for specific version change
    let version_change = results.iter().find(|r| {
        matches!(r, DiffResult::Modified(path, old_val, new_val)
            if path.contains("version") && old_val == &json!("1.0") && new_val == &json!("2.0"))
    });
    assert!(version_change.is_some());
}

// ============================================================================
// AI/ML SPECIFIC TESTS - NumPy Arrays
// ============================================================================

#[test]
fn test_numpy_array_comparison() {
    let old = TestFixtures::numpy_array_old();
    let new = TestFixtures::numpy_array_new();

    let options = DiffOptions {
        diffai_options: Some(DiffaiSpecificOptions {
            scientific_precision: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    assert!(!results.is_empty());

    // Should detect array shape changes
    let shape_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("shape")))
        .count();
    assert!(shape_changes > 0);

    // Should detect dtype changes
    let dtype_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("dtype")))
        .count();
    assert!(dtype_changes > 0);

    // Should detect new arrays
    let added_arrays = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Added(path, _) if path.contains("weights")))
        .count();
    assert!(added_arrays > 0);
}

#[test]
fn test_numpy_statistics_comparison() {
    let old = json!({
        "array": {
            "statistics": {
                "mean": 0.5,
                "std": 0.2,
                "min": 0.0,
                "max": 1.0
            }
        }
    });

    let new = json!({
        "array": {
            "statistics": {
                "mean": 0.48,
                "std": 0.22,
                "min": 0.0,
                "max": 1.0
            }
        }
    });

    let options = DiffOptions {
        epsilon: Some(0.001), // Very precise comparison
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    // Should detect statistical changes
    assert!(!results.is_empty());
}

// ============================================================================
// AI/ML SPECIFIC TESTS - MATLAB Files
// ============================================================================

#[test]
fn test_matlab_file_comparison() {
    let old = TestFixtures::matlab_file_old();
    let new = TestFixtures::matlab_file_new();

    let results = diff(&old, &new, None).unwrap();

    assert!(!results.is_empty());

    // Should detect network type change
    let network_changes = results
        .iter()
        .filter(|r| {
            matches!(r, DiffResult::Modified(path, old_val, new_val)
                if path.contains("network.type")
                    && old_val == &json!("feedforward")
                    && new_val == &json!("convolutional"))
        })
        .count();
    assert!(network_changes > 0);

    // Should detect activation function change
    let activation_changes = results
        .iter()
        .filter(|r| {
            matches!(r, DiffResult::Modified(path, old_val, new_val)
                if path.contains("activation")
                    && old_val == &json!("relu")
                    && new_val == &json!("tanh"))
        })
        .count();
    assert!(activation_changes > 0);
}

// ============================================================================
// TRAINING METRICS COMPARISON TESTS
// ============================================================================

#[test]
fn test_training_metrics_comparison() {
    let old = TestFixtures::training_metrics_old();
    let new = TestFixtures::training_metrics_new();

    let diffai_options = DiffaiSpecificOptions {
        loss_tracking: Some(true),
        accuracy_tracking: Some(true),
        optimizer_comparison: Some(true),
        learning_rate_tracking: Some(true),
        ..Default::default()
    };

    let options = DiffOptions {
        diffai_options: Some(diffai_options),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    assert!(!results.is_empty());

    // Should detect optimizer change
    let optimizer_changes = results
        .iter()
        .filter(|r| {
            matches!(r, DiffResult::Modified(path, old_val, new_val)
                if path.contains("optimizer") && old_val == &json!("Adam") && new_val == &json!("SGD"))
        })
        .count();
    assert!(optimizer_changes > 0);

    // Should detect learning rate change
    let lr_changes = results
        .iter()
        .filter(|r| {
            matches!(r, 
            DiffResult::LearningRateChanged(_, old_lr, new_lr) 
            if *old_lr == 0.001 && *new_lr == 0.01)
        })
        .count();
    assert!(lr_changes > 0);
}

#[test]
fn test_training_history_arrays() {
    let old = json!({
        "training": {
            "loss": [2.5, 1.8, 1.2, 0.9, 0.7],
            "accuracy": [0.2, 0.4, 0.6, 0.75, 0.82]
        }
    });

    let new = json!({
        "training": {
            "loss": [2.3, 1.5, 1.0, 0.7, 0.5],
            "accuracy": [0.25, 0.45, 0.65, 0.80, 0.88]
        }
    });

    let results = diff(&old, &new, None).unwrap();

    // Should detect changes in loss and accuracy arrays
    assert!(!results.is_empty());

    let loss_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("loss[")))
        .count();
    assert!(loss_changes > 0);

    let accuracy_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("accuracy[")))
        .count();
    assert!(accuracy_changes > 0);
}

// ============================================================================
// MODEL ARCHITECTURE COMPARISON TESTS
// ============================================================================

#[test]
fn test_model_architecture_comparison() {
    let old = TestFixtures::model_architecture_old();
    let new = TestFixtures::model_architecture_new();

    let results = diff(&old, &new, None).unwrap();

    assert!(!results.is_empty());

    // Should detect model type change
    let type_changes = results
        .iter()
        .filter(|r| {
            matches!(r, DiffResult::Modified(path, old_val, new_val)
                if path.contains("model.type")
                    && old_val == &json!("sequential")
                    && new_val == &json!("functional"))
        })
        .count();
    assert!(type_changes > 0);

    // Should detect added layers (conv2, dropout layers)
    let added_layers = results
        .iter()
        .filter(|r| {
            matches!(r, DiffResult::Added(path, _) 
                if path.contains("layers[") && (path.contains("conv2") || path.contains("dropout")))
        })
        .count();
    assert!(added_layers > 0);

    // Should detect filter count changes
    let filter_changes = results
        .iter()
        .filter(|r| {
            matches!(r, DiffResult::Modified(path, old_val, new_val)
                if path.contains("filters") && old_val == &json!(32) && new_val == &json!(64))
        })
        .count();
    assert!(filter_changes > 0);
}

// ============================================================================
// OPTIONS TESTING - diffai Specific Options
// ============================================================================

#[test]
fn test_tensor_comparison_mode() {
    let old = json!({
        "tensor": {
            "shape": [100, 200],
            "data": [1.0, 2.0, 3.0]
        }
    });

    let new = json!({
        "tensor": {
            "shape": [100, 300], // Shape changed
            "data": [1.1, 2.1, 3.1] // Data changed
        }
    });

    // Test shape-only mode
    let options = DiffOptions {
        diffai_options: Some(DiffaiSpecificOptions {
            tensor_comparison_mode: Some("shape".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();

    // Should primarily focus on shape changes
    let shape_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("shape")))
        .count();
    assert!(shape_changes > 0);
}

#[test]
fn test_ml_analysis_enabled() {
    let old = json!({
        "model": {
            "learning_rate": 0.001,
            "loss": 0.5,
            "accuracy": 0.8,
            "weights": {"layer1": 0.1}
        }
    });

    let new = json!({
        "model": {
            "learning_rate": 0.01,
            "loss": 0.3,
            "accuracy": 0.9,
            "weights": {"layer1": 0.2}
        }
    });

    // With ML analysis enabled
    let ml_options = DiffOptions {
        diffai_options: Some(DiffaiSpecificOptions {
            ml_analysis_enabled: Some(true),
            learning_rate_tracking: Some(true),
            loss_tracking: Some(true),
            accuracy_tracking: Some(true),
            weight_threshold: Some(0.05),
            ..Default::default()
        }),
        ..Default::default()
    };

    let ml_results = diff(&old, &new, Some(&ml_options)).unwrap();

    // Should use ML-specific diff result types
    let ml_specific_results = ml_results
        .iter()
        .filter(|r| {
            matches!(
                r,
                DiffResult::LearningRateChanged(_, _, _)
                    | DiffResult::LossChange(_, _, _)
                    | DiffResult::AccuracyChange(_, _, _)
                    | DiffResult::WeightSignificantChange(_, _)
            )
        })
        .count();

    assert!(ml_specific_results > 0);

    // Without ML analysis
    let regular_options = DiffOptions::default();
    let regular_results = diff(&old, &new, Some(&regular_options)).unwrap();

    // Should use regular diff result types
    let regular_result_types = regular_results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(_, _, _)))
        .count();

    assert!(regular_result_types > 0);
}

// ============================================================================
// OUTPUT FORMAT TESTS
// ============================================================================

#[test]
fn test_diffai_output_format() {
    let old = json!({"model": "old"});
    let new = json!({"model": "new"});

    let results = diff(&old, &new, None).unwrap();

    // Test diffai-specific format
    let formatted = format_output(&results, OutputFormat::Diffai).unwrap();
    assert!(!formatted.is_empty());
    assert!(formatted.contains("Modified"));

    // Test all supported formats
    for format in OutputFormat::value_variants() {
        let output = format_output(&results, *format).unwrap();
        assert!(!output.is_empty());
    }
}

#[test]
fn test_output_format_parsing() {
    assert_eq!(
        OutputFormat::parse_format("diffai").unwrap(),
        OutputFormat::Diffai
    );
    assert_eq!(
        OutputFormat::parse_format("json").unwrap(),
        OutputFormat::Json
    );
    assert_eq!(
        OutputFormat::parse_format("yaml").unwrap(),
        OutputFormat::Yaml
    );
    assert_eq!(
        OutputFormat::parse_format("unified").unwrap(),
        OutputFormat::Diffai
    );

    assert!(OutputFormat::parse_format("invalid").is_err());
}

// ============================================================================
// INTEGRATION TESTS WITH FIXTURES
// ============================================================================

#[test]
fn test_comprehensive_ml_workflow() {
    // Test a complete ML model comparison workflow
    let old_model = TestFixtures::pytorch_model_old();
    let new_model = TestFixtures::pytorch_model_new();

    let comprehensive_options = DiffOptions {
        diffai_options: Some(DiffaiSpecificOptions {
            ml_analysis_enabled: Some(true),
            tensor_comparison_mode: Some("both".to_string()),
            learning_rate_tracking: Some(true),
            optimizer_comparison: Some(true),
            loss_tracking: Some(true),
            accuracy_tracking: Some(true),
            weight_threshold: Some(0.01),
            activation_analysis: Some(true),
            ..Default::default()
        }),
        epsilon: Some(0.001),
        output_format: Some(OutputFormat::Json),
        ..Default::default()
    };

    let results = diff(&old_model, &new_model, Some(&comprehensive_options)).unwrap();

    assert!(!results.is_empty());

    // Should detect multiple types of ML changes
    let ml_change_types = results
        .iter()
        .fold(std::collections::HashSet::new(), |mut acc, r| {
            let change_type = match r {
                DiffResult::LearningRateChanged(_, _, _) => "learning_rate",
                DiffResult::LossChange(_, _, _) => "loss",
                DiffResult::AccuracyChange(_, _, _) => "accuracy",
                DiffResult::WeightSignificantChange(_, _) => "weight",
                DiffResult::Modified(_, _, _) => "modified",
                DiffResult::Added(_, _) => "added",
                DiffResult::Removed(_, _) => "removed",
                _ => "other",
            };
            acc.insert(change_type);
            acc
        });

    // Should detect multiple types of changes in a comprehensive ML comparison
    assert!(ml_change_types.len() >= 3);
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[test]
fn test_large_ml_model_performance() {
    // Generate large model data
    let large_old = ml_generators::generate_model_weights(vec![1000, 500, 250, 100, 10]);
    let large_new = ml_generators::generate_model_weights(vec![1000, 600, 300, 100, 10]); // Different architecture

    let start = std::time::Instant::now();
    let results = diff(&large_old, &large_new, None).unwrap();
    let duration = start.elapsed();

    assert!(!results.is_empty());
    assert!(duration.as_secs() < 5); // Should complete within 5 seconds
}

#[test]
fn test_deep_model_structure_performance() {
    // Generate deeply nested model structure
    let mut deep_old = json!({});
    let mut deep_new = json!({});

    let mut current_old = &mut deep_old;
    let mut current_new = &mut deep_new;

    for i in 0..50 {
        // 50 levels deep
        let layer_name = format!("layer_{i}");
        current_old[&layer_name] = json!({
            "weights": {"mean": 0.01 * i as f64},
            "next": {}
        });
        current_new[&layer_name] = json!({
            "weights": {"mean": 0.011 * i as f64}, // Slightly different
            "next": {}
        });

        current_old = &mut current_old[&layer_name]["next"];
        current_new = &mut current_new[&layer_name]["next"];
    }

    let start = std::time::Instant::now();
    let results = diff(&deep_old, &deep_new, None).unwrap();
    let duration = start.elapsed();

    assert!(!results.is_empty());
    assert!(duration.as_secs() < 3); // Should handle deep nesting efficiently
}
