use diffai_core::*;
use serde_json::json;
use std::path::Path;

#[path = "fixtures.rs"]
mod fixtures;
use fixtures::{ml_generators, TestFixtures};

// ============================================================================
// UNIFIED API CORE TESTS - Basic Functionality
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
fn test_diff_added_removed() {
    let old = json!({"name": "Alice"});
    let new = json!({"name": "Alice", "age": 30});

    let results = diff(&old, &new, None).unwrap();

    assert_eq!(results.len(), 1);
    match &results[0] {
        DiffResult::Added(path, value) => {
            assert_eq!(path, "age");
            assert_eq!(value, &json!(30));
        }
        _ => panic!("Expected Added result"),
    }

    // Test removal
    let results = diff(&new, &old, None).unwrap();
    assert_eq!(results.len(), 1);
    match &results[0] {
        DiffResult::Removed(path, value) => {
            assert_eq!(path, "age");
            assert_eq!(value, &json!(30));
        }
        _ => panic!("Expected Removed result"),
    }
}

#[test]
fn test_diff_type_changed() {
    let old = json!({"value": "30"});
    let new = json!({"value": 30});

    let results = diff(&old, &new, None).unwrap();

    assert_eq!(results.len(), 1);
    match &results[0] {
        DiffResult::TypeChanged(path, old_val, new_val) => {
            assert_eq!(path, "value");
            assert_eq!(old_val, &json!("30"));
            assert_eq!(new_val, &json!(30));
        }
        _ => panic!("Expected TypeChanged result"),
    }
}

#[test]
fn test_diff_no_changes() {
    let old = json!({"name": "Alice", "age": 30});
    let new = json!({"name": "Alice", "age": 30});

    let results = diff(&old, &new, None).unwrap();
    assert_eq!(results.len(), 0);
}

// ============================================================================
// ML ANALYSIS FEATURES TESTS - Automatic Detection
// ============================================================================

#[test]
fn test_tensor_stats_changed_detection() {
    let old = json!({
        "layers": {
            "conv1.weight": {
                "shape": [64, 3, 7, 7],
                "data": [0.1, 0.2, 0.15, 0.18],
                "dtype": "float32"
            }
        }
    });

    let new = json!({
        "layers": {
            "conv1.weight": {
                "shape": [64, 3, 7, 7],
                "data": [0.2, 0.3, 0.25, 0.28],
                "dtype": "float32"
            }
        }
    });

    let results = diff(&old, &new, None).unwrap();

    // Should contain TensorStatsChanged result for weight changes
    let tensor_stats_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::TensorStatsChanged(_, _, _)))
        .count();

    assert!(tensor_stats_changes > 0, "Should detect tensor statistics changes");
}

#[test]
fn test_model_architecture_changed_detection() {
    let old = json!({
        "model_info": {
            "architecture": "ResNet18",
            "layers": ["conv1", "bn1", "relu", "maxpool", "layer1"]
        }
    });

    let new = json!({
        "model_info": {
            "architecture": "ResNet50",
            "layers": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"]
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let architecture_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::ModelArchitectureChanged(_, _, _)))
        .count();

    assert!(architecture_changes > 0, "Should detect model architecture changes");
}

#[test]
fn test_learning_rate_changed_detection() {
    let old = json!({
        "optimizer": {
            "type": "Adam",
            "learning_rate": 0.001
        }
    });

    let new = json!({
        "optimizer": {
            "type": "Adam",
            "learning_rate": 0.01
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let lr_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)))
        .count();

    assert!(lr_changes > 0, "Should detect learning rate changes");
}

#[test]
fn test_weight_significant_change_detection() {
    let old = json!({
        "weights": {
            "layer1": 0.1,
            "layer2": 0.05
        }
    });

    let new = json!({
        "weights": {
            "layer1": 0.2,   // 0.1 change - significant
            "layer2": 0.051  // 0.001 change - not significant
        }
    });

    let results = diff(&old, &new, None).unwrap();

    let significant_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::WeightSignificantChange(_, _)))
        .count();

    assert!(significant_changes > 0, "Should detect significant weight changes");
}

// ============================================================================
// TENSOR STATISTICS TESTS
// ============================================================================

#[test]
fn test_tensor_stats_calculation() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let shape = vec![5];
    let dtype = "float32".to_string();

    let stats = TensorStats::new(&data, shape.clone(), dtype.clone());

    assert_eq!(stats.mean, 3.0);
    assert_eq!(stats.min, 1.0);
    assert_eq!(stats.max, 5.0);
    assert_eq!(stats.shape, shape);
    assert_eq!(stats.dtype, dtype);
    assert_eq!(stats.element_count, 5);
    
    // Standard deviation should be sqrt(2) = ~1.414
    assert!((stats.std - 1.414).abs() < 0.01);
}

#[test]
fn test_tensor_stats_empty_data() {
    let data = vec![];
    let shape = vec![0];
    let dtype = "float32".to_string();

    let stats = TensorStats::new(&data, shape.clone(), dtype.clone());

    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std, 0.0);
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
    assert_eq!(stats.element_count, 0);
}

// ============================================================================
// OUTPUT FORMAT TESTS
// ============================================================================

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

    assert!(OutputFormat::parse_format("invalid").is_err());
}

#[test]
fn test_format_output_basic() {
    let results = vec![
        DiffResult::Modified("name".to_string(), json!("old"), json!("new")),
        DiffResult::Added("age".to_string(), json!(30)),
    ];

    // Test all supported formats
    for format in OutputFormat::value_variants() {
        let output = format_output(&results, *format).unwrap();
        assert!(!output.is_empty(), "Output should not be empty for format: {:?}", format);
    }
}

// ============================================================================
// DIFF OPTIONS TESTS
// ============================================================================

#[test]
fn test_diff_options_default() {
    let options = DiffOptions::default();
    
    assert_eq!(options.epsilon, None);
    assert!(options.ignore_keys_regex.is_none()); 
    assert_eq!(options.output_format, None);
    assert_eq!(options.use_memory_optimization, None);
}

#[test]
fn test_diff_with_epsilon() {
    let old = json!({"value": 1.0001});
    let new = json!({"value": 1.0002});

    // Without epsilon - should detect change
    let results = diff(&old, &new, None).unwrap();
    assert!(!results.is_empty());

    // With epsilon - should ignore small change
    let options = DiffOptions {
        epsilon: Some(0.001),
        ..Default::default()
    };
    let results = diff(&old, &new, Some(&options)).unwrap();
    assert!(results.is_empty(), "Small changes within epsilon should be ignored");
}

// ============================================================================
// FILE FORMAT HANDLING TESTS (Basic Path Extension Check)
// ============================================================================

#[test]
fn test_path_extension_recognition() {
    let pytorch_files = vec!["model.pt", "checkpoint.pth"];
    let safetensors_files = vec!["model.safetensors"];
    let numpy_files = vec!["data.npy", "arrays.npz"];
    let matlab_files = vec!["network.mat"];

    // Test that we can distinguish file types by extension
    for filename in pytorch_files {
        let path = Path::new(filename);
        assert!(path.extension().is_some());
        assert!(matches!(path.extension().unwrap().to_str(), Some("pt") | Some("pth")));
    }

    for filename in safetensors_files {
        let path = Path::new(filename);
        assert_eq!(path.extension().unwrap().to_str(), Some("safetensors"));
    }

    for filename in numpy_files {
        let path = Path::new(filename);
        assert!(matches!(path.extension().unwrap().to_str(), Some("npy") | Some("npz")));
    }

    for filename in matlab_files {
        let path = Path::new(filename);
        assert_eq!(path.extension().unwrap().to_str(), Some("mat"));
    }
}

#[test]
fn test_path_without_extension() {
    let path = Path::new("test_file");
    assert!(path.extension().is_none(), "File without extension should return None");
}

// ============================================================================
// MEMORY OPTIMIZATION TESTS
// ============================================================================

#[test]
fn test_memory_optimization_flag() {
    let old = json!({"data": [1, 2, 3, 4, 5]});
    let new = json!({"data": [1, 2, 3, 4, 6]});

    let options = DiffOptions {
        use_memory_optimization: Some(true),
        ..Default::default()
    };

    let results = diff(&old, &new, Some(&options)).unwrap();
    assert!(!results.is_empty());
}

// ============================================================================
// COMPLEX NESTED STRUCTURE TESTS
// ============================================================================

#[test]
fn test_nested_structure_diff() {
    let old = json!({
        "model": {
            "layers": {
                "conv1": {
                    "weights": {"mean": 0.1, "std": 0.05},
                    "bias": {"mean": 0.0, "std": 0.01}
                },
                "fc": {
                    "weights": {"mean": 0.0, "std": 0.02}
                }
            },
            "optimizer": {
                "type": "Adam",
                "lr": 0.001
            }
        }
    });

    let new = json!({
        "model": {
            "layers": {
                "conv1": {
                    "weights": {"mean": 0.12, "std": 0.06},  // Changed
                    "bias": {"mean": 0.0, "std": 0.01}
                },
                "fc": {
                    "weights": {"mean": 0.0, "std": 0.02}
                },
                "dropout": {  // Added layer
                    "rate": 0.5
                }
            },
            "optimizer": {
                "type": "SGD",  // Changed
                "lr": 0.01      // Changed
            }
        }
    });

    let results = diff(&old, &new, None).unwrap();
    
    assert!(!results.is_empty());
    
    // Should detect multiple types of changes
    let change_types: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            DiffResult::Modified(_, _, _) => "modified",
            DiffResult::Added(_, _) => "added",
            DiffResult::LearningRateChanged(_, _, _) => "lr_changed",
            _ => "other"
        })
        .collect();
    
    assert!(change_types.len() >= 2, "Should detect multiple types of changes");
}

// ============================================================================
// PYTORCH SPECIFIC TESTS USING FIXTURES
// ============================================================================

#[test]
fn test_pytorch_model_fixture_comparison() {
    let old_model = TestFixtures::pytorch_model_old();
    let new_model = TestFixtures::pytorch_model_new();

    let results = diff(&old_model, &new_model, None).unwrap();
    assert!(!results.is_empty());

    // Should detect optimizer change (Adam -> SGD)
    let optimizer_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("optimizer.type")))
        .count();
    assert!(optimizer_changes > 0);

    // Should detect learning rate change
    let lr_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)))
        .count();
    assert!(lr_changes > 0);
}

// ============================================================================
// SAFETENSORS SPECIFIC TESTS USING FIXTURES
// ============================================================================

#[test]
fn test_safetensors_model_fixture_comparison() {
    let old_model = TestFixtures::safetensors_model_old();
    let new_model = TestFixtures::safetensors_model_new();

    let results = diff(&old_model, &new_model, None).unwrap();
    assert!(!results.is_empty());

    // Should detect tensor shape changes
    let shape_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("shape")))
        .count();
    assert!(shape_changes > 0);

    // Should detect new tensors
    let added_tensors = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Added(path, _) if path.contains("new_layer")))
        .count();
    assert!(added_tensors > 0);
}

// ============================================================================
// NUMPY SPECIFIC TESTS USING FIXTURES
// ============================================================================

#[test]
fn test_numpy_array_fixture_comparison() {
    let old_array = TestFixtures::numpy_array_old();
    let new_array = TestFixtures::numpy_array_new();

    let results = diff(&old_array, &new_array, None).unwrap();
    assert!(!results.is_empty());

    // Should detect array shape changes
    let shape_changes = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("shape")))
        .count();
    assert!(shape_changes > 0);

    // Should detect new arrays
    let added_arrays = results
        .iter()
        .filter(|r| matches!(r, DiffResult::Added(path, _) if path.contains("weights")))
        .count();
    assert!(added_arrays > 0);
}

// ============================================================================
// MATLAB SPECIFIC TESTS USING FIXTURES
// ============================================================================

#[test]
fn test_matlab_file_fixture_comparison() {
    let old_file = TestFixtures::matlab_file_old();
    let new_file = TestFixtures::matlab_file_new();

    let results = diff(&old_file, &new_file, None).unwrap();
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
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

#[test]
fn test_large_model_performance() {
    let large_old = ml_generators::generate_model_weights(vec![1000, 500, 100]);
    let large_new = ml_generators::generate_model_weights(vec![1000, 600, 100]);

    let start = std::time::Instant::now();
    let results = diff(&large_old, &large_new, None).unwrap();
    let duration = start.elapsed();

    assert!(!results.is_empty());
    assert!(duration.as_secs() < 5, "Should complete within 5 seconds");
}

#[test]
fn test_deep_nested_structure_performance() {
    let mut deep_old = json!({});
    let mut deep_new = json!({});

    // Create 20 levels of nesting 
    let mut current_old = &mut deep_old;
    let mut current_new = &mut deep_new;

    for i in 0..20 {
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
    assert!(duration.as_secs() < 3, "Should handle deep nesting efficiently");
}

// ============================================================================
// COMPREHENSIVE WORKFLOW TESTS
// ============================================================================

#[test]
fn test_comprehensive_ml_workflow() {
    let old_model = TestFixtures::pytorch_model_old();
    let new_model = TestFixtures::pytorch_model_new();

    let results = diff(&old_model, &new_model, None).unwrap();
    assert!(!results.is_empty());

    // Should detect multiple types of ML changes
    let change_types: std::collections::HashSet<_> = results
        .iter()
        .map(|r| match r {
            DiffResult::LearningRateChanged(_, _, _) => "learning_rate",
            DiffResult::WeightSignificantChange(_, _) => "weight",
            DiffResult::Modified(_, _, _) => "modified",
            DiffResult::Added(_, _) => "added",
            _ => "other",
        })
        .collect();

    assert!(change_types.len() >= 2, "Should detect multiple types of changes");
}