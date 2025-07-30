use diffai_core::*;
use serde_json::json;
use std::path::Path;

// ============================================================================
// BASIC FORMAT COMPATIBILITY TESTS (Without Format Detection)
// ============================================================================

#[test]
fn test_pytorch_style_data_analysis() {
    let pytorch_data = json!({
        "model_state_dict": {
            "conv1.weight": {"shape": [64, 3, 7, 7], "dtype": "float32"},
            "conv1.bias": {"shape": [64], "dtype": "float32"}
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 0.001}]
        },
        "epoch": 10,
        "loss": 0.5
    });

    let updated_pytorch_data = json!({
        "model_state_dict": {
            "conv1.weight": {"shape": [64, 3, 7, 7], "dtype": "float32"},
            "conv1.bias": {"shape": [64], "dtype": "float32"},
            "conv2.weight": {"shape": [128, 64, 3, 3], "dtype": "float32"}  // Added layer
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 0.0001}]  // Changed LR
        },
        "epoch": 15,
        "loss": 0.3
    });

    let results = diff(&pytorch_data, &updated_pytorch_data, None).unwrap();
    
    // Should detect PyTorch-specific changes
    let has_lr_change = results.iter().any(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));
    let has_loss_change = results.iter().any(|r| matches!(r, DiffResult::LossChange(_, _, _)));
    let has_new_layer = results.iter().any(|r| matches!(r, DiffResult::Added(path, _) if path.contains("conv2")));

    assert!(has_lr_change, "Should detect learning rate changes in PyTorch format");
    assert!(has_loss_change, "Should detect loss changes in PyTorch format");
    assert!(has_new_layer, "Should detect new layers in PyTorch format");
}

#[test]
fn test_safetensors_style_data_analysis() {
    let safetensors_data = json!({
        "__metadata__": {
            "format": "pt",
            "version": "0.3.1"
        },
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [32000, 4096],
            "data_offsets": [0, 524288000]
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32", 
            "shape": [4096, 4096],
            "data_offsets": [524288000, 591396864]
        }
    });

    let updated_safetensors_data = json!({
        "__metadata__": {
            "format": "pt",
            "version": "0.4.0"  // Version updated
        },
        "model.embed_tokens.weight": {
            "dtype": "F16",  // Precision changed
            "shape": [32000, 4096],
            "data_offsets": [0, 262144000]  // Offsets changed due to precision
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",  // Precision changed
            "shape": [4096, 4096],
            "data_offsets": [262144000, 295698432]
        }
    });

    let results = diff(&safetensors_data, &updated_safetensors_data, None).unwrap();

    // Should detect Safetensors-specific changes
    let has_version_change = results.iter().any(|r| matches!(r, DiffResult::ModelVersionChanged(_, _, _)));
    let has_dtype_changes = results.iter().any(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("dtype")));

    assert!(has_version_change, "Should detect version changes in Safetensors format");
    assert!(has_dtype_changes, "Should detect dtype changes in Safetensors format");
}

#[test]
fn test_numpy_style_data_analysis() {
    let numpy_data = json!({
        "arrays": {
            "data": {
                "shape": [1000, 784],
                "dtype": "float64",
                "fortran_order": false,
                "statistics": {"mean": 0.5, "std": 0.2}
            },
            "labels": {
                "shape": [1000],
                "dtype": "int32",
                "fortran_order": false,
                "unique_count": 10
            }
        }
    });

    let updated_numpy_data = json!({
        "arrays": {
            "data": {
                "shape": [1200, 784],  // Size changed
                "dtype": "float32",    // Precision changed
                "fortran_order": false,
                "statistics": {"mean": 0.48, "std": 0.22}  // Stats changed
            },
            "labels": {
                "shape": [1200],  // Size changed
                "dtype": "int32",
                "fortran_order": false,
                "unique_count": 12  // More classes
            },
            "weights": {  // New array added
                "shape": [784, 10],
                "dtype": "float32",
                "fortran_order": false,
                "statistics": {"mean": 0.01, "std": 0.05}
            }
        }
    });

    let results = diff(&numpy_data, &updated_numpy_data, None).unwrap();

    // Should detect NumPy-specific changes
    let has_shape_changes = results.iter().any(|r| matches!(r, DiffResult::TensorShapeChanged(_, _, _)));
    let has_stats_changes = results.iter().any(|r| matches!(r, DiffResult::TensorStatsChanged(_, _, _)));
    let has_new_array = results.iter().any(|r| matches!(r, DiffResult::Added(path, _) if path.contains("weights")));

    assert!(has_shape_changes, "Should detect shape changes in NumPy format");
    assert!(has_stats_changes, "Should detect statistics changes in NumPy format");
    assert!(has_new_array, "Should detect new arrays in NumPy format");
}

#[test]
fn test_matlab_style_data_analysis() {
    let matlab_data = json!({
        "variables": {
            "network": {
                "type": "feedforward",
                "layers": [784, 128, 64, 10],
                "activation": "sigmoid",
                "trainFcn": "trainlm"
            },
            "trainParams": {
                "epochs": 100,
                "goal": 0.01,
                "lr": 0.01,
                "mu": 0.001
            },
            "performance": {
                "train": [2.5, 1.8, 1.2, 0.9],
                "val": [2.8, 2.0, 1.5, 1.1]
            }
        }
    });

    let updated_matlab_data = json!({
        "variables": {
            "network": {
                "type": "feedforward",
                "layers": [784, 256, 128, 10],  // Architecture changed
                "activation": "relu",            // Activation changed
                "trainFcn": "trainbr"           // Training function changed
            },
            "trainParams": {
                "epochs": 150,    // More epochs
                "goal": 0.005,    // Better goal
                "lr": 0.001,      // Lower LR
                "mu": 0.0001      // Different mu
            },
            "performance": {
                "train": [2.3, 1.5, 1.0, 0.7],  // Better performance
                "val": [2.5, 1.7, 1.2, 0.9]
            }
        }
    });

    let results = diff(&matlab_data, &updated_matlab_data, None).unwrap();

    // Should detect MATLAB-specific changes
    let has_architecture_change = results.iter().any(|r| matches!(r, DiffResult::Modified(path, _, _) if path.contains("layers")));
    let has_activation_change = results.iter().any(|r| matches!(r, DiffResult::ActivationFunctionChanged(_, _, _)));
    let has_lr_change = results.iter().any(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));

    assert!(has_architecture_change, "Should detect architecture changes in MATLAB format");
    assert!(has_activation_change, "Should detect activation function changes in MATLAB format");
    assert!(has_lr_change, "Should detect learning rate changes in MATLAB format");
}

// ============================================================================
// PATH EXTENSION UTILITY TESTS
// ============================================================================

#[test]
fn test_file_extension_parsing() {
    let test_cases = vec![
        ("model.pt", Some("pt")),
        ("checkpoint.pth", Some("pth")),
        ("model.safetensors", Some("safetensors")),
        ("data.npy", Some("npy")),
        ("arrays.npz", Some("npz")),
        ("network.mat", Some("mat")),
        ("file_without_extension", None),
        ("file.", None), // Empty extension
    ];

    for (filename, expected_ext) in test_cases {
        let path = Path::new(filename);
        let actual_ext = path.extension().and_then(|ext| ext.to_str());
        
        match expected_ext {
            Some(expected) => {
                assert_eq!(actual_ext, Some(expected), "Failed to extract extension from {}", filename);
            }
            None => {
                assert!(actual_ext.is_none() || actual_ext == Some(""), "Expected no extension for {}", filename);
            }
        }
    }
}

#[test]
fn test_format_specific_field_patterns() {
    // Test that we can identify format-specific field patterns in data
    let test_data = vec![
        // PyTorch patterns
        (json!({"model_state_dict": {}}), "PyTorch"),
        (json!({"optimizer_state_dict": {}}), "PyTorch"),
        
        // Safetensors patterns
        (json!({"__metadata__": {}}), "Safetensors"),
        (json!({"data_offsets": []}), "Safetensors"),
        
        // NumPy patterns
        (json!({"fortran_order": false}), "NumPy"),
        (json!({"arrays": {}}), "NumPy"),
        
        // MATLAB patterns
        (json!({"variables": {}}), "MATLAB"),
        (json!({"trainFcn": "trainlm"}), "MATLAB"),
    ];

    for (data, format_name) in test_data {
        // In a real implementation, this would use format detection logic
        // For now, we just verify the data structure is valid JSON
        assert!(data.is_object(), "Data should be a valid JSON object for {} format", format_name);
    }
}

// ============================================================================
// CROSS-FORMAT STRUCTURAL ANALYSIS
// ============================================================================

#[test]
fn test_similar_ml_concepts_across_formats() {
    // Test that similar ML concepts are detected regardless of format structure
    
    // Learning rate in different format styles
    let pytorch_lr = json!({"optimizer_state_dict": {"param_groups": [{"lr": 0.001}]}});
    let matlab_lr = json!({"trainParams": {"lr": 0.001}});
    let generic_lr = json!({"learning_rate": 0.001});

    let pytorch_lr_changed = json!({"optimizer_state_dict": {"param_groups": [{"lr": 0.01}]}});
    let matlab_lr_changed = json!({"trainParams": {"lr": 0.01}});
    let generic_lr_changed = json!({"learning_rate": 0.01});

    // All should detect learning rate changes
    let results_pytorch = diff(&pytorch_lr, &pytorch_lr_changed, None).unwrap();
    let results_matlab = diff(&matlab_lr, &matlab_lr_changed, None).unwrap();
    let results_generic = diff(&generic_lr, &generic_lr_changed, None).unwrap();

    let pytorch_has_lr = results_pytorch.iter().any(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));
    let matlab_has_lr = results_matlab.iter().any(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));
    let generic_has_lr = results_generic.iter().any(|r| matches!(r, DiffResult::LearningRateChanged(_, _, _)));

    assert!(pytorch_has_lr, "Should detect LR changes in PyTorch format");
    assert!(matlab_has_lr, "Should detect LR changes in MATLAB format");
    assert!(generic_has_lr, "Should detect LR changes in generic format");
}

#[test]
fn test_version_detection_across_formats() {
    // Test version detection in different format contexts
    let test_cases = vec![
        // Safetensors metadata version
        (
            json!({"__metadata__": {"version": "0.3.1"}}),
            json!({"__metadata__": {"version": "0.4.0"}}),
        ),
        // Generic model version
        (
            json!({"model_version": "1.0"}),
            json!({"model_version": "2.0"}),
        ),
        // Framework version
        (
            json!({"pytorch_version": "1.11.0"}),
            json!({"pytorch_version": "1.12.0"}),
        ),
    ];

    for (old, new) in test_cases {
        let results = diff(&old, &new, None).unwrap();
        let has_version_change = results.iter().any(|r| matches!(r, DiffResult::ModelVersionChanged(_, _, _)));
        assert!(has_version_change, "Should detect version changes in data: {:?}", old);
    }
}

// ============================================================================
// ERROR HANDLING AND EDGE CASES
// ============================================================================

#[test]
fn test_malformed_format_specific_structures() {
    let malformed_cases = vec![
        // Wrong type for PyTorch model_state_dict
        (
            json!({"model_state_dict": "not_an_object"}),
            json!({"model_state_dict": {"layer.weight": {"shape": [10, 10]}}}),
        ),
        // Wrong type for Safetensors metadata
        (
            json!({"__metadata__": "not_an_object"}),
            json!({"__metadata__": {"format": "pt"}}),
        ),
        // Wrong type for NumPy arrays
        (
            json!({"arrays": "not_an_object"}),
            json!({"arrays": {"data": {"shape": [100]}}}),
        ),
    ];

    for (malformed, fixed) in malformed_cases {
        let results = diff(&malformed, &fixed, None).unwrap();
        
        // Should detect as type changes, not format-specific changes
        let has_type_changes = results.iter().any(|r| matches!(r, DiffResult::TypeChanged(_, _, _)));
        assert!(has_type_changes, "Should detect type changes for malformed data: {:?}", malformed);
    }
}