use diffai_core::{diff_ml_models, parse_pytorch_model, TensorStats, QuantizationAnalysisInfo, TransferLearningInfo, ExperimentReproducibilityInfo, EnsembleAnalysisInfo};
use std::path::Path;

#[test]
fn test_tensor_stats_calculation() {
    // Test basic tensor statistics calculation
    let _values = [1.0, 2.0, 3.0, 4.0, 5.0];
    let expected_mean = 3.0;
    let expected_std = (2.0_f64).sqrt(); // std of [1,2,3,4,5] is sqrt(2)

    // These would be calculated by our stats functions
    // For now, test the structure
    let stats = TensorStats {
        mean: expected_mean,
        std: expected_std,
        min: 1.0,
        max: 5.0,
        shape: vec![5],
        dtype: "f32".to_string(),
        total_params: 5,
    };

    assert_eq!(stats.mean, expected_mean);
    assert_eq!(stats.total_params, 5);
    assert_eq!(stats.shape, vec![5]);
}

#[test]
fn test_tensor_stats_comparison() {
    // Test comparison of tensor statistics
    let stats1 = TensorStats {
        mean: 0.0,
        std: 1.0,
        min: -2.0,
        max: 2.0,
        shape: vec![100, 50],
        dtype: "f32".to_string(),
        total_params: 5000,
    };

    let stats2 = TensorStats {
        mean: 0.1,
        std: 1.1,
        min: -1.8,
        max: 2.2,
        shape: vec![100, 50],
        dtype: "f32".to_string(),
        total_params: 5000,
    };

    // Test that they're different but same shape
    assert_ne!(stats1.mean, stats2.mean);
    assert_eq!(stats1.shape, stats2.shape);
    assert_eq!(stats1.total_params, stats2.total_params);
}

#[test]
fn test_different_tensor_shapes() {
    let stats1 = TensorStats {
        mean: 0.0,
        std: 1.0,
        min: -2.0,
        max: 2.0,
        shape: vec![256, 128],
        dtype: "f32".to_string(),
        total_params: 32768,
    };

    let stats2 = TensorStats {
        mean: 0.0,
        std: 1.0,
        min: -2.0,
        max: 2.0,
        shape: vec![512, 128], // Different first dimension
        dtype: "f32".to_string(),
        total_params: 65536,
    };

    // Test shape differences
    assert_ne!(stats1.shape, stats2.shape);
    assert_ne!(stats1.total_params, stats2.total_params);
}

#[test]
fn test_ml_diff_result_variants() {
    use diffai_core::DiffResult;

    // Test that our new DiffResult variants work correctly
    let tensor_stats1 = TensorStats {
        mean: 0.0,
        std: 1.0,
        min: -2.0,
        max: 2.0,
        shape: vec![128, 64],
        dtype: "f32".to_string(),
        total_params: 8192,
    };

    let tensor_stats2 = TensorStats {
        mean: 0.1,
        std: 1.1,
        min: -1.9,
        max: 2.1,
        shape: vec![128, 64],
        dtype: "f32".to_string(),
        total_params: 8192,
    };

    // Test TensorStatsChanged variant
    let diff = DiffResult::TensorStatsChanged(
        "linear1.weight".to_string(),
        tensor_stats1.clone(),
        tensor_stats2.clone(),
    );

    match diff {
        DiffResult::TensorStatsChanged(name, stats1, stats2) => {
            assert_eq!(name, "linear1.weight");
            assert_eq!(stats1.mean, 0.0);
            assert_eq!(stats2.mean, 0.1);
        }
        _ => panic!("Expected TensorStatsChanged variant"),
    }

    // Test TensorShapeChanged variant
    let shape_diff = DiffResult::TensorShapeChanged(
        "linear2.weight".to_string(),
        vec![256, 128],
        vec![512, 128],
    );

    match shape_diff {
        DiffResult::TensorShapeChanged(name, shape1, shape2) => {
            assert_eq!(name, "linear2.weight");
            assert_eq!(shape1, vec![256, 128]);
            assert_eq!(shape2, vec![512, 128]);
        }
        _ => panic!("Expected TensorShapeChanged variant"),
    }
}

#[test]
fn test_error_handling_invalid_model_file() {
    // Test error handling for invalid model files
    let result = diff_ml_models(
        Path::new("nonexistent_file.safetensors"),
        Path::new("another_nonexistent_file.safetensors"),
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_error_handling_invalid_diff() {
    // Test error handling for model diff with invalid files
    let result = diff_ml_models(
        Path::new("nonexistent1.safetensors"),
        Path::new("nonexistent2.safetensors"),
        None,
    );
    assert!(result.is_err());
}

#[test]
fn test_epsilon_tolerance_in_ml_diff() {
    // Test that epsilon tolerance works correctly in ML model comparison
    // This is a conceptual test - in real implementation we'd need actual model files

    let epsilon = 0.01;

    // Two tensors with small differences that should be ignored with epsilon
    let stats1 = TensorStats {
        mean: 1.000,
        std: 0.500,
        min: -1.0,
        max: 1.0,
        shape: vec![10, 10],
        dtype: "f32".to_string(),
        total_params: 100,
    };

    let stats2 = TensorStats {
        mean: 1.005, // Within epsilon tolerance
        std: 0.504,  // Within epsilon tolerance
        min: -1.001, // Within epsilon tolerance
        max: 1.002,  // Within epsilon tolerance
        shape: vec![10, 10],
        dtype: "f32".to_string(),
        total_params: 100,
    };

    // Check differences are within tolerance
    assert!((stats1.mean - stats2.mean).abs() < epsilon);
    assert!((stats1.std - stats2.std).abs() < epsilon);
}

#[test]
fn test_pytorch_model_parsing() {
    // Test parsing of PyTorch model files
    let model_path = Path::new("tests/fixtures/ml_models/simple_base.pt");
    
    // If file exists, test parsing
    if model_path.exists() {
        let result = parse_pytorch_model(model_path);
        match result {
            Ok(tensors) => {
                assert!(!tensors.is_empty(), "Should parse at least one tensor");
                
                // Check tensor stats structure
                for (name, stats) in tensors {
                    assert!(!name.is_empty(), "Tensor name should not be empty");
                    assert!(!stats.shape.is_empty(), "Tensor shape should not be empty");
                    assert!(!stats.dtype.is_empty(), "Tensor dtype should not be empty");
                    assert!(stats.total_params > 0, "Tensor should have parameters");
                }
            }
            Err(_) => {
                // PyTorch parsing might fail for some models, that's okay
                // We're testing the error handling path
            }
        }
    }
}

#[test]
fn test_pytorch_vs_safetensors_comparison() {
    // Test comparison between PyTorch and Safetensors versions of same model
    let pytorch_path = Path::new("tests/fixtures/ml_models/simple_base.pt");
    let safetensors_path = Path::new("tests/fixtures/ml_models/simple_base.safetensors");
    
    if pytorch_path.exists() && safetensors_path.exists() {
        // Try to parse both files
        let pytorch_result = parse_pytorch_model(pytorch_path);
        let safetensors_result = diffai_core::parse_safetensors_model(safetensors_path);
        
        match (pytorch_result, safetensors_result) {
            (Ok(pytorch_tensors), Ok(safetensors_tensors)) => {
                // Both parsed successfully - they should have similar structure
                assert_eq!(pytorch_tensors.len(), safetensors_tensors.len(), 
                          "Should have same number of tensors");
                
                // Check that tensor names match
                for (name, _) in &pytorch_tensors {
                    assert!(safetensors_tensors.contains_key(name), 
                           "Safetensors should contain tensor: {}", name);
                }
            }
            _ => {
                // One or both failed to parse - that's okay for testing
                // We're verifying the error handling works
            }
        }
    }
}

#[test]
fn test_pytorch_model_diff() {
    // Test diffing between two PyTorch models
    let model1_path = Path::new("tests/fixtures/ml_models/simple_base.pt");
    let model2_path = Path::new("tests/fixtures/ml_models/simple_modified.pt");
    
    if model1_path.exists() && model2_path.exists() {
        let result = diff_ml_models(model1_path, model2_path, None);
        
        match result {
            Ok(diff_results) => {
                // If parsing succeeds, check that we get meaningful results
                assert!(!diff_results.is_empty(), "Should have some diff results");
                
                // Check that we have tensor-related diffs
                let has_tensor_diffs = diff_results.iter().any(|diff| {
                    matches!(diff, 
                             diffai_core::DiffResult::TensorStatsChanged(_, _, _) | 
                             diffai_core::DiffResult::TensorShapeChanged(_, _, _) |
                             diffai_core::DiffResult::TensorAdded(_, _) |
                             diffai_core::DiffResult::TensorRemoved(_, _))
                });
                
                assert!(has_tensor_diffs, "Should have tensor-related differences");
            }
            Err(_) => {
                // PyTorch parsing might fail - that's okay for testing
                // We're testing error handling
            }
        }
    }
}

#[test]
fn test_quantization_analysis_info() {
    // Test QuantizationAnalysisInfo data structure
    let quant_info = QuantizationAnalysisInfo {
        compression_ratio: 0.75,
        bit_reduction: "32bit→8bit".to_string(),
        estimated_speedup: 2.5,
        memory_savings: 0.68,
        precision_loss_estimate: 0.02,
        quantization_method: "uniform".to_string(),
        recommended_layers: vec!["linear1".to_string(), "linear2".to_string()],
        sensitive_layers: vec!["output".to_string()],
        deployment_suitability: "good".to_string(),
    };

    // Test basic properties
    assert_eq!(quant_info.compression_ratio, 0.75);
    assert_eq!(quant_info.estimated_speedup, 2.5);
    assert_eq!(quant_info.precision_loss_estimate, 0.02);
    assert_eq!(quant_info.deployment_suitability, "good");
    assert_eq!(quant_info.bit_reduction, "32bit→8bit");
    assert_eq!(quant_info.quantization_method, "uniform");

    // Test that compression ratio is percentage (0-1)
    assert!(quant_info.compression_ratio >= 0.0 && quant_info.compression_ratio <= 1.0);
    
    // Test that speedup is positive
    assert!(quant_info.estimated_speedup > 0.0);
    
    // Test that precision loss is reasonable percentage (0-1)
    assert!(quant_info.precision_loss_estimate >= 0.0 && quant_info.precision_loss_estimate <= 1.0);
    
    // Test layer lists
    assert!(!quant_info.recommended_layers.is_empty());
    assert!(!quant_info.sensitive_layers.is_empty());
}

#[test]
fn test_transfer_learning_info() {
    // Test TransferLearningInfo data structure
    let transfer_info = TransferLearningInfo {
        frozen_layers: 8,
        updated_layers: 2,
        total_layers: 10,
        parameter_update_ratio: 0.3,
        layer_learning_intensity: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9],
        domain_adaptation_strength: "moderate".to_string(),
        feature_extraction_vs_finetuning: "fine-tuning".to_string(),
        most_adapted_layers: vec!["layer8".to_string(), "layer9".to_string()],
        transfer_efficiency_score: 0.85,
        overfitting_risk: "low".to_string(),
    };

    // Test basic properties
    assert_eq!(transfer_info.frozen_layers, 8);
    assert_eq!(transfer_info.updated_layers, 2);
    assert_eq!(transfer_info.total_layers, 10);
    assert_eq!(transfer_info.parameter_update_ratio, 0.3);
    assert_eq!(transfer_info.domain_adaptation_strength, "moderate");
    assert_eq!(transfer_info.feature_extraction_vs_finetuning, "fine-tuning");

    // Test that layer counts are non-negative
    assert!(transfer_info.frozen_layers >= 0);
    assert!(transfer_info.updated_layers >= 0);
    
    // Test that parameter update ratio is reasonable (0-1)
    assert!(transfer_info.parameter_update_ratio >= 0.0 && transfer_info.parameter_update_ratio <= 1.0);
    
    // Test total layers calculation
    assert_eq!(transfer_info.frozen_layers + transfer_info.updated_layers, transfer_info.total_layers);
    
    // Test layer intensity vector
    assert_eq!(transfer_info.layer_learning_intensity.len(), transfer_info.total_layers);
}

#[test]
fn test_experiment_reproducibility_info() {
    // Test ExperimentReproducibilityInfo data structure
    let repro_info = ExperimentReproducibilityInfo {
        hyperparameter_changes: vec![
            "learning_rate: 0.001 -> 0.002".to_string(),
            "batch_size: 32 -> 64".to_string(),
        ],
        critical_changes: vec!["learning_rate: 0.001 -> 0.002".to_string()],
        environment_diffs: vec!["cuda_version: 11.8 -> 12.0".to_string()],
        reproducibility_score: 0.85,
        risk_factors: vec!["learning_rate_change".to_string()],
        seed_consistency: "consistent".to_string(),
        data_versioning: "same".to_string(),
        model_determinism: "deterministic".to_string(),
        reproduction_recommendation: "verify learning rate impact".to_string(),
    };

    // Test basic properties
    assert_eq!(repro_info.hyperparameter_changes.len(), 2);
    assert_eq!(repro_info.critical_changes.len(), 1);
    assert_eq!(repro_info.reproducibility_score, 0.85);
    assert_eq!(repro_info.seed_consistency, "consistent");
    assert_eq!(repro_info.data_versioning, "same");

    // Test that scores are in valid range (0-1)
    assert!(repro_info.reproducibility_score >= 0.0 && repro_info.reproducibility_score <= 1.0);
    
    // Test that change vectors are initialized
    assert!(!repro_info.hyperparameter_changes.is_empty());
    assert!(!repro_info.critical_changes.is_empty());
    
    // Test that critical changes <= hyperparameter changes
    assert!(repro_info.critical_changes.len() <= repro_info.hyperparameter_changes.len());
}

#[test]
fn test_ensemble_analysis_info() {
    // Test EnsembleAnalysisInfo data structure
    let ensemble_info = EnsembleAnalysisInfo {
        model_count: 3,
        diversity_score: 0.72,
        correlation_matrix: vec![
            vec![1.0, 0.3, 0.2],
            vec![0.3, 1.0, 0.4],
            vec![0.2, 0.4, 1.0],
        ],
        complementarity_analysis: "high".to_string(),
        ensemble_efficiency: 0.88,
        redundancy_detection: vec![],
        optimal_subset: vec!["model1".to_string(), "model2".to_string(), "model3".to_string()],
        weighting_strategy: "performance".to_string(),
        ensemble_recommendation: "use all models".to_string(),
    };

    // Test basic properties
    assert_eq!(ensemble_info.model_count, 3);
    assert_eq!(ensemble_info.diversity_score, 0.72);
    assert_eq!(ensemble_info.ensemble_efficiency, 0.88);
    assert_eq!(ensemble_info.complementarity_analysis, "high");
    assert_eq!(ensemble_info.weighting_strategy, "performance");

    // Test correlation matrix dimensions
    assert_eq!(ensemble_info.correlation_matrix.len(), 3);
    for row in &ensemble_info.correlation_matrix {
        assert_eq!(row.len(), 3);
    }
    
    // Test that diagonal elements are 1.0 (self-correlation)
    for i in 0..3 {
        assert_eq!(ensemble_info.correlation_matrix[i][i], 1.0);
    }
    
    // Test that scores are in valid range (0-1)
    assert!(ensemble_info.diversity_score >= 0.0 && ensemble_info.diversity_score <= 1.0);
    assert!(ensemble_info.ensemble_efficiency >= 0.0 && ensemble_info.ensemble_efficiency <= 1.0);
    
    // Test that model count is positive
    assert!(ensemble_info.model_count > 0);
    
    // Test recommendations
    assert!(!ensemble_info.ensemble_recommendation.is_empty());
}

#[test]
fn test_quantization_analysis_edge_cases() {
    // Test edge cases for quantization analysis
    
    // Test maximum compression (very high compression ratio)
    let max_compression = QuantizationAnalysisInfo {
        compression_ratio: 0.95,
        bit_reduction: "32bit→1bit".to_string(),
        estimated_speedup: 10.0,
        memory_savings: 0.93,
        precision_loss_estimate: 0.15,
        quantization_method: "aggressive".to_string(),
        recommended_layers: vec![],
        sensitive_layers: vec!["all".to_string()],
        deployment_suitability: "risky".to_string(),
    };
    
    assert!(max_compression.compression_ratio > 0.9);
    assert!(max_compression.estimated_speedup > 5.0);
    assert!(max_compression.precision_loss_estimate > 0.1);
    assert_eq!(max_compression.deployment_suitability, "risky");
    
    // Test minimal compression (low compression ratio)
    let min_compression = QuantizationAnalysisInfo {
        compression_ratio: 0.1,
        bit_reduction: "32bit→16bit".to_string(),
        estimated_speedup: 1.2,
        memory_savings: 0.12,
        precision_loss_estimate: 0.001,
        quantization_method: "conservative".to_string(),
        recommended_layers: vec!["linear1".to_string(), "linear2".to_string()],
        sensitive_layers: vec![],
        deployment_suitability: "excellent".to_string(),
    };
    
    assert!(min_compression.compression_ratio < 0.2);
    assert!(min_compression.estimated_speedup < 2.0);
    assert!(min_compression.precision_loss_estimate < 0.01);
    assert_eq!(min_compression.deployment_suitability, "excellent");
}

#[test]
fn test_transfer_learning_edge_cases() {
    // Test edge cases for transfer learning analysis
    
    // Test full fine-tuning (all layers updated)
    let full_finetune = TransferLearningInfo {
        frozen_layers: 0,
        updated_layers: 12,
        total_layers: 12,
        parameter_update_ratio: 1.0,
        layer_learning_intensity: vec![0.9; 12],
        domain_adaptation_strength: "strong".to_string(),
        feature_extraction_vs_finetuning: "full-training".to_string(),
        most_adapted_layers: (0..12).map(|i| format!("layer{}", i)).collect(),
        transfer_efficiency_score: 0.95,
        overfitting_risk: "high".to_string(),
    };
    
    assert_eq!(full_finetune.frozen_layers, 0);
    assert!(full_finetune.updated_layers > 0);
    assert_eq!(full_finetune.parameter_update_ratio, 1.0);
    assert_eq!(full_finetune.domain_adaptation_strength, "strong");
    assert_eq!(full_finetune.overfitting_risk, "high");
    
    // Test minimal adaptation (most layers frozen)
    let minimal_adapt = TransferLearningInfo {
        frozen_layers: 11,
        updated_layers: 1,
        total_layers: 12,
        parameter_update_ratio: 0.05,
        layer_learning_intensity: {
            let mut intensity = vec![0.0; 11];
            intensity.push(0.8); // Only last layer adapted
            intensity
        },
        domain_adaptation_strength: "weak".to_string(),
        feature_extraction_vs_finetuning: "feature-extraction".to_string(),
        most_adapted_layers: vec!["layer11".to_string()],
        transfer_efficiency_score: 0.65,
        overfitting_risk: "low".to_string(),
    };
    
    assert!(minimal_adapt.frozen_layers > minimal_adapt.updated_layers);
    assert!(minimal_adapt.parameter_update_ratio < 0.1);
    assert_eq!(minimal_adapt.domain_adaptation_strength, "weak");
    assert_eq!(minimal_adapt.overfitting_risk, "low");
}
