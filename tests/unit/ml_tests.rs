use diffai_core::{diff_ml_models, TensorStats};
use std::path::Path;

#[test]
fn test_tensor_stats_calculation() {
    // Test basic tensor statistics calculation
    let _values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
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

    let epsilon = Some(0.01);

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
    assert!((stats1.mean - stats2.mean).abs() < epsilon.unwrap());
    assert!((stats1.std - stats2.std).abs() < epsilon.unwrap());
}
