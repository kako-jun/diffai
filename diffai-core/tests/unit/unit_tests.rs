// Unit tests for diffai core components
// Test individual functions and modules in isolation

#[cfg(test)]
mod tests {
    use diffai_core::{
        diff_basic, diff_ml_models_enhanced, parse_csv, parse_xml, AnomalyInfo, ConvergenceInfo,
        DiffResult, LearningProgressInfo, MemoryAnalysisInfo, ModelInfo, TensorStats,
    };
    use serde_json::Value;
    use std::collections::HashMap;

    mod diff_result_tests {
        use super::*;

        #[test]
        fn test_diff_result_serialization() {
            let result = DiffResult::Added(
                "test_key".to_string(),
                Value::String("test_value".to_string()),
            );
            let serialized = serde_json::to_string(&result).unwrap();
            assert!(serialized.contains("Added"));
            assert!(serialized.contains("test_key"));
            assert!(serialized.contains("test_value"));
        }

        #[test]
        fn test_tensor_stats_creation() {
            let stats = TensorStats {
                mean: 0.0,
                std: 0.5,
                min: -1.0,
                max: 1.0,
                shape: vec![10, 20, 30],
                dtype: "float32".to_string(),
                total_params: 6000, // 10 * 20 * 30
            };

            assert_eq!(stats.shape, vec![10, 20, 30]);
            assert_eq!(stats.dtype, "float32");
            assert_eq!(stats.mean, 0.0);
        }

        #[test]
        fn test_tensor_shape_changed() {
            let old_shape = vec![100, 200];
            let new_shape = vec![150, 200];
            let result = DiffResult::TensorShapeChanged(
                "layer1.weight".to_string(),
                old_shape.clone(),
                new_shape.clone(),
            );

            match result {
                DiffResult::TensorShapeChanged(name, old, new) => {
                    assert_eq!(name, "layer1.weight");
                    assert_eq!(old, old_shape);
                    assert_eq!(new, new_shape);
                }
                _ => panic!("Expected TensorShapeChanged"),
            }
        }
    }

    mod ml_analysis_tests {
        use super::*;

        #[test]
        fn test_model_info_creation() {
            let model_info = ModelInfo {
                total_params: 1000000,
                trainable_params: 900000,
                model_size_mb: 50.0,
                architecture_hash: "abc123".to_string(),
                layer_count: 12,
                layer_types: vec!["Linear".to_string(), "ReLU".to_string()],
            };

            assert_eq!(model_info.total_params, 1000000);
            assert_eq!(model_info.trainable_params, 900000);
            assert!(model_info.model_size_mb > 0.0);
        }

        #[test]
        fn test_learning_progress_info() {
            let progress = LearningProgressInfo {
                loss_trend: "improving".to_string(),
                parameter_update_magnitude: 0.5,
                gradient_norm_ratio: 1.05,
                convergence_speed: 0.9,
                training_efficiency: 0.85,
                learning_rate_schedule: "constant".to_string(),
                momentum_coefficient: 0.9,
                weight_decay_effect: 0.01,
                batch_size_impact: 32,
                optimization_algorithm: "adam".to_string(),
            };

            assert_eq!(progress.loss_trend, "improving"); // Loss should decrease
            assert!(progress.parameter_update_magnitude > 0.0); // Parameters should update
            assert!(progress.training_efficiency > 0.0);
        }

        #[test]
        fn test_convergence_analysis() {
            let convergence = ConvergenceInfo {
                convergence_status: "converging".to_string(),
                parameter_stability: 0.95,
                loss_volatility: 0.05,
                gradient_consistency: 0.9,
                plateau_detection: false,
                overfitting_risk: "low".to_string(),
                early_stopping_recommendation: "continue".to_string(),
                convergence_speed_estimate: 0.8,
                remaining_iterations: 100,
                confidence_interval: (0.85, 0.95),
            };

            assert_eq!(convergence.convergence_status, "converging");
            assert!(convergence.parameter_stability > 0.9);
            assert!(!convergence.plateau_detection);
        }

        #[test]
        fn test_anomaly_detection() {
            let anomaly = AnomalyInfo {
                anomaly_type: "gradient_explosion".to_string(),
                severity: "high".to_string(),
                affected_layers: vec!["layer1".to_string(), "layer2".to_string()],
                detection_confidence: 0.95,
                anomaly_magnitude: 5.0,
                temporal_pattern: "sudden".to_string(),
                root_cause_analysis: "Learning rate too high".to_string(),
                recommended_action: "Reduce learning rate".to_string(),
                recovery_probability: 0.8,
                prevention_suggestions: vec!["Use gradient clipping".to_string()],
            };

            assert_eq!(anomaly.anomaly_type, "gradient_explosion");
            assert_eq!(anomaly.severity, "high");
            assert!(anomaly.detection_confidence > 0.9);
        }

        #[test]
        fn test_memory_analysis() {
            let memory = MemoryAnalysisInfo {
                memory_delta_bytes: 512 * 1024 * 1024, // 512MB
                peak_memory_usage: 2560 * 1024 * 1024, // 2560MB
                memory_efficiency_ratio: 0.8,
                gpu_memory_utilization: 0.75,
                memory_fragmentation_level: 0.1,
                cache_efficiency: 0.9,
                memory_leak_indicators: vec![],
                optimization_opportunities: vec![],
                estimated_gpu_memory_mb: 2048.0,
                memory_recommendation: "Memory usage is within acceptable limits".to_string(),
            };

            assert!(memory.memory_delta_bytes > 0);
            assert!(memory.peak_memory_usage > 0);
            assert!(memory.memory_efficiency_ratio > 0.0);
        }
    }

    mod data_parsing_tests {
        use super::*;

        #[test]
        fn test_csv_parsing() {
            let csv_data = "name,age,score\nAlice,25,95.5\nBob,30,87.2";
            let result = parse_csv(csv_data);

            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert!(parsed.is_array());

            let array = parsed.as_array().unwrap();
            assert_eq!(array.len(), 2);
        }

        #[test]
        fn test_xml_parsing() {
            let xml_data =
                r#"<root><item id="1"><name>Test</name><value>123</value></item></root>"#;
            let result = parse_xml(xml_data);

            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert!(parsed.is_object());
        }

        #[test]
        fn test_diff_values_basic() {
            let old_value = serde_json::json!({"name": "Alice", "age": 25});
            let new_value = serde_json::json!({"name": "Alice", "age": 26});

            let results = diff_basic(&old_value, &new_value);
            assert_eq!(results.len(), 1);

            match &results[0] {
                DiffResult::Modified(path, old, new) => {
                    assert_eq!(path, "age");
                    assert_eq!(old, &Value::Number(25.into()));
                    assert_eq!(new, &Value::Number(26.into()));
                }
                _ => panic!("Expected Modified result"),
            }
        }

        #[test]
        fn test_diff_values_added() {
            let old_value = serde_json::json!({"name": "Alice"});
            let new_value = serde_json::json!({"name": "Alice", "age": 25});

            let results = diff_basic(&old_value, &new_value);
            assert_eq!(results.len(), 1);

            match &results[0] {
                DiffResult::Added(path, value) => {
                    assert_eq!(path, "age");
                    assert_eq!(value, &Value::Number(25.into()));
                }
                _ => panic!("Expected Added result"),
            }
        }

        #[test]
        fn test_diff_values_removed() {
            let old_value = serde_json::json!({"name": "Alice", "age": 25});
            let new_value = serde_json::json!({"name": "Alice"});

            let results = diff_basic(&old_value, &new_value);
            assert_eq!(results.len(), 1);

            match &results[0] {
                DiffResult::Removed(path, value) => {
                    assert_eq!(path, "age");
                    assert_eq!(value, &Value::Number(25.into()));
                }
                _ => panic!("Expected Removed result"),
            }
        }
    }

    mod scientific_data_tests {
        use super::*;

        #[test]
        fn test_numeric_precision() {
            let old_value = serde_json::json!(1.0);
            let new_value = serde_json::json!(2.0);

            let results = diff_basic(&old_value, &new_value);
            assert_eq!(results.len(), 1);

            // Test that small differences are still detected
            let old_small = serde_json::json!(0.001);
            let new_small = serde_json::json!(0.002);
            let results_small = diff_basic(&old_small, &new_small);
            assert_eq!(results_small.len(), 1);
        }

        #[test]
        fn test_array_comparison() {
            let old_array = serde_json::json!([1, 2, 3]);
            let new_array = serde_json::json!([1, 2, 4]);

            let results = diff_basic(&old_array, &new_array);
            assert_eq!(results.len(), 1);

            match &results[0] {
                DiffResult::Modified(path, old, new) => {
                    assert_eq!(path, "[2]");
                    assert_eq!(old, &Value::Number(3.into()));
                    assert_eq!(new, &Value::Number(4.into()));
                }
                _ => panic!("Expected Modified result for array element"),
            }
        }
    }
}
