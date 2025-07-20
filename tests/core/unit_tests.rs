// Unit tests for diffai core components
// Test individual functions and modules in isolation

#[cfg(test)]
mod tests {
    use diffai_core::{
        DiffResult, TensorStats, ModelInfo, LearningProgressInfo,
        ConvergenceInfo, AnomalyInfo, MemoryAnalysisInfo,
        parse_csv, parse_xml, diff_values, diff_ml_models_enhanced
    };
    use serde_json::Value;
    use std::collections::HashMap;

    mod diff_result_tests {
        use super::*;

        #[test]
        fn test_diff_result_serialization() {
            let result = DiffResult::Added("test_key".to_string(), Value::String("test_value".to_string()));
            let serialized = serde_json::to_string(&result).unwrap();
            assert!(serialized.contains("Added"));
            assert!(serialized.contains("test_key"));
            assert!(serialized.contains("test_value"));
        }

        #[test]
        fn test_tensor_stats_creation() {
            let stats = TensorStats {
                shape: vec![10, 20, 30],
                dtype: "float32".to_string(),
                min_val: -1.0,
                max_val: 1.0,
                mean: 0.0,
                std: 0.5,
                norm: 10.0,
                sparsity: 0.1,
                zero_count: 100,
                inf_count: 0,
                nan_count: 0,
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
                new_shape.clone()
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
                loss_change: -0.5,
                accuracy_change: 0.1,
                convergence_rate: 0.05,
                training_stability: 0.9,
                overfitting_risk: 0.2,
            };

            assert!(progress.loss_change < 0.0); // Loss should decrease
            assert!(progress.accuracy_change > 0.0); // Accuracy should increase
            assert!(progress.training_stability > 0.0);
        }

        #[test]
        fn test_convergence_analysis() {
            let convergence = ConvergenceInfo {
                is_converged: true,
                convergence_epoch: Some(50),
                plateau_detection: false,
                gradient_norm_trend: "decreasing".to_string(),
                loss_plateau_duration: 0,
            };

            assert!(convergence.is_converged);
            assert_eq!(convergence.convergence_epoch, Some(50));
            assert!(!convergence.plateau_detection);
        }

        #[test]
        fn test_anomaly_detection() {
            let anomaly = AnomalyInfo {
                anomaly_detected: true,
                anomaly_type: "gradient_explosion".to_string(),
                severity: "high".to_string(),
                affected_layers: vec!["layer1".to_string(), "layer2".to_string()],
                confidence: 0.95,
            };

            assert!(anomaly.anomaly_detected);
            assert_eq!(anomaly.anomaly_type, "gradient_explosion");
            assert_eq!(anomaly.severity, "high");
            assert!(anomaly.confidence > 0.9);
        }

        #[test]
        fn test_memory_analysis() {
            let memory = MemoryAnalysisInfo {
                memory_usage_mb: 2048.0,
                memory_change_mb: 512.0,
                peak_memory_mb: 2560.0,
                memory_efficiency: 0.8,
                memory_fragmentation: 0.1,
            };

            assert!(memory.memory_usage_mb > 0.0);
            assert!(memory.memory_change_mb > 0.0);
            assert!(memory.peak_memory_mb >= memory.memory_usage_mb);
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
            let xml_data = r#"<root><item id="1"><name>Test</name><value>123</value></item></root>"#;
            let result = parse_xml(xml_data);
            
            assert!(result.is_ok());
            let parsed = result.unwrap();
            assert!(parsed.is_object());
        }

        #[test]
        fn test_diff_values_basic() {
            let old_value = serde_json::json!({"name": "Alice", "age": 25});
            let new_value = serde_json::json!({"name": "Alice", "age": 26});
            
            let results = diff_values("", &old_value, &new_value, &None, 0.0);
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
            
            let results = diff_values("", &old_value, &new_value, &None, 0.0);
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
            
            let results = diff_values("", &old_value, &new_value, &None, 0.0);
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
        fn test_epsilon_tolerance() {
            let old_value = serde_json::json!(1.0001);
            let new_value = serde_json::json!(1.0002);
            
            // With strict tolerance
            let results_strict = diff_values("", &old_value, &new_value, &None, 0.00001);
            assert_eq!(results_strict.len(), 1);
            
            // With loose tolerance
            let results_loose = diff_values("", &old_value, &new_value, &None, 0.001);
            assert_eq!(results_loose.len(), 0);
        }

        #[test]
        fn test_array_comparison() {
            let old_array = serde_json::json!([1, 2, 3]);
            let new_array = serde_json::json!([1, 2, 4]);
            
            let results = diff_values("", &old_array, &new_array, &None, 0.0);
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