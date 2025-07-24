use serde_json::{json, Value};
use std::path::Path;

/// Common test fixtures shared across diffai core/python/js tests
/// These fixtures are compatible with CLI fixtures but focused on AI/ML unified API testing

pub struct TestFixtures;

impl TestFixtures {
    /// Get path to shared CLI fixtures directory
    pub fn cli_fixtures_dir() -> &'static str {
        "../../tests/fixtures"
    }
    
    /// Load JSON file from CLI fixtures
    pub fn load_cli_fixture(filename: &str) -> Value {
        let path = format!("{}/{}", Self::cli_fixtures_dir(), filename);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Failed to read fixture: {}", path));
        
        if filename.ends_with(".json") {
            serde_json::from_str(&content).unwrap()
        } else {
            panic!("Only JSON fixtures supported in unified API tests")
        }
    }
    
    /// Basic configuration comparison fixtures (shared with diffx)
    pub fn config_v1() -> Value {
        Self::load_cli_fixture("config_v1.json")
    }
    
    pub fn config_v2() -> Value {
        Self::load_cli_fixture("config_v2.json")
    }
    
    pub fn config_v3() -> Value {
        Self::load_cli_fixture("config_v3.json")
    }
    
    /// AI/ML specific test fixtures
    
    /// PyTorch model metadata fixtures
    pub fn pytorch_model_old() -> Value {
        json!({
            "model_type": "pytorch",
            "model_info": {
                "architecture": "ResNet",
                "layers": [
                    {
                        "name": "conv1",
                        "type": "Conv2d",
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size": [7, 7],
                        "weights": {
                            "shape": [64, 3, 7, 7],
                            "mean": 0.01,
                            "std": 0.1
                        }
                    },
                    {
                        "name": "fc",
                        "type": "Linear",
                        "in_features": 512,
                        "out_features": 1000,
                        "weights": {
                            "shape": [1000, 512],
                            "mean": 0.0,
                            "std": 0.05
                        }
                    }
                ],
                "optimizer": {
                    "type": "Adam",
                    "learning_rate": 0.001,
                    "beta1": 0.9,
                    "beta2": 0.999
                },
                "loss_function": "CrossEntropyLoss",
                "training": {
                    "epoch": 10,
                    "loss": 0.25,
                    "accuracy": 0.92
                }
            }
        })
    }
    
    pub fn pytorch_model_new() -> Value {
        json!({
            "model_type": "pytorch",
            "model_info": {
                "architecture": "ResNet",
                "layers": [
                    {
                        "name": "conv1",
                        "type": "Conv2d",
                        "in_channels": 3,
                        "out_channels": 64,
                        "kernel_size": [7, 7],
                        "weights": {
                            "shape": [64, 3, 7, 7],
                            "mean": 0.015,  // Changed
                            "std": 0.12     // Changed
                        }
                    },
                    {
                        "name": "fc",
                        "type": "Linear",
                        "in_features": 512,
                        "out_features": 1000,
                        "weights": {
                            "shape": [1000, 512],
                            "mean": 0.002,  // Changed
                            "std": 0.048    // Changed
                        }
                    }
                ],
                "optimizer": {
                    "type": "SGD",        // Changed from Adam
                    "learning_rate": 0.01, // Changed
                    "momentum": 0.9       // Added
                },
                "loss_function": "CrossEntropyLoss",
                "training": {
                    "epoch": 15,          // Changed
                    "loss": 0.18,         // Improved
                    "accuracy": 0.95      // Improved
                }
            }
        })
    }
    
    /// SafeTensors model fixtures
    pub fn safetensors_model_old() -> Value {
        json!({
            "model_type": "safetensors",
            "tensors": {
                "embedding.weight": {
                    "shape": [50000, 768],
                    "dtype": "float32"
                },
                "encoder.layer.0.attention.self.query.weight": {
                    "shape": [768, 768],
                    "dtype": "float32"
                },
                "encoder.layer.0.attention.self.key.weight": {
                    "shape": [768, 768],
                    "dtype": "float32"
                },
                "classifier.weight": {
                    "shape": [2, 768],
                    "dtype": "float32"
                }
            },
            "metadata": {
                "model_name": "bert-base",
                "version": "1.0",
                "total_params": 110000000
            }
        })
    }
    
    pub fn safetensors_model_new() -> Value {
        json!({
            "model_type": "safetensors",
            "tensors": {
                "embedding.weight": {
                    "shape": [50000, 1024], // Changed dimension
                    "dtype": "float32"
                },
                "encoder.layer.0.attention.self.query.weight": {
                    "shape": [1024, 1024], // Changed dimension
                    "dtype": "float16"     // Changed precision
                },
                "encoder.layer.0.attention.self.key.weight": {
                    "shape": [1024, 1024], // Changed dimension
                    "dtype": "float16"     // Changed precision
                },
                "classifier.weight": {
                    "shape": [2, 1024],    // Changed dimension
                    "dtype": "float32"
                },
                "new_layer.weight": {      // Added new tensor
                    "shape": [1024, 512],
                    "dtype": "float32"
                }
            },
            "metadata": {
                "model_name": "bert-large", // Changed model
                "version": "2.0",           // Changed version
                "total_params": 340000000   // Changed param count
            }
        })
    }
    
    /// NumPy array fixtures
    pub fn numpy_array_old() -> Value {
        json!({
            "model_type": "numpy",
            "arrays": {
                "data": {
                    "shape": [1000, 784],
                    "dtype": "float64",
                    "statistics": {
                        "mean": 0.5,
                        "std": 0.2,
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "labels": {
                    "shape": [1000],
                    "dtype": "int32",
                    "unique_values": 10
                }
            }
        })
    }
    
    pub fn numpy_array_new() -> Value {
        json!({
            "model_type": "numpy",
            "arrays": {
                "data": {
                    "shape": [1200, 784],    // Changed size
                    "dtype": "float32",      // Changed precision
                    "statistics": {
                        "mean": 0.48,         // Changed
                        "std": 0.22,          // Changed
                        "min": 0.0,
                        "max": 1.0
                    }
                },
                "labels": {
                    "shape": [1200],         // Changed size
                    "dtype": "int32",
                    "unique_values": 12      // Changed
                },
                "weights": {                 // Added new array
                    "shape": [784, 10],
                    "dtype": "float32",
                    "statistics": {
                        "mean": 0.01,
                        "std": 0.05,
                        "min": -0.1,
                        "max": 0.1
                    }
                }
            }
        })
    }
    
    /// MATLAB file fixtures
    pub fn matlab_file_old() -> Value {
        json!({
            "model_type": "matlab",
            "variables": {
                "network": {
                    "type": "feedforward",
                    "layers": [784, 128, 64, 10],
                    "activation": "relu",
                    "output_activation": "softmax"
                },
                "weights": {
                    "W1": {"shape": [128, 784], "type": "double"},
                    "b1": {"shape": [128, 1], "type": "double"},
                    "W2": {"shape": [64, 128], "type": "double"},
                    "b2": {"shape": [64, 1], "type": "double"},
                    "W3": {"shape": [10, 64], "type": "double"},
                    "b3": {"shape": [10, 1], "type": "double"}
                },
                "training_params": {
                    "epochs": 100,
                    "learning_rate": 0.01,
                    "batch_size": 32
                }
            }
        })
    }
    
    pub fn matlab_file_new() -> Value {
        json!({
            "model_type": "matlab",
            "variables": {
                "network": {
                    "type": "convolutional",  // Changed architecture
                    "layers": [784, 256, 128, 10], // Changed layer sizes
                    "activation": "tanh",     // Changed activation
                    "output_activation": "softmax"
                },
                "weights": {
                    "W1": {"shape": [256, 784], "type": "single"}, // Changed precision
                    "b1": {"shape": [256, 1], "type": "single"},
                    "W2": {"shape": [128, 256], "type": "single"}, // Changed dimensions
                    "b2": {"shape": [128, 1], "type": "single"},
                    "W3": {"shape": [10, 128], "type": "single"}, // Changed dimensions
                    "b3": {"shape": [10, 1], "type": "single"}
                },
                "training_params": {
                    "epochs": 150,            // Changed
                    "learning_rate": 0.001,   // Changed
                    "batch_size": 64,         // Changed
                    "regularization": 0.01    // Added
                }
            }
        })
    }
    
    /// Training metrics comparison fixtures
    pub fn training_metrics_old() -> Value {
        json!({
            "experiment": {
                "name": "baseline_experiment",
                "model": "resnet50",
                "dataset": "imagenet",
                "metrics": {
                    "training": {
                        "loss": [2.5, 1.8, 1.2, 0.9, 0.7],
                        "accuracy": [0.2, 0.4, 0.6, 0.75, 0.82]
                    },
                    "validation": {
                        "loss": [2.8, 2.0, 1.5, 1.1, 0.9],
                        "accuracy": [0.18, 0.35, 0.55, 0.70, 0.78]
                    },
                    "hyperparameters": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "optimizer": "Adam",
                        "weight_decay": 0.0001
                    }
                }
            }
        })
    }
    
    pub fn training_metrics_new() -> Value {
        json!({
            "experiment": {
                "name": "improved_experiment",
                "model": "resnet50",
                "dataset": "imagenet",
                "metrics": {
                    "training": {
                        "loss": [2.3, 1.5, 1.0, 0.7, 0.5],    // Improved
                        "accuracy": [0.25, 0.45, 0.65, 0.80, 0.88] // Improved
                    },
                    "validation": {
                        "loss": [2.5, 1.7, 1.2, 0.9, 0.7],    // Improved
                        "accuracy": [0.22, 0.40, 0.60, 0.75, 0.83] // Improved
                    },
                    "hyperparameters": {
                        "learning_rate": 0.01,     // Changed
                        "batch_size": 64,          // Changed
                        "optimizer": "SGD",        // Changed
                        "weight_decay": 0.0005,    // Changed
                        "momentum": 0.9            // Added
                    }
                }
            }
        })
    }
    
    /// Model architecture comparison fixtures
    pub fn model_architecture_old() -> Value {
        json!({
            "model": {
                "name": "custom_cnn",
                "type": "sequential",
                "layers": [
                    {
                        "name": "input",
                        "type": "Input",
                        "shape": [224, 224, 3]
                    },
                    {
                        "name": "conv1",
                        "type": "Conv2D",
                        "filters": 32,
                        "kernel_size": [3, 3],
                        "activation": "relu"
                    },
                    {
                        "name": "pool1",
                        "type": "MaxPooling2D",
                        "pool_size": [2, 2]
                    },
                    {
                        "name": "flatten",
                        "type": "Flatten"
                    },
                    {
                        "name": "dense1",
                        "type": "Dense",
                        "units": 128,
                        "activation": "relu"
                    },
                    {
                        "name": "output",
                        "type": "Dense",
                        "units": 10,
                        "activation": "softmax"
                    }
                ]
            }
        })
    }
    
    pub fn model_architecture_new() -> Value {
        json!({
            "model": {
                "name": "improved_cnn",  // Changed name
                "type": "functional",    // Changed type
                "layers": [
                    {
                        "name": "input",
                        "type": "Input",
                        "shape": [224, 224, 3]
                    },
                    {
                        "name": "conv1",
                        "type": "Conv2D",
                        "filters": 64,       // Increased filters
                        "kernel_size": [3, 3],
                        "activation": "relu"
                    },
                    {
                        "name": "conv2",     // Added new layer
                        "type": "Conv2D",
                        "filters": 64,
                        "kernel_size": [3, 3],
                        "activation": "relu"
                    },
                    {
                        "name": "pool1",
                        "type": "MaxPooling2D",
                        "pool_size": [2, 2]
                    },
                    {
                        "name": "dropout1", // Added dropout
                        "type": "Dropout",
                        "rate": 0.25
                    },
                    {
                        "name": "flatten",
                        "type": "Flatten"
                    },
                    {
                        "name": "dense1",
                        "type": "Dense",
                        "units": 256,       // Increased units
                        "activation": "relu"
                    },
                    {
                        "name": "dropout2", // Added dropout
                        "type": "Dropout",
                        "rate": 0.5
                    },
                    {
                        "name": "output",
                        "type": "Dense",
                        "units": 10,
                        "activation": "softmax"
                    }
                ]
            }
        })
    }
}

/// Helper functions for AI/ML test data generation
pub mod ml_generators {
    use super::*;
    
    pub fn generate_tensor_data(shape: Vec<usize>, dtype: &str) -> Value {
        json!({
            "shape": shape,
            "dtype": dtype,
            "size": shape.iter().product::<usize>()
        })
    }
    
    pub fn generate_model_weights(layer_sizes: Vec<usize>) -> Value {
        let mut weights = serde_json::Map::new();
        
        for i in 0..layer_sizes.len() - 1 {
            let weight_name = format!("layer_{}_weight", i);
            let bias_name = format!("layer_{}_bias", i);
            
            weights.insert(weight_name, json!({
                "shape": [layer_sizes[i + 1], layer_sizes[i]],
                "dtype": "float32"
            }));
            
            weights.insert(bias_name, json!({
                "shape": [layer_sizes[i + 1]],
                "dtype": "float32"
            }));
        }
        
        json!(weights)
    }
    
    pub fn generate_training_history(epochs: usize) -> Value {
        let mut history = serde_json::Map::new();
        
        // Generate mock training loss (decreasing)
        let train_loss: Vec<Value> = (0..epochs)
            .map(|i| json!(2.0 * (-0.1 * i as f64).exp()))
            .collect();
        
        // Generate mock validation loss (decreasing but higher)
        let val_loss: Vec<Value> = (0..epochs)
            .map(|i| json!(2.2 * (-0.08 * i as f64).exp()))
            .collect();
        
        // Generate mock accuracies (increasing)
        let train_acc: Vec<Value> = (0..epochs)
            .map(|i| json!(1.0 - (-0.15 * i as f64).exp()))
            .collect();
        
        let val_acc: Vec<Value> = (0..epochs)
            .map(|i| json!(1.0 - 1.1 * (-0.12 * i as f64).exp()))
            .collect();
        
        history.insert("loss".to_string(), json!(train_loss));
        history.insert("val_loss".to_string(), json!(val_loss));
        history.insert("accuracy".to_string(), json!(train_acc));
        history.insert("val_accuracy".to_string(), json!(val_acc));
        
        json!(history)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pytorch_fixtures() {
        let old = TestFixtures::pytorch_model_old();
        let new = TestFixtures::pytorch_model_new();
        
        assert_eq!(old["model_type"], json!("pytorch"));
        assert_eq!(new["model_type"], json!("pytorch"));
        assert_ne!(old["model_info"]["optimizer"], new["model_info"]["optimizer"]);
    }
    
    #[test]
    fn test_safetensors_fixtures() {
        let old = TestFixtures::safetensors_model_old();
        let new = TestFixtures::safetensors_model_new();
        
        assert_eq!(old["model_type"], json!("safetensors"));
        assert_eq!(new["model_type"], json!("safetensors"));
        
        // Should have different tensor shapes
        let old_embedding = &old["tensors"]["embedding.weight"]["shape"];
        let new_embedding = &new["tensors"]["embedding.weight"]["shape"];
        assert_ne!(old_embedding, new_embedding);
    }
    
    #[test]
    fn test_ml_generators() {
        let tensor = ml_generators::generate_tensor_data(vec![100, 200], "float32");
        assert_eq!(tensor["shape"], json!([100, 200]));
        assert_eq!(tensor["size"], json!(20000));
        
        let weights = ml_generators::generate_model_weights(vec![784, 128, 10]);
        assert!(weights["layer_0_weight"].is_object());
        assert!(weights["layer_1_weight"].is_object());
    }
    
    #[test]
    fn test_training_history_generator() {
        let history = ml_generators::generate_training_history(5);
        
        if let Value::Array(losses) = &history["loss"] {
            assert_eq!(losses.len(), 5);
            // Loss should generally decrease
            let first = losses[0].as_f64().unwrap();
            let last = losses[4].as_f64().unwrap();
            assert!(first > last);
        } else {
            panic!("Expected loss to be an array");
        }
    }
}