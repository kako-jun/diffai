import pytest
import json
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import diffai_python
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # Import the locally built Rust binding directly 
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "target" / "debug"))
    import diffai_python
except ImportError:
    pytest.skip("diffai_python Rust module not built", allow_module_level=True)

# ============================================================================
# TEST FIXTURES - Shared with Core Tests (AI/ML Focus)
# ============================================================================

class TestFixtures:
    """
    Python equivalent of Rust fixtures for diffai unified API testing.
    Focuses on AI/ML specific test data.
    """
    
    @staticmethod
    def load_cli_fixture(filename):
        """Load JSON file from CLI fixtures directory"""
        fixtures_dir = Path(__file__).parent.parent.parent / "tests" / "fixtures"
        fixture_path = fixtures_dir / filename
        
        if not fixture_path.exists():
            pytest.skip(f"CLI fixture not found: {fixture_path}")
        
        with open(fixture_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def config_v1():
        return TestFixtures.load_cli_fixture("config_v1.json")
    
    @staticmethod
    def config_v2():
        return TestFixtures.load_cli_fixture("config_v2.json")
    
    # AI/ML specific fixtures
    
    @staticmethod
    def pytorch_model_old():
        return {
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
        }
    
    @staticmethod
    def pytorch_model_new():
        return {
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
                            "mean": 0.015,  # Changed
                            "std": 0.12     # Changed
                        }
                    },
                    {
                        "name": "fc",
                        "type": "Linear",
                        "in_features": 512,
                        "out_features": 1000,
                        "weights": {
                            "shape": [1000, 512],
                            "mean": 0.002,  # Changed
                            "std": 0.048    # Changed
                        }
                    }
                ],
                "optimizer": {
                    "type": "SGD",        # Changed from Adam
                    "learning_rate": 0.01, # Changed
                    "momentum": 0.9       # Added
                },
                "loss_function": "CrossEntropyLoss",
                "training": {
                    "epoch": 15,          # Changed
                    "loss": 0.18,         # Improved
                    "accuracy": 0.95      # Improved
                }
            }
        }
    
    @staticmethod
    def safetensors_model_old():
        return {
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
        }
    
    @staticmethod
    def safetensors_model_new():
        return {
            "model_type": "safetensors",
            "tensors": {
                "embedding.weight": {
                    "shape": [50000, 1024], # Changed dimension
                    "dtype": "float32"
                },
                "encoder.layer.0.attention.self.query.weight": {
                    "shape": [1024, 1024], # Changed dimension
                    "dtype": "float16"     # Changed precision
                },
                "classifier.weight": {
                    "shape": [2, 1024],    # Changed dimension
                    "dtype": "float32"
                },
                "new_layer.weight": {      # Added new tensor
                    "shape": [1024, 512],
                    "dtype": "float32"
                }
            },
            "metadata": {
                "model_name": "bert-large", # Changed model
                "version": "2.0",           # Changed version
                "total_params": 340000000   # Changed param count
            }
        }
    
    @staticmethod
    def training_metrics_old():
        return {
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
        }
    
    @staticmethod
    def training_metrics_new():
        return {
            "experiment": {
                "name": "improved_experiment",
                "model": "resnet50",
                "dataset": "imagenet",
                "metrics": {
                    "training": {
                        "loss": [2.3, 1.5, 1.0, 0.7, 0.5],    # Improved
                        "accuracy": [0.25, 0.45, 0.65, 0.80, 0.88] # Improved
                    },
                    "validation": {
                        "loss": [2.5, 1.7, 1.2, 0.9, 0.7],    # Improved
                        "accuracy": [0.22, 0.40, 0.60, 0.75, 0.83] # Improved
                    },
                    "hyperparameters": {
                        "learning_rate": 0.01,     # Changed
                        "batch_size": 64,          # Changed
                        "optimizer": "SGD",        # Changed
                        "weight_decay": 0.0005,    # Changed
                        "momentum": 0.9            # Added
                    }
                }
            }
        }

# ============================================================================
# UNIFIED API TESTS - Core Functionality
# ============================================================================

class TestUnifiedAPI:
    """Test the unified diff() function with Python bindings for AI/ML"""
    
    def test_diff_basic_modification(self):
        old = {"name": "Alice", "age": 30}
        new = {"name": "Alice", "age": 31}
        
        results = diffai_python.diff(old, new)
        
        assert len(results) == 1
        result = results[0]
        assert result["type"] == "Modified"
        assert result["path"] == "age"
        assert result["old_value"] == 30
        assert result["new_value"] == 31
    
    def test_diff_ai_ml_specific_results(self):
        old = {"learning_rate": 0.001, "accuracy": 0.85}
        new = {"learning_rate": 0.01, "accuracy": 0.92}
        
        results = diffai_python.diff(
            old, new,
            learning_rate_tracking=True,
            accuracy_tracking=True
        )
        
        assert len(results) == 2
        
        # Check for learning rate change
        lr_result = next((r for r in results if r["type"] == "LearningRateChanged"), None)
        assert lr_result is not None
        assert lr_result["old_learning_rate"] == 0.001
        assert lr_result["new_learning_rate"] == 0.01
        
        # Check for accuracy change
        acc_result = next((r for r in results if r["type"] == "AccuracyChange"), None)
        assert acc_result is not None
        assert acc_result["old_accuracy"] == 0.85
        assert acc_result["new_accuracy"] == 0.92
    
    def test_diff_weight_threshold(self):
        old = {"weights": {"layer1": 0.1, "layer2": 0.05}}
        new = {"weights": {"layer1": 0.2, "layer2": 0.051}}
        
        results = diffai_python.diff(
            old, new,
            weight_threshold=0.05  # Only changes > 0.05 are significant
        )
        
        # Should only detect layer1 as significant change (0.1 difference > 0.05 threshold)
        significant_changes = [r for r in results if r["type"] == "WeightSignificantChange"]
        assert len(significant_changes) == 1
        
        change = significant_changes[0]
        assert "layer1" in change["path"]
        assert change["magnitude"] == 0.1

# ============================================================================
# AI/ML SPECIFIC TESTS - PyTorch Models
# ============================================================================

class TestPyTorchModels:
    """Test PyTorch model comparison functionality"""
    
    def test_pytorch_model_comparison(self):
        old = TestFixtures.pytorch_model_old()
        new = TestFixtures.pytorch_model_new()
        
        results = diffai_python.diff(
            old, new,
            ml_analysis_enabled=True,
            learning_rate_tracking=True,
            optimizer_comparison=True,
            loss_tracking=True,
            accuracy_tracking=True
        )
        
        assert len(results) > 0
        
        # Should detect optimizer change (Adam -> SGD)
        optimizer_changes = [r for r in results 
                           if r["type"] == "Modified" and "optimizer.type" in r["path"]]
        assert len(optimizer_changes) > 0
        
        # Should detect learning rate change
        lr_changes = [r for r in results if r["type"] == "LearningRateChanged"]
        assert len(lr_changes) > 0
    
    def test_pytorch_layer_weight_changes(self):
        old = {
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
        }
        
        new = {
            "layers": {
                "conv1": {
                    "weights": {
                        "mean": 0.015,  # Small change
                        "std": 0.12     # Small change
                    }
                },
                "fc": {
                    "weights": {
                        "mean": 0.1,    # Large change
                        "std": 0.15     # Large change
                    }
                }
            }
        }
        
        results = diffai_python.diff(
            old, new,
            weight_threshold=0.05,
            epsilon=0.001
        )
        
        # Should detect significant changes in fc layer
        significant_changes = [r for r in results if r["type"] == "WeightSignificantChange"]
        assert len(significant_changes) > 0

# ============================================================================
# AI/ML SPECIFIC TESTS - SafeTensors Models
# ============================================================================

class TestSafeTensorsModels:
    """Test SafeTensors model comparison functionality"""
    
    def test_safetensors_model_comparison(self):
        old = TestFixtures.safetensors_model_old()
        new = TestFixtures.safetensors_model_new()
        
        results = diffai_python.diff(
            old, new,
            tensor_comparison_mode="both"
        )
        
        assert len(results) > 0
        
        # Should detect tensor shape changes
        shape_changes = [r for r in results 
                        if r["type"] == "Modified" and "shape" in r["path"]]
        assert len(shape_changes) > 0
        
        # Should detect dtype changes  
        dtype_changes = [r for r in results 
                        if r["type"] == "Modified" and "dtype" in r["path"]]
        assert len(dtype_changes) > 0
        
        # Should detect new tensors
        added_tensors = [r for r in results 
                        if r["type"] == "Added" and "new_layer" in r["path"]]
        assert len(added_tensors) > 0
    
    def test_safetensors_metadata_comparison(self):
        old = TestFixtures.safetensors_model_old()
        new = TestFixtures.safetensors_model_new()
        
        results = diffai_python.diff(old, new)
        
        # Should detect metadata changes
        metadata_changes = [r for r in results 
                          if r["type"] == "Modified" and "metadata" in r["path"]]
        assert len(metadata_changes) > 0
        
        # Check for specific version change
        version_change = next((r for r in results 
                             if r["type"] == "Modified" and "version" in r["path"] 
                             and r["old_value"] == "1.0" and r["new_value"] == "2.0"), None)
        assert version_change is not None

# ============================================================================
# TRAINING METRICS COMPARISON TESTS
# ============================================================================

class TestTrainingMetrics:
    """Test training metrics comparison functionality"""
    
    def test_training_metrics_comparison(self):
        old = TestFixtures.training_metrics_old()
        new = TestFixtures.training_metrics_new()
        
        results = diffai_python.diff(
            old, new,
            loss_tracking=True,
            accuracy_tracking=True,
            optimizer_comparison=True,
            learning_rate_tracking=True
        )
        
        assert len(results) > 0
        
        # Should detect optimizer change
        optimizer_changes = [r for r in results 
                           if r["type"] == "Modified" and "optimizer" in r["path"]
                           and r["old_value"] == "Adam" and r["new_value"] == "SGD"]
        assert len(optimizer_changes) > 0
        
        # Should detect learning rate change
        lr_changes = [r for r in results if r["type"] == "LearningRateChanged"
                     and r["old_learning_rate"] == 0.001 and r["new_learning_rate"] == 0.01]
        assert len(lr_changes) > 0
    
    def test_training_history_arrays(self):
        old = {
            "training": {
                "loss": [2.5, 1.8, 1.2, 0.9, 0.7],
                "accuracy": [0.2, 0.4, 0.6, 0.75, 0.82]
            }
        }
        
        new = {
            "training": {
                "loss": [2.3, 1.5, 1.0, 0.7, 0.5],
                "accuracy": [0.25, 0.45, 0.65, 0.80, 0.88]
            }
        }
        
        results = diffai_python.diff(old, new)
        
        # Should detect changes in loss and accuracy arrays
        assert len(results) > 0
        
        loss_changes = [r for r in results 
                       if r["type"] == "Modified" and "loss[" in r["path"]]
        assert len(loss_changes) > 0
        
        accuracy_changes = [r for r in results 
                          if r["type"] == "Modified" and "accuracy[" in r["path"]]
        assert len(accuracy_changes) > 0

# ============================================================================
# PYTHON TYPE CONVERSION TESTS (AI/ML Focus)
# ============================================================================

class TestPythonTypeConversion:
    """Test Python <-> Rust type conversion for AI/ML data types"""
    
    def test_numpy_like_arrays(self):
        """Test handling of numpy-like array data"""
        old = {
            "tensor": {
                "data": [[1.0, 2.0], [3.0, 4.0]],
                "shape": [2, 2],
                "dtype": "float32"
            }
        }
        
        new = {
            "tensor": {
                "data": [[1.1, 2.1], [3.1, 4.1]],
                "shape": [2, 2],
                "dtype": "float64"  # Changed precision
            }
        }
        
        results = diffai_python.diff(old, new, epsilon=0.05)
        
        # Should detect dtype change
        dtype_changes = [r for r in results 
                        if r["type"] == "Modified" and "dtype" in r["path"]]
        assert len(dtype_changes) > 0
    
    def test_ml_metadata_handling(self):
        """Test handling of ML-specific metadata"""
        old = {
            "model_metadata": {
                "framework": "pytorch",
                "version": "1.9.0",
                "device": "cuda:0",
                "mixed_precision": False
            }
        }
        
        new = {
            "model_metadata": {
                "framework": "pytorch",
                "version": "2.0.0",      # Changed
                "device": "cuda:1",      # Changed
                "mixed_precision": True  # Changed
            }
        }
        
        results = diffai_python.diff(old, new)
        
        assert len(results) == 3  # Three changes
        
        # Check each change
        version_change = next((r for r in results if "version" in r["path"]), None)
        assert version_change is not None
        assert version_change["old_value"] == "1.9.0"
        assert version_change["new_value"] == "2.0.0"

# ============================================================================
# OPTIONS TESTING - diffai Specific Options
# ============================================================================

class TestDiffaiOptions:
    """Test diffai-specific options functionality"""
    
    def test_tensor_comparison_mode(self):
        old = {
            "tensor": {
                "shape": [100, 200],
                "data": [1.0, 2.0, 3.0]
            }
        }
        
        new = {
            "tensor": {
                "shape": [100, 300], # Shape changed
                "data": [1.1, 2.1, 3.1] # Data changed
            }
        }
        
        # Test shape-only mode
        results = diffai_python.diff(
            old, new,
            tensor_comparison_mode="shape"
        )
        
        # Should primarily focus on shape changes
        shape_changes = [r for r in results 
                        if r["type"] == "Modified" and "shape" in r["path"]]
        assert len(shape_changes) > 0
    
    def test_ml_analysis_enabled(self):
        old = {
            "model": {
                "learning_rate": 0.001,
                "loss": 0.5,
                "accuracy": 0.8,
                "weights": {"layer1": 0.1}
            }
        }
        
        new = {
            "model": {
                "learning_rate": 0.01,
                "loss": 0.3,
                "accuracy": 0.9,
                "weights": {"layer1": 0.2}
            }
        }
        
        # With ML analysis enabled
        ml_results = diffai_python.diff(
            old, new,
            ml_analysis_enabled=True,
            learning_rate_tracking=True,
            loss_tracking=True,
            accuracy_tracking=True,
            weight_threshold=0.05
        )
        
        # Should use ML-specific diff result types
        ml_specific_results = [r for r in ml_results if r["type"] in [
            "LearningRateChanged", "LossChange", "AccuracyChange", "WeightSignificantChange"
        ]]
        assert len(ml_specific_results) > 0
        
        # Without ML analysis
        regular_results = diffai_python.diff(old, new)
        
        # Should use regular diff result types
        regular_result_types = [r for r in regular_results if r["type"] == "Modified"]
        assert len(regular_result_types) > 0
    
    def test_scientific_precision(self):
        old = {
            "measurements": {
                "precision": 1e-10,
                "recall": 0.99999999,
                "f1_score": 0.999999995
            }
        }
        
        new = {
            "measurements": {
                "precision": 1.1e-10,      # Very small change
                "recall": 0.999999991,     # Very small change
                "f1_score": 0.999999996    # Very small change
            }
        }
        
        # With scientific precision
        precise_results = diffai_python.diff(
            old, new,
            scientific_precision=True,
            epsilon=1e-12  # Very small epsilon
        )
        
        # Should detect very small changes
        assert len(precise_results) > 0
        
        # Without scientific precision (larger epsilon)
        regular_results = diffai_python.diff(
            old, new,
            epsilon=1e-8  # Larger epsilon
        )
        
        # Should detect fewer or no changes
        assert len(regular_results) <= len(precise_results)

# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in diffai Python bindings"""
    
    def test_invalid_regex_pattern(self):
        old = {"test": "value"}
        new = {"test": "value2"}
        
        with pytest.raises(Exception):  # Should raise ValueError for invalid regex
            diffai_python.diff(old, new, ignore_keys_regex="[invalid_regex")
    
    def test_invalid_output_format(self):
        old = {"test": "value"}
        new = {"test": "value2"}
        
        with pytest.raises(Exception):  # Should raise ValueError for invalid format
            diffai_python.diff(old, new, output_format="invalid_format")
    
    def test_invalid_tensor_comparison_mode(self):
        old = {"tensor": {"shape": [10]}}
        new = {"tensor": {"shape": [20]}}
        
        # Should handle invalid mode gracefully or raise clear error
        try:
            results = diffai_python.diff(old, new, tensor_comparison_mode="invalid_mode")
            # If it doesn't raise an error, it should still work
            assert isinstance(results, list)
        except Exception as e:
            assert "invalid_mode" in str(e).lower() or "mode" in str(e).lower()

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete AI/ML workflows"""
    
    def test_comprehensive_ml_workflow(self):
        """Test a complete ML model comparison workflow"""
        old_model = TestFixtures.pytorch_model_old()
        new_model = TestFixtures.pytorch_model_new()
        
        results = diffai_python.diff(
            old_model, new_model,
            ml_analysis_enabled=True,
            tensor_comparison_mode="both",
            learning_rate_tracking=True,
            optimizer_comparison=True,
            loss_tracking=True,
            accuracy_tracking=True,
            weight_threshold=0.01,
            activation_analysis=True,
            epsilon=0.001,
            output_format="json"
        )
        
        assert len(results) > 0
        
        # Should detect multiple types of ML changes
        change_types = {r["type"] for r in results}
        
        # Should detect multiple types of changes in a comprehensive ML comparison
        assert len(change_types) >= 2
    
    def test_real_world_model_evolution(self):
        """Simulate real-world model evolution scenario"""
        # Model v1: Simple architecture
        model_v1 = {
            "architecture": {
                "type": "sequential",
                "layers": [
                    {"type": "dense", "units": 128, "activation": "relu"},
                    {"type": "dropout", "rate": 0.2},
                    {"type": "dense", "units": 10, "activation": "softmax"}
                ]
            },
            "training": {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 50,
                "loss": 0.45,
                "accuracy": 0.87
            }
        }
        
        # Model v2: Improved architecture and hyperparameters
        model_v2 = {
            "architecture": {
                "type": "functional",  # Changed
                "layers": [
                    {"type": "dense", "units": 256, "activation": "relu"},  # Increased units
                    {"type": "batch_norm"},  # Added batch normalization
                    {"type": "dropout", "rate": 0.3},  # Increased dropout
                    {"type": "dense", "units": 128, "activation": "relu"},  # Added layer
                    {"type": "dropout", "rate": 0.2},
                    {"type": "dense", "units": 10, "activation": "softmax"}
                ]
            },
            "training": {
                "optimizer": "adamw",  # Changed optimizer
                "learning_rate": 0.0005,  # Reduced learning rate
                "batch_size": 64,  # Increased batch size
                "epochs": 100,  # More epochs
                "loss": 0.32,  # Improved loss
                "accuracy": 0.94  # Improved accuracy
            }
        }
        
        results = diffai_python.diff(
            model_v1, model_v2,
            ml_analysis_enabled=True,
            learning_rate_tracking=True,
            loss_tracking=True,
            accuracy_tracking=True
        )
        
        assert len(results) > 0
        
        # Should detect architectural improvements
        arch_changes = [r for r in results if "architecture" in r["path"]]
        assert len(arch_changes) > 0
        
        # Should detect training improvements
        training_improvements = [r for r in results if r["type"] in 
                               ["LearningRateChanged", "LossChange", "AccuracyChange"]]
        assert len(training_improvements) > 0

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests for AI/ML data comparison"""
    
    @pytest.mark.slow
    def test_large_tensor_comparison(self):
        """Test performance with large tensor-like data"""
        # Simulate large model weights
        old = {
            "layers": {
                f"layer_{i}": {
                    "weights": [[j + 0.001 * i for j in range(100)] for _ in range(100)],
                    "bias": [0.1 * i for _ in range(100)]
                }
                for i in range(50)  # 50 layers
            }
        }
        
        new = {
            "layers": {
                f"layer_{i}": {
                    "weights": [[j + 0.001 * i + 0.01 for j in range(100)] for _ in range(100)],  # Small change
                    "bias": [0.1 * i + 0.001 for _ in range(100)]  # Small change
                }
                for i in range(50)  # 50 layers
            }
        }
        
        import time
        start_time = time.time()
        results = diffai_python.diff(old, new, weight_threshold=0.005, epsilon=0.001)
        end_time = time.time()
        
        assert len(results) > 0  # Should detect changes
        assert end_time - start_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.slow
    def test_deep_model_structure_performance(self):
        """Test performance with deeply nested model structures"""
        def create_deep_model(depth, base_value):
            if depth == 0:
                return {"value": base_value, "weights": [base_value] * 10}
            
            return {
                "layer": create_deep_model(depth - 1, base_value),
                "params": {"learning_rate": 0.001 + base_value * 0.001}
            }
        
        old = create_deep_model(30, 0.1)  # 30 levels deep
        new = create_deep_model(30, 0.11) # Slightly different values
        
        import time
        start_time = time.time()
        results = diffai_python.diff(old, new, epsilon=0.001)
        end_time = time.time()
        
        assert len(results) > 0  # Should find differences
        assert end_time - start_time < 5.0  # Should handle deep nesting efficiently

if __name__ == "__main__":
    pytest.main([__file__])