#!/usr/bin/env python3
"""
diffai-python Examples - UNIFIED API DESIGN

Demonstrates native Python API usage for AI/ML model comparison
Users load model data themselves and call the unified diff() function
"""

import json
import tempfile
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union
from diffai_python import diff

def print_header(title: str) -> None:
    """Print a formatted header"""
    print(f"\n{title}")
    print("=" * len(title))

def print_example(title: str, description: str) -> None:
    """Print example title and description"""
    print(f"\n{title}")
    print(f"   {description}")

def print_results(results: List[Dict[str, Any]]) -> None:
    """Print diff results in a formatted way"""
    if not results:
        print("   No differences found.")
        return
    
    print("   ML Differences:")
    for result in results:
        result_type = result.get('type', 'unknown')
        path = result.get('path', '')
        
        if result_type == 'TensorShapeChanged':
            old_shape = result.get('old_shape', [])
            new_shape = result.get('new_shape', [])
            print(f"   📐 Tensor Shape: {path}")
            print(f"      {old_shape} → {new_shape}")
        elif result_type == 'WeightSignificantChange':
            magnitude = result.get('magnitude', 0)
            print(f"   ⚖️  Weight Change: {path}")
            print(f"      Magnitude: {magnitude:.6e}")
            if 'statistics' in result:
                stats = result['statistics']
                print(f"      Mean: {stats.get('mean_change', 0):.6e}")
                print(f"      Changed: {stats.get('changed_elements', 0)}/{stats.get('total_elements', 0)}")
        elif result_type == 'ModelArchitectureChanged':
            print(f"   🏗️  Architecture: {path}")
            print(f"      {result.get('old_value')} → {result.get('new_value')}")
        elif result_type == 'added':
            print(f"   ➕ Added: {path} = {result.get('new_value')}")
        elif result_type == 'removed':
            print(f"   ➖ Removed: {path} = {result.get('old_value')}")
        elif result_type == 'modified':
            print(f"   🔄 Modified: {path}")
            print(f"      Old: {result.get('old_value')}")
            print(f"      New: {result.get('new_value')}")
        else:
            print(f"   • {result}")

def example_model_architecture_comparison():
    """Compare model architectures using unified API"""
    print_example(
        "Model Architecture Evolution",
        "Compare BERT base to large architecture using unified diff() function"
    )
    
    # BERT Base model metadata
    bert_base = {
        "model_name": "bert-base-uncased",
        "model_type": "bert",
        "architecture": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "vocab_size": 30522
        },
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "epochs": 3,
            "optimizer": "adam"
        },
        "performance": {
            "accuracy": 0.92,
            "f1_score": 0.91,
            "inference_time": 120  # ms
        }
    }
    
    # BERT Large model metadata
    bert_large = {
        "model_name": "bert-large-uncased",
        "model_type": "bert",
        "architecture": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "max_position_embeddings": 512,
            "vocab_size": 30522
        },
        "training": {
            "learning_rate": 1e-5,
            "batch_size": 16,
            "epochs": 3,
            "optimizer": "adamw"
        },
        "performance": {
            "accuracy": 0.95,
            "f1_score": 0.94,
            "inference_time": 280  # ms
        }
    }
    
    # Use unified diff() API with ML analysis
    results = diff(
        bert_base, 
        bert_large,
        ml_analysis_enabled=True,
        scientific_precision=True,
        show_types=True
    )
    print_results(results)

def example_training_checkpoint_analysis():
    """Analyze training checkpoint differences"""
    print_example(
        "Training Progress Monitoring",
        "Track model improvements across training checkpoints"
    )
    
    # Epoch 1 checkpoint
    checkpoint_1 = {
        "epoch": 1,
        "step": 1000,
        "model_state": {
            "embeddings.word_embeddings.weight": {
                "shape": [30522, 768],
                "mean": 0.0012,
                "std": 0.0234,
                "norm": 15.67
            },
            "encoder.layer.0.attention.self.query.weight": {
                "shape": [768, 768], 
                "mean": 0.0001,
                "std": 0.0156,
                "norm": 24.32
            },
            "encoder.layer.0.attention.self.query.bias": {
                "shape": [768],
                "mean": 0.0,
                "std": 0.0089,
                "norm": 0.245
            }
        },
        "optimizer_state": {
            "learning_rate": 2e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01
        },
        "metrics": {
            "train_loss": 3.45,
            "train_accuracy": 0.65,
            "val_loss": 3.67,
            "val_accuracy": 0.62,
            "perplexity": 31.5
        }
    }
    
    # Epoch 10 checkpoint (after training)
    checkpoint_10 = {
        "epoch": 10,
        "step": 10000,
        "model_state": {
            "embeddings.word_embeddings.weight": {
                "shape": [30522, 768],
                "mean": 0.0015,
                "std": 0.0245,
                "norm": 15.89
            },
            "encoder.layer.0.attention.self.query.weight": {
                "shape": [768, 768],
                "mean": 0.0003,
                "std": 0.0167,
                "norm": 24.58
            },
            "encoder.layer.0.attention.self.query.bias": {
                "shape": [768],
                "mean": 0.001,
                "std": 0.0092,
                "norm": 0.267
            }
        },
        "optimizer_state": {
            "learning_rate": 5e-6,  # Decayed
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.01
        },
        "metrics": {
            "train_loss": 1.23,
            "train_accuracy": 0.94,
            "val_loss": 1.45,
            "val_accuracy": 0.91,
            "perplexity": 4.2
        }
    }
    
    # Analyze training progress with gradient analysis
    results = diff(
        checkpoint_1,
        checkpoint_10,
        ml_analysis_enabled=True,
        gradient_analysis=True,
        statistical_summary=True,
        weight_threshold=1e-4
    )
    print_results(results)

def example_model_quantization_impact():
    """Analyze the impact of model quantization"""
    print_example(
        "Quantization Impact Analysis",
        "Compare FP32 vs INT8 quantized model for precision loss assessment"
    )
    
    # Original FP32 model
    fp32_model = {
        "precision": "float32",
        "model_size_mb": 440,
        "weights": {
            "linear1.weight": {
                "dtype": "float32",
                "shape": [1024, 768],
                "sample_values": [0.123456789, -0.987654321, 0.456789123],
                "min": -2.34567,
                "max": 2.34567,
                "mean": 0.000123,
                "std": 0.234567
            },
            "linear1.bias": {
                "dtype": "float32", 
                "shape": [1024],
                "sample_values": [0.001234, -0.005678, 0.009012],
                "min": -0.1,
                "max": 0.1,
                "mean": 0.0,
                "std": 0.01
            }
        },
        "performance": {
            "inference_time_ms": 125,
            "memory_usage_mb": 1200,
            "accuracy": 0.945,
            "throughput_samples_sec": 800
        }
    }
    
    # Quantized INT8 model
    int8_model = {
        "precision": "int8",
        "model_size_mb": 110,  # 4x smaller
        "weights": {
            "linear1.weight": {
                "dtype": "int8",
                "shape": [1024, 768],
                "sample_values": [0.123, -0.988, 0.457],  # Reduced precision
                "min": -2.35,
                "max": 2.35,
                "mean": 0.000120,  # Slight difference
                "std": 0.234
            },
            "linear1.bias": {
                "dtype": "int8",
                "shape": [1024], 
                "sample_values": [0.001, -0.006, 0.009],  # Reduced precision
                "min": -0.1,
                "max": 0.1,
                "mean": 0.0,
                "std": 0.01
            }
        },
        "performance": {
            "inference_time_ms": 45,  # 3x faster
            "memory_usage_mb": 300,   # 4x less memory
            "accuracy": 0.938,        # Slight accuracy drop
            "throughput_samples_sec": 2400  # 3x higher throughput
        }
    }
    
    # Compare with higher epsilon for quantization tolerance
    results = diff(
        fp32_model,
        int8_model,
        epsilon=1e-3,  # Higher tolerance for quantization
        ml_analysis_enabled=True,
        scientific_precision=True
    )
    print_results(results)

def example_tensor_comparison():
    """Compare tensor data with statistical analysis"""
    print_example(
        "Tensor Weight Analysis",
        "Compare model weights with statistical significance testing"
    )
    
    # Simulate tensor data before fine-tuning
    np.random.seed(42)
    weights_before = {
        "layer1": {
            "weight": {
                "shape": [256, 128],
                "data": np.random.normal(0, 0.1, (256, 128)).tolist(),
                "requires_grad": True
            },
            "bias": {
                "shape": [256],
                "data": np.zeros(256).tolist(),
                "requires_grad": True
            }
        },
        "layer2": {
            "weight": {
                "shape": [128, 64],
                "data": np.random.normal(0, 0.1, (128, 64)).tolist(),
                "requires_grad": True
            },
            "bias": {
                "shape": [128],
                "data": np.zeros(128).tolist(),
                "requires_grad": True
            }
        }
    }
    
    # Simulate tensor data after fine-tuning
    np.random.seed(123)  # Different seed for different weights
    weights_after = {
        "layer1": {
            "weight": {
                "shape": [256, 128],
                "data": (np.random.normal(0, 0.1, (256, 128)) + 
                        np.random.normal(0, 0.01, (256, 128))).tolist(),  # Small changes
                "requires_grad": True
            },
            "bias": {
                "shape": [256],
                "data": np.random.normal(0, 0.001, 256).tolist(),  # No longer zero
                "requires_grad": True
            }
        },
        "layer2": {
            "weight": {
                "shape": [128, 64], 
                "data": (np.random.normal(0, 0.1, (128, 64)) +
                        np.random.normal(0, 0.02, (128, 64))).tolist(),  # Larger changes
                "requires_grad": True
            },
            "bias": {
                "shape": [128],
                "data": np.random.normal(0, 0.002, 128).tolist(),
                "requires_grad": True
            }
        },
        "layer3": {  # New layer added
            "weight": {
                "shape": [64, 32],
                "data": np.random.normal(0, 0.1, (64, 32)).tolist(),
                "requires_grad": True
            },
            "bias": {
                "shape": [64],
                "data": np.zeros(64).tolist(),
                "requires_grad": True
            }
        }
    }
    
    # Analyze with tensor-specific options
    results = diff(
        weights_before,
        weights_after,
        ml_analysis_enabled=True,
        tensor_comparison_mode="statistical",
        weight_threshold=1e-5,
        statistical_summary=True
    )
    print_results(results)

def example_model_format_comparison():
    """Compare different model formats and versions"""
    print_example(
        "Model Format and Version Comparison",
        "Compare different serialization formats and model versions"
    )
    
    # PyTorch model metadata
    pytorch_model = {
        "framework": "pytorch",
        "version": "2.0.1",
        "model_format": "state_dict",
        "serialization": "pickle",
        "model_info": {
            "total_params": 110_000_000,
            "trainable_params": 110_000_000,
            "model_size_mb": 440,
            "layers": 12
        },
        "compatibility": {
            "torch_version": ">=1.9.0",
            "python_version": ">=3.8",
            "cuda_compute_capability": ">=6.1"
        }
    }
    
    # Safetensors model metadata
    safetensors_model = {
        "framework": "safetensors",
        "version": "0.3.1",
        "model_format": "safetensors",
        "serialization": "safetensors",
        "model_info": {
            "total_params": 110_000_000,
            "trainable_params": 110_000_000,
            "model_size_mb": 438,  # Slightly smaller due to efficient format
            "layers": 12
        },
        "compatibility": {
            "torch_version": ">=1.13.0",
            "python_version": ">=3.8",
            "cuda_compute_capability": ">=6.1"
        },
        "safety_features": {
            "memory_safe_loading": True,
            "lazy_loading": True,
            "cross_platform": True
        }
    }
    
    # Compare model formats
    results = diff(
        pytorch_model,
        safetensors_model,
        ml_analysis_enabled=True,
        model_format="auto"
    )
    print_results(results)

def example_hyperparameter_optimization():
    """Track hyperparameter optimization experiments"""
    print_example(
        "Hyperparameter Optimization Tracking",
        "Compare different hyperparameter configurations and their results"
    )
    
    # Baseline configuration
    baseline_config = {
        "experiment_id": "baseline",
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 32,
            "max_length": 128,
            "dropout": 0.1,
            "warmup_steps": 1000,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8
        },
        "results": {
            "final_accuracy": 0.924,
            "final_loss": 0.234,
            "best_val_accuracy": 0.918,
            "training_time_hours": 4.5,
            "convergence_epoch": 8,
            "memory_usage_gb": 12.4
        },
        "metrics_history": {
            "train_accuracy": [0.65, 0.78, 0.85, 0.89, 0.91, 0.92, 0.923, 0.924],
            "val_accuracy": [0.62, 0.75, 0.82, 0.87, 0.89, 0.91, 0.915, 0.918]
        }
    }
    
    # Optimized configuration  
    optimized_config = {
        "experiment_id": "optimized_v1",
        "hyperparameters": {
            "learning_rate": 3e-5,     # Increased
            "batch_size": 16,          # Decreased for stability
            "max_length": 256,         # Increased
            "dropout": 0.05,           # Decreased
            "warmup_steps": 2000,      # Increased
            "weight_decay": 0.005,     # Decreased
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-6       # Decreased
        },
        "results": {
            "final_accuracy": 0.951,   # Improved!
            "final_loss": 0.156,       # Improved!
            "best_val_accuracy": 0.945, # Improved!
            "training_time_hours": 6.2,  # Longer due to larger sequences
            "convergence_epoch": 6,      # Faster convergence
            "memory_usage_gb": 18.7     # Higher due to larger batch
        },
        "metrics_history": {
            "train_accuracy": [0.69, 0.82, 0.89, 0.93, 0.947, 0.951],
            "val_accuracy": [0.66, 0.79, 0.86, 0.92, 0.941, 0.945]
        }
    }
    
    # Compare configurations
    results = diff(
        baseline_config,
        optimized_config,
        epsilon=1e-6,
        ml_analysis_enabled=True,
        show_types=False
    )
    print_results(results)

def example_model_deployment_comparison():
    """Compare model configurations for different deployment environments"""
    print_example(
        "Deployment Environment Comparison",
        "Compare model configurations across development, staging, and production"
    )
    
    # Development environment
    dev_config = {
        "environment": "development",
        "model_config": {
            "precision": "float32",
            "batch_size": 8,
            "max_sequence_length": 128,
            "use_cache": False,
            "debug_mode": True
        },
        "inference_config": {
            "device": "cpu",
            "num_workers": 1,
            "timeout_seconds": 60,
            "enable_profiling": True
        },
        "monitoring": {
            "log_level": "DEBUG", 
            "metrics_enabled": True,
            "detailed_timing": True
        }
    }
    
    # Production environment
    prod_config = {
        "environment": "production",
        "model_config": {
            "precision": "float16",    # Optimized for speed
            "batch_size": 64,          # Larger batches
            "max_sequence_length": 512, # Support longer sequences
            "use_cache": True,         # Enable caching
            "debug_mode": False        # Disabled for security
        },
        "inference_config": {
            "device": "cuda",          # GPU acceleration
            "num_workers": 8,          # Parallel processing
            "timeout_seconds": 10,     # Stricter timeout
            "enable_profiling": False  # Disabled for performance
        },
        "monitoring": {
            "log_level": "INFO",       # Less verbose
            "metrics_enabled": True,
            "detailed_timing": False   # Disabled for performance
        },
        "scaling": {                   # Production-only config
            "auto_scaling": True,
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70
        }
    }
    
    # Compare deployment configurations
    results = diff(
        dev_config,
        prod_config,
        ml_analysis_enabled=True,
        show_unchanged=False
    )
    print_results(results)

def main():
    """Run all examples"""
    print("=" * 70)
    print("diffai-python Native API Examples - UNIFIED API DESIGN")
    print("=" * 70)
    print("\nAll examples use only the unified diff() function.")
    print("Users load model data themselves using appropriate ML libraries.")
    
    examples = [
        example_model_architecture_comparison,
        example_training_checkpoint_analysis,
        example_model_quantization_impact,
        example_tensor_comparison,
        example_model_format_comparison,
        example_hyperparameter_optimization,
        example_model_deployment_comparison,
    ]
    
    for example_func in examples:
        try:
            print_header(f"Example: {example_func.__name__.replace('example_', '').replace('_', ' ').title()}")
            example_func()
        except Exception as e:
            print(f"\n❌ ERROR in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print_header("Summary")
    print("✅ All examples use the unified diff() API only")
    print("🤖 Users handle model loading with ML libraries (PyTorch, TensorFlow, etc.)")  
    print("📊 AI/ML-specific analysis and statistics")
    print("🚀 Ready for MLOps and model monitoring pipelines")
    
    print("\nML-Specific Benefits:")
    print("  • Tensor shape and data comparison")
    print("  • Statistical significance testing")
    print("  • Weight change magnitude analysis")
    print("  • Training progress monitoring")
    print("  • Model architecture evolution tracking")
    print("  • Quantization impact assessment")
    print("  • Hyperparameter optimization tracking")
    
    print("\nMLOps Use Cases:")
    print("  • Model version comparison")
    print("  • Training checkpoint analysis")
    print("  • A/B testing for model variants")
    print("  • Deployment configuration validation")
    print("  • Performance regression detection")
    print("  • Model drift monitoring")
    
    print("\nFor more information:")
    print("  • Documentation: https://github.com/kako-jun/diffai")
    print("  • PyPI Package: https://pypi.org/project/diffai-python/")
    print("  • Issues: https://github.com/kako-jun/diffai/issues")

if __name__ == "__main__":
    main()