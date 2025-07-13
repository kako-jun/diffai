#!/usr/bin/env python3
"""Create test safetensors files for testing diffai ML comparison functionality."""

import numpy as np
from safetensors.numpy import save_file

# Create a simple model with different tensors
def create_test_model(filename, seed=42):
    np.random.seed(seed)
    
    tensors = {
        "linear1.weight": np.random.randn(256, 128).astype(np.float32),
        "linear1.bias": np.random.randn(256).astype(np.float32),
        "linear2.weight": np.random.randn(128, 256).astype(np.float32),
        "linear2.bias": np.random.randn(128).astype(np.float32),
        "output.weight": np.random.randn(10, 128).astype(np.float32),
        "output.bias": np.random.randn(10).astype(np.float32),
    }
    
    save_file(tensors, filename)
    print(f"Created {filename}")

# Create test models
if __name__ == "__main__":
    # Model 1: Base model
    create_test_model("test_model1.safetensors", seed=42)
    
    # Model 2: Same architecture, different weights (simulating fine-tuning)
    create_test_model("test_model2.safetensors", seed=123)
    
    # Model 3: Different architecture (different layer sizes)
    np.random.seed(999)
    tensors_diff_arch = {
        "linear1.weight": np.random.randn(512, 128).astype(np.float32),  # Different size
        "linear1.bias": np.random.randn(512).astype(np.float32),
        "linear2.weight": np.random.randn(128, 512).astype(np.float32),
        "linear2.bias": np.random.randn(128).astype(np.float32),
        "output.weight": np.random.randn(10, 128).astype(np.float32),
        "output.bias": np.random.randn(10).astype(np.float32),
        "new_layer.weight": np.random.randn(64, 64).astype(np.float32),  # New layer
    }
    save_file(tensors_diff_arch, "test_model3.safetensors")
    print("Created test_model3.safetensors with different architecture")