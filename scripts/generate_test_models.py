#!/usr/bin/env python3
"""
Generate real AI/ML test model files for diffai testing.

This script creates actual PyTorch and Safetensors model files
with realistic tensor data for comprehensive testing.
"""

import torch
import torch.nn as nn
import numpy as np
from safetensors.torch import save_file as save_safetensors
import os
import json

def create_simple_model():
    """Create a simple neural network model."""
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    return SimpleNet()

def create_transformer_model():
    """Create a small transformer-like model."""
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super(SimpleTransformer, self).__init__()
            self.embedding = nn.Embedding(1000, 128)
            self.transformer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
            self.classifier = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.transformer(x)
            x = self.classifier(x.mean(dim=1))
            return x
    
    return SimpleTransformer()

def add_training_noise(model, noise_scale=0.01):
    """Add small random noise to simulate training progress."""
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * noise_scale)

def main():
    """Generate test model files."""
    
    # Create output directory
    os.makedirs("../tests/fixtures/ml_models", exist_ok=True)
    
    print("🔧 Generating ML test data...")
    
    # ==========================================
    # 1. Basic model comparison test files
    # ==========================================
    print("📊 Creating basic model comparison files...")
    
    # Base model
    base_model = create_simple_model()
    
    # Save as PyTorch
    torch.save(base_model.state_dict(), "../tests/fixtures/ml_models/simple_base.pt")
    
    # Save as Safetensors
    save_safetensors(base_model.state_dict(), "../tests/fixtures/ml_models/simple_base.safetensors")
    
    # Modified model (slightly different weights)
    modified_model = create_simple_model()
    modified_model.load_state_dict(base_model.state_dict())
    add_training_noise(modified_model, noise_scale=0.05)
    
    torch.save(modified_model.state_dict(), "../tests/fixtures/ml_models/simple_modified.pt")
    save_safetensors(modified_model.state_dict(), "../tests/fixtures/ml_models/simple_modified.safetensors")
    
    # ==========================================
    # 2. Different architecture models
    # ==========================================
    print("🏗️ Creating different architecture models...")
    
    # Transformer model
    transformer_model = create_transformer_model()
    torch.save(transformer_model.state_dict(), "../tests/fixtures/ml_models/transformer.pt")
    save_safetensors(transformer_model.state_dict(), "../tests/fixtures/ml_models/transformer.safetensors")
    
    # ==========================================
    # 3. Training progression models
    # ==========================================
    print("📈 Creating training progression models...")
    
    # Initial model (epoch 0)
    initial_model = create_simple_model()
    torch.save(initial_model.state_dict(), "../tests/fixtures/ml_models/checkpoint_epoch_0.pt")
    save_safetensors(initial_model.state_dict(), "../tests/fixtures/ml_models/checkpoint_epoch_0.safetensors")
    
    # Training progression (epoch 10)
    epoch_10_model = create_simple_model()
    epoch_10_model.load_state_dict(initial_model.state_dict())
    add_training_noise(epoch_10_model, noise_scale=0.02)
    torch.save(epoch_10_model.state_dict(), "../tests/fixtures/ml_models/checkpoint_epoch_10.pt")
    save_safetensors(epoch_10_model.state_dict(), "../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors")
    
    # Further training (epoch 50)
    epoch_50_model = create_simple_model()
    epoch_50_model.load_state_dict(epoch_10_model.state_dict())
    add_training_noise(epoch_50_model, noise_scale=0.03)
    torch.save(epoch_50_model.state_dict(), "../tests/fixtures/ml_models/checkpoint_epoch_50.pt")
    save_safetensors(epoch_50_model.state_dict(), "../tests/fixtures/ml_models/checkpoint_epoch_50.safetensors")
    
    # ==========================================
    # 4. Quantization models
    # ==========================================
    print("🔢 Creating quantization test models...")
    
    # Full precision model
    fp32_model = create_simple_model()
    torch.save(fp32_model.state_dict(), "../tests/fixtures/ml_models/model_fp32.pt")
    save_safetensors(fp32_model.state_dict(), "../tests/fixtures/ml_models/model_fp32.safetensors")
    
    # Simulated quantized model (scaled weights)
    quantized_model = create_simple_model()
    quantized_model.load_state_dict(fp32_model.state_dict())
    
    # Simulate quantization by scaling and adding noise
    with torch.no_grad():
        for param in quantized_model.parameters():
            # Quantize to int8 range then back to float32
            scale = param.abs().max() / 127.0
            quantized = torch.round(param / scale).clamp(-128, 127)
            param.copy_(quantized * scale)
    
    torch.save(quantized_model.state_dict(), "../tests/fixtures/ml_models/model_quantized.pt")
    save_safetensors(quantized_model.state_dict(), "../tests/fixtures/ml_models/model_quantized.safetensors")
    
    # ==========================================
    # 5. Anomaly detection models
    # ==========================================
    print("🚨 Creating anomaly detection test models...")
    
    # Normal model
    normal_model = create_simple_model()
    torch.save(normal_model.state_dict(), "../tests/fixtures/ml_models/normal_model.pt")
    save_safetensors(normal_model.state_dict(), "../tests/fixtures/ml_models/normal_model.safetensors")
    
    # Anomalous model (extreme weights)
    anomalous_model = create_simple_model()
    anomalous_model.load_state_dict(normal_model.state_dict())
    
    # Create anomalous weights (exploding gradients simulation)
    with torch.no_grad():
        for name, param in anomalous_model.named_parameters():
            if 'fc1' in name:  # Affect first layer dramatically
                param.mul_(100.0)  # Extremely large weights
    
    torch.save(anomalous_model.state_dict(), "../tests/fixtures/ml_models/anomalous_model.pt")
    save_safetensors(anomalous_model.state_dict(), "../tests/fixtures/ml_models/anomalous_model.safetensors")
    
    # ==========================================
    # 6. Different sizes for memory analysis
    # ==========================================
    print("💾 Creating memory analysis test models...")
    
    # Small model
    class SmallModel(nn.Module):
        def __init__(self):
            super(SmallModel, self).__init__()
            self.fc = nn.Linear(10, 5)
    
    small_model = SmallModel()
    torch.save(small_model.state_dict(), "../tests/fixtures/ml_models/small_model.pt")
    save_safetensors(small_model.state_dict(), "../tests/fixtures/ml_models/small_model.safetensors")
    
    # Large model
    class LargeModel(nn.Module):
        def __init__(self):
            super(LargeModel, self).__init__()
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 64)
            self.fc5 = nn.Linear(64, 10)
    
    large_model = LargeModel()
    torch.save(large_model.state_dict(), "../tests/fixtures/ml_models/large_model.pt")
    save_safetensors(large_model.state_dict(), "../tests/fixtures/ml_models/large_model.safetensors")
    
    # ==========================================
    # 7. Generate metadata file
    # ==========================================
    print("📋 Creating metadata file...")
    
    metadata = {
        "generated_by": "diffai test data generator",
        "description": "Real ML model files for comprehensive testing",
        "models": {
            "simple_base": {
                "architecture": "SimpleNet",
                "parameters": sum(p.numel() for p in base_model.parameters()),
                "layers": 3,
                "purpose": "Base model for comparison tests"
            },
            "simple_modified": {
                "architecture": "SimpleNet", 
                "parameters": sum(p.numel() for p in modified_model.parameters()),
                "layers": 3,
                "purpose": "Modified version for diff testing"
            },
            "transformer": {
                "architecture": "SimpleTransformer",
                "parameters": sum(p.numel() for p in transformer_model.parameters()),
                "purpose": "Different architecture comparison"
            },
            "checkpoints": {
                "purpose": "Training progression analysis",
                "epochs": [0, 10, 50]
            },
            "quantization": {
                "fp32": "Full precision model",
                "quantized": "Simulated int8 quantized model"
            },
            "anomaly": {
                "normal": "Standard weights",
                "anomalous": "Extreme weights (gradient explosion simulation)"
            },
            "memory": {
                "small": f"Parameters: {sum(p.numel() for p in small_model.parameters())}",
                "large": f"Parameters: {sum(p.numel() for p in large_model.parameters())}"
            }
        }
    }
    
    with open("../tests/fixtures/ml_models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Test data generation complete!")
    print(f"📁 Generated files in: tests/fixtures/ml_models/")
    
    # Show file sizes
    import glob
    files = glob.glob("../tests/fixtures/ml_models/*")
    print("\n📊 Generated files:")
    for file in sorted(files):
        size = os.path.getsize(file)
        print(f"  {os.path.basename(file)}: {size:,} bytes")

if __name__ == "__main__":
    main()