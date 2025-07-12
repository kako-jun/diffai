#!/usr/bin/env python3
"""
Download small real models from HuggingFace for diffai testing

This script uses uv for dependency management and downloads models
in both PyTorch and Safetensors formats for comprehensive testing.

Usage:
    # Setup with uv
    uv sync
    
    # Run the script
    uv run python download_models.py
"""

from huggingface_hub import hf_hub_download
import os
import ssl
import urllib3
from pathlib import Path

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variable to disable SSL verification
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

# Create output directory
OUTPUT_DIR = Path(__file__).parent

def download_model_safely(repo_id: str, filename: str, local_dir: str, description: str):
    """Safely download a model with error handling and progress info"""
    try:
        output_path = OUTPUT_DIR / local_dir
        output_path.mkdir(exist_ok=True)
        
        print(f"📥 Downloading {description}...")
        print(f"   Repo: {repo_id}")
        print(f"   File: {filename}")
        
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(output_path),
            local_dir_use_symlinks=False
        )
        
        file_size = os.path.getsize(model_file) / 1024 / 1024  # MB
        print(f"✅ Downloaded: {model_file} ({file_size:.1f} MB)")
        
        return model_file
        
    except Exception as e:
        print(f"❌ Error downloading {description}: {e}")
        return None

def download_small_model():
    """Download a small BERT model in safetensors format"""
    model_file = download_model_safely(
        repo_id="distilbert-base-uncased",
        filename="model.safetensors",
        local_dir="distilbert_base",
        description="DistilBERT-base (Safetensors)"
    )
    
    # Also download config for reference
    if model_file:
        config_file = download_model_safely(
            repo_id="distilbert-base-uncased",
            filename="config.json", 
            local_dir="distilbert_base",
            description="DistilBERT config"
        )
    
    return model_file

def download_tiny_model():
    """Download a small model in PyTorch format for testing PyTorch support"""
    return download_model_safely(
        repo_id="microsoft/DialoGPT-small",
        filename="pytorch_model.bin",
        local_dir="dialogpt_small",
        description="DialoGPT-small (PyTorch format)"
    )

def download_gpt2_small():
    """Download GPT-2 small model in safetensors format"""
    return download_model_safely(
        repo_id="openai-community/gpt2",
        filename="model.safetensors",
        local_dir="gpt2_small",
        description="GPT-2 small (Safetensors)"
    )

def download_alternative_models():
    """Download additional models for comprehensive testing"""
    models = []
    
    # Try to get a smaller model first
    tiny_gpt2 = download_model_safely(
        repo_id="sshleifer/tiny-gpt2",
        filename="pytorch_model.bin",
        local_dir="tiny_gpt2",
        description="Tiny GPT-2 (PyTorch, very small)"
    )
    if tiny_gpt2:
        models.append(("Tiny GPT-2", tiny_gpt2))
    
    # Try another small safetensors model
    distilgpt2 = download_model_safely(
        repo_id="distilgpt2",
        filename="model.safetensors", 
        local_dir="distilgpt2",
        description="DistilGPT-2 (Safetensors)"
    )
    if distilgpt2:
        models.append(("DistilGPT-2", distilgpt2))
    
    return models

if __name__ == "__main__":
    print("🚀 Downloading real ML models for diffai testing...")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Try to download different models
    models = []
    
    # Priority 1: DistilBERT (most likely to work, safetensors)
    print("\n🎯 Phase 1: Downloading primary test models")
    distilbert = download_small_model()
    if distilbert:
        models.append(("DistilBERT (Safetensors)", distilbert))
    
    # Priority 2: Small PyTorch model for PyTorch testing
    tiny_model = download_tiny_model()
    if tiny_model:
        models.append(("DialoGPT-small (PyTorch)", tiny_model))
    
    # Priority 3: GPT-2 small (if available)
    print("\n🎯 Phase 2: Downloading additional models")
    gpt2 = download_gpt2_small()
    if gpt2:
        models.append(("GPT-2 small (Safetensors)", gpt2))
    
    # Priority 4: Alternative models
    print("\n🎯 Phase 3: Downloading alternative models")
    alternative_models = download_alternative_models()
    models.extend(alternative_models)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Successfully downloaded {len(models)} models:")
    total_size = 0
    for name, path in models:
        size_mb = os.path.getsize(path) / 1024 / 1024
        total_size += size_mb
        print(f"  ✅ {name}")
        print(f"     📁 {path}")
        print(f"     📏 Size: {size_mb:.1f} MB")
        print()
    
    print(f"💾 Total size: {total_size:.1f} MB")
    print()
    print("🎉 Ready for diffai testing!")
    print()
    print("💡 Usage examples:")
    if len(models) >= 2:
        model1_path = models[0][1]
        model2_path = models[1][1]
        print(f"   diffai '{model1_path}' '{model2_path}' --stats")
        print(f"   diffai '{model1_path}' '{model2_path}' --learning-progress --output json")
    
    print("\n🔧 To run this script:")
    print("   cd real_models_test/")
    print("   uv sync")
    print("   uv run python download_models.py")