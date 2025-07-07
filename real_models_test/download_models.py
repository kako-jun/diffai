#!/usr/bin/env python3
"""
Download small real models from HuggingFace for diffai testing
"""

from huggingface_hub import hf_hub_download
import os
import ssl
import urllib3

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variable to disable SSL verification
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

def download_small_model():
    """Download a small BERT model in safetensors format"""
    try:
        # Download DistilBERT model (small, fast, safetensors available)
        print("Downloading DistilBERT-base model...")
        model_file = hf_hub_download(
            repo_id="distilbert-base-uncased",
            filename="model.safetensors",
            local_dir="distilbert_base",
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {model_file}")
        
        # Also download config for reference
        config_file = hf_hub_download(
            repo_id="distilbert-base-uncased", 
            filename="config.json",
            local_dir="distilbert_base",
            local_dir_use_symlinks=False
        )
        print(f"Downloaded config: {config_file}")
        
        return model_file
        
    except Exception as e:
        print(f"Error downloading DistilBERT: {e}")
        return None

def download_tiny_model():
    """Download an even smaller model for testing"""
    try:
        # Download a tiny BERT model
        print("Downloading TinyBERT model...")
        model_file = hf_hub_download(
            repo_id="huawei-noah/TinyBERT_General_4L_312D",
            filename="pytorch_model.bin",  # This might be PyTorch format
            local_dir="tinybert",
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {model_file}")
        return model_file
        
    except Exception as e:
        print(f"Error downloading TinyBERT: {e}")
        return None

def download_gpt2_small():
    """Download GPT-2 small model in safetensors format"""
    try:
        print("Downloading GPT-2 small model...")
        model_file = hf_hub_download(
            repo_id="openai-community/gpt2",
            filename="model.safetensors",
            local_dir="gpt2_small",
            local_dir_use_symlinks=False
        )
        print(f"Downloaded: {model_file}")
        return model_file
        
    except Exception as e:
        print(f"Error downloading GPT-2: {e}")
        return None

if __name__ == "__main__":
    print("Downloading real ML models for diffai testing...")
    
    # Try to download different models
    models = []
    
    # DistilBERT (most likely to work)
    distilbert = download_small_model()
    if distilbert:
        models.append(("DistilBERT", distilbert))
    
    # GPT-2 small
    gpt2 = download_gpt2_small()
    if gpt2:
        models.append(("GPT-2", gpt2))
    
    # Print summary
    print(f"\nSuccessfully downloaded {len(models)} models:")
    for name, path in models:
        print(f"  {name}: {path}")
        print(f"    Size: {os.path.getsize(path) / 1024 / 1024:.1f} MB")
    
    print("\nReady for diffai testing!")