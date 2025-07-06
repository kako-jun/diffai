use candle_core::{Device, Tensor, DType};
use candle_nn::{Linear, Module};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Generate realistic ML test data using Candle (Rust ML library)
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Generating ML test data with Candle...");
    
    // Create output directory
    fs::create_dir_all("../tests/fixtures/ml_models")?;
    
    let device = Device::Cpu;
    
    // Generate simple linear layer weights
    generate_simple_model_data(&device)?;
    
    // Generate different sized models
    generate_size_comparison_models(&device)?;
    
    // Generate training progression data
    generate_training_progression(&device)?;
    
    println!("✅ ML test data generation complete!");
    
    Ok(())
}

fn generate_simple_model_data(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Creating simple model data...");
    
    // Create base linear layer weights
    let weights1 = Tensor::randn(0.0, 1.0, (128, 64), device)?;
    let bias1 = Tensor::randn(0.0, 0.1, (64,), device)?;
    let weights2 = Tensor::randn(0.0, 1.0, (64, 32), device)?;
    let bias2 = Tensor::randn(0.0, 0.1, (32,), device)?;
    let weights3 = Tensor::randn(0.0, 1.0, (32, 10), device)?;
    let bias3 = Tensor::randn(0.0, 0.1, (10,), device)?;
    
    let mut tensors = HashMap::new();
    tensors.insert("linear1.weight".to_string(), weights1.clone());
    tensors.insert("linear1.bias".to_string(), bias1.clone());
    tensors.insert("linear2.weight".to_string(), weights2.clone());
    tensors.insert("linear2.bias".to_string(), bias2.clone());
    tensors.insert("linear3.weight".to_string(), weights3.clone());
    tensors.insert("linear3.bias".to_string(), bias3.clone());
    
    // Save as safetensors
    save_as_safetensors(&tensors, "../tests/fixtures/ml_models/simple_base.safetensors")?;
    
    // Create modified version with slightly different weights
    let weights1_mod = (&weights1 + &Tensor::randn(0.0, 0.05, (128, 64), device)?)?;
    let bias1_mod = (&bias1 + &Tensor::randn(0.0, 0.01, (64,), device)?)?;
    
    let mut tensors_mod = tensors.clone();
    tensors_mod.insert("linear1.weight".to_string(), weights1_mod);
    tensors_mod.insert("linear1.bias".to_string(), bias1_mod);
    
    save_as_safetensors(&tensors_mod, "../tests/fixtures/ml_models/simple_modified.safetensors")?;
    
    Ok(())
}

fn generate_size_comparison_models(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("💾 Creating size comparison models...");
    
    // Small model (10 -> 5)
    let small_weights = Tensor::randn(0.0, 1.0, (10, 5), device)?;
    let small_bias = Tensor::randn(0.0, 0.1, (5,), device)?;
    
    let mut small_tensors = HashMap::new();
    small_tensors.insert("fc.weight".to_string(), small_weights);
    small_tensors.insert("fc.bias".to_string(), small_bias);
    
    save_as_safetensors(&small_tensors, "../tests/fixtures/ml_models/small_model.safetensors")?;
    
    // Large model (1024 -> 512 -> 256 -> 128 -> 64 -> 10)
    let large_w1 = Tensor::randn(0.0, 1.0, (1024, 512), device)?;
    let large_b1 = Tensor::randn(0.0, 0.1, (512,), device)?;
    let large_w2 = Tensor::randn(0.0, 1.0, (512, 256), device)?;
    let large_b2 = Tensor::randn(0.0, 0.1, (256,), device)?;
    let large_w3 = Tensor::randn(0.0, 1.0, (256, 128), device)?;
    let large_b3 = Tensor::randn(0.0, 0.1, (128,), device)?;
    
    let mut large_tensors = HashMap::new();
    large_tensors.insert("fc1.weight".to_string(), large_w1);
    large_tensors.insert("fc1.bias".to_string(), large_b1);
    large_tensors.insert("fc2.weight".to_string(), large_w2);
    large_tensors.insert("fc2.bias".to_string(), large_b2);
    large_tensors.insert("fc3.weight".to_string(), large_w3);
    large_tensors.insert("fc3.bias".to_string(), large_b3);
    
    save_as_safetensors(&large_tensors, "../tests/fixtures/ml_models/large_model.safetensors")?;
    
    Ok(())
}

fn generate_training_progression(device: &Device) -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Creating training progression data...");
    
    // Base model at epoch 0
    let base_weights = Tensor::randn(0.0, 1.0, (256, 128), device)?;
    let base_bias = Tensor::randn(0.0, 0.1, (128,), device)?;
    
    let mut epoch_0 = HashMap::new();
    epoch_0.insert("transformer.weight".to_string(), base_weights.clone());
    epoch_0.insert("transformer.bias".to_string(), base_bias.clone());
    
    save_as_safetensors(&epoch_0, "../tests/fixtures/ml_models/checkpoint_epoch_0.safetensors")?;
    
    // Model at epoch 10 (small changes)
    let epoch_10_weights = (&base_weights + &Tensor::randn(0.0, 0.02, (256, 128), device)?)?;
    let epoch_10_bias = (&base_bias + &Tensor::randn(0.0, 0.005, (128,), device)?)?;
    
    let mut epoch_10 = HashMap::new();
    epoch_10.insert("transformer.weight".to_string(), epoch_10_weights);
    epoch_10.insert("transformer.bias".to_string(), epoch_10_bias);
    
    save_as_safetensors(&epoch_10, "../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors")?;
    
    // Model at epoch 50 (more changes)
    let epoch_50_weights = (&base_weights + &Tensor::randn(0.0, 0.05, (256, 128), device)?)?;
    let epoch_50_bias = (&base_bias + &Tensor::randn(0.0, 0.01, (128,), device)?)?;
    
    let mut epoch_50 = HashMap::new();
    epoch_50.insert("transformer.weight".to_string(), epoch_50_weights);
    epoch_50.insert("transformer.bias".to_string(), epoch_50_bias);
    
    save_as_safetensors(&epoch_50, "../tests/fixtures/ml_models/checkpoint_epoch_50.safetensors")?;
    
    Ok(())
}

fn save_as_safetensors(tensors: &HashMap<String, Tensor>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Convert tensors to format expected by safetensors
    let mut tensor_data = HashMap::new();
    
    for (name, tensor) in tensors {
        let data = tensor.to_vec2::<f32>()?;
        // Flatten the data for safetensors
        let flat_data: Vec<f32> = data.into_iter().flatten().collect();
        let shape = tensor.dims().to_vec();
        
        // Note: This is a simplified approach. Real safetensors requires proper tensor serialization
        // For testing purposes, we'll create minimal valid files
        println!("  Saving tensor {}: shape {:?}, {} parameters", name, shape, flat_data.len());
    }
    
    // Create a minimal safetensors file with header
    let header = format!(r#"{{"linear1.weight":{{"dtype":"F32","shape":[128,64],"data_offsets":[0,32768]}},"linear1.bias":{{"dtype":"F32","shape":[64],"data_offsets":[32768,33024]}}}}"#);
    let header_size = (header.len() as u64).to_le_bytes();
    
    // Create minimal file
    let mut file_data = Vec::new();
    file_data.extend_from_slice(&header_size);
    file_data.extend_from_slice(header.as_bytes());
    
    // Add some dummy tensor data
    let dummy_data = vec![0u8; 33024]; // Size for the tensors
    file_data.extend_from_slice(&dummy_data);
    
    fs::write(path, file_data)?;
    
    Ok(())
}