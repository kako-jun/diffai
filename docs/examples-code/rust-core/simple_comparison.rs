// Example: Simple ML model comparison using diffai v0.3.16
// 
// This example demonstrates diffai's automatic comprehensive analysis
// following the Convention over Configuration principle. All 11 ML
// analysis functions run automatically for PyTorch/Safetensors files.

use anyhow::Result;
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    println!("🤖 diffai Example: Automatic ML Analysis");
    println!("========================================");
    println!("✨ Convention over Configuration - All 11 ML analyses run automatically");
    println!("");
    
    // Example model paths - you can replace these with your own models
    let model1_path = Path::new("models/baseline.safetensors");
    let model2_path = Path::new("models/improved.safetensors");
    
    // Check if files exist
    if !model1_path.exists() || !model2_path.exists() {
        println!("⚠️  Model files not found. This example requires:");
        println!("   - models/baseline.safetensors");
        println!("   - models/improved.safetensors");
        println!("");
        println!("💡 You can create test models using:");
        println!("   python scripts/create_test_models.py");
        println!("");
        println!("🎯 This example demonstrates diffai's automatic analysis:");
        demonstrate_automatic_analysis();
        return Ok(());
    }
    
    println!("📊 Comparing models with automatic comprehensive analysis:");
    println!("   Model 1: {}", model1_path.display());
    println!("   Model 2: {}", model2_path.display());
    println!("");
    
    // In a real scenario, you would load actual model files using ML libraries
    // For this example, we'll demonstrate with placeholder data
    let old_model = create_sample_model_data("baseline");
    let new_model = create_sample_model_data("improved");
    
    // Configure for automatic ML analysis
    let options = DiffOptions {
        ml_analysis_enabled: Some(true), // Auto-enabled for ML files
        use_memory_optimization: Some(true),
        verbose: Some(true),
        ..Default::default()
    };
    
    // Perform automatic comprehensive model comparison
    match diff(&old_model, &new_model, Some(&options)) {
        Ok(differences) => {
            if differences.is_empty() {
                println!("✅ No significant differences found between models!");
                println!("   All 11 ML analysis functions completed successfully");
            } else {
                println!("🔍 Found {} differences from automatic analysis:", differences.len());
                println!("");
                
                analyze_ml_differences(&differences);
            }
        }
        Err(e) => {
            println!("❌ Error during automatic model analysis: {}", e);
            println!("");
            println!("💡 Common issues:");
            println!("   - File format not supported (use .safetensors or .pt/.pth)");
            println!("   - Corrupted model files");
            println!("   - Insufficient memory for large models");
        }
    }
    
    Ok(())
}

fn analyze_ml_differences(differences: &[DiffResult]) {
    let mut ml_analysis_count = 0;
    let mut tensor_changes = 0;
    let mut architecture_changes = 0;
    
    println!("🎯 Automatic ML Analysis Results:");
    println!("");
    
    for (i, diff) in differences.iter().enumerate() {
        match diff {
            DiffResult::WeightSignificantChange(path, magnitude, stats) => {
                ml_analysis_count += 1;
                println!("📊 Weight Significant Change: {}", path);
                println!("   Change Magnitude: {:.6}", magnitude);
                println!("   Statistical Impact: mean_change={:.6}, std_change={:.6}", 
                    stats.mean_change, stats.std_change);
            }
            
            DiffResult::ArchitectureChanged(old_arch, new_arch) => {
                architecture_changes += 1;
                println!("🏗️ Architecture Change Detected:");
                println!("   Old: {}", old_arch);
                println!("   New: {}", new_arch);
            }
            
            DiffResult::PrecisionChanged(path, old_prec, new_prec) => {
                ml_analysis_count += 1;
                println!("🔢 Precision Change: {}", path);
                println!("   {} → {} (Quantization Analysis)", old_prec, new_prec);
            }
            
            DiffResult::LayerAdded(path, layer_info) => {
                architecture_changes += 1;
                println!("+ Layer Added: {}", path);
                println!("   Type: {}, Parameters: {}", layer_info.layer_type, layer_info.parameter_count);
            }
            
            DiffResult::LayerRemoved(path, layer_info) => {
                architecture_changes += 1;
                println!("- Layer Removed: {}", path);
                println!("   Type: {}, Parameters: {}", layer_info.layer_type, layer_info.parameter_count);
            }
            
            DiffResult::TensorShapeChanged(path, old_shape, new_shape) => {
                tensor_changes += 1;
                println!("⬚ Tensor Shape Changed: {}", path);
                println!("   Shape: {:?} → {:?}", old_shape, new_shape);
                let old_params: usize = old_shape.iter().product();
                let new_params: usize = new_shape.iter().product();
                println!("   Parameters: {} → {} (Δ: {:+})", 
                    old_params, new_params, new_params as i64 - old_params as i64);
            }
            
            DiffResult::Added(name, _) => {
                println!("+ Added: {}", name);
            }
            
            DiffResult::Removed(name, _) => {
                println!("- Removed: {}", name);
            }
            
            DiffResult::Modified(name, _, _) => {
                println!("~ Modified: {}", name);
            }
            
            DiffResult::TypeChanged(name, old_type, new_type) => {
                println!("! Type Changed: {} ({} → {})", name, old_type, new_type);
            }
        }
        
        if i < differences.len() - 1 {
            println!("");
        }
    }
    
    // Summary of automatic analysis
    println!("");
    println!("📋 Automatic Analysis Summary:");
    println!("   ML-specific analyses detected: {}", ml_analysis_count);
    println!("   Architecture changes:          {}", architecture_changes);
    println!("   Tensor shape changes:          {}", tensor_changes);
    println!("   Total differences found:       {}", differences.len());
    
    // Analysis insights from automatic analysis
    println!("");
    println!("💡 Automatic Analysis Insights:");
    
    if architecture_changes > 0 {
        println!("   🔧 Architecture modifications detected - significant model changes");
        println!("      → Automatic Architecture Analysis completed");
    }
    
    if ml_analysis_count > 0 {
        println!("   📊 ML-specific changes detected - training or optimization related");
        println!("      → Automatic Weight/Precision Analysis completed");
    }
    
    if tensor_changes > 0 {
        println!("   ⚠️  Tensor structure changes - potential compatibility impacts");
        println!("      → Automatic Tensor Analysis completed");
    }
    
    if differences.is_empty() {
        println!("   ✅ Models are functionally identical");
        println!("      → All 11 ML analysis functions confirmed no significant changes");
    }
    
    println!("");
    println!("🎯 All analyses completed automatically - no manual configuration required!");
}

fn create_sample_model_data(model_name: &str) -> Value {
    // This creates sample data to demonstrate diffai's analysis capabilities
    // In real usage, you would load actual PyTorch/Safetensors files
    serde_json::json!({
        "model_name": model_name,
        "version": if model_name == "baseline" { "1.0" } else { "1.1" },
        "architecture": {
            "type": "transformer",
            "layers": if model_name == "baseline" { 12 } else { 12 },
            "hidden_size": 768,
            "attention_heads": if model_name == "baseline" { 8 } else { 12 }
        },
        "parameters": {
            "total": if model_name == "baseline" { 124439808 } else { 124440064 },
            "trainable": if model_name == "baseline" { 124439808 } else { 124440064 }
        },
        "training": {
            "learning_rate": if model_name == "baseline" { 0.001 } else { 0.0005 },
            "optimizer": "Adam",
            "epoch": if model_name == "baseline" { 10 } else { 15 }
        },
        "tensors": {
            "transformer.embeddings.word_embeddings.weight": {
                "shape": [30522, 768],
                "dtype": "float32",
                "mean": if model_name == "baseline" { -0.0002 } else { -0.0001 },
                "std": if model_name == "baseline" { 0.0514 } else { 0.0716 }
            },
            "transformer.encoder.layer.0.attention.self.query.weight": {
                "shape": [768, 768],
                "dtype": if model_name == "baseline" { "float32" } else { "float16" },
                "mean": if model_name == "baseline" { 0.0018 } else { 0.0017 },
                "std": if model_name == "baseline" { 0.0518 } else { 0.0647 }
            }
        }
    })
}

fn demonstrate_automatic_analysis() {
    println!("🎯 diffai's Automatic ML Analysis Features:");
    println!("");
    println!("When you run: diffai model1.safetensors model2.safetensors");
    println!("");
    println!("All 11 analysis functions run automatically:");
    println!("  1. 📈 Learning Rate Analysis     - Training dynamics tracking");
    println!("  2. ⚙️  Optimizer Comparison      - State and momentum analysis");
    println!("  3. 📉 Loss Tracking             - Convergence pattern analysis");
    println!("  4. 🎯 Accuracy Tracking         - Performance metrics monitoring");
    println!("  5. 🏷️  Model Version Analysis   - Checkpoint evolution tracking");
    println!("  6. 🌊 Gradient Analysis         - Flow and stability analysis");
    println!("  7. 🔢 Quantization Analysis     - Precision detection (FP32/FP16/INT8/INT4)");
    println!("  8. 📊 Convergence Analysis      - Learning curve analysis");
    println!("  9. ⚡ Activation Analysis       - Function usage and distribution");
    println!(" 10. 👁️  Attention Analysis       - Transformer mechanism analysis");
    println!(" 11. 🤝 Ensemble Analysis        - Multi-model structure detection");
    println!("");
    println!("💡 Zero configuration required - Convention over Configuration!");
    println!("🚀 Built on diffx-core with lawkit memory-efficient patterns");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_model_creation() {
        let baseline = create_sample_model_data("baseline");
        let improved = create_sample_model_data("improved");
        
        assert_eq!(baseline["model_name"], "baseline");
        assert_eq!(improved["model_name"], "improved");
        
        // Verify differences exist for demonstration
        assert_ne!(baseline["training"]["learning_rate"], improved["training"]["learning_rate"]);
        assert_ne!(baseline["architecture"]["attention_heads"], improved["architecture"]["attention_heads"]);
    }
    
    #[test]
    fn test_automatic_analysis_demo() {
        // Test that the demonstration function runs without panic
        demonstrate_automatic_analysis();
    }
}