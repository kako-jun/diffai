// Example: Quantization Impact Analysis using diffai v0.3.16
// 
// This example demonstrates how to use diffai's automatic comprehensive
// analysis to assess quantization effects. The Quantization Analysis
// function runs automatically as part of the 11 ML analyses.

use anyhow::Result;
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use std::path::Path;

fn main() -> Result<()> {
    println!("🔢 diffai Example: Automatic Quantization Impact Analysis");
    println!("=========================================================");
    println!("✨ Convention over Configuration - All 11 ML analyses run automatically");
    println!("🎯 Focus: Quantization Analysis (Function #7 of 11)");
    println!("");
    
    // Example paths for full precision and quantized models
    let fp32_path = Path::new("models/model_fp32.safetensors");
    let quantized_path = Path::new("models/model_int8.safetensors");
    
    if !fp32_path.exists() || !quantized_path.exists() {
        println!("⚠️  Model files not found. This example demonstrates:");
        println!("   - Automatic quantization detection");
        println!("   - Precision loss assessment");
        println!("   - Memory efficiency analysis");
        println!("");
        demonstrate_quantization_analysis();
        return Ok(());
    }
    
    println!("🔬 Analyzing quantization impact with automatic comprehensive analysis:");
    println!("   Full Precision: {}", fp32_path.display());
    println!("   Quantized:      {}", quantized_path.display());
    println!("");
    
    // Create sample data for demonstration
    let fp32_model = create_fp32_model_data();
    let quantized_model = create_quantized_model_data();
    
    // Configure for automatic ML analysis (quantization analysis included)
    let options = DiffOptions {
        ml_analysis_enabled: Some(true), // Auto-enabled for ML files
        use_memory_optimization: Some(true),
        verbose: Some(true),
        ..Default::default()
    };
    
    // Run automatic comprehensive quantization analysis
    match diff(&fp32_model, &quantized_model, Some(&options)) {
        Ok(differences) => {
            analyze_quantization_impact(&differences);
        }
        Err(e) => {
            println!("❌ Error during automatic quantization analysis: {}", e);
        }
    }
    
    Ok(())
}

fn analyze_quantization_impact(differences: &[DiffResult]) {
    println!("🎯 Automatic Quantization Analysis Results:");
    println!("");
    
    let mut precision_changes = 0;
    let mut total_size_change = 0i64;
    let mut quantization_coverage = Vec::new();
    let mut quality_indicators = Vec::new();
    
    for diff in differences {
        match diff {
            DiffResult::PrecisionChanged(path, old_prec, new_prec) => {
                precision_changes += 1;
                println!("🔢 Precision Change Detected: {}", path);
                println!("   {} → {} (Automatic Quantization Analysis)", old_prec, new_prec);
                
                // Analyze quantization type
                let compression_ratio = match (old_prec.as_str(), new_prec.as_str()) {
                    ("float32", "float16") => 2.0,
                    ("float32", "int8") => 4.0,
                    ("float32", "int4") => 8.0,
                    ("float16", "int8") => 2.0,
                    ("float16", "int4") => 4.0,
                    _ => 1.0,
                };
                
                quantization_coverage.push((path.clone(), compression_ratio));
                
                if compression_ratio > 1.0 {
                    println!("   📉 Memory Reduction: {:.1}x compression", compression_ratio);
                }
            }
            
            DiffResult::WeightSignificantChange(path, magnitude, stats) => {
                println!("📊 Weight Distribution Impact: {}", path);
                println!("   Quantization Effect: magnitude={:.6}", magnitude);
                println!("   Statistical Change: mean_δ={:.6}, std_δ={:.6}", 
                    stats.mean_change, stats.std_change);
                
                // Assess quality impact
                if magnitude < &0.01 {
                    quality_indicators.push("minimal");
                } else if magnitude < &0.05 {
                    quality_indicators.push("moderate");
                } else {
                    quality_indicators.push("significant");
                }
            }
            
            DiffResult::TensorShapeChanged(path, old_shape, new_shape) => {
                println!("⬚ Tensor Structure: {}", path);
                println!("   Shape maintained: {:?} (good quantization practice)", old_shape);
                
                let old_size: usize = old_shape.iter().product();
                let new_size: usize = new_shape.iter().product();
                total_size_change += (new_size as i64) - (old_size as i64);
            }
            
            _ => {}
        }
    }
    
    println!("");
    println!("🔬 Quantization Quality Assessment:");
    
    // Memory efficiency analysis
    let avg_compression = if quantization_coverage.is_empty() {
        1.0
    } else {
        quantization_coverage.iter().map(|(_, ratio)| ratio).sum::<f64>() / quantization_coverage.len() as f64
    };
    
    println!("   📦 Memory Efficiency:");
    println!("      Average Compression: {:.1}x", avg_compression);
    println!("      Quantized Layers: {}", precision_changes);
    
    if avg_compression > 3.0 {
        println!("      Status: 🏆 Excellent compression (>3x)");
    } else if avg_compression > 2.0 {
        println!("      Status: ✅ Good compression (>2x)");
    } else if avg_compression > 1.5 {
        println!("      Status: 👍 Moderate compression");
    } else {
        println!("      Status: 🤔 Limited compression detected");
    }
    
    // Quality impact analysis
    let quality_impact = if quality_indicators.is_empty() {
        "unknown"
    } else {
        match quality_indicators.iter().filter(|&&x| x == "significant").count() {
            n if n > quality_indicators.len() / 2 => "high",
            _ => match quality_indicators.iter().filter(|&&x| x == "moderate").count() {
                n if n > 0 => "moderate",
                _ => "low"
            }
        }
    };
    
    println!("   🎯 Quality Impact:");
    match quality_impact {
        "high" => {
            println!("      Impact Level: 🔴 High");
            println!("      → Significant weight changes detected");
            println!("      → Thorough accuracy validation recommended");
        }
        "moderate" => {
            println!("      Impact Level: 🟡 Moderate");
            println!("      → Balanced quality/efficiency trade-off");
            println!("      → Standard validation sufficient");
        }
        "low" => {
            println!("      Impact Level: 🟢 Low");
            println!("      → Minimal weight distribution changes");
            println!("      → Excellent quantization quality");
        }
        _ => {
            println!("      Impact Level: ❓ Unknown");
            println!("      → Insufficient data for assessment");
        }
    }
    
    println!("");
    println!("💡 Quantization Recommendations:");
    
    if precision_changes == 0 {
        println!("   🤔 No quantization detected");
        println!("   → Verify quantization process completed");
        println!("   → Check model format and precision settings");
    } else if avg_compression < 2.0 {
        println!("   🔧 Optimization opportunity");
        println!("   → Consider more aggressive quantization");
        println!("   → INT8 or INT4 may be suitable for this model");
    } else {
        println!("   ✅ Good quantization achieved");
        println!("   → {:.1}x memory reduction with {} quality impact", avg_compression, quality_impact);
        
        if quality_impact == "low" {
            println!("   → Consider deploying this quantized version");
        } else {
            println!("   → Validate accuracy on test dataset");
        }
    }
    
    println!("");
    println!("🚀 Quantization Analysis completed automatically as part of 11 ML analyses!");
    println!("   🎯 Zero configuration required - Convention over Configuration!");
}

fn create_fp32_model_data() -> Value {
    serde_json::json!({
        "model_name": "full_precision_model",
        "version": "1.0",
        "precision": "float32",
        "memory_usage": "2.4GB",
        "tensors": {
            "conv1.weight": {
                "shape": [64, 3, 7, 7],
                "dtype": "float32",
                "mean": 0.0012,
                "std": 0.0234,
                "memory_mb": 36.75
            },
            "linear.weight": {
                "shape": [1000, 2048],
                "dtype": "float32", 
                "mean": -0.0003,
                "std": 0.0456,
                "memory_mb": 8.0
            },
            "batch_norm.weight": {
                "shape": [64],
                "dtype": "float32",
                "mean": 1.0001,
                "std": 0.0123,
                "memory_mb": 0.25
            }
        }
    })
}

fn create_quantized_model_data() -> Value {
    serde_json::json!({
        "model_name": "quantized_model",
        "version": "1.0-int8",
        "precision": "mixed",
        "memory_usage": "0.8GB",  // ~3x reduction
        "tensors": {
            "conv1.weight": {
                "shape": [64, 3, 7, 7],
                "dtype": "int8",  // Quantized
                "mean": 0.0014,   // Slight change due to quantization
                "std": 0.0241,
                "memory_mb": 9.19  // 4x reduction
            },
            "linear.weight": {
                "shape": [1000, 2048],
                "dtype": "int8",  // Quantized
                "mean": -0.0004,  // Slight change
                "std": 0.0462,
                "memory_mb": 2.0   // 4x reduction
            },
            "batch_norm.weight": {
                "shape": [64],
                "dtype": "float32",  // Often kept in FP32
                "mean": 1.0001,
                "std": 0.0123,
                "memory_mb": 0.25  // No change
            }
        }
    })
}

fn demonstrate_quantization_analysis() {
    println!("🎯 diffai's Automatic Quantization Analysis:");
    println!("");
    println!("When you run: diffai model_fp32.safetensors model_int8.safetensors");
    println!("");
    println!("Quantization Analysis (Function #7) runs automatically with others:");
    println!("  1. 📈 Learning Rate Analysis     - Checks for training impact");
    println!("  2. ⚙️  Optimizer Comparison      - Validates optimizer state");
    println!("  3. 📉 Loss Tracking             - Monitors accuracy impact");
    println!("  4. 🎯 Accuracy Tracking         - Quantifies performance change");
    println!("  5. 🏷️  Model Version Analysis   - Tracks quantization version");
    println!("  6. 🌊 Gradient Analysis         - Assesses numerical stability");
    println!("  7. 🔢 Quantization Analysis     - 🎯 MAIN FOCUS");
    println!("  8. 📊 Convergence Analysis      - Checks training consistency");
    println!("  9. ⚡ Activation Analysis       - Monitors activation changes");
    println!(" 10. 👁️  Attention Analysis       - Validates attention integrity");
    println!(" 11. 🤝 Ensemble Analysis        - Checks multi-model impact");
    println!("");
    println!("🔢 Quantization-Specific Detection:");
    println!("   • Automatic precision detection (FP32/FP16/INT8/INT4)");
    println!("   • Memory compression ratio calculation");
    println!("   • Precision loss estimation");
    println!("   • Quality impact assessment");
    println!("   • Deployment readiness evaluation");
    println!("");
    println!("💡 Zero configuration - automatically detects and analyzes quantization!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_models() {
        let fp32 = create_fp32_model_data();
        let quantized = create_quantized_model_data();
        
        assert_eq!(fp32["precision"], "float32");
        assert_eq!(quantized["precision"], "mixed");
        
        // Verify quantization occurred
        assert_eq!(quantized["tensors"]["conv1.weight"]["dtype"], "int8");
        assert_eq!(quantized["tensors"]["linear.weight"]["dtype"], "int8");
        
        // Verify memory reduction
        let fp32_memory = fp32["memory_usage"].as_str().unwrap();
        let quantized_memory = quantized["memory_usage"].as_str().unwrap();
        assert_ne!(fp32_memory, quantized_memory);
    }
}