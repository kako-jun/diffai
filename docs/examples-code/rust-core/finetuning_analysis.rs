// Example: Fine-tuning Analysis using diffai v0.3.16
// 
// This example demonstrates how to use diffai's automatic comprehensive
// analysis to validate fine-tuning results. All 11 ML analysis functions
// run automatically following Convention over Configuration.

use anyhow::Result;
use diffai_core::{diff, DiffOptions, DiffResult};
use serde_json::Value;
use std::path::Path;

fn main() -> Result<()> {
    println!("🎯 diffai Example: Automatic Fine-tuning Analysis");
    println!("=================================================");
    println!("✨ Convention over Configuration - All 11 ML analyses run automatically");
    println!("");
    
    // Example paths for pre-trained and fine-tuned models
    let pretrained_path = Path::new("models/pretrained.safetensors");
    let finetuned_path = Path::new("models/finetuned.safetensors");
    
    if !pretrained_path.exists() || !finetuned_path.exists() {
        println!("⚠️  Model files not found. This example demonstrates:");
        println!("   - Automatic fine-tuning validation");
        println!("   - Learning progress analysis");
        println!("   - Training stability assessment");
        println!("");
        demonstrate_finetuning_analysis();
        return Ok(());
    }
    
    println!("🔬 Analyzing fine-tuning results with automatic comprehensive analysis:");
    println!("   Pre-trained: {}", pretrained_path.display());
    println!("   Fine-tuned:  {}", finetuned_path.display());
    println!("");
    
    // Create sample data for demonstration
    let pretrained_model = create_pretrained_model_data();
    let finetuned_model = create_finetuned_model_data();
    
    // Configure for automatic ML analysis (especially fine-tuning focused)
    let options = DiffOptions {
        ml_analysis_enabled: Some(true), // Auto-enabled for ML files
        use_memory_optimization: Some(true),
        verbose: Some(true),
        ..Default::default()
    };
    
    // Run automatic comprehensive fine-tuning analysis
    match diff(&pretrained_model, &finetuned_model, Some(&options)) {
        Ok(differences) => {
            analyze_finetuning_results(&differences);
        }
        Err(e) => {
            println!("❌ Error during automatic fine-tuning analysis: {}", e);
        }
    }
    
    Ok(())
}

fn analyze_finetuning_results(differences: &[DiffResult]) {
    println!("🎯 Automatic Fine-tuning Analysis Results:");
    println!("");
    
    let mut learning_indicators = 0;
    let mut architecture_stability = true;
    let mut training_progression = Vec::new();
    
    for diff in differences {
        match diff {
            DiffResult::WeightSignificantChange(path, magnitude, stats) => {
                learning_indicators += 1;
                println!("📊 Learning Progress Detected: {}", path);
                println!("   Weight Evolution: magnitude={:.6}", magnitude);
                println!("   Training Impact: mean_change={:.6}, std_change={:.6}", 
                    stats.mean_change, stats.std_change);
                
                // Analyze learning quality
                if magnitude < &0.01 {
                    training_progression.push("stable");
                } else if magnitude < &0.1 {
                    training_progression.push("moderate");
                } else {
                    training_progression.push("significant");
                }
            }
            
            DiffResult::ArchitectureChanged(old_arch, new_arch) => {
                architecture_stability = false;
                println!("⚠️  Architecture Change During Fine-tuning:");
                println!("   This is unusual for standard fine-tuning!");
                println!("   Old: {}", old_arch);
                println!("   New: {}", new_arch);
            }
            
            DiffResult::PrecisionChanged(path, old_prec, new_prec) => {
                println!("🔢 Precision Optimization: {}", path);
                println!("   {} → {} (Quantization during fine-tuning)", old_prec, new_prec);
            }
            
            _ => {}
        }
    }
    
    println!("");
    println!("🔬 Fine-tuning Quality Assessment:");
    
    if architecture_stability {
        println!("   ✅ Architecture Stability: Maintained (good fine-tuning practice)");
    } else {
        println!("   ⚠️  Architecture Changes: Detected (unusual for fine-tuning)");
    }
    
    let avg_progression = if training_progression.is_empty() {
        "none"
    } else {
        match training_progression.iter().filter(|&&x| x == "significant").count() {
            n if n > training_progression.len() / 2 => "aggressive",
            _ => match training_progression.iter().filter(|&&x| x == "moderate").count() {
                n if n > 0 => "balanced",
                _ => "conservative"
            }
        }
    };
    
    match avg_progression {
        "aggressive" => {
            println!("   🔥 Training Intensity: Aggressive");
            println!("      → High learning rate or many epochs");
            println!("      → Monitor for overfitting");
        }
        "balanced" => {
            println!("   ⚖️  Training Intensity: Balanced");
            println!("      → Optimal fine-tuning approach");
            println!("      → Good learning/stability trade-off");
        }
        "conservative" => {
            println!("   🛡️  Training Intensity: Conservative");
            println!("      → Low learning rate or few epochs");
            println!("      → May benefit from additional training");
        }
        _ => {
            println!("   🤔 Training Intensity: Minimal");
            println!("      → Very little change detected");
            println!("      → Consider increasing learning rate");
        }
    }
    
    println!("");
    println!("🎯 Automatic Analysis Summary:");
    println!("   Learning indicators: {}", learning_indicators);
    println!("   Architecture stability: {}", if architecture_stability { "✅" } else { "⚠️" });
    println!("   Training progression: {}", avg_progression);
    
    println!("");
    println!("💡 Fine-tuning Recommendations:");
    
    if learning_indicators == 0 {
        println!("   🤔 No significant learning detected");
        println!("   → Check learning rate and training duration");
        println!("   → Verify dataset quality and size");
    } else if learning_indicators < 5 {
        println!("   👍 Focused learning detected");
        println!("   → Specific layers/components updated");
        println!("   → Good for targeted fine-tuning");
    } else {
        println!("   🔥 Extensive learning detected");
        println!("   → Widespread parameter updates");
        println!("   → Monitor validation performance");
    }
    
    println!("");
    println!("🚀 All 11 ML analyses completed automatically for fine-tuning validation!");
}

fn create_pretrained_model_data() -> Value {
    serde_json::json!({
        "model_name": "pretrained_base",
        "version": "1.0",
        "training": {
            "learning_rate": 0.001,
            "optimizer": "Adam",
            "epoch": 0,
            "training_stage": "pretrained"
        },
        "performance": {
            "accuracy": 0.85,
            "loss": 0.45
        },
        "tensors": {
            "transformer.embeddings.word_embeddings.weight": {
                "shape": [30522, 768],
                "dtype": "float32",
                "mean": 0.0001,
                "std": 0.0512
            },
            "classifier.weight": {
                "shape": [2, 768],
                "dtype": "float32",
                "mean": 0.0023,
                "std": 0.0892
            }
        }
    })
}

fn create_finetuned_model_data() -> Value {
    serde_json::json!({
        "model_name": "finetuned_specialist",
        "version": "1.1",
        "training": {
            "learning_rate": 0.0001,  // Typical fine-tuning LR
            "optimizer": "Adam",
            "epoch": 5,
            "training_stage": "finetuned"
        },
        "performance": {
            "accuracy": 0.92,  // Improved after fine-tuning
            "loss": 0.21
        },
        "tensors": {
            "transformer.embeddings.word_embeddings.weight": {
                "shape": [30522, 768],
                "dtype": "float32",
                "mean": 0.0002,  // Slight change
                "std": 0.0515
            },
            "classifier.weight": {
                "shape": [2, 768],
                "dtype": "float32",
                "mean": 0.0156,  // Significant change (task-specific layer)
                "std": 0.1234
            }
        }
    })
}

fn demonstrate_finetuning_analysis() {
    println!("🎯 diffai's Automatic Fine-tuning Analysis:");
    println!("");
    println!("When you run: diffai pretrained.safetensors finetuned.safetensors");
    println!("");
    println!("All 11 analysis functions automatically assess fine-tuning quality:");
    println!("  1. 📈 Learning Rate Analysis     - Detects fine-tuning learning rates");
    println!("  2. ⚙️  Optimizer Comparison      - Tracks optimizer state evolution");
    println!("  3. 📉 Loss Tracking             - Monitors training convergence");
    println!("  4. 🎯 Accuracy Tracking         - Validates performance improvements");
    println!("  5. 🏷️  Model Version Analysis   - Tracks fine-tuning progression");
    println!("  6. 🌊 Gradient Analysis         - Ensures healthy gradient flow");
    println!("  7. 🔢 Quantization Analysis     - Detects precision optimizations");
    println!("  8. 📊 Convergence Analysis      - Assesses training stability");
    println!("  9. ⚡ Activation Analysis       - Monitors activation changes");
    println!(" 10. 👁️  Attention Analysis       - Tracks attention pattern shifts");
    println!(" 11. 🤝 Ensemble Analysis        - Detects ensemble fine-tuning");
    println!("");
    println!("🔬 Fine-tuning Specific Insights:");
    println!("   • Identifies which layers learned most");
    println!("   • Assesses learning intensity and quality");
    println!("   • Validates architecture stability");
    println!("   • Monitors for overfitting indicators");
    println!("");
    println!("💡 Zero configuration - Convention over Configuration for fine-tuning!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_data_creation() {
        let pretrained = create_pretrained_model_data();
        let finetuned = create_finetuned_model_data();
        
        assert_eq!(pretrained["training"]["training_stage"], "pretrained");
        assert_eq!(finetuned["training"]["training_stage"], "finetuned");
        
        // Verify fine-tuning characteristics
        assert!(finetuned["performance"]["accuracy"].as_f64().unwrap() > 
                pretrained["performance"]["accuracy"].as_f64().unwrap());
        assert!(finetuned["performance"]["loss"].as_f64().unwrap() < 
                pretrained["performance"]["loss"].as_f64().unwrap());
    }
}