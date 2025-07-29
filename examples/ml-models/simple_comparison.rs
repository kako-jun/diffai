// Example: Simple ML model comparison using diffai
// 
// This example demonstrates basic usage of diffai for comparing
// machine learning models, specifically focusing on tensor statistics
// and shape differences.

use anyhow::Result;
use diffai_core::{diff_ml_models, DiffResult};
use std::path::Path;

fn main() -> Result<()> {
    println!("🤖 diffai Example: Simple Model Comparison");
    println!("==========================================");
    
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
        return Ok(());
    }
    
    println!("📊 Comparing models:");
    println!("   Model 1: {}", model1_path.display());
    println!("   Model 2: {}", model2_path.display());
    println!("");
    
    // Set epsilon tolerance for floating-point comparisons
    let epsilon = Some(1e-6);
    
    // Perform model comparison
    match diff_ml_models(model1_path, model2_path, epsilon) {
        Ok(differences) => {
            if differences.is_empty() {
                println!("✅ No significant differences found between models!");
                println!("   (within epsilon tolerance: {:?})", epsilon);
            } else {
                println!("🔍 Found {} differences:", differences.len());
                println!("");
                
                analyze_differences(&differences);
            }
        }
        Err(e) => {
            println!("❌ Error comparing models: {}", e);
            println!("");
            println!("💡 Common issues:");
            println!("   - File format not supported (use .safetensors or .pt/.pth)");
            println!("   - Corrupted model files");
            println!("   - Insufficient memory for large models");
        }
    }
    
    Ok(())
}

fn analyze_differences(differences: &[DiffResult]) {
    let mut tensor_stats_changes = 0;
    let mut tensor_shape_changes = 0;
    let mut tensor_additions = 0;
    let mut tensor_removals = 0;
    
    for (i, diff) in differences.iter().enumerate() {
        match diff {
            DiffResult::TensorStatsChanged(name, stats1, stats2) => {
                tensor_stats_changes += 1;
                println!("📊 Tensor Statistics Changed: {}", name);
                println!("   Mean:     {:.6} → {:.6} (Δ: {:+.6})", 
                    stats1.mean, stats2.mean, stats2.mean - stats1.mean);
                println!("   Std Dev:  {:.6} → {:.6} (Δ: {:+.6})", 
                    stats1.std, stats2.std, stats2.std - stats1.std);
                println!("   Min:      {:.6} → {:.6} (Δ: {:+.6})", 
                    stats1.min, stats2.min, stats2.min - stats1.min);
                println!("   Max:      {:.6} → {:.6} (Δ: {:+.6})", 
                    stats1.max, stats2.max, stats2.max - stats1.max);
                println!("   Shape:    {:?} ({})", stats1.shape, format_shape_info(&stats1.shape));
                println!("   Params:   {}", stats1.total_params);
            }
            
            DiffResult::TensorShapeChanged(name, shape1, shape2) => {
                tensor_shape_changes += 1;
                println!("⬚ Tensor Shape Changed: {}", name);
                println!("   Old shape: {:?} ({})", shape1, format_shape_info(shape1));
                println!("   New shape: {:?} ({})", shape2, format_shape_info(shape2));
                let params1: usize = shape1.iter().product();
                let params2: usize = shape2.iter().product();
                println!("   Parameters: {} → {} (Δ: {:+})", params1, params2, params2 as i64 - params1 as i64);
            }
            
            DiffResult::Added(name, _) => {
                tensor_additions += 1;
                println!("+ Tensor Added: {}", name);
                if name.starts_with("tensor.") {
                    let tensor_name = &name[7..]; // Remove "tensor." prefix
                    println!("   New layer or parameter: {}", tensor_name);
                }
            }
            
            DiffResult::Removed(name, _) => {
                tensor_removals += 1;
                println!("- Tensor Removed: {}", name);
                if name.starts_with("tensor.") {
                    let tensor_name = &name[7..]; // Remove "tensor." prefix
                    println!("   Removed layer or parameter: {}", tensor_name);
                }
            }
            
            DiffResult::Modified(name, _, _) => {
                println!("~ Modified: {}", name);
            }
            
            DiffResult::TypeChanged(name, _, _) => {
                println!("! Type Changed: {}", name);
            }
            
            DiffResult::ModelArchitectureChanged(name, arch1, arch2) => {
                println!("🏗️ Architecture Changed: {}", name);
                println!("   Total parameters: {} → {}", arch1.total_parameters, arch2.total_parameters);
                println!("   Layer count: {} → {}", arch1.layer_count, arch2.layer_count);
                println!("   Model size: {} → {} bytes", arch1.model_size_bytes, arch2.model_size_bytes);
            }
        }
        
        if i < differences.len() - 1 {
            println!("");
        }
    }
    
    // Summary
    println!("");
    println!("📋 Summary:");
    println!("   Tensor statistics changes: {}", tensor_stats_changes);
    println!("   Tensor shape changes:      {}", tensor_shape_changes);
    println!("   Tensors added:             {}", tensor_additions);
    println!("   Tensors removed:           {}", tensor_removals);
    
    // Analysis insights
    println!("");
    println!("💡 Analysis Insights:");
    
    if tensor_shape_changes > 0 {
        println!("   ⚠️  Shape changes detected - this likely indicates architectural modifications");
    }
    
    if tensor_additions > 0 || tensor_removals > 0 {
        println!("   ⚠️  Layer additions/removals detected - significant model changes");
    }
    
    if tensor_stats_changes > 0 && tensor_shape_changes == 0 && tensor_additions == 0 && tensor_removals == 0 {
        println!("   ✅ Only statistical changes - likely fine-tuning or continued training");
    }
    
    if differences.is_empty() {
        println!("   ✅ Models are functionally identical within the specified tolerance");
    }
}

fn format_shape_info(shape: &[usize]) -> String {
    let total_params: usize = shape.iter().product();
    match shape.len() {
        0 => "scalar".to_string(),
        1 => format!("vector, {total_params} elements"),
        2 => format!("matrix, {}×{} = {total_params} elements", shape[0], shape[1]),
        3 => format!("3D tensor, {}×{}×{} = {total_params} elements", shape[0], shape[1], shape[2]),
        4 => format!("4D tensor, {}×{}×{}×{} = {total_params} elements", shape[0], shape[1], shape[2], shape[3]),
        _ => format!("{}D tensor, {total_params} elements", shape.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_shape_info() {
        assert_eq!(format_shape_info(&[]), "scalar");
        assert_eq!(format_shape_info(&[10]), "vector, 10 elements");
        assert_eq!(format_shape_info(&[3, 4]), "matrix, 3×4 = 12 elements");
        assert_eq!(format_shape_info(&[2, 3, 4]), "3D tensor, 2×3×4 = 24 elements");
        assert_eq!(format_shape_info(&[1, 2, 3, 4]), "4D tensor, 1×2×3×4 = 24 elements");
        assert_eq!(format_shape_info(&[1, 2, 3, 4, 5]), "5D tensor, 120 elements");
    }
}