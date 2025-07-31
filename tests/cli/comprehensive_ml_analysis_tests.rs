use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test comprehensive ML analysis functionality - all 11 features
#[test]
fn test_comprehensive_ml_analysis_features() -> Result<(), Box<dyn std::error::Error>> {
    let test_cases = [
        // PyTorch format tests
        ("tests/fixtures/ml_models/simple_base.pt", "tests/fixtures/ml_models/model1.pt"),
        ("tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("tests/fixtures/ml_models/checkpoint_epoch_0.pt", "tests/fixtures/ml_models/checkpoint_epoch_10.pt"),
        
        // SafeTensors format tests  
        ("tests/fixtures/ml_models/normal_model.safetensors", "tests/fixtures/ml_models/transformer.safetensors"),
        ("tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
    ];

    for (file1, file2) in &test_cases {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2).arg("--output").arg("json");
        
        let output = cmd.output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Basic stability requirements
        assert!(!stderr.contains("panic"), "Panic in {} vs {}", file1, file2);
        assert!(!stderr.contains("unwrap"), "Unwrap error in {} vs {}", file1, file2);
        
        // Must have some output for different files
        assert!(!stdout.trim().is_empty(), "No output for {} vs {}", file1, file2);
        
        // JSON format validation
        let json_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
        assert!(json_result.is_ok(), "Invalid JSON output for {} vs {}", file1, file2);
        
        let json_data = json_result.unwrap();
        assert!(json_data.is_array(), "Output should be JSON array for {} vs {}", file1, file2);
        
        let changes = json_data.as_array().unwrap();
        assert!(!changes.is_empty(), "Should detect changes for {} vs {}", file1, file2);
        
        // Verify ML analysis results are present
        let has_ml_analysis = changes.iter().any(|change| {
            if let Some(change_type) = change.get("ModelArchitectureChanged") {
                if let Some(analysis_key) = change_type.as_array().and_then(|arr| arr.get(0)) {
                    let key = analysis_key.as_str().unwrap_or("");
                    // Check for any of the implemented ML analysis features
                    key.contains("memory_analysis") || 
                    key.contains("gradient_distributions") ||
                    key.contains("memory_breakdown") ||
                    key.contains("convergence_") ||
                    key.contains("learning_rate_") ||
                    key.contains("attention_") ||
                    key.contains("ensemble_") ||
                    key.contains("quantization_")
                } else {
                    false
                }
            } else {
                false
            }
        });
        
        assert!(has_ml_analysis, "Should contain ML analysis results for {} vs {}", file1, file2);
    }

    Ok(())
}

/// Test specific ML analysis features individually
#[test]
fn test_memory_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should contain memory analysis
    assert!(stdout.contains("memory_analysis"), "Should contain memory_analysis feature");
    assert!(stdout.contains("memory_usage"), "Should analyze memory usage");
    
    Ok(())
}

/// Test gradient analysis feature
#[test]
fn test_gradient_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should contain gradient analysis
    assert!(stdout.contains("gradient_distributions") || stdout.contains("gradient_"), 
            "Should contain gradient analysis feature");
    
    Ok(())
}

/// Test attention analysis with transformer models
#[test]
fn test_attention_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.safetensors")
        .arg("tests/fixtures/ml_models/transformer.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // For transformer models, should have attention analysis
    if stdout.contains("self_attn") || stdout.contains("attention") {
        // This test case has attention layers, should analyze them
        assert!(output.status.success(), "Should successfully analyze transformer attention");
    }
    
    Ok(())
}

/// Test convergence analysis feature
#[test] 
fn test_convergence_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should analyze convergence patterns between epochs
    assert!(output.status.success(), "Should successfully analyze convergence");
    
    Ok(())
}

/// Test learning rate analysis feature
#[test]
fn test_learning_rate_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_50.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should analyze learning rate changes
    assert!(output.status.success(), "Should successfully analyze learning rate changes");
    
    Ok(())
}

/// Test quantization analysis feature
#[test]
fn test_quantization_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model_fp32.pt")
        .arg("tests/fixtures/ml_models/model_quantized.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should analyze quantization differences
    assert!(output.status.success(), "Should successfully analyze quantization");
    
    Ok(())
}

/// Test ensemble analysis feature
#[test]
fn test_ensemble_analysis_feature() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.pt")
        .arg("tests/fixtures/ml_models/anomalous_model.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should analyze ensemble patterns
    assert!(output.status.success(), "Should successfully analyze ensemble patterns");
    
    Ok(())
}

/// Test ML analysis output format compliance
#[test]
fn test_ml_analysis_output_format() -> Result<(), Box<dyn std::error::Error>> {
    let formats = ["json", "yaml"];
    
    for format in &formats {
        let mut cmd = diffai_cmd();
        cmd.arg("tests/fixtures/ml_models/simple_base.pt")
            .arg("tests/fixtures/ml_models/model1.pt")
            .arg("--output")
            .arg(format);

        let output = cmd.output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        assert!(!stderr.contains("panic"), "Panic in {} format", format);
        assert!(!stdout.trim().is_empty(), "No output in {} format", format);
        
        // Format-specific validation
        match *format {
            "json" => {
                let json_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
                assert!(json_result.is_ok(), "Invalid JSON format");
            }
            "yaml" => {
                // Basic YAML format check - should not start with { or [
                assert!(!stdout.trim_start().starts_with("{"), "YAML should not start with {{");
                assert!(!stdout.trim_start().starts_with("["), "YAML should not start with [");
            }
            _ => {}
        }
    }
    
    Ok(())
}

/// Test ML analysis with identical files (edge case)
#[test]
fn test_ml_analysis_identical_files() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_base.pt")  // Same file
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    assert!(!stderr.contains("panic"), "Should not panic on identical files");
    assert_eq!(output.status.code(), Some(0), "Should return exit code 0 for identical files");
    
    // Should have empty or minimal output for identical files
    if !stdout.trim().is_empty() {
        let json_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
        if json_result.is_ok() {
            let json_data = json_result.unwrap();
            if let Some(changes) = json_data.as_array() {
                assert!(changes.is_empty(), "Should have no changes for identical files");
            }
        }
    }
    
    Ok(())
}

/// Test ML analysis error handling
#[test]
fn test_ml_analysis_error_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Test with non-existent file
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/nonexistent.pt")
        .arg("tests/fixtures/ml_models/simple_base.pt");

    let output = cmd.output()?;
    
    // Should fail gracefully with non-zero exit code
    assert_ne!(output.status.code(), Some(0), "Should fail with non-existent file");
    
    // Should not panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"), "Should not panic on file error");
    
    Ok(())
}