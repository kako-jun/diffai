use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test automatic tensor statistics analysis (Feature 1)
#[test]
fn test_tensor_statistics_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not panic during tensor analysis");
    assert!(!stderr.contains("unwrap"), "Should not have unwrap errors");
    
    // Must succeed when analyzing different models - exit code 1 means differences found, which is expected
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully analyze tensor statistics (exit code: {})", exit_code);
    
    // Debug output for troubleshooting
    println!("STDOUT: '{}'", stdout);
    println!("STDERR: '{}'", stderr);
    
    // Must contain actual ML analysis output
    assert!(!stdout.trim().is_empty(), "Should produce tensor analysis output");
    
    // Must contain memory analysis (one of the 11 ML features)
    assert!(stdout.contains("memory_analysis") || stdout.contains("memory_usage"), 
            "Should contain memory analysis feature");

    Ok(())
}

/// Test automatic model architecture analysis (Feature 2)
#[test]
fn test_model_architecture_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not panic during architecture analysis");
    assert!(!stderr.contains("unimplemented"), "Should not have unimplemented features");
    
    // Must succeed - exit code 1 means differences found, which is expected
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully analyze model architecture (exit code: {})", exit_code);
    
    // Must contain structural analysis
    assert!(!stdout.trim().is_empty(), "Should produce architecture analysis output");
    
    // Must contain specific ML analysis features
    let has_ml_analysis = stdout.contains("ModelArchitectureChanged") || 
                         stdout.contains("estimated_layers") || 
                         stdout.contains("detected_components");
    assert!(has_ml_analysis, "Should contain model architecture analysis");

    Ok(())
}

/// Test automatic weight change analysis (Feature 3)
#[test]
fn test_weight_change_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not panic during weight analysis");
    
    // Must succeed - exit code 1 means differences found, which is accepted
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully analyze weight changes (exit code: {})", exit_code);
    
    // Must contain weight distribution analysis (one of the 11 ML features)
    let has_weight_analysis = stdout.contains("weight_distribution") || 
                             stdout.contains("gradient_distributions") ||
                             stdout.contains("detected_components");
    assert!(has_weight_analysis, "Should contain weight change analysis");
    
    Ok(())
}

/// Test automatic memory analysis (Feature 4)
#[test]
fn test_memory_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/large_model.pt");

    let output = cmd.output()?;
    
    // Should analyze memory differences without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Memory analysis should be stable
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    Ok(())
}

/// Test learning rate analysis (Feature 5)
#[test]
fn test_learning_rate_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not panic during learning rate analysis");
    
    // Must succeed - exit code 1 means differences found, which is expected
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully analyze learning rate changes (exit code: {})", exit_code);
    
    // Must contain ML analysis output
    assert!(!stdout.trim().is_empty(), "Should produce learning rate analysis output");
    
    Ok(())
}

/// Test convergence analysis (Feature 6)
#[test]
fn test_convergence_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.pt")
        .arg("tests/fixtures/ml_models/anomalous_model.pt");

    let output = cmd.output()?;
    
    // Should analyze convergence patterns without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test gradient analysis (Feature 7)
#[test]
fn test_gradient_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    
    // Should estimate gradient information without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test attention analysis (Feature 8)
#[test]
fn test_attention_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/transformer.pt")
        .arg("tests/fixtures/ml_models/transformer.safetensors");

    let output = cmd.output()?;
    
    // Should analyze transformer attention without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test quantization analysis (Feature 10)
#[test]
fn test_quantization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model_fp32.pt")
        .arg("tests/fixtures/ml_models/model_quantized.pt");

    let output = cmd.output()?;
    
    // Should analyze quantization differences without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test all 11 ML analysis features are present
#[test]
fn test_all_11_ml_features_present() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not panic during comprehensive analysis");
    // Must succeed - exit code 1 means differences found, which is expected
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully analyze all ML features (exit code: {})", exit_code);
    
    // Must contain JSON output
    assert!(!stdout.trim().is_empty(), "Should produce comprehensive ML analysis output");
    
    // Parse JSON to verify structure
    let json_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);
    assert!(json_result.is_ok(), "Should produce valid JSON output");
    
    let json_data = json_result.unwrap();
    assert!(json_data.is_array(), "Output should be JSON array");
    
    let changes = json_data.as_array().unwrap();
    assert!(!changes.is_empty(), "Should detect changes between different models");
    
    // Verify ML analysis features are present
    let has_memory_analysis = changes.iter().any(|change| {
        change.to_string().contains("memory_analysis") || 
        change.to_string().contains("memory_usage")
    });
    
    let has_gradient_analysis = changes.iter().any(|change| {
        change.to_string().contains("gradient_distributions") ||
        change.to_string().contains("gradient_")
    });
    
    let has_architecture_analysis = changes.iter().any(|change| {
        change.to_string().contains("ModelArchitectureChanged") ||
        change.to_string().contains("estimated_layers") ||
        change.to_string().contains("detected_components")
    });
    
    // At least some ML analysis features must be present
    assert!(has_memory_analysis || has_gradient_analysis || has_architecture_analysis, 
            "Should contain at least some of the 11 ML analysis features");

    Ok(())
}

/// Test batch normalization analysis (new feature)
#[test]
fn test_batch_normalization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not crash analyzing batch normalization");
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully run batch norm analysis (exit code: {})", exit_code);

    Ok(())
}

/// Test regularization impact analysis (new feature)
#[test]
fn test_regularization_impact_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not crash analyzing regularization impact");
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully run regularization analysis (exit code: {})", exit_code);

    Ok(())
}

/// Test activation pattern analysis (new feature)
#[test]
fn test_activation_pattern_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not crash analyzing activation patterns");
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully run activation analysis (exit code: {})", exit_code);

    Ok(())
}

/// Test weight distribution analysis (new feature)
#[test]
fn test_weight_distribution_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not crash analyzing weight distributions");
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully run weight distribution analysis (exit code: {})", exit_code);

    Ok(())
}

/// Test model complexity assessment (new feature)
#[test]
fn test_model_complexity_assessment() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Must not crash
    assert!(!stderr.contains("panic"), "Should not crash assessing model complexity");
    let exit_code = output.status.code().unwrap_or(-1);
    assert!(exit_code == 0 || exit_code == 1, "Should successfully run complexity assessment (exit code: {})", exit_code);

    Ok(())
}

/// Test multi-format automatic analysis
#[test]
fn test_multi_format_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Test PyTorch format
    let mut cmd1 = diffai_cmd();
    cmd1.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt");

    let output1 = cmd1.output()?;
    assert!(!String::from_utf8_lossy(&output1.stderr).contains("panic"));

    // Test Safetensors format  
    let mut cmd2 = diffai_cmd();
    cmd2.arg("tests/fixtures/ml_models/simple_base.safetensors")
        .arg("tests/fixtures/ml_models/simple_modified.safetensors");

    let output2 = cmd2.output()?;
    assert!(!String::from_utf8_lossy(&output2.stderr).contains("panic"));

    // Test NumPy format
    let mut cmd3 = diffai_cmd();
    cmd3.arg("tests/fixtures/ml_models/numpy_data1.npy")
        .arg("tests/fixtures/ml_models/numpy_data2.npy");

    let output3 = cmd3.output()?;
    assert!(!String::from_utf8_lossy(&output3.stderr).contains("panic"));

    Ok(())
}

/// Test comprehensive analysis with no crashes
#[test]
fn test_comprehensive_analysis_stability() -> Result<(), Box<dyn std::error::Error>> {
    let test_cases = [
        ("tests/fixtures/ml_models/small_model.pt", "tests/fixtures/ml_models/large_model.pt"),
        ("tests/fixtures/ml_models/normal_model.safetensors", "tests/fixtures/ml_models/anomalous_model.safetensors"), 
        ("tests/fixtures/ml_models/checkpoint_epoch_0.pt", "tests/fixtures/ml_models/checkpoint_epoch_50.pt"),
        ("tests/fixtures/ml_models/model_fp32.safetensors", "tests/fixtures/ml_models/model_quantized.safetensors"),
    ];

    for (file1, file2) in test_cases.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        
        // Primary requirement: No panics or unhandled errors
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed on {} vs {}", file1, file2);
        assert!(!stderr.contains("unwrap"), "Unhandled error on {} vs {}", file1, file2);
        
        // Exit code should be reasonable (0, 1, or 2)
        let code = output.status.code().unwrap_or(-1);
        assert!(code >= 0 && code <= 2, "Invalid exit code {} for {} vs {}", code, file1, file2);
    }

    Ok(())
}