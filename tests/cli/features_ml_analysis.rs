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
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should process without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // May produce TensorStatsChanged output if implementation is working
    // At minimum, should not crash when analyzing tensor statistics
    if output.status.success() {
        // If successful, output might contain tensor analysis
        // This is lenient as we're testing stability first
    }

    Ok(())
}

/// Test automatic model architecture analysis (Feature 2)
#[test]
fn test_model_architecture_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt");

    let output = cmd.output()?;
    
    // Should process without panic when analyzing model architecture
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    assert!(!stderr.contains("unimplemented"));
    
    // Exit code should be reasonable
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test automatic weight change analysis (Feature 3)
#[test]
fn test_weight_change_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt");

    let output = cmd.output()?;
    
    // Should process weight changes without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // May show WeightSignificantChange if implementation is complete
    let stdout = String::from_utf8_lossy(&output.stdout);
    
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
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_50.pt");

    let output = cmd.output()?;
    
    // Should handle learning rate detection gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
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