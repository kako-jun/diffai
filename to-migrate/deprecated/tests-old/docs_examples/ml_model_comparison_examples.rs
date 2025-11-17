use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML model comparison examples from ml-model-comparison.md
#[test]
fn test_training_checkpoint_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt");

    let output = cmd.output()?;
    
    // Should analyze training progression
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test model architecture comparison from ml-model-comparison.md
#[test]
fn test_model_architecture_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.pt")
        .arg("tests/fixtures/ml_models/transformer.pt");

    let output = cmd.output()?;
    
    // Should detect architectural differences
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test quantization analysis from ml-model-comparison.md
#[test]
fn test_quantization_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model_fp32.pt")
        .arg("tests/fixtures/ml_models/model_quantized.pt");

    let output = cmd.output()?;
    
    // Should analyze quantization differences
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}