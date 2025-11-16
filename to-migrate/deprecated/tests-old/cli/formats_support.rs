use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test PyTorch format support (.pt files)
#[test]
fn test_pytorch_format_support() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    
    // Should handle PyTorch files without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    assert!(!stderr.contains("unimplemented"));
    
    // Exit code should be 0 (no diff) or 1 (diff found)
    assert!(matches!(output.status.code(), Some(0) | Some(1)));

    Ok(())
}

/// Test Safetensors format support (.safetensors files)
#[test]
fn test_safetensors_format_support() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.safetensors")
        .arg("tests/fixtures/ml_models/model_fp32.safetensors");

    let output = cmd.output()?;
    
    // Should handle Safetensors files without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    assert!(!stderr.contains("unimplemented"));
    
    // Exit code should be 0 (no diff) or 1 (diff found)
    assert!(matches!(output.status.code(), Some(0) | Some(1)));

    Ok(())
}

/// Test NumPy format support (.npy files)
#[test]
fn test_numpy_format_support() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_data1.npy")
        .arg("tests/fixtures/ml_models/numpy_data2.npy");

    let output = cmd.output()?;
    
    // Should handle NumPy files without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be valid
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test NumPy compressed format support (.npz files)
#[test]
fn test_numpy_compressed_format_support() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_model1.npz")
        .arg("tests/fixtures/ml_models/numpy_model2.npz");

    let output = cmd.output()?;
    
    // Should handle NumPy compressed files without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be valid
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test MATLAB format support (.mat files)
#[test]
fn test_matlab_format_support() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/matlab_data1.mat")
        .arg("tests/fixtures/ml_models/matlab_data2.mat");

    let output = cmd.output()?;
    
    // Should handle MATLAB files without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be valid
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test cross-format comparison (should work or fail gracefully)
#[test]
fn test_cross_format_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/small_model.safetensors");

    let output = cmd.output()?;
    
    // Should handle cross-format comparison gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test explicit format specification for PyTorch
#[test]
fn test_explicit_pytorch_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--format")
        .arg("pytorch");

    let output = cmd.output()?;
    
    // Should handle explicit format specification
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test explicit format specification for Safetensors
#[test]
fn test_explicit_safetensors_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.safetensors")
        .arg("tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--format")
        .arg("safetensors");

    let output = cmd.output()?;
    
    // Should handle explicit format specification
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test explicit format specification for NumPy
#[test]
fn test_explicit_numpy_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_small1.npy")
        .arg("tests/fixtures/ml_models/numpy_small2.npy")
        .arg("--format")
        .arg("numpy");

    let output = cmd.output()?;
    
    // Should handle explicit format specification
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test explicit format specification for MATLAB
#[test]
fn test_explicit_matlab_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/matlab_simple1.mat")
        .arg("tests/fixtures/ml_models/matlab_simple2.mat")
        .arg("--format")
        .arg("matlab");

    let output = cmd.output()?;
    
    // Should handle explicit format specification
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    Ok(())
}

/// Test invalid format specification
#[test]
fn test_invalid_format_specification() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--format")
        .arg("invalid_format");

    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("invalid").or(predicates::str::contains("Invalid")));

    Ok(())
}