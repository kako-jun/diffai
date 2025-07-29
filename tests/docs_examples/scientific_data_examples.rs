use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test scientific data examples from scientific-data.md
#[test]
fn test_numpy_scientific_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_data1.npy")
        .arg("tests/fixtures/ml_models/numpy_data2.npy");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test MATLAB scientific data from scientific-data.md  
#[test]
fn test_matlab_scientific_data() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/matlab_data1.mat")
        .arg("tests/fixtures/ml_models/matlab_data2.mat");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}