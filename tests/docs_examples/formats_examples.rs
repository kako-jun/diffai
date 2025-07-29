use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test format support examples from formats.md
#[test]
fn test_all_supported_formats() -> Result<(), Box<dyn std::error::Error>> {
    let format_tests = [
        ("PyTorch", "tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("Safetensors", "tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
        ("NumPy", "tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
        ("MATLAB", "tests/fixtures/ml_models/matlab_data1.mat", "tests/fixtures/ml_models/matlab_data2.mat"),
    ];

    for (format_name, file1, file2) in format_tests.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed to handle {} format", format_name);
    }

    Ok(())
}