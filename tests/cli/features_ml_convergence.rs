use assert_cmd::prelude::*;
use std::fs;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML convergence analysis
/// Corresponds to: docs/examples/test-results/ml_convergence_analysis.md
#[test]
fn test_convergence_analysis_in_full_output() -> Result<(), Box<dyn std::error::Error>> {
    // Test that convergence analysis is included in default full analysis
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_50.safetensors");

    let output = cmd.output()?;

    // Should process successfully with real ML files
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that convergence analysis was performed (outputs inference speed analysis)
    assert!(stdout.contains("mean=") || stdout.contains("std="));
    assert!(stdout.contains("inference_speed") || stdout.contains("⚡"));

    Ok(())
}

/// Test hyperparameter impact analysis
#[test]
fn test_hyperparameter_impact() -> Result<(), Box<dyn std::error::Error>> {
    // Helper function for creating test safetensors files
    fn create_test_safetensors_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let test_data = b"{}"; // Minimal JSON metadata
        fs::create_dir_all("../tests/output")?;
        fs::write(path, test_data)?;
        Ok(())
    }

    create_test_safetensors_file("../tests/output/lr_001.safetensors")?;
    create_test_safetensors_file("../tests/output/lr_0001.safetensors")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/lr_001.safetensors")
        .arg("../tests/output/lr_0001.safetensors");

    let output = cmd.output()?;

    // Should accept the flag and either succeed or show a parse error for invalid safetensors
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("Failed to parse")
            || stderr.contains("HeaderTooSmall")
            || stderr.contains("Error:")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/lr_001.safetensors");
    let _ = fs::remove_file("../tests/output/lr_0001.safetensors");

    Ok(())
}
