use assert_cmd::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML memory analysis
/// Corresponds to: docs/examples/test-results/ml_memory_analysis.md
#[test]
fn test_memory_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Use real ML model files for testing memory analysis
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("../tests/fixtures/ml_models/large_model.safetensors");

    let output = cmd.output()?;

    // Should process successfully with real ML files
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that memory analysis was performed (shows tensor changes and review friendly output)
    assert!(stdout.contains("+") || stdout.contains("-"));
    assert!(stdout.contains("review_friendly"));

    Ok(())
}
