use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML architecture comparison
/// Corresponds to: docs/examples/test-results/ml_architecture_comparison.md
#[test]
fn test_architecture_comparison() -> Result<(), Box<dyn std::error::Error>> {
    // Use real ML model files for testing architecture comparison
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/transformer.safetensors");

    let output = cmd.output()?;

    // Should process successfully with real ML files
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check that architecture comparison was performed (shows tensor changes and deployment readiness)
    assert!(stdout.contains("+") || stdout.contains("-"));
    assert!(stdout.contains("deployment_readiness"));

    Ok(())
}

/// Test deployment readiness analysis
#[test]
fn test_deployment_readiness() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;
    
    fs::create_dir_all("../tests/output")?;
    fs::write(
        "../tests/output/model_a.json",
        r#"{"status": "ready", "version": "1.0"}"#,
    )?;
    fs::write(
        "../tests/output/model_b.json",
        r#"{"status": "candidate", "version": "2.0"}"#,
    )?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/model_a.json")
        .arg("../tests/output/model_b.json");

    let output = cmd.output()?;

    // Should accept the flag and process successfully
    assert!(output.status.success());

    // Clean up
    let _ = fs::remove_file("../tests/output/model_a.json");
    let _ = fs::remove_file("../tests/output/model_b.json");

    Ok(())
}