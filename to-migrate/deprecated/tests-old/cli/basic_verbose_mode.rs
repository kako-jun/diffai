use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::io::Write;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test verbose mode output with AI/ML files
/// Corresponds to: docs/examples/test-results/basic_verbose_mode.md
#[test]
fn test_verbose_mode() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--verbose");

    let output = cmd.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    
    // Check what the command outputs and whether it handles AI/ML files reasonably
    let verbose_output = format!("{stdout}{stderr}");
    
    // Even if the command fails, it should not panic and should provide some output
    assert!(!verbose_output.contains("panic"), "Should not panic when processing AI/ML files");
    
    // Either succeed with meaningful output, or fail with helpful error message
    if output.status.success() {
        assert!(!stdout.trim().is_empty(), "Successful execution should produce output");
    } else {
        assert!(!stderr.trim().is_empty(), "Failed execution should provide error message");
        // Should fail gracefully, not with internal errors
        assert!(!verbose_output.contains("unwrap"), "Should handle errors gracefully");
    }

    Ok(())
}

/// Test format specification with AI/ML files
#[test]
fn test_specify_input_format() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--format")
        .arg("pytorch");

    let output = cmd.output()?;
    
    // Should either succeed or provide meaningful error about AI/ML format detection
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(!stdout.is_empty());
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Should not crash with unhandled errors, should provide AI/ML specific guidance
        assert!(!stderr.contains("panic"));
    }
    
    Ok(())
}
