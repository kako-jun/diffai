#[allow(unused_imports)]
use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test help command output
/// Corresponds to: docs/examples/test-results/cli_help_output.md
#[test]
fn test_help_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("diffai"))
        .stdout(predicate::str::contains("Usage:"))
        .stdout(predicate::str::contains("Arguments:"))
        .stdout(predicate::str::contains("Options:"));

    Ok(())
}

/// Test help flag variations
#[test]
fn test_help_flag_variations() -> Result<(), Box<dyn std::error::Error>> {
    // Test -h short flag
    let mut cmd = diffai_cmd();
    cmd.arg("-h");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("diffai"))
        .stdout(predicate::str::contains("Usage:"));

    Ok(())
}

/// Test no arguments shows help
#[test]
fn test_no_arguments() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();

    let output = cmd.output()?;

    // Should either show help or error message with usage info
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(stdout.contains("Usage:") || stderr.contains("Usage:") || stderr.contains("required"));

    Ok(())
}
