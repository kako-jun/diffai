use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test version command output
/// Corresponds to: docs/examples/test-results/cli_version_info.md
#[test]
fn test_version_command() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("diffai"))
        .stdout(predicate::str::contains("0."));

    Ok(())
}

/// Test version flag variations
#[test]
fn test_version_flag_variations() -> Result<(), Box<dyn std::error::Error>> {
    // Test -V short flag
    let mut cmd = diffai_cmd();
    cmd.arg("-V");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("diffai"));

    Ok(())
}
