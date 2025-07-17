use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test CLI options from reference documentation
/// Corresponds to: docs/reference/cli-reference.md
#[test]
fn test_cli_reference_options() -> Result<(), Box<dyn std::error::Error>> {
    // Test epsilon option from CLI reference
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/data1.json")
        .arg("../tests/fixtures/data2.json")
        .arg("--epsilon")
        .arg("0.00001");
    
    cmd.assert()
        .success();
    
    Ok(())
}

/// Test array ID key option from reference
#[test]
fn test_array_id_key_reference() -> Result<(), Box<dyn std::error::Error>> {
    // Test array-id-key option from CLI reference
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/users1.json")
        .arg("../tests/fixtures/users2.json")
        .arg("--array-id-key")
        .arg("id");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("[id="));
    
    Ok(())
}

/// Test ignore keys regex option from reference
#[test]
fn test_ignore_keys_regex_reference() -> Result<(), Box<dyn std::error::Error>> {
    // Test ignore-keys-regex option from CLI reference
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/file1.json")
        .arg("../tests/fixtures/file2.json")
        .arg("--ignore-keys-regex")
        .arg("^age$");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("~ age:").not());
    
    Ok(())
}