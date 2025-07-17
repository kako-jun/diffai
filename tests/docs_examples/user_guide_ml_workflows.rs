use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test ML workflow examples from user guide
/// Corresponds to: docs/user-guide/ml-workflows.md
#[test]
fn test_ml_workflow_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Test ML model comparison workflow from docs
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("mean=").or(predicate::str::contains("std=")));
    
    Ok(())
}

/// Test ML workflow with JSON output
#[test]
fn test_ml_workflow_json_output() -> Result<(), Box<dyn std::error::Error>> {
    // Test ML workflow with JSON output example from docs
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::starts_with("["))
        .stdout(predicate::str::contains("TensorStatsChanged").or(predicate::str::contains("MemoryAnalysis")));
    
    Ok(())
}