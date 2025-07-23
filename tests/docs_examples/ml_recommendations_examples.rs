use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::NamedTempFile;
use std::io::Write;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

// Helper function to create temporary files for testing
fn create_temp_file(content: &str, suffix: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(suffix).expect("Failed to create temp file");
    writeln!(file, "{}", content).expect("Failed to write to temp file");
    file
}

/// Test case 1: diffai baseline.safetensors candidate.safetensors
#[test]
fn test_deployment_recommendations() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"model": {"performance": 0.85, "memory": 512}}"#, ".json");
    let candidate = create_temp_file(r#"{"model": {"performance": 0.75, "memory": 1024}}"#, ".json");
    
    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(candidate.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("performance:"))
        .stdout(predicates::str::contains("memory:"));
    
    Ok(())
}

/// Test case 2: diffai model_v1.pt model_v2.pt --output json | jq '.recommendations[]'
#[test]
fn test_json_recommendations_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"recommendations": {"enabled": true, "level": "high"}}"#, ".json");
    let file2 = create_temp_file(r#"{"recommendations": {"enabled": true, "level": "critical"}}"#, ".json");
    
    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--output").arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());
    
    Ok(())
}

/// Test case 3: diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt
#[test]
fn test_training_progress_recommendations() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint1 = create_temp_file(r#"{"epoch": 10, "loss": 0.5, "accuracy": 0.80}"#, ".json");
    let checkpoint2 = create_temp_file(r#"{"epoch": 20, "loss": 0.3, "accuracy": 0.85}"#, ".json");
    
    let mut cmd = diffai_cmd();
    cmd.arg(checkpoint1.path()).arg(checkpoint2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("epoch:"))
        .stdout(predicates::str::contains("loss:"))
        .stdout(predicates::str::contains("accuracy:"));
    
    Ok(())
}