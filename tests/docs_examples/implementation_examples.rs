use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

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

/// Test case 1: diff_ml_models_enhanced (Phase 2 implementation status example)
#[test]
fn test_ml_models_enhanced() -> Result<(), Box<dyn std::error::Error>> {
    let model1 = create_temp_file(
        r#"{"model": {"enhanced": true, "version": "2.4"}}"#,
        ".json",
    );
    let model2 = create_temp_file(
        r#"{"model": {"enhanced": true, "version": "2.5"}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(model1.path()).arg(model2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("version:"));

    Ok(())
}
