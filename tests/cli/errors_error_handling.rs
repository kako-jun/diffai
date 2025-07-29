use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test handling of non-existent files
#[test]
fn test_nonexistent_file_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("nonexistent1.safetensors").arg("nonexistent2.safetensors");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("No such file").or(predicate::str::contains("not found")));

    Ok(())
}

/// Test handling of unsupported JSON format (diffai is AI/ML only)
#[test]
fn test_unsupported_json_error() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary JSON file (unsupported format for diffai)
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/unsupported.json", r#"{"test": "data"}"#)?;
    fs::write("../tests/output/unsupported2.json", r#"{"test": "data2"}"#)?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/unsupported.json")
        .arg("../tests/output/unsupported2.json");

    let output = cmd.output()?;

    // Should show unsupported format error and suggest diffx
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() 
        && (stderr.contains("Unsupported file format") 
            || stderr.contains("diffx")
            || stderr.contains("AI/ML file formats"))
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/unsupported.json");
    let _ = fs::remove_file("../tests/output/unsupported2.json");

    Ok(())
}

/// Test handling of unsupported file formats
#[test]
fn test_unsupported_format_error() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary unknown format file
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/unknown.xyz", "unknown format data")?;
    fs::write("../tests/output/unknown2.xyz", "unknown format data2")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/unknown.xyz")
        .arg("../tests/output/unknown2.xyz");

    let output = cmd.output()?;

    // Should handle unknown format gracefully or show error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("format")
            || stderr.contains("Could not infer")
            || stderr.contains("unsupported")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/unknown.xyz");
    let _ = fs::remove_file("../tests/output/unknown2.xyz");

    Ok(())
}

/// Test handling of invalid command line arguments
#[test]
fn test_invalid_argument_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--invalid-flag");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("unrecognized").or(predicate::str::contains("invalid")));

    Ok(())
}

/// Test handling of insufficient arguments
#[test]
fn test_insufficient_arguments_error() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("only-one-file.safetensors");

    let output = cmd.output()?;

    // Should show error about needing two files
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() || stderr.contains("required") || stderr.contains("Usage:"));

    Ok(())
}

/// Test handling of empty files
#[test]
fn test_empty_file_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary empty safetensors file
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/empty.safetensors", "")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/empty.safetensors")
        .arg("../tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;

    // Should handle empty files gracefully or show appropriate error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() && (stderr.contains("empty") || stderr.contains("parse") || stderr.contains("invalid")));

    // Clean up
    let _ = fs::remove_file("../tests/output/empty.safetensors");

    Ok(())
}

/// Test error handling for permission denied
#[test]
#[cfg(unix)]
fn test_permission_denied_error() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;

    // Create temporary safetensors file with no read permissions
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/no_read.safetensors", b"fake safetensors data")?;

    let mut perms = fs::metadata("../tests/output/no_read.safetensors")?.permissions();
    perms.set_mode(0o000); // No permissions
    fs::set_permissions("../tests/output/no_read.safetensors", perms)?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/no_read.safetensors")
        .arg("../tests/fixtures/ml_models/model1.pt");

    let output = cmd.output()?;

    // Should show permission error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && (stderr.contains("permission") || stderr.contains("denied"))
    );

    // Clean up (restore permissions first)
    let mut perms = fs::metadata("../tests/output/no_read.safetensors")?.permissions();
    perms.set_mode(0o644);
    fs::set_permissions("../tests/output/no_read.safetensors", perms)?;
    let _ = fs::remove_file("../tests/output/no_read.safetensors");

    Ok(())
}
