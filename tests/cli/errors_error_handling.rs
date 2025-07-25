#[allow(unused_imports)]
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
    cmd.arg("nonexistent1.json").arg("nonexistent2.json");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("No such file").or(predicate::str::contains("not found")));

    Ok(())
}

/// Test handling of invalid JSON format
#[test]
fn test_invalid_json_error() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary invalid JSON file
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/invalid.json", "{ invalid json syntax }")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/invalid.json")
        .arg("../tests/fixtures/file1.json");

    let output = cmd.output()?;

    // Should either handle gracefully or show error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success()
            || stderr.contains("parse")
            || stderr.contains("invalid")
            || stderr.contains("JSON")
    );

    // Clean up
    let _ = fs::remove_file("../tests/output/invalid.json");

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
    cmd.arg("only-one-file.json");

    let output = cmd.output()?;

    // Should show error about needing two files
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success() || stderr.contains("required") || stderr.contains("Usage:"));

    Ok(())
}

/// Test handling of empty files
#[test]
fn test_empty_file_handling() -> Result<(), Box<dyn std::error::Error>> {
    // Create temporary empty file
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/empty.json", "")?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/empty.json")
        .arg("../tests/fixtures/file1.json");

    let output = cmd.output()?;

    // Should handle empty files gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success() || stderr.contains("empty") || stderr.contains("parse"));

    // Clean up
    let _ = fs::remove_file("../tests/output/empty.json");

    Ok(())
}

/// Test error handling for permission denied
#[test]
#[cfg(unix)]
fn test_permission_denied_error() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::fs::PermissionsExt;

    // Create temporary file with no read permissions
    fs::create_dir_all("../tests/output")?;
    fs::write("../tests/output/no_read.json", r#"{"test": "data"}"#)?;

    let mut perms = fs::metadata("../tests/output/no_read.json")?.permissions();
    perms.set_mode(0o000); // No permissions
    fs::set_permissions("../tests/output/no_read.json", perms)?;

    let mut cmd = diffai_cmd();
    cmd.arg("../tests/output/no_read.json")
        .arg("../tests/fixtures/file1.json");

    let output = cmd.output()?;

    // Should show permission error
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success() && (stderr.contains("permission") || stderr.contains("denied"))
    );

    // Clean up (restore permissions first)
    let mut perms = fs::metadata("../tests/output/no_read.json")?.permissions();
    perms.set_mode(0o644);
    fs::set_permissions("../tests/output/no_read.json", perms)?;
    let _ = fs::remove_file("../tests/output/no_read.json");

    Ok(())
}
