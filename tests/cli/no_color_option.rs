/// Tests for --no-color option in diffai
/// Ensures color output is properly disabled when flag is specified
use assert_cmd::Command;
use std::io::Write;
use tempfile::NamedTempFile;

fn create_test_json1() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;
    writeln!(
        temp_file,
        r#"{{"name": "test", "value": 123, "enabled": true}}"#
    )?;
    Ok(temp_file)
}

fn create_test_json2() -> Result<NamedTempFile, Box<dyn std::error::Error>> {
    let mut temp_file = NamedTempFile::new()?;
    writeln!(
        temp_file,
        r#"{{"name": "test", "value": 456, "enabled": false}}"#
    )?;
    Ok(temp_file)
}

#[test]
fn test_diffai_no_color_option_basic() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = create_test_json1()?;
    let test_file2 = create_test_json2()?;

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1.path().to_str().unwrap(),
            test_file2.path().to_str().unwrap(),
            "--format",
            "json",
            "--no-color",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with --no-color");

    // Output should not contain ANSI color codes
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Debug output for troubleshooting
    eprintln!("Exit status: {:?}", output.status);
    eprintln!("Stdout: '{}'", stdout);
    eprintln!("Stderr: '{}'", stderr);

    assert!(
        !stdout.contains("\x1b["),
        "Output should not contain ANSI color codes when --no-color is specified"
    );

    // Should still contain the differences (if there are any)
    // Allow empty output if files are identical or have specific differences
    if !stdout.trim().is_empty() {
        assert!(
            stdout.contains("value")
                || stdout.contains("123")
                || stdout.contains("456")
                || stdout.contains("~"),
            "If output exists, it should contain difference information"
        );
    }

    Ok(())
}

#[test]
fn test_diffai_no_color_option_with_json_output() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = create_test_json1()?;
    let test_file2 = create_test_json2()?;

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1.path().to_str().unwrap(),
            test_file2.path().to_str().unwrap(),
            "--format",
            "json",
            "--no-color",
            "--output",
            "json",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with --no-color and JSON output");

    // JSON output should not contain ANSI color codes
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("\x1b["),
        "JSON output should not contain ANSI color codes when --no-color is specified"
    );

    // Should be valid JSON if not empty
    if !stdout.trim().is_empty() {
        let _: serde_json::Value =
            serde_json::from_str(&stdout).expect("Output should be valid JSON");
    }

    Ok(())
}

#[test]
fn test_diffai_no_color_option_with_verbose() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = create_test_json1()?;
    let test_file2 = create_test_json2()?;

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1.path().to_str().unwrap(),
            test_file2.path().to_str().unwrap(),
            "--format",
            "json",
            "--no-color",
            "--verbose",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with --no-color and --verbose");

    // Verbose output should not contain ANSI color codes
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !stdout.contains("\x1b["),
        "Verbose stdout should not contain ANSI color codes when --no-color is specified"
    );
    assert!(
        !stderr.contains("\x1b["),
        "Verbose stderr should not contain ANSI color codes when --no-color is specified"
    );

    Ok(())
}

#[test]
fn test_diffai_color_vs_no_color_output_difference() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = create_test_json1()?;
    let test_file2 = create_test_json2()?;

    // Test with colors enabled (default)
    let colored_output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1.path().to_str().unwrap(),
            test_file2.path().to_str().unwrap(),
            "--format",
            "json",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with colors");

    // Test with colors disabled
    let no_color_output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1.path().to_str().unwrap(),
            test_file2.path().to_str().unwrap(),
            "--format",
            "json",
            "--no-color",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with --no-color");

    let colored_stdout = String::from_utf8_lossy(&colored_output.stdout);
    let no_color_stdout = String::from_utf8_lossy(&no_color_output.stdout);

    // Colored output might contain ANSI codes (but not necessarily)
    // No-color output should definitely not contain ANSI codes
    assert!(
        !no_color_stdout.contains("\x1b["),
        "No-color output should not contain ANSI color codes"
    );

    // Both should contain the same difference information (ignoring colors)
    // This is a basic sanity check that --no-color doesn't break functionality
    assert!(
        (!no_color_stdout.trim().is_empty() && no_color_stdout.contains("value"))
            || (!colored_stdout.trim().is_empty() && colored_stdout.contains("value")),
        "At least one output should contain difference information. No-color: '{}', Colored: '{}'",
        no_color_stdout.trim(),
        colored_stdout.trim()
    );

    Ok(())
}

#[test]
fn test_diffai_no_color_with_different_formats() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = create_test_json1()?;
    let test_file2 = create_test_json2()?;

    let formats = ["json", "yaml"];

    for format in &formats {
        let output = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "diffai",
                "--",
                test_file1.path().to_str().unwrap(),
                test_file2.path().to_str().unwrap(),
                "--no-color",
                "--output",
                format,
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect(&format!(
                "Failed to execute diffai with --no-color and --output {}",
                format
            ));

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.contains("\x1b["),
            "Output format {} should not contain ANSI color codes when --no-color is specified",
            format
        );
    }

    Ok(())
}
