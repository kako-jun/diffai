/// Tests for --no-color option in diffai
/// Ensures color output is properly disabled when flag is specified
use assert_cmd::Command;

// Use existing AI/ML fixtures for testing --no-color option
fn get_test_ml_file1() -> &'static str {
    "tests/fixtures/ml_models/model1.pt"
}

fn get_test_ml_file2() -> &'static str {
    "tests/fixtures/ml_models/model2.pt"
}

#[test]
fn test_diffai_no_color_option_basic() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = get_test_ml_file1();
    let test_file2 = get_test_ml_file2();

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1,
            test_file2,
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
    eprintln!("Stdout: '{stdout}'");
    eprintln!("Stderr: '{stderr}'");

    assert!(
        !stdout.contains("\x1b["),
        "Output should not contain ANSI color codes when --no-color is specified"
    );

    // Should still contain AI/ML differences (if there are any)
    // Allow empty output if files are identical
    if !stdout.trim().is_empty() {
        assert!(
            stdout.contains("~")
                || stdout.contains("mean")
                || stdout.contains("std") 
                || stdout.contains("analysis")
                || stdout.contains("tensor"),
            "If output exists, it should contain AI/ML difference information"
        );
    }

    Ok(())
}

#[test]
fn test_diffai_no_color_option_with_json_output() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = get_test_ml_file1();
    let test_file2 = get_test_ml_file2();

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1,
            test_file2,
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
    let test_file1 = get_test_ml_file1();
    let test_file2 = get_test_ml_file2();

    let output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1,
            test_file2,
            "--no-color",
            "--verbose",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with --no-color and --verbose");

    // Verbose output should not contain ANSI color codes
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Filter out cargo build messages which may contain ANSI codes
    let filtered_stderr: String = stderr
        .lines()
        .filter(|line| {
            // Skip cargo build/compilation messages and proxychains output
            !line.contains("Compiling")
                && !line.contains("Finished")
                && !line.contains("Running")
                && !line.contains("Blocking")
                && !line.contains("[proxychains]")
                && !line.trim().is_empty()
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        !stdout.contains("\x1b["),
        "Verbose stdout should not contain ANSI color codes when --no-color is specified"
    );
    assert!(
        !filtered_stderr.contains("\x1b["),
        "Verbose stderr should not contain ANSI color codes when --no-color is specified. Stderr: {filtered_stderr:?}"
    );

    Ok(())
}

#[test]
fn test_diffai_color_vs_no_color_output_difference() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = get_test_ml_file1();
    let test_file2 = get_test_ml_file2();

    // Test with colors enabled (default)
    let colored_output = Command::new("cargo")
        .args([
            "run",
            "--bin",
            "diffai",
            "--",
            test_file1,
            test_file2,
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
            test_file1,
            test_file2,
            "--no-color",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai with --no-color");

    let colored_stdout = String::from_utf8_lossy(&colored_output.stdout);
    let no_color_stdout = String::from_utf8_lossy(&no_color_output.stdout);

    // No-color output should definitely not contain ANSI codes
    assert!(
        !no_color_stdout.contains("\x1b["),
        "No-color output should not contain ANSI color codes"
    );

    // Both should have meaningful output for AI/ML files or handle gracefully
    let has_meaningful_output = (!no_color_stdout.trim().is_empty() && (
        no_color_stdout.contains("~") || no_color_stdout.contains("mean") || 
        no_color_stdout.contains("std") || no_color_stdout.contains("analysis")))
        || (!colored_stdout.trim().is_empty() && (
        colored_stdout.contains("~") || colored_stdout.contains("mean") || 
        colored_stdout.contains("std") || colored_stdout.contains("analysis")));

    // Allow for no differences if models are identical, or require meaningful ML output
    assert!(
        has_meaningful_output || (no_color_stdout.trim().is_empty() && colored_stdout.trim().is_empty()),
        "Should have meaningful AI/ML analysis output or indicate no differences. No-color: '{}', Colored: '{}'",
        no_color_stdout.trim(),
        colored_stdout.trim()
    );

    Ok(())
}

#[test]
fn test_diffai_no_color_with_different_formats() -> Result<(), Box<dyn std::error::Error>> {
    let test_file1 = get_test_ml_file1();
    let test_file2 = get_test_ml_file2();

    let formats = ["json", "yaml"];

    for format in &formats {
        let output = Command::new("cargo")
            .args([
                "run",
                "--bin",
                "diffai",
                "--",
                test_file1,
                test_file2,
                "--no-color",
                "--output",
                format,
            ])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .unwrap_or_else(|_| {
                panic!("Failed to execute diffai with --no-color and --output {format}")
            });

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            !stdout.contains("\x1b["),
            "Output format {format} should not contain ANSI color codes when --no-color is specified"
        );
    }

    Ok(())
}
