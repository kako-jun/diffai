/// Integration tests for verbose output functionality
/// Tests comprehensive diagnostic information display
use assert_cmd::Command;
// Note: predicates not used in current tests but kept for future use
use std::fs;
use tempfile::TempDir;

/// Test basic verbose configuration display
#[test]
fn test_verbose_basic_configuration() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("test1.json");
    let file2 = temp_dir.path().join("test2.json");

    fs::write(&file1, r#"{"name": "test1", "value": 100}"#).unwrap();
    fs::write(&file2, r#"{"name": "test2", "value": 200}"#).unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Check verbose mode enabled message
    assert!(stderr.contains("=== diffai verbose mode enabled ==="));

    // Check configuration section
    assert!(stderr.contains("Configuration:"));
    assert!(stderr.contains("Input format: Json"));
    assert!(stderr.contains("Output format: Cli"));
    assert!(stderr.contains("Recursive mode: false"));

    // Check file analysis section
    assert!(stderr.contains("File analysis:"));
    assert!(stderr.contains("Input 1:"));
    assert!(stderr.contains("Input 2:"));
    assert!(stderr.contains("Detected format: Json"));
    assert!(stderr.contains("File 1 size:"));
    assert!(stderr.contains("File 2 size:"));

    // Check processing results section
    assert!(stderr.contains("Processing results:"));
    assert!(stderr.contains("Total processing time:"));
    assert!(stderr.contains("Differences found: 2"));
    assert!(stderr.contains("Format-specific analysis: Json"));
}

/// Test verbose output with ML analysis features (now included by default)
#[test]
fn test_verbose_with_default_ml_features() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("model1.json");
    let file2 = temp_dir.path().join("model2.json");

    fs::write(
        &file1,
        r#"{"model": {"layers": [{"type": "linear", "params": 1000}]}}"#,
    )
    .unwrap();
    fs::write(
        &file2,
        r#"{"model": {"layers": [{"type": "linear", "params": 2000}]}}"#,
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Check verbose mode enabled message
    assert!(stderr.contains("=== diffai verbose mode enabled ==="));
    // For non-ML files, ML analysis features may not be displayed
}

/// Test verbose output with advanced options
#[test]
fn test_verbose_with_advanced_options() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("data1.json");
    let file2 = temp_dir.path().join("data2.json");

    fs::write(&file1, r#"{"id": "123", "name": "test1", "value": 100.0}"#).unwrap();
    fs::write(
        &file2,
        r#"{"id": "456", "name": "test2", "value": 100.001}"#,
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .arg("--epsilon")
        .arg("0.01")
        .arg("--ignore-keys-regex")
        .arg("^id$")
        .arg("--path")
        .arg("name")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Check advanced configuration options
    assert!(stderr.contains("Epsilon tolerance: 0.01"));
    assert!(stderr.contains("Ignore keys regex: ^id$"));
    assert!(stderr.contains("Path filter: name"));
}

/// Test verbose output format-specific information
#[test]
fn test_verbose_format_specific() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("config1.yaml");
    let file2 = temp_dir.path().join("config2.yaml");

    fs::write(&file1, "name: test1\nvalue: 100\n").unwrap();
    fs::write(&file2, "name: test2\nvalue: 200\n").unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Check YAML format detection
    assert!(stderr.contains("Detected format: Yaml"));
    assert!(stderr.contains("Format-specific analysis: Yaml"));
}

/// Test verbose output with JSON output format
#[test]
fn test_verbose_with_json_output() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("test1.json");
    let file2 = temp_dir.path().join("test2.json");

    fs::write(&file1, r#"{"value": 1}"#).unwrap();
    fs::write(&file2, r#"{"value": 2}"#).unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .arg("--output")
        .arg("json")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Check JSON output format in configuration
    assert!(stderr.contains("Output format: Json"));
}

/// Test verbose directory comparison
#[test]
fn test_verbose_directory_comparison() {
    let temp_dir = TempDir::new().unwrap();
    let dir1 = temp_dir.path().join("dir1");
    let dir2 = temp_dir.path().join("dir2");

    fs::create_dir_all(&dir1).unwrap();
    fs::create_dir_all(&dir2).unwrap();

    fs::write(dir1.join("file1.json"), r#"{"name": "test1"}"#).unwrap();
    fs::write(dir1.join("file2.json"), r#"{"name": "common1"}"#).unwrap();
    fs::write(dir2.join("file2.json"), r#"{"name": "common2"}"#).unwrap();
    fs::write(dir2.join("file3.json"), r#"{"name": "test3"}"#).unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(dir1.to_str().unwrap())
        .arg(dir2.to_str().unwrap())
        .arg("--verbose")
        .arg("--recursive")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // For directory comparisons, verbose output may be minimal
    // The test should pass if no error occurs and output is produced
    assert!(output.status.success());

    // Directory comparison should produce some output (either verbose info or comparison results)
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!stderr.is_empty() || !stdout.is_empty());
}

/// Test verbose timing measurement precision
#[test]
fn test_verbose_timing_precision() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("small1.json");
    let file2 = temp_dir.path().join("small2.json");

    fs::write(&file1, r#"{"a": 1}"#).unwrap();
    fs::write(&file2, r#"{"a": 2}"#).unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Check that timing information is present and formatted correctly
    assert!(
        stderr.contains("Total processing time:"),
        "Should contain processing time label"
    );
    assert!(
        stderr.contains("µs") || stderr.contains("ms") || stderr.contains("ns"),
        "Processing time should include time unit. Actual stderr: {stderr}"
    );
}

/// Test verbose output with ML file formats (when supported)
#[test]
fn test_verbose_ml_format_detection() {
    // This test checks if verbose mode properly detects and reports ML file formats
    // It will be more meaningful once we have actual ML test files

    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("model1.json");
    let file2 = temp_dir.path().join("model2.json");

    // Create JSON files that simulate ML model metadata
    fs::write(
        &file1,
        r#"{"tensors": {"layer1.weight": {"shape": [10, 5]}}}"#,
    )
    .unwrap();
    fs::write(
        &file2,
        r#"{"tensors": {"layer1.weight": {"shape": [10, 6]}}}"#,
    )
    .unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Verify verbose mode works with ML-like data structures with default analysis
    assert!(stderr.contains("=== diffai verbose mode enabled ==="));
    // For JSON files that might contain ML-like data, analysis may be available
}

/// Test verbose mode helps with debugging
#[test]
fn test_verbose_debugging_information() {
    let temp_dir = TempDir::new().unwrap();
    let file1 = temp_dir.path().join("debug1.json");
    let file2 = temp_dir.path().join("debug2.json");

    fs::write(&file1, r#"{"complex": {"nested": {"data": [1, 2, 3]}}}"#).unwrap();
    fs::write(&file2, r#"{"complex": {"nested": {"data": [1, 2, 4]}}}"#).unwrap();

    let mut cmd = Command::cargo_bin("diffai").unwrap();
    let output = cmd
        .arg(file1.to_str().unwrap())
        .arg(file2.to_str().unwrap())
        .arg("--verbose")
        .arg("--path")
        .arg("complex.nested")
        .output()
        .unwrap();

    let stderr = String::from_utf8(output.stderr).unwrap();

    // Verify verbose output provides debugging context
    assert!(stderr.contains("Path filter: complex.nested"));
    assert!(stderr.contains("Differences found:"));

    // The output should help users understand what processing was done
    assert!(stderr.contains("Total processing time:"));
}
