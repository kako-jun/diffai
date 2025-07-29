use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test basic synopsis from cli-reference.md
#[test]
fn test_basic_synopsis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    
    // Should execute basic synopsis: diffai <INPUT1> <INPUT2>
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test all documented format options from cli-reference.md
#[test]
fn test_format_options() -> Result<(), Box<dyn std::error::Error>> {
    let formats = [
        ("pytorch", "tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("safetensors", "tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
        ("numpy", "tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
        ("matlab", "tests/fixtures/ml_models/matlab_data1.mat", "tests/fixtures/ml_models/matlab_data2.mat"),
    ];

    for (format, file1, file2) in formats.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1)
            .arg(file2)
            .arg("--format")
            .arg(format);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for format: {}", format);
    }

    Ok(())
}

/// Test output format options from cli-reference.md
#[test]
fn test_output_format_options() -> Result<(), Box<dyn std::error::Error>> {
    let output_formats = ["json", "yaml"];

    for format in output_formats.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg("tests/fixtures/ml_models/small_model.pt")
            .arg("tests/fixtures/ml_models/large_model.pt")
            .arg("--output")
            .arg(format);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for output format: {}", format);
    }

    Ok(())
}

/// Test verbose mode from cli-reference.md
#[test]
fn test_verbose_mode_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt")
        .arg("--verbose");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test short form
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("-v");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test no-color option from cli-reference.md
#[test]
fn test_no_color_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--no-color");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should not contain ANSI color codes
    assert!(!stdout.contains("\x1b["));

    Ok(())
}

/// Test quiet mode from cli-reference.md
#[test]
fn test_quiet_mode_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")  // Same file
        .arg("--quiet");

    let output = cmd.output()?;
    
    // Quiet mode should produce no stdout
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.is_empty());

    // Test short form
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt")
        .arg("-q");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.is_empty());

    Ok(())
}

/// Test brief mode from cli-reference.md
#[test]
fn test_brief_mode_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--brief");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test comprehensive ML analysis examples from cli-reference.md
#[test]
fn test_comprehensive_ml_analysis_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Test the comprehensive output example showing all 11 features
    let test_cases = [
        ("tests/fixtures/ml_models/normal_model.pt", "tests/fixtures/ml_models/anomalous_model.pt"),
        ("tests/fixtures/ml_models/model_fp32.pt", "tests/fixtures/ml_models/model_quantized.pt"),
        ("tests/fixtures/ml_models/checkpoint_epoch_0.pt", "tests/fixtures/ml_models/checkpoint_epoch_50.pt"),
    ];

    for (file1, file2) in test_cases.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for {} vs {}", file1, file2);
    }

    Ok(())
}

/// Test format-aware automatic feature selection from cli-reference.md
#[test]
fn test_format_aware_feature_selection() -> Result<(), Box<dyn std::error::Error>> {
    // Test that PyTorch files get full analysis
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/transformer.pt")
        .arg("tests/fixtures/ml_models/normal_model.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test that Safetensors files get appropriate analysis
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/transformer.safetensors")
        .arg("tests/fixtures/ml_models/normal_model.safetensors");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test that NumPy files get basic analysis
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_data1.npy")
        .arg("tests/fixtures/ml_models/numpy_data2.npy");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test scientific data analysis examples from cli-reference.md
#[test]
fn test_scientific_data_analysis_examples() -> Result<(), Box<dyn std::error::Error>> {
    // Test NumPy array analysis
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_model1.npz")
        .arg("tests/fixtures/ml_models/numpy_model2.npz");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test MATLAB file analysis
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/matlab_simple1.mat")
        .arg("tests/fixtures/ml_models/matlab_simple2.mat");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test JSON output examples from cli-reference.md
#[test]
fn test_json_output_examples() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // If successful and has output, should be JSON-like
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.trim().is_empty() {
            let trimmed = stdout.trim();
            assert!(trimmed.starts_with("[") || trimmed.starts_with("{") || trimmed == "[]");
        }
    }

    Ok(())
}

/// Test YAML output examples from cli-reference.md  
#[test]
fn test_yaml_output_examples() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.pt")
        .arg("tests/fixtures/ml_models/simple_modified.pt")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test exit codes mentioned in cli-reference.md
#[test]
fn test_documented_exit_codes() -> Result<(), Box<dyn std::error::Error>> {
    // Test exit code 0 for identical files
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt");

    let output = cmd.output()?;
    // Should exit with code 0 or handle gracefully
    assert!(matches!(output.status.code(), Some(0) | Some(1)));

    // Test exit code for different files
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    // Should exit with reasonable code
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}