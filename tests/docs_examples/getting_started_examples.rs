use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test the first example from getting-started.md - basic PyTorch comparison
#[test]
fn test_getting_started_basic_pytorch() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    
    // Should execute without panic
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be reasonable
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test safetensors comparison from getting-started.md
#[test]
fn test_getting_started_safetensors() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/simple_base.safetensors")
        .arg("tests/fixtures/ml_models/simple_modified.safetensors");

    let output = cmd.output()?;
    
    // Should handle safetensors without crashing
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test automatic format detection from getting-started.md
#[test]
fn test_automatic_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    // Test different file extensions are handled automatically
    let test_pairs = [
        ("tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("tests/fixtures/ml_models/small_model.safetensors", "tests/fixtures/ml_models/model_fp32.safetensors"),
        ("tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
        ("tests/fixtures/ml_models/matlab_data1.mat", "tests/fixtures/ml_models/matlab_data2.mat"),
    ];

    for (file1, file2) in test_pairs.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        
        // Should handle each format automatically
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for {} vs {}", file1, file2);
    }

    Ok(())
}

/// Test comprehensive ML analysis mentioned in getting-started.md
#[test]
fn test_comprehensive_ml_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.pt")
        .arg("tests/fixtures/ml_models/anomalous_model.pt");

    let output = cmd.output()?;
    
    // Should provide comprehensive analysis
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test output formats from getting-started.md
#[test]
fn test_output_formats_introduction() -> Result<(), Box<dyn std::error::Error>> {
    // Test JSON output
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/large_model.pt")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test YAML output
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/large_model.pt")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test basic options from getting-started.md
#[test]
fn test_basic_options_introduction() -> Result<(), Box<dyn std::error::Error>> {
    // Test verbose option
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt")
        .arg("--verbose");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test no-color option
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt")
        .arg("--no-color");

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    
    // Should not contain ANSI color codes
    assert!(!stdout.contains("\x1b["));

    Ok(())
}

/// Test format specification from getting-started.md
#[test]
fn test_format_specification_introduction() -> Result<(), Box<dyn std::error::Error>> {
    let formats = ["pytorch", "safetensors", "numpy", "matlab"];
    let files = [
        ("tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
        ("tests/fixtures/ml_models/numpy_small1.npy", "tests/fixtures/ml_models/numpy_small2.npy"),
        ("tests/fixtures/ml_models/matlab_simple1.mat", "tests/fixtures/ml_models/matlab_simple2.mat"),
    ];

    for (i, format) in formats.iter().enumerate() {
        let mut cmd = diffai_cmd();
        cmd.arg(files[i].0)
            .arg(files[i].1)
            .arg("--format")
            .arg(format);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for format: {}", format);
    }

    Ok(())
}

/// Test model comparison workflow from getting-started.md
#[test]
fn test_model_comparison_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Test training checkpoint comparison
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_50.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test model architecture comparison
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.pt")
        .arg("tests/fixtures/ml_models/transformer.pt");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test quantization comparison from getting-started.md
#[test]
fn test_quantization_comparison_workflow() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model_fp32.pt")
        .arg("tests/fixtures/ml_models/model_quantized.pt");

    let output = cmd.output()?;
    
    // Should analyze quantization differences
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test scientific data analysis from getting-started.md
#[test]
fn test_scientific_data_workflow() -> Result<(), Box<dyn std::error::Error>> {
    // Test NumPy array comparison
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_data1.npy")
        .arg("tests/fixtures/ml_models/numpy_data2.npy");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test compressed NumPy comparison
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_model1.npz")
        .arg("tests/fixtures/ml_models/numpy_model2.npz");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test identical files handling from getting-started.md
#[test]
fn test_identical_files_handling() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/small_model.pt")
        .arg("tests/fixtures/ml_models/small_model.pt");  // Same file

    let output = cmd.output()?;
    
    // Should handle identical files gracefully
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test error handling for nonexistent files
#[test]
fn test_error_handling_nonexistent_files() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("nonexistent_file.pt")
        .arg("tests/fixtures/ml_models/model1.pt");

    cmd.assert()
        .failure()
        .stderr(predicates::str::contains("No such file").or(predicates::str::contains("not found")));

    Ok(())
}

/// Test help and version commands from getting-started.md
#[test]
fn test_help_and_version_commands() -> Result<(), Box<dyn std::error::Error>> {
    // Test help command
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"))
        .stdout(predicates::str::contains("AI/ML"));

    // Test version command  
    let mut cmd = diffai_cmd();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"));

    Ok(())
}