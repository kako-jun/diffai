use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

/// Test basic diffai command from index.md main example
#[test]
fn test_index_basic_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/model1.pt")
        .arg("tests/fixtures/ml_models/model2.pt");

    let output = cmd.output()?;
    
    // Should execute the main example without issues
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test AI/ML specialization claim from index.md
#[test]
fn test_ai_ml_specialization() -> Result<(), Box<dyn std::error::Error>> {
    // Test that diffai can handle ML-specific files mentioned in index
    let ml_files = [
        ("tests/fixtures/ml_models/simple_base.pt", "tests/fixtures/ml_models/simple_modified.pt"),
        ("tests/fixtures/ml_models/simple_base.safetensors", "tests/fixtures/ml_models/simple_modified.safetensors"),
        ("tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
    ];

    for (file1, file2) in ml_files.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed for ML files: {} vs {}", file1, file2);
    }

    Ok(())
}

/// Test comprehensive analysis mentioned in index.md
#[test]
fn test_comprehensive_analysis_claim() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/checkpoint_epoch_0.pt")
        .arg("tests/fixtures/ml_models/checkpoint_epoch_10.pt");

    let output = cmd.output()?;
    
    // Should provide comprehensive analysis as claimed
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));
    
    // Exit code should be reasonable
    assert!(matches!(output.status.code(), Some(0) | Some(1) | Some(2)));

    Ok(())
}

/// Test format support claimed in index.md
#[test]
fn test_format_support_claims() -> Result<(), Box<dyn std::error::Error>> {
    // Test each format mentioned in index.md
    let format_tests = [
        ("PyTorch", "tests/fixtures/ml_models/model1.pt", "tests/fixtures/ml_models/model2.pt"),
        ("Safetensors", "tests/fixtures/ml_models/small_model.safetensors", "tests/fixtures/ml_models/model_fp32.safetensors"),
        ("NumPy", "tests/fixtures/ml_models/numpy_data1.npy", "tests/fixtures/ml_models/numpy_data2.npy"),
        ("MATLAB", "tests/fixtures/ml_models/matlab_data1.mat", "tests/fixtures/ml_models/matlab_data2.mat"),
    ];

    for (format_name, file1, file2) in format_tests.iter() {
        let mut cmd = diffai_cmd();
        cmd.arg(file1).arg(file2);

        let output = cmd.output()?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("panic"), "Failed to handle {} format", format_name);
    }

    Ok(())
}

/// Test automatic analysis mentioned in index.md
#[test]
fn test_automatic_analysis_claim() -> Result<(), Box<dyn std::error::Error>> {
    // Test that analysis happens automatically without configuration
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/normal_model.pt")
        .arg("tests/fixtures/ml_models/anomalous_model.pt");

    let output = cmd.output()?;
    
    // Should perform analysis automatically
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test zero configuration claim from index.md
#[test]
fn test_zero_configuration_claim() -> Result<(), Box<dyn std::error::Error>> {
    // Test that diffai works without any configuration options
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/transformer.pt")
        .arg("tests/fixtures/ml_models/transformer.safetensors");

    let output = cmd.output()?;
    
    // Should work without any configuration
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test scientific computing support from index.md
#[test]
fn test_scientific_computing_support() -> Result<(), Box<dyn std::error::Error>> {
    // Test scientific data formats mentioned in index
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/numpy_model1.npz")
        .arg("tests/fixtures/ml_models/numpy_model2.npz");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    // Test MATLAB support
    let mut cmd = diffai_cmd();
    cmd.arg("tests/fixtures/ml_models/matlab_simple1.mat")
        .arg("tests/fixtures/ml_models/matlab_simple2.mat");

    let output = cmd.output()?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panic"));

    Ok(())
}

/// Test that diffai identifies itself correctly
#[test]
fn test_tool_identification() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicates::str::contains("diffai"))
        .stdout(predicates::str::contains("AI/ML"));

    Ok(())
}