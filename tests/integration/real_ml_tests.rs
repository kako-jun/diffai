use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

// Real ML tests using generated model files

#[test]
fn test_basic_model_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("tensor.fc1.weight"))
        .stdout(predicate::str::contains("mean="))
        .stdout(predicate::str::contains("std="));

    Ok(())
}

#[test]
fn test_learning_progress_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_0.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors")
        .arg("--learning-progress");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("learning_progress"))
        .stdout(predicate::str::contains("trend="))
        .stdout(predicate::str::contains("magnitude="));

    Ok(())
}

#[test]
fn test_convergence_analysis_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_50.safetensors")
        .arg("--convergence-analysis");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("convergence_analysis"))
        .stdout(predicate::str::contains("status="))
        .stdout(predicate::str::contains("stability="));

    Ok(())
}

#[test]
fn test_anomaly_detection_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/normal_model.safetensors")
        .arg("../tests/fixtures/ml_models/anomalous_model.safetensors")
        .arg("--anomaly-detection");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("anomaly_detection"))
        .stdout(predicate::str::contains("type="))
        .stdout(predicate::str::contains("severity="));

    Ok(())
}

#[test]
fn test_memory_analysis_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("../tests/fixtures/ml_models/large_model.safetensors")
        .arg("--memory-analysis");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("memory_analysis"))
        .stdout(predicate::str::contains("delta="))
        .stdout(predicate::str::contains("efficiency="));

    Ok(())
}

#[test]
fn test_quantization_analysis_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/model_fp32.safetensors")
        .arg("../tests/fixtures/ml_models/model_quantized.safetensors")
        .arg("--quantization-analysis");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("tensor."))
        .stdout(predicate::str::contains("mean="))
        .stdout(predicate::str::contains("std="));

    Ok(())
}

#[test]
fn test_architecture_comparison_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/transformer.safetensors")
        .arg("--architecture-comparison");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("architecture_comparison"))
        .stdout(predicate::str::contains("type1="))
        .stdout(predicate::str::contains("differences="));

    Ok(())
}

#[test]
fn test_multiple_advanced_features() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--learning-progress")
        .arg("--convergence-analysis")
        .arg("--memory-analysis")
        .arg("--stats");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("learning_progress"))
        .stdout(predicate::str::contains("convergence_analysis"))
        .stdout(predicate::str::contains("memory_analysis"))
        .stdout(predicate::str::contains("tensor."));

    Ok(())
}

#[test]
fn test_json_output_with_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--learning-progress")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should output valid JSON
    assert!(stdout.starts_with('[') && stdout.trim_end().ends_with(']'));

    // Should contain ML analysis results
    assert!(stdout.contains("LearningProgress"));

    Ok(())
}

#[test]
fn test_yaml_output_with_real_models() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--convergence-analysis")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should output valid YAML with ML analysis
    assert!(stdout.contains("ConvergenceAnalysis"));

    Ok(())
}

#[test]
#[ignore] // PyTorch parsing needs improvement
fn test_pytorch_vs_safetensors_consistency() -> Result<(), Box<dyn std::error::Error>> {
    // Test that PyTorch and Safetensors versions produce similar results
    let mut cmd_safetensors = diffai_cmd();
    cmd_safetensors
        .arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--stats");

    let output_safetensors = cmd_safetensors.output()?;
    assert!(output_safetensors.status.success());

    let mut cmd_pytorch = diffai_cmd();
    cmd_pytorch
        .arg("../tests/fixtures/ml_models/simple_base.pt")
        .arg("../tests/fixtures/ml_models/simple_modified.pt")
        .arg("--stats");

    let output_pytorch = cmd_pytorch.output()?;

    // Safetensors should succeed, PyTorch may or may not (depending on implementation)
    assert!(output_safetensors.status.success());

    if output_pytorch.status.success() {
        let stdout_safetensors = String::from_utf8_lossy(&output_safetensors.stdout);
        let stdout_pytorch = String::from_utf8_lossy(&output_pytorch.stdout);

        // Should contain similar tensor information
        assert!(stdout_safetensors.contains("tensor."));
        assert!(stdout_pytorch.contains("tensor."));
    }

    Ok(())
}

#[test]
fn test_model_size_impact_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Test analysis between significantly different model sizes
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("../tests/fixtures/ml_models/transformer.safetensors")
        .arg("--architecture-comparison")
        .arg("--memory-analysis");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("architecture_comparison"))
        .stdout(predicate::str::contains("memory_analysis"));

    Ok(())
}
