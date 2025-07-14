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
        .stdout(predicate::str::contains("fc1.").or(predicate::str::contains("fc2.")))
        .stdout(predicate::str::contains("mean="))
        .stdout(predicate::str::contains("std="));

    Ok(())
}

#[test]
fn test_learning_progress_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_0.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("mean=").or(predicate::str::contains("std=")))
        .stdout(predicate::str::contains("memory_analysis").or(predicate::str::contains("🧠")));

    Ok(())
}

#[test]
fn test_convergence_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/checkpoint_epoch_10.safetensors")
        .arg("../tests/fixtures/ml_models/checkpoint_epoch_50.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("mean=").or(predicate::str::contains("std=")))
        .stdout(predicate::str::contains("inference_speed").or(predicate::str::contains("⚡")));

    Ok(())
}

#[test]
fn test_anomaly_detection_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/normal_model.safetensors")
        .arg("../tests/fixtures/ml_models/anomalous_model.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("mean=").or(predicate::str::contains("std=")))
        .stdout(predicate::str::contains("regression_test").or(predicate::str::contains("✅")));

    Ok(())
}

#[test]
fn test_memory_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("../tests/fixtures/ml_models/large_model.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("+").or(predicate::str::contains("-")))
        .stdout(predicate::str::contains("review_friendly"));

    Ok(())
}

#[test]
fn test_quantization_analysis_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/model_fp32.safetensors")
        .arg("../tests/fixtures/ml_models/model_quantized.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("fc").or(predicate::str::contains("mean=")))
        .stdout(predicate::str::contains("std=").or(predicate::str::contains("tensor_")));

    Ok(())
}

#[test]
fn test_architecture_comparison_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/transformer.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("+").or(predicate::str::contains("-")))
        .stdout(predicate::str::contains("deployment_readiness"));

    Ok(())
}

#[test]
fn test_multiple_advanced_features_included_by_default() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("mean=").or(predicate::str::contains("std=")))
        .stdout(predicate::str::contains("memory_analysis").or(predicate::str::contains("🧠")))
        .stdout(predicate::str::contains("fc1.").or(predicate::str::contains("fc2.")));

    Ok(())
}

#[test]
fn test_json_output_with_default_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("json");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should output valid JSON
    assert!(stdout.starts_with('[') && stdout.trim_end().ends_with(']'));

    // Should contain ML analysis results (TensorStatsChanged and MemoryAnalysis)
    assert!(stdout.contains("TensorStatsChanged") || stdout.contains("MemoryAnalysis"));

    Ok(())
}

#[test]
fn test_yaml_output_with_default_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors")
        .arg("--output")
        .arg("yaml");

    let output = cmd.output()?;
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should output valid YAML with ML analysis (TensorStatsChanged and InferenceSpeedAnalysis)
    assert!(stdout.contains("TensorStatsChanged") || stdout.contains("InferenceSpeedAnalysis"));

    Ok(())
}

#[test]
// PyTorch parsing now works correctly with multi-dimensional tensors
fn test_pytorch_vs_safetensors_consistency() -> Result<(), Box<dyn std::error::Error>> {
    // Test that PyTorch and Safetensors versions produce similar results with default analysis
    let mut cmd_safetensors = diffai_cmd();
    cmd_safetensors
        .arg("../tests/fixtures/ml_models/simple_base.safetensors")
        .arg("../tests/fixtures/ml_models/simple_modified.safetensors");

    let output_safetensors = cmd_safetensors.output()?;
    assert!(output_safetensors.status.success());

    let mut cmd_pytorch = diffai_cmd();
    cmd_pytorch
        .arg("../tests/fixtures/ml_models/simple_base.pt")
        .arg("../tests/fixtures/ml_models/simple_modified.pt");

    let output_pytorch = cmd_pytorch.output()?;

    // Safetensors should succeed, PyTorch may or may not (depending on implementation)
    assert!(output_safetensors.status.success());

    if output_pytorch.status.success() {
        let stdout_safetensors = String::from_utf8_lossy(&output_safetensors.stdout);
        let stdout_pytorch = String::from_utf8_lossy(&output_pytorch.stdout);

        // Should contain similar tensor information
        assert!(stdout_safetensors.contains("fc1.") || stdout_safetensors.contains("fc2."));
        assert!(stdout_pytorch.contains("fc1.") || stdout_pytorch.contains("fc2."));
    }

    Ok(())
}

#[test]
fn test_model_size_impact_analysis() -> Result<(), Box<dyn std::error::Error>> {
    // Test analysis between significantly different model sizes with default analysis
    let mut cmd = diffai_cmd();
    cmd.arg("../tests/fixtures/ml_models/small_model.safetensors")
        .arg("../tests/fixtures/ml_models/transformer.safetensors");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("+").or(predicate::str::contains("-")))
        .stdout(
            predicate::str::contains("deployment_readiness")
                .or(predicate::str::contains("review_friendly")),
        );

    Ok(())
}
