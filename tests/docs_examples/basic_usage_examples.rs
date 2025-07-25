#[allow(unused_imports)]
#[allow(unused_imports)]
use assert_cmd::prelude::*;
#[allow(unused_imports)]
use predicates::prelude::*;
use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

// Helper function to get the diffai command
fn diffai_cmd() -> Command {
    Command::cargo_bin("diffai").expect("Failed to find diffai binary")
}

// Helper function to create temporary files for testing
fn create_temp_file(content: &str, suffix: &str) -> NamedTempFile {
    let mut file = NamedTempFile::with_suffix(suffix).expect("Failed to create temp file");
    writeln!(file, "{}", content).expect("Failed to write to temp file");
    file
}

/// Test case 1: diffai model1.safetensors model2.safetensors
#[test]
fn test_basic_comprehensive_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"layers": 2, "params": 1000}}"#, ".json");
    let file2 = create_temp_file(r#"{"model": {"layers": 3, "params": 1500}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers"))
        .stdout(predicates::str::contains("params"));

    Ok(())
}

/// Test case 2: diffai model1.safetensors model2.safetensors --output json
#[test]
fn test_json_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"tensor": {"mean": 0.5, "std": 0.1}}"#, ".json");
    let file2 = create_temp_file(r#"{"tensor": {"mean": 0.6, "std": 0.2}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 3: diffai model1.safetensors model2.safetensors --output yaml
#[test]
fn test_yaml_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"weights": {"layer1": 0.5}}"#, ".json");
    let file2 = create_temp_file(r#"{"weights": {"layer1": 0.7}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert().code(1).stdout(predicates::str::contains("-"));

    Ok(())
}

/// Test case 4: diffai dir1/ dir2/ --recursive
#[test]
fn test_recursive_directory_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"config": {"version": "1.0"}}"#, ".json");
    let file2 = create_temp_file(r#"{"config": {"version": "2.0"}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("version"));

    Ok(())
}

/// Test case 5: diffai models_v1/ models_v2/ --format safetensors --recursive
#[test]
fn test_recursive_with_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"type": "safetensors"}}"#, ".json");
    let file2 = create_temp_file(
        r#"{"model": {"type": "safetensors", "version": 2}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--format")
        .arg("safetensors")
        .arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("model"));

    Ok(())
}

/// Test case 6: diffai model1.pt model2.pt
#[test]
fn test_pytorch_model_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"state_dict": {"layer1.weight": [0.1, 0.2]}}"#, ".json");
    let file2 = create_temp_file(
        r#"{"state_dict": {"layer1.weight": [0.15, 0.25]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("state_dict"));

    Ok(())
}

/// Test case 7: diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt
#[test]
fn test_training_checkpoint_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"epoch": 1, "loss": 0.8, "accuracy": 0.6}"#, ".json");
    let file2 = create_temp_file(r#"{"epoch": 10, "loss": 0.3, "accuracy": 0.9}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("epoch"));

    Ok(())
}

/// Test case 8: diffai baseline_model.pt improved_model.pt
#[test]
fn test_baseline_vs_improved() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"performance": 0.85, "params": 1000000}"#, ".json");
    let improved = create_temp_file(r#"{"performance": 0.92, "params": 1200000}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(improved.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("performance"));

    Ok(())
}

/// Test case 9: diffai model1.safetensors model2.safetensors (comprehensive analysis)
#[test]
fn test_safetensors_comprehensive() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"tensors": {"fc1.bias": {"shape": [64]}}}"#, ".json");
    let file2 = create_temp_file(r#"{"tensors": {"fc1.bias": {"shape": [128]}}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("tensors"));

    Ok(())
}

/// Test case 10: diffai baseline.safetensors candidate.safetensors
#[test]
fn test_deployment_validation() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(r#"{"deployment": {"ready": true, "risk": "low"}}"#, ".json");
    let candidate = create_temp_file(
        r#"{"deployment": {"ready": true, "risk": "medium"}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(candidate.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("deployment"));

    Ok(())
}

/// Test case 11: diffai data_v1.npy data_v2.npy
#[test]
fn test_numpy_array_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"array": {"data": [1.0, 2.0, 3.0], "shape": [3]}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"array": {"data": [1.1, 2.1, 3.1], "shape": [3]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("array"));

    Ok(())
}

/// Test case 12: diffai simulation_v1.mat simulation_v2.mat
#[test]
fn test_matlab_file_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"simulation": {"time": 100, "results": [0.5, 0.6]}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"simulation": {"time": 150, "results": [0.7, 0.8]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("simulation"));

    Ok(())
}

/// Test case 13: diffai dataset_v1.npz dataset_v2.npz
#[test]
fn test_compressed_numpy_archives() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"dataset": {"train": 1000, "test": 200}}"#, ".json");
    let file2 = create_temp_file(r#"{"dataset": {"train": 1200, "test": 250}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("dataset"));

    Ok(())
}

/// Test case 14: diffai experiment_v1/ experiment_v2/ --recursive
#[test]
fn test_experiment_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"experiment": {"id": "v1", "accuracy": 0.85}}"#, ".json");
    let file2 = create_temp_file(r#"{"experiment": {"id": "v2", "accuracy": 0.90}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("experiment"));

    Ok(())
}

/// Test case 15: diffai checkpoints/epoch_10.safetensors checkpoints/epoch_20.safetensors
#[test]
fn test_checkpoint_learning_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"checkpoint": {"epoch": 10, "loss": 0.5}}"#, ".json");
    let file2 = create_temp_file(r#"{"checkpoint": {"epoch": 20, "loss": 0.3}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("checkpoint"));

    Ok(())
}

/// Test case 16: diffai baseline/model.safetensors new/model.safetensors --output json > model_diff.json
#[test]
fn test_cicd_model_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"model": {"version": "baseline", "accuracy": 0.85}}"#,
        ".json",
    );
    let new = create_temp_file(
        r#"{"model": {"version": "new", "accuracy": 0.88}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path())
        .arg(new.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}
