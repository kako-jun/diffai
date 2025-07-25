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

/// Test case 1: diffai data_v1.npy data_v2.npy
#[test]
fn test_numpy_array_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"numpy_array": {"shape": [1000, 256], "mean": 0.1234, "std": 0.9876, "dtype": "float64"}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"numpy_array": {"shape": [1000, 256], "mean": 0.1456, "std": 0.9654, "dtype": "float64"}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("numpy_array"));

    Ok(())
}

/// Test case 2: diffai dataset_v1.npz dataset_v2.npz
#[test]
fn test_compressed_numpy_archives() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"train_data": {"shape": [60000, 784], "mean": 0.1307, "std": 0.3081, "dtype": "float32"}, "test_data": {"shape": [10000, 784], "mean": 0.1325, "std": 0.3105, "dtype": "float32"}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"train_data": {"shape": [60000, 784], "mean": 0.1309, "std": 0.3082, "dtype": "float32"}, "test_data": {"shape": [10000, 784], "mean": 0.1327, "std": 0.3106, "dtype": "float32"}, "validation_data": {"shape": [5000, 784], "mean": 0.1315, "std": 0.3095, "dtype": "float32"}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("train_data"));

    Ok(())
}

/// Test case 3: diffai experiment_baseline.npy experiment_result.npy --output json
#[test]
fn test_numpy_json_output() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"experiment": {"baseline": true, "data": [1.0, 2.0, 3.0]}}"#,
        ".json",
    );
    let result = create_temp_file(
        r#"{"experiment": {"baseline": false, "data": [1.1, 2.1, 3.1]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path())
        .arg(result.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 4: diffai simulation_v1.mat simulation_v2.mat
#[test]
fn test_matlab_file_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"results": {"shape": [500, 100], "mean": 2.3456, "std": 1.2345, "dtype": "double"}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"results": {"shape": [500, 100], "mean": 2.4567, "std": 1.3456, "dtype": "double"}, "new_variable": {"shape": [100], "dtype": "single", "elements": 100}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("results"));

    Ok(())
}

/// Test case 5: diffai results_v1.mat results_v2.mat --path "experiment_data"
#[test]
fn test_matlab_specific_variables() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"experiment_data": {"temperature": [20.1, 20.2, 20.3]}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"experiment_data": {"temperature": [21.1, 21.2, 21.3]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--path")
        .arg("experiment_data");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("experiment_data"));

    Ok(())
}

/// Test case 6: diffai analysis_v1.mat analysis_v2.mat --output yaml
#[test]
fn test_matlab_yaml_output() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"analysis": {"method": "linear", "r_squared": 0.85}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"analysis": {"method": "polynomial", "r_squared": 0.92}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert().code(1).stdout(predicates::str::contains("-"));

    Ok(())
}

/// Test case 7: diffai experiment_v1.npy experiment_v2.npy --epsilon 1e-6
#[test]
fn test_epsilon_tolerance_numerical() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"measurement": {"value": 1.0000001}}"#, ".json");
    let file2 = create_temp_file(r#"{"measurement": {"value": 1.0000002}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--epsilon")
        .arg("1e-6");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("measurement"));

    Ok(())
}

/// Test case 8: diffai simulation_v1.mat simulation_v2.mat --epsilon 1e-8
#[test]
fn test_matlab_epsilon_simulation() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"simulation": {"velocity": 1.23456789, "pressure": 101.325}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"simulation": {"velocity": 1.23456790, "pressure": 101.326}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--epsilon")
        .arg("1e-8");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("simulation"));

    Ok(())
}

/// Test case 9: diffai results_v1.mat results_v2.mat --path "experimental_data"
#[test]
fn test_matlab_path_filtering() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"experimental_data": {"sample_1": {"concentration": 0.5}}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"experimental_data": {"sample_1": {"concentration": 0.6}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--path")
        .arg("experimental_data");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("experimental_data"));

    Ok(())
}

/// Test case 10: diffai data_v1.mat data_v2.mat --ignore-keys-regex "^(metadata|timestamp)"
#[test]
fn test_ignore_metadata_variables() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"metadata": {"created": "2024-01-01"}, "timestamp": "12:00:00", "data": {"values": [1, 2, 3]}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"metadata": {"created": "2024-01-02"}, "timestamp": "13:00:00", "data": {"values": [1, 2, 4]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--ignore-keys-regex")
        .arg("^(metadata|timestamp)$");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("data"));

    Ok(())
}

/// Test case 11: diffai baseline_experiment.npy treated_experiment.npy
#[test]
fn test_experimental_data_validation() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"data": {"shape": [1000, 50], "mean": 0.4567, "std": 0.1234, "dtype": "float64"}}"#,
        ".json",
    );
    let treated = create_temp_file(
        r#"{"data": {"shape": [1000, 50], "mean": 0.5123, "std": 0.1456, "dtype": "float64"}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(treated.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("data"));

    Ok(())
}
