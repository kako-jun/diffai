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

/// Test case 1: diffai model1.pt model2.pt
#[test]
fn test_pytorch_models_comprehensive() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"state_dict": {"fc1.weight": [0.1, 0.2], "fc1.bias": [0.01]}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"state_dict": {"fc1.weight": [0.15, 0.25], "fc1.bias": [0.02]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("state_dict"));

    Ok(())
}

/// Test case 2: diffai model1.safetensors model2.safetensors
#[test]
fn test_safetensors_models_comprehensive() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"tensors": {"layer1.weight": {"shape": [64, 32]}}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"tensors": {"layer1.weight": {"shape": [64, 64]}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("tensors"));

    Ok(())
}

/// Test case 3: diffai pretrained.safetensors finetuned.safetensors
#[test]
fn test_automatic_format_detection() -> Result<(), Box<dyn std::error::Error>> {
    let pretrained = create_temp_file(
        r#"{"model": {"pretrained": true, "accuracy": 0.85}}"#,
        ".json",
    );
    let finetuned = create_temp_file(
        r#"{"model": {"pretrained": false, "accuracy": 0.92}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(pretrained.path()).arg(finetuned.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("accuracy"));

    Ok(())
}

/// Test case 4: diffai model1.safetensors model2.safetensors --epsilon 1e-6
#[test]
fn test_epsilon_tolerance_minor() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"weights": {"layer1": 0.1000000}}"#, ".json");
    let file2 = create_temp_file(r#"{"weights": {"layer1": 0.1000001}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--epsilon")
        .arg("1e-6");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("weights"));

    Ok(())
}

/// Test case 5: diffai fp32_model.safetensors int8_model.safetensors --epsilon 0.1
#[test]
fn test_quantization_analysis_epsilon() -> Result<(), Box<dyn std::error::Error>> {
    let fp32 = create_temp_file(
        r#"{"model": {"precision": "fp32", "weights": [0.123, 0.456]}}"#,
        ".json",
    );
    let int8 = create_temp_file(
        r#"{"model": {"precision": "int8", "weights": [0.12, 0.46]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(fp32.path())
        .arg(int8.path())
        .arg("--epsilon")
        .arg("0.1");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("precision"));

    Ok(())
}

/// Test case 6: diffai model1.pt model2.pt --output json
#[test]
fn test_json_output_automation() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"layers": {"conv1": {"filters": 32}}}"#, ".json");
    let file2 = create_temp_file(r#"{"layers": {"conv1": {"filters": 64}}}"#, ".json");

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

/// Test case 7: diffai model1.pt model2.pt --output yaml
#[test]
fn test_yaml_output_readability() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"parameters": {"learning_rate": 0.01}}"#, ".json");
    let file2 = create_temp_file(r#"{"parameters": {"learning_rate": 0.001}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--output")
        .arg("yaml");
    cmd.assert().code(1).stdout(predicates::str::contains("-"));

    Ok(())
}

/// Test case 8: diffai model1.pt model2.pt --output json > changes.json
#[test]
fn test_pipe_to_file() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"metrics": {"loss": 0.5}}"#, ".json");
    let file2 = create_temp_file(r#"{"metrics": {"loss": 0.3}}"#, ".json");

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

/// Test case 9: diffai model1.safetensors model2.safetensors --path "classifier"
#[test]
fn test_focus_specific_layers() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"classifier": {"weight": [0.1, 0.2]}}"#, ".json");
    let file2 = create_temp_file(r#"{"classifier": {"weight": [0.15, 0.25]}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--path")
        .arg("classifier");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("classifier"));

    Ok(())
}

/// Test case 10: diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(timestamp|_metadata)"
#[test]
fn test_ignore_metadata() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"timestamp": "2024-01-01", "weights": {"layer1": 0.5}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"timestamp": "2024-01-02", "weights": {"layer1": 0.6}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--ignore-keys-regex")
        .arg("^timestamp$");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("weights"));

    Ok(())
}

/// Test case 11: diffai pretrained_bert.safetensors finetuned_bert.safetensors
#[test]
fn test_finetuning_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let pretrained = create_temp_file(
        r#"{"bert": {"encoder": {"attention": {"query": {"weight": [0.001]}}}}, "classifier": {"weight": [0.0]}}"#,
        ".json",
    );
    let finetuned = create_temp_file(
        r#"{"bert": {"encoder": {"attention": {"query": {"weight": [0.0023]}}}}, "classifier": {"weight": [0.0145]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(pretrained.path()).arg(finetuned.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("bert"));

    Ok(())
}

/// Test case 12: diffai model_fp32.safetensors model_int8.safetensors --epsilon 0.1
#[test]
fn test_quantization_impact_assessment() -> Result<(), Box<dyn std::error::Error>> {
    let fp32 = create_temp_file(
        r#"{"conv1": {"weight": {"mean": 0.0045, "std": 0.2341}}}"#,
        ".json",
    );
    let int8 = create_temp_file(
        r#"{"conv1": {"weight": {"mean": 0.0043, "std": 0.2298}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(fp32.path())
        .arg(int8.path())
        .arg("--epsilon")
        .arg("0.1");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("conv1"));

    Ok(())
}

/// Test case 13: diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt
#[test]
fn test_training_progress_tracking() -> Result<(), Box<dyn std::error::Error>> {
    let epoch10 = create_temp_file(
        r#"{"layers": {"0": {"weight": {"mean": -0.0012, "std": 1.2341}}, "1": {"bias": {"mean": 0.1234, "std": 0.4567}}}}"#,
        ".json",
    );
    let epoch50 = create_temp_file(
        r#"{"layers": {"0": {"weight": {"mean": 0.0034, "std": 0.8907}}, "1": {"bias": {"mean": 0.0567, "std": 0.3210}}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(epoch10.path()).arg(epoch50.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("layers"));

    Ok(())
}

/// Test case 14: diffai resnet50.safetensors efficientnet_b0.safetensors
#[test]
fn test_architecture_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let resnet = create_temp_file(
        r#"{"features": {"conv1": {"weight": {"shape": [64, 3, 7, 7]}}, "layer4": {"2": {"downsample": {"0": {"weight": {"shape": [2048, 1024, 1, 1]}}}}}}}"#,
        ".json",
    );
    let efficientnet = create_temp_file(
        r#"{"features": {"conv1": {"weight": {"shape": [32, 3, 3, 3]}}, "mbconv": {"expand_conv": {"weight": {"shape": [96, 32, 1, 1]}}}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(resnet.path()).arg(efficientnet.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("features"));

    Ok(())
}

/// Test case 15: diffai --recursive model_dir1/ model_dir2/
#[test]
fn test_recursive_mode_large_models() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"large_model": {"size": "1GB", "parameters": 1000000}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"large_model": {"size": "1.2GB", "parameters": 1200000}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path()).arg(file2.path()).arg("--recursive");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("large_model"));

    Ok(())
}

/// Test case 16: diffai model1.safetensors model2.safetensors --path "tensor.classifier"
#[test]
fn test_focus_analysis_specific_parts() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"tensor": {"classifier": {"weight": [0.1, 0.2]}}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"tensor": {"classifier": {"weight": [0.15, 0.25]}}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--path")
        .arg("tensor.classifier");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("classifier"));

    Ok(())
}

/// Test case 17: diffai model1.safetensors model2.safetensors --epsilon 1e-3
#[test]
fn test_higher_epsilon_faster_comparison() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"precision": 0.001234}}"#, ".json");
    let file2 = create_temp_file(r#"{"model": {"precision": 0.001567}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--epsilon")
        .arg("1e-3");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("model"));

    Ok(())
}

/// Test case 18: diffai --verbose model1.safetensors model2.safetensors
#[test]
fn test_verbose_mode_processing_info() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"processing": {"stage": "training"}}"#, ".json");
    let file2 = create_temp_file(r#"{"processing": {"stage": "validation"}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg("--verbose").arg(file1.path()).arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("processing"));

    Ok(())
}

/// Test case 19: diffai --architecture-comparison model1.safetensors model2.safetensors
#[test]
fn test_architecture_differences_only() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(
        r#"{"architecture": {"type": "transformer", "layers": 12}}"#,
        ".json",
    );
    let file2 = create_temp_file(
        r#"{"architecture": {"type": "transformer", "layers": 24}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(file1.path())
        .arg(file2.path())
        .arg("--architecture-comparison");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("architecture"));

    Ok(())
}

/// Test case 20: diffai model1.safetensors model2.safetensors --output json (subprocess run)
#[test]
fn test_subprocess_run_json() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"model": {"version": "1.0"}}"#, ".json");
    let file2 = create_temp_file(r#"{"model": {"version": "2.0"}}"#, ".json");

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

/// Test case 21: diffai models/baseline.safetensors models/candidate.safetensors --output json > model_diff.json
#[test]
fn test_cicd_compare_models() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"model": {"type": "baseline", "accuracy": 0.85}}"#,
        ".json",
    );
    let candidate = create_temp_file(
        r#"{"model": {"type": "candidate", "accuracy": 0.88}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path())
        .arg(candidate.path())
        .arg("--output")
        .arg("json");
    cmd.assert()
        .code(1)
        .stdout(predicates::str::is_match(r#"\[.*\]"#).unwrap());

    Ok(())
}

/// Test case 22: diffai model.safetensors model.safetensors
#[test]
fn test_single_model_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let file = create_temp_file(r#"{"model": {"layers": 6, "parameters": 100000}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg(file.path()).arg(file.path());
    cmd.assert().code(0);

    Ok(())
}

/// Test case 23: diffai --format safetensors model1.safetensors model2.safetensors
#[test]
fn test_explicit_format() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"safetensors": {"format": "explicit"}}"#, ".json");
    let file2 = create_temp_file(
        r#"{"safetensors": {"format": "explicit", "version": 2}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg("--format")
        .arg("safetensors")
        .arg(file1.path())
        .arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("safetensors"));

    Ok(())
}

/// Test case 24: diffai --epsilon 1e-3 large1.safetensors large2.safetensors
#[test]
fn test_memory_optimization_epsilon() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"large": {"tensor": [0.001, 0.002, 0.003]}}"#, ".json");
    let file2 = create_temp_file(
        r#"{"large": {"tensor": [0.0015, 0.0025, 0.0035]}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg("--epsilon")
        .arg("1e-3")
        .arg(file1.path())
        .arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("large"));

    Ok(())
}

/// Test case 25: diffai --path "tensor.classifier" large1.safetensors large2.safetensors
#[test]
fn test_memory_optimization_path() -> Result<(), Box<dyn std::error::Error>> {
    let file1 = create_temp_file(r#"{"tensor": {"classifier": {"weight": [0.1]}}}"#, ".json");
    let file2 = create_temp_file(r#"{"tensor": {"classifier": {"weight": [0.2]}}}"#, ".json");

    let mut cmd = diffai_cmd();
    cmd.arg("--path")
        .arg("tensor.classifier")
        .arg(file1.path())
        .arg(file2.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("classifier"));

    Ok(())
}

/// Test case 26: diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors
#[test]
fn test_comprehensive_analysis_automatic() -> Result<(), Box<dyn std::error::Error>> {
    let epoch10 = create_temp_file(
        r#"{"checkpoint": {"epoch": 10, "loss": 0.5, "accuracy": 0.8}}"#,
        ".json",
    );
    let epoch20 = create_temp_file(
        r#"{"checkpoint": {"epoch": 20, "loss": 0.3, "accuracy": 0.9}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(epoch10.path()).arg(epoch20.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("checkpoint"));

    Ok(())
}

/// Test case 27: diffai baseline.safetensors experiment.safetensors
#[test]
fn test_experimental_comparison_automatic() -> Result<(), Box<dyn std::error::Error>> {
    let baseline = create_temp_file(
        r#"{"experiment": {"type": "baseline", "performance": 0.85}}"#,
        ".json",
    );
    let experiment = create_temp_file(
        r#"{"experiment": {"type": "enhanced", "performance": 0.92}}"#,
        ".json",
    );

    let mut cmd = diffai_cmd();
    cmd.arg(baseline.path()).arg(experiment.path());
    cmd.assert()
        .code(1)
        .stdout(predicates::str::contains("experiment"));

    Ok(())
}
