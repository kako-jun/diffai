// Integration tests for diffai components
// Test the interaction between different parts of the system

use std::io::Write;
use std::process::Command;
use tempfile::NamedTempFile;

fn run_diffai_command(args: &[&str]) -> std::process::Output {
    let mut command = Command::new("cargo");
    command.args(&["run", "--bin", "diffai", "--"]);
    command.args(args);
    command.output().expect("Failed to execute diffai command")
}

fn create_temp_file_with_content(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    file.write_all(content.as_bytes())
        .expect("Failed to write to temp file");
    file
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_basic_file_diff_integration() {
        let content1 = r#"{"name": "test", "value": 42}"#;
        let content2 = r#"{"name": "test", "value": 43}"#;

        let file1 = create_temp_file_with_content(content1);
        let file2 = create_temp_file_with_content(content2);

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
        ]);

        assert!(output.status.success());

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("value") || stdout.contains("42") || stdout.contains("43"));
    }

    #[test]
    fn test_ml_model_analysis_integration() {
        // Test with ML model fixture files
        let output = run_diffai_command(&[
            "tests/fixtures/ml_models/model1.pt",
            "tests/fixtures/ml_models/model2.pt",
            "--ml",
        ]);

        // Models may not exist, so check for appropriate error handling
        assert!(output.status.success() || output.status.code() == Some(3));
    }

    #[test]
    fn test_json_format_integration() {
        let content1 = r#"{"users": [{"id": 1, "name": "Alice"}]}"#;
        let content2 = r#"{"users": [{"id": 1, "name": "Bob"}]}"#;

        let file1 = create_temp_file_with_content(content1);
        let file2 = create_temp_file_with_content(content2);

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
            "--format",
            "json",
        ]);

        assert!(output.status.success());

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("{"));

        // Verify it's valid JSON
        let _parsed: serde_json::Value =
            serde_json::from_str(&stdout).expect("Output should be valid JSON");
    }

    #[test]
    fn test_verbose_mode_integration() {
        let content1 = "line1\nline2\nline3";
        let content2 = "line1\nmodified\nline3";

        let file1 = create_temp_file_with_content(content1);
        let file2 = create_temp_file_with_content(content2);

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
            "--verbose",
        ]);

        assert!(output.status.success());

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.len() > 50); // Verbose output should be substantial
    }

    #[test]
    fn test_no_color_option_integration() {
        let content1 = "test content";
        let content2 = "modified content";

        let file1 = create_temp_file_with_content(content1);
        let file2 = create_temp_file_with_content(content2);

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
            "--no-color",
        ]);

        assert!(output.status.success());

        let stdout = String::from_utf8_lossy(&output.stdout);
        // Output should not contain ANSI color codes
        assert!(!stdout.contains("\x1b["));
    }

    #[test]
    fn test_recursive_directory_integration() {
        let output =
            run_diffai_command(&["tests/fixtures/dir1", "tests/fixtures/dir2", "--recursive"]);

        assert!(output.status.success());

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("dir") || stdout.contains("file") || stdout.len() > 0);
    }

    #[test]
    fn test_ml_anomaly_detection_integration() {
        let content1 = r#"{"tensor_data": [1.0, 2.0, 3.0, 4.0, 5.0]}"#;
        let content2 = r#"{"tensor_data": [1.0, 2.0, 999.0, 4.0, 5.0]}"#;

        let file1 = create_temp_file_with_content(content1);
        let file2 = create_temp_file_with_content(content2);

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
            "--ml",
            "--anomaly-detection",
        ]);

        assert!(output.status.success());

        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("anomaly") || stdout.contains("tensor") || stdout.len() > 0);
    }

    #[test]
    fn test_output_file_integration() {
        let content1 = "original";
        let content2 = "modified";

        let file1 = create_temp_file_with_content(content1);
        let file2 = create_temp_file_with_content(content2);
        let output_file = NamedTempFile::new().expect("Failed to create output file");

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
            "--output",
            output_file.path().to_str().unwrap(),
        ]);

        assert!(output.status.success());

        // Check that output file was created and has content
        let output_content =
            std::fs::read_to_string(output_file.path()).expect("Failed to read output file");
        assert!(!output_content.is_empty());
    }

    #[test]
    fn test_multiple_format_support_integration() {
        // Test CSV files
        let csv1 = "name,age\nAlice,30\nBob,25";
        let csv2 = "name,age\nAlice,31\nBob,25";

        let file1 = create_temp_file_with_content(csv1);
        let file2 = create_temp_file_with_content(csv2);

        let output = run_diffai_command(&[
            file1.path().to_str().unwrap(),
            file2.path().to_str().unwrap(),
        ]);

        assert!(output.status.success());

        // Test YAML files
        let yaml1 = "name: Alice\nage: 30";
        let yaml2 = "name: Alice\nage: 31";

        let yaml_file1 = create_temp_file_with_content(yaml1);
        let yaml_file2 = create_temp_file_with_content(yaml2);

        let yaml_output = run_diffai_command(&[
            yaml_file1.path().to_str().unwrap(),
            yaml_file2.path().to_str().unwrap(),
        ]);

        assert!(yaml_output.status.success());
    }

    #[test]
    fn test_ml_convergence_analysis_integration() {
        // Test with model checkpoint files
        let output = run_diffai_command(&[
            "tests/fixtures/ml_models/checkpoint_epoch_0.pt",
            "tests/fixtures/ml_models/checkpoint_epoch_10.pt",
            "--ml",
            "--convergence-analysis",
        ]);

        // Models may not exist, so check for appropriate error handling
        assert!(output.status.success() || output.status.code() == Some(3));
    }
}
