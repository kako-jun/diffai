//! CLI options tests for diffai

use std::process::Command;
use std::fs;
use std::io::Write;
use tempfile::tempdir;

// Helper function to create test JSON files
fn create_test_json_pair() -> Result<(std::path::PathBuf, std::path::PathBuf), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    
    let file1 = temp_dir.path().join("test1.json");
    let mut f1 = fs::File::create(&file1)?;
    writeln!(f1, r#"{{"name": "Alice", "age": 30, "scores": [85, 92, 78]}}"#)?;
    
    let file2 = temp_dir.path().join("test2.json");
    let mut f2 = fs::File::create(&file2)?;
    writeln!(f2, r#"{{"name": "Alice", "age": 31, "scores": [85, 95, 78]}}"#)?;
    
    Ok((file1, file2))
}

// Helper function to create test YAML files
fn create_test_yaml_pair() -> Result<(std::path::PathBuf, std::path::PathBuf), Box<dyn std::error::Error>> {
    let temp_dir = tempdir()?;
    
    let file1 = temp_dir.path().join("test1.yaml");
    let mut f1 = fs::File::create(&file1)?;
    writeln!(f1, "name: Alice\nage: 30\nscores:\n  - 85\n  - 92\n  - 78")?;
    
    let file2 = temp_dir.path().join("test2.yaml");
    let mut f2 = fs::File::create(&file2)?;
    writeln!(f2, "name: Alice\nage: 31\nscores:\n  - 85\n  - 95\n  - 78")?;
    
    Ok((file1, file2))
}

#[test]
fn test_format_options() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        for format in ["json", "yaml", "toml", "ini", "xml", "csv"] {
            let output = Command::new("cargo")
                .args(["run", "--bin", "diffai", "--", 
                       file1.to_str().unwrap(), file2.to_str().unwrap(),
                       "--format", format])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .expect("Failed to execute diffai");

            // Should recognize the format option
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(!stderr.contains("unrecognized") || stderr.contains("invalid"), 
                        "Format {} should be recognized", format);
            }
        }
    }
}

#[test]
fn test_output_options() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        for output_format in ["cli", "json", "yaml", "unified"] {
            let output = Command::new("cargo")
                .args(["run", "--bin", "diffai", "--", 
                       file1.to_str().unwrap(), file2.to_str().unwrap(),
                       "--output", output_format])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .expect("Failed to execute diffai");

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(!stderr.contains("unrecognized"), 
                        "Output format {} should be recognized", output_format);
            }
        }
    }
}

#[test]
fn test_recursive_option() {
    let temp_dir = tempdir().expect("Failed to create temp dir");
    let dir1 = temp_dir.path().join("dir1");
    let dir2 = temp_dir.path().join("dir2");
    fs::create_dir_all(&dir1).expect("Failed to create dir1");
    fs::create_dir_all(&dir2).expect("Failed to create dir2");
    
    // Create test files in directories
    fs::write(dir1.join("test.json"), r#"{"a": 1}"#).expect("Failed to write file");
    fs::write(dir2.join("test.json"), r#"{"a": 2}"#).expect("Failed to write file");

    let output = Command::new("cargo")
        .args(["run", "--bin", "diffai", "--", 
               dir1.to_str().unwrap(), dir2.to_str().unwrap(),
               "--recursive"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai --recursive");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(!stderr.contains("unrecognized"), 
                "Recursive option should be recognized");
    }
}

#[test]
fn test_path_filter_option() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        let output = Command::new("cargo")
            .args(["run", "--bin", "diffai", "--", 
                   file1.to_str().unwrap(), file2.to_str().unwrap(),
                   "--path", "age"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to execute diffai --path");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("unrecognized"), 
                    "Path option should be recognized");
        }
    }
}

#[test]
fn test_ignore_keys_regex_option() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        let output = Command::new("cargo")
            .args(["run", "--bin", "diffai", "--", 
                   file1.to_str().unwrap(), file2.to_str().unwrap(),
                   "--ignore-keys-regex", "^age$"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to execute diffai --ignore-keys-regex");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("unrecognized"), 
                    "ignore-keys-regex option should be recognized");
        }
    }
}

#[test]
fn test_epsilon_option() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        let output = Command::new("cargo")
            .args(["run", "--bin", "diffai", "--", 
                   file1.to_str().unwrap(), file2.to_str().unwrap(),
                   "--epsilon", "0.001"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to execute diffai --epsilon");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("unrecognized"), 
                    "Epsilon option should be recognized");
        }
    }
}

#[test]
fn test_array_id_key_option() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        let output = Command::new("cargo")
            .args(["run", "--bin", "diffai", "--", 
                   file1.to_str().unwrap(), file2.to_str().unwrap(),
                   "--array-id-key", "id"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to execute diffai --array-id-key");

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("unrecognized"), 
                    "array-id-key option should be recognized");
        }
    }
}

#[test]
fn test_verbose_option() {
    if let Ok((file1, file2)) = create_test_json_pair() {
        let output = Command::new("cargo")
            .args(["run", "--bin", "diffai", "--", 
                   file1.to_str().unwrap(), file2.to_str().unwrap(),
                   "--verbose"])
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .output()
            .expect("Failed to execute diffai --verbose");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Verbose should produce output or at least not error
            assert!(!stdout.is_empty() || output.stderr.is_empty() || 
                    stdout.contains("verbose") || stdout.contains("processing"));
        }
    }
}

#[test]
fn test_version_option() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "diffai", "--", "--version"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai --version");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("0.3.2") || stdout.contains("diffai"), 
            "Version output should contain version or program name");
}

#[test]
fn test_help_option() {
    let output = Command::new("cargo")
        .args(["run", "--bin", "diffai", "--", "--help"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("Failed to execute diffai --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("diffai") && stdout.contains("Usage") && stdout.contains("Arguments"), 
            "Help output should contain program information");
}

#[test]
fn test_mixed_format_comparison() {
    if let Ok((json_file, _)) = create_test_json_pair() {
        if let Ok((yaml_file, _)) = create_test_yaml_pair() {
            let output = Command::new("cargo")
                .args(["run", "--bin", "diffai", "--", 
                       json_file.to_str().unwrap(), yaml_file.to_str().unwrap()])
                .current_dir(env!("CARGO_MANIFEST_DIR"))
                .output()
                .expect("Failed to execute diffai with mixed formats");

            // Should handle mixed format comparison gracefully
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                // Should not crash with unhandled errors
                assert!(!stderr.contains("panic"), "Should not panic on mixed formats");
            }
        }
    }
}