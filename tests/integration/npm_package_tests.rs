/**
 * Integration tests for diffai npm package
 *
 * Tests the Node.js wrapper functionality and binary download mechanisms
 */
use std::path::Path;

#[allow(dead_code)]
fn create_test_package_json(dir: &Path) -> std::io::Result<()> {
    let package_json_content = r#"{
  "name": "diffai-test",
  "version": "1.0.0",
  "private": true,
  "dependencies": {}
}"#;

    std::fs::write(dir.join("package.json"), package_json_content)?;
    Ok(())
}

#[allow(dead_code)]
fn create_test_files(dir: &Path) -> std::io::Result<()> {
    let json1 = r#"{"model": "test", "version": 1, "params": 1000}"#;
    let json2 = r#"{"model": "test", "version": 2, "params": 2000}"#;

    std::fs::write(dir.join("test1.json"), json1)?;
    std::fs::write(dir.join("test2.json"), json2)?;
    Ok(())
}

#[test]
fn test_npm_package_structure() {
    let npm_dir = Path::new("../diffai-npm");

    // Check that npm package files exist
    assert!(npm_dir.exists(), "diffai-npm directory should exist");
    assert!(
        npm_dir.join("package.json").exists(),
        "package.json should exist"
    );
    assert!(npm_dir.join("index.js").exists(), "index.js should exist");
    assert!(npm_dir.join("lib.js").exists(), "lib.js should exist");
    assert!(
        npm_dir.join("scripts/download-binary.js").exists(),
        "download-binary.js should exist"
    );
    assert!(npm_dir.join("README.md").exists(), "README.md should exist");
}

#[test]
fn test_npm_package_json_validity() {
    let package_json_path = Path::new("../diffai-npm/package.json");

    if package_json_path.exists() {
        let content = std::fs::read_to_string(package_json_path)
            .expect("Should be able to read package.json");

        let json: serde_json::Value =
            serde_json::from_str(&content).expect("package.json should be valid JSON");

        // Check required fields
        assert!(json["name"].is_string(), "Should have name field");
        assert!(json["version"].is_string(), "Should have version field");
        assert!(
            json["description"].is_string(),
            "Should have description field"
        );
        assert!(json["bin"].is_object(), "Should have bin field");
        assert!(json["keywords"].is_array(), "Should have keywords field");

        // Check main field points to lib.js
        assert!(json["main"].is_string(), "Should have main field");
        assert_eq!(
            json["main"].as_str().unwrap(),
            "lib.js",
            "Main should point to lib.js"
        );

        // Check version matches Cargo.toml
        let version = json["version"].as_str().unwrap();
        assert_eq!(version, "0.2.7", "Version should match Cargo.toml");
    }
}

#[test]
fn test_npm_index_js_executable() {
    let index_js_path = Path::new("../diffai-npm/index.js");

    if index_js_path.exists() {
        let content =
            std::fs::read_to_string(index_js_path).expect("Should be able to read index.js");

        // Check for required components
        assert!(
            content.contains("#!/usr/bin/env node"),
            "Should have shebang"
        );
        assert!(content.contains("spawn"), "Should use child_process.spawn");
        assert!(content.contains("diffai"), "Should reference diffai binary");
        assert!(
            content.contains("process.argv"),
            "Should pass through arguments"
        );
        assert!(content.contains("process.exit"), "Should handle exit codes");
    }
}

#[test]
fn test_npm_lib_js_structure() {
    let lib_js_path = Path::new("../diffai-npm/lib.js");

    if lib_js_path.exists() {
        let content = std::fs::read_to_string(lib_js_path).expect("Should be able to read lib.js");

        // Check for JavaScript API components
        assert!(
            content.contains("module.exports"),
            "Should export JavaScript API"
        );
        assert!(
            content.contains("function diff("),
            "Should have diff function"
        );
        assert!(
            content.contains("function diffString("),
            "Should have diffString function"
        );
        assert!(
            content.contains("function isDiffaiAvailable("),
            "Should have isDiffaiAvailable function"
        );
        assert!(
            content.contains("function getVersion("),
            "Should have getVersion function"
        );
        assert!(
            content.contains("class DiffaiError"),
            "Should have DiffaiError class"
        );
        assert!(content.contains("spawn"), "Should use child_process.spawn");
        assert!(content.contains("@typedef"), "Should have JSDoc types");
        assert!(
            content.contains("DiffaiOptions"),
            "Should have options type definition"
        );
        assert!(
            content.contains("output: 'json'"),
            "Should support JSON output"
        );
        assert!(content.contains("stats"), "Should support stats option");
        assert!(
            content.contains("architectureComparison"),
            "Should support ML analysis options"
        );
    }
}

#[test]
fn test_npm_download_script_structure() {
    let download_script_path = Path::new("../diffai-npm/scripts/download-binary.js");

    if download_script_path.exists() {
        let content = std::fs::read_to_string(download_script_path)
            .expect("Should be able to read download-binary.js");

        // Check for required functionality
        assert!(
            content.contains("getPlatformInfo"),
            "Should detect platform"
        );
        assert!(content.contains("downloadFile"), "Should download files");
        assert!(
            content.contains("https://github.com"),
            "Should use GitHub releases"
        );
        assert!(content.contains("windows"), "Should support Windows");
        assert!(content.contains("macos"), "Should support macOS");
        assert!(content.contains("linux"), "Should support Linux");
        assert!(content.contains("x86_64"), "Should support x86_64");
        assert!(content.contains("aarch64"), "Should support ARM64");
    }
}

#[test]
#[ignore] // Requires Node.js environment
fn test_npm_package_installation() {
    // This test would require a full Node.js environment
    // and should be run in CI/CD with Node.js installed

    // For now, just verify the npm package structure exists
    let npm_src = Path::new("../diffai-npm");

    if npm_src.exists() {
        assert!(npm_src.join("package.json").exists());
        assert!(npm_src.join("index.js").exists());
        assert!(npm_src.join("scripts/download-binary.js").exists());
    }
}

#[test]
#[ignore] // Requires Node.js environment
fn test_npm_package_usage() {
    // This test would verify actual npm package usage
    // Would test: node diffai-npm/index.js test1.json test2.json
    // This requires Node.js to be available in the test environment

    let npm_index = Path::new("../diffai-npm/index.js");
    if npm_index.exists() {
        // For now, just verify the file is executable-like
        let content = std::fs::read_to_string(npm_index).expect("Should read index.js");
        assert!(content.contains("#!/usr/bin/env node"));
    }
}

#[test]
fn test_npm_test_script_exists() {
    let test_script_path = Path::new("../diffai-npm/test.js");

    if test_script_path.exists() {
        let content =
            std::fs::read_to_string(test_script_path).expect("Should be able to read test.js");

        // Check for test components
        assert!(content.contains("runTest"), "Should have test runner");
        assert!(content.contains("--version"), "Should test version command");
        assert!(content.contains("--help"), "Should test help command");
        assert!(content.contains("spawn"), "Should use child_process");

        // Check for JavaScript API tests
        assert!(
            content.contains("require('./lib.js')"),
            "Should import lib.js"
        );
        assert!(
            content.contains("isDiffaiAvailable"),
            "Should test JavaScript API"
        );
        assert!(
            content.contains("diffString"),
            "Should test diffString function"
        );
        assert!(
            content.contains("DiffaiError"),
            "Should test error handling"
        );
    }
}

#[test]
fn test_npm_readme_completeness() {
    let readme_path = Path::new("../diffai-npm/README.md");

    if readme_path.exists() {
        let content =
            std::fs::read_to_string(readme_path).expect("Should be able to read README.md");

        // Check for essential documentation sections
        assert!(content.contains("# diffai"), "Should have main title");
        assert!(
            content.contains("## Installation"),
            "Should have installation section"
        );
        assert!(
            content.contains("npm install"),
            "Should show npm install command"
        );
        assert!(content.contains("## Usage"), "Should have usage section");
        assert!(content.contains("```bash"), "Should have code examples");
        assert!(content.contains("--stats"), "Should document stats option");
        assert!(
            content.contains("architecture-comparison"),
            "Should document Phase 3 features"
        );
        assert!(
            content.contains("safetensors"),
            "Should mention supported formats"
        );
        assert!(
            content.contains("pytorch") || content.contains("PyTorch"),
            "Should mention PyTorch support"
        );
        assert!(
            content.contains("numpy") || content.contains("NumPy"),
            "Should mention NumPy support"
        );
        assert!(content.contains("LICENSE"), "Should mention license");
    }
}

#[test]
fn test_npm_package_dependencies() {
    let package_json_path = Path::new("../diffai-npm/package.json");

    if package_json_path.exists() {
        let content = std::fs::read_to_string(package_json_path)
            .expect("Should be able to read package.json");

        let json: serde_json::Value =
            serde_json::from_str(&content).expect("package.json should be valid JSON");

        // Check that there are no runtime dependencies (pure wrapper)
        let dependencies = json.get("dependencies");
        if let Some(deps) = dependencies {
            assert!(
                deps.as_object().unwrap().is_empty(),
                "npm package should have no runtime dependencies"
            );
        }

        // Check engines requirement
        if let Some(engines) = json.get("engines") {
            assert!(
                engines["node"].is_string(),
                "Should specify Node.js version requirement"
            );
        }
    }
}

#[test]
fn test_npm_platform_support_metadata() {
    let package_json_path = Path::new("../diffai-npm/package.json");

    if package_json_path.exists() {
        let content = std::fs::read_to_string(package_json_path)
            .expect("Should be able to read package.json");

        let json: serde_json::Value =
            serde_json::from_str(&content).expect("package.json should be valid JSON");

        // Check OS and CPU support
        if let Some(os) = json.get("os") {
            let os_list = os.as_array().unwrap();
            assert!(os_list.iter().any(|v| v.as_str() == Some("linux")));
            assert!(os_list.iter().any(|v| v.as_str() == Some("darwin")));
            assert!(os_list.iter().any(|v| v.as_str() == Some("win32")));
        }

        if let Some(cpu) = json.get("cpu") {
            let cpu_list = cpu.as_array().unwrap();
            assert!(cpu_list.iter().any(|v| v.as_str() == Some("x64")));
            assert!(cpu_list.iter().any(|v| v.as_str() == Some("arm64")));
        }
    }
}
