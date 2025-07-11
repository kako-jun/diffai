/**
 * Integration tests for diffai Python package
 *
 * Tests the Python wrapper functionality, API structure, and package integrity
 */
use std::path::Path;

#[allow(dead_code)]
fn create_test_files(dir: &Path) -> std::io::Result<()> {
    let json1 = r#"{"model": "test", "version": 1, "params": 1000}"#;
    let json2 = r#"{"model": "test", "version": 2, "params": 2000}"#;

    std::fs::write(dir.join("test1.json"), json1)?;
    std::fs::write(dir.join("test2.json"), json2)?;
    Ok(())
}

#[test]
fn test_python_package_structure() {
    let python_dir = Path::new("diffai-python");

    // Check that Python package files exist
    assert!(python_dir.exists(), "diffai-python directory should exist");
    assert!(
        python_dir.join("pyproject.toml").exists(),
        "pyproject.toml should exist"
    );
    assert!(
        python_dir.join("src/diffai/__init__.py").exists(),
        "__init__.py should exist"
    );
    assert!(
        python_dir.join("src/diffai/diffai.py").exists(),
        "diffai.py should exist"
    );
    assert!(
        python_dir.join("src/diffai/compat.py").exists(),
        "compat.py should exist"
    );
    assert!(
        python_dir.join("src/diffai/installer.py").exists(),
        "installer.py should exist"
    );
    assert!(
        python_dir.join("README.md").exists(),
        "README.md should exist"
    );
    assert!(
        python_dir.join("test_integration.py").exists(),
        "test_integration.py should exist"
    );
}

#[test]
fn test_pyproject_toml_validity() {
    let pyproject_path = Path::new("diffai-python/pyproject.toml");

    if pyproject_path.exists() {
        let content =
            std::fs::read_to_string(pyproject_path).expect("Should be able to read pyproject.toml");

        // Parse TOML
        let toml: toml::Value =
            toml::from_str(&content).expect("pyproject.toml should be valid TOML");

        // Check required project fields
        let project = &toml["project"];
        assert!(project["name"].is_str(), "Should have name field");
        assert!(project["version"].is_str(), "Should have version field");
        assert!(
            project["description"].is_str(),
            "Should have description field"
        );
        assert!(project["readme"].is_str(), "Should have readme field");
        assert!(project["license"].is_str(), "Should have license field");
        assert!(
            project["requires-python"].is_str(),
            "Should have Python version requirement"
        );

        // Check version matches Cargo.toml
        let version = project["version"].as_str().unwrap();
        assert_eq!(version, "0.2.6", "Version should match Cargo.toml");

        // Check build system
        let build_system = &toml["build-system"];
        assert!(
            build_system["requires"].is_array(),
            "Should specify build requirements"
        );
        assert!(
            build_system["build-backend"].is_str(),
            "Should specify build backend"
        );

        // Check classifiers
        if let Some(classifiers) = project["classifiers"].as_array() {
            assert!(!classifiers.is_empty(), "Should have classifiers");

            let classifier_strings: Vec<&str> =
                classifiers.iter().filter_map(|v| v.as_str()).collect();

            assert!(
                classifier_strings.iter().any(|s| s.contains("Python :: 3")),
                "Should support Python 3"
            );
            assert!(
                classifier_strings.iter().any(|s| s.contains("MIT License")),
                "Should specify MIT license"
            );
        }

        // Check keywords
        if let Some(keywords) = project["keywords"].as_array() {
            let keyword_strings: Vec<&str> = keywords.iter().filter_map(|v| v.as_str()).collect();

            assert!(
                keyword_strings.iter().any(|s| s.contains("ai")),
                "Should have AI keyword"
            );
            assert!(
                keyword_strings.iter().any(|s| s.contains("ml")),
                "Should have ML keyword"
            );
            assert!(
                keyword_strings.iter().any(|s| s.contains("diff")),
                "Should have diff keyword"
            );
        }
    }
}

#[test]
fn test_python_init_py_structure() {
    let init_path = Path::new("diffai-python/src/diffai/__init__.py");

    if init_path.exists() {
        let content =
            std::fs::read_to_string(init_path).expect("Should be able to read __init__.py");

        // Check for essential imports and exports
        assert!(
            content.contains("from .diffai import"),
            "Should import from diffai module"
        );
        assert!(content.contains("diff"), "Should export diff function");
        assert!(content.contains("DiffOptions"), "Should export DiffOptions");
        assert!(content.contains("DiffResult"), "Should export DiffResult");
        assert!(
            content.contains("OutputFormat"),
            "Should export OutputFormat"
        );
        assert!(
            content.contains("DiffaiError"),
            "Should export error classes"
        );
        assert!(content.contains("__all__"), "Should have __all__ list");
        assert!(content.contains("__version__"), "Should export version");

        // Check for backward compatibility
        assert!(
            content.contains("from .compat import"),
            "Should import compatibility layer"
        );
        assert!(
            content.contains("diffai_diff"),
            "Should export legacy functions"
        );
    }
}

#[test]
fn test_python_diffai_py_structure() {
    let diffai_path = Path::new("diffai-python/src/diffai/diffai.py");

    if diffai_path.exists() {
        let content =
            std::fs::read_to_string(diffai_path).expect("Should be able to read diffai.py");

        // Check for essential classes and functions
        assert!(
            content.contains("class DiffOptions"),
            "Should define DiffOptions class"
        );
        assert!(
            content.contains("class DiffResult"),
            "Should define DiffResult class"
        );
        assert!(
            content.contains("class OutputFormat"),
            "Should define OutputFormat enum"
        );
        assert!(content.contains("def diff("), "Should define diff function");
        assert!(
            content.contains("def verify_installation("),
            "Should define verify_installation"
        );
        assert!(
            content.contains("def run_diffai("),
            "Should define run_diffai"
        );

        // Check for all CLI options (from actual CLI)
        assert!(
            content.contains("stats: bool"),
            "Should support stats option"
        );
        assert!(
            content.contains("architecture_comparison: bool"),
            "Should support architecture comparison"
        );
        assert!(
            content.contains("memory_analysis: bool"),
            "Should support memory analysis"
        );
        assert!(
            content.contains("anomaly_detection: bool"),
            "Should support anomaly detection"
        );
        assert!(
            content.contains("convergence_analysis: bool"),
            "Should support convergence analysis"
        );
        assert!(
            content.contains("gradient_analysis: bool"),
            "Should support gradient analysis"
        );
        assert!(
            content.contains("similarity_matrix: bool"),
            "Should support similarity matrix"
        );

        // Check for error handling
        assert!(
            content.contains("class DiffaiError"),
            "Should define base error class"
        );
        assert!(
            content.contains("class BinaryNotFoundError"),
            "Should define binary error"
        );
        assert!(
            content.contains("class InvalidInputError"),
            "Should define input error"
        );

        // Check version
        assert!(
            content.contains("__version__ = \"0.2.6\""),
            "Should have correct version"
        );
    }
}

#[test]
fn test_python_installer_py_functionality() {
    let installer_path = Path::new("diffai-python/src/diffai/installer.py");

    if installer_path.exists() {
        let content =
            std::fs::read_to_string(installer_path).expect("Should be able to read installer.py");

        // Check for platform detection
        assert!(
            content.contains("def get_platform_info"),
            "Should have platform detection"
        );
        assert!(content.contains("platform.system()"), "Should detect OS");
        assert!(
            content.contains("platform.machine()"),
            "Should detect architecture"
        );

        // Check for supported platforms
        assert!(content.contains("Windows"), "Should support Windows");
        assert!(content.contains("Darwin"), "Should support macOS");
        assert!(content.contains("Linux"), "Should support Linux");
        assert!(content.contains("x86_64"), "Should support x86_64");
        assert!(content.contains("aarch64"), "Should support ARM64");

        // Check for download functionality
        assert!(
            content.contains("def download_file"),
            "Should have download function"
        );
        assert!(
            content.contains("urllib.request"),
            "Should use urllib for downloads"
        );
        assert!(
            content.contains("def extract_archive"),
            "Should extract archives"
        );
        assert!(content.contains("zipfile"), "Should handle ZIP files");
        assert!(content.contains("tarfile"), "Should handle TAR files");

        // Check for binary verification
        assert!(
            content.contains("def verify_binary"),
            "Should verify downloaded binary"
        );
        assert!(
            content.contains("--version"),
            "Should test binary with --version"
        );

        // Check for GitHub releases integration
        assert!(content.contains("github.com"), "Should use GitHub releases");
        assert!(
            content.contains("PACKAGE_VERSION"),
            "Should use package version"
        );
    }
}

#[test]
fn test_python_compat_py_backward_compatibility() {
    let compat_path = Path::new("diffai-python/src/diffai/compat.py");

    if compat_path.exists() {
        let content =
            std::fs::read_to_string(compat_path).expect("Should be able to read compat.py");

        // Check for legacy function definitions
        assert!(
            content.contains("def diffai_diff("),
            "Should have legacy diff function"
        );
        assert!(
            content.contains("def diffai_diff_files("),
            "Should have legacy file diff"
        );
        assert!(
            content.contains("def check_diffai_binary("),
            "Should have binary check"
        );
        assert!(
            content.contains("def get_stats("),
            "Should have legacy stats function"
        );
        assert!(
            content.contains("def get_json_diff("),
            "Should have legacy JSON diff"
        );
        assert!(
            content.contains("def compare_models("),
            "Should have legacy model comparison"
        );

        // Check for legacy configuration
        assert!(
            content.contains("class LegacyDiffConfig"),
            "Should have legacy config class"
        );

        // Check for proper imports from new API
        assert!(
            content.contains("from .diffai import"),
            "Should import from new API"
        );
        assert!(
            content.contains("DiffOptions"),
            "Should use new DiffOptions"
        );
        assert!(
            content.contains("OutputFormat"),
            "Should use new OutputFormat"
        );
    }
}

#[test]
fn test_python_readme_completeness() {
    let readme_path = Path::new("diffai-python/README.md");

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
            content.contains("pip install"),
            "Should show pip install command"
        );
        assert!(content.contains("## Usage"), "Should have usage section");
        assert!(
            content.contains("```python"),
            "Should have Python code examples"
        );
        assert!(
            content.contains("import diffai"),
            "Should show import statement"
        );
        assert!(content.contains("diffai.diff"), "Should show basic usage");
        assert!(
            content.contains("DiffOptions"),
            "Should document DiffOptions"
        );
        assert!(
            content.contains("OutputFormat"),
            "Should document OutputFormat"
        );

        // Check for ML analysis documentation
        assert!(
            content.contains("stats=True"),
            "Should document stats option"
        );
        assert!(
            content.contains("architecture_comparison"),
            "Should document Phase 3 features"
        );
        assert!(
            content.contains("memory_analysis"),
            "Should document memory analysis"
        );
        assert!(
            content.contains("anomaly_detection"),
            "Should document anomaly detection"
        );

        // Check for supported formats
        assert!(
            content.contains("safetensors"),
            "Should mention Safetensors support"
        );
        assert!(
            content.contains("pytorch"),
            "Should mention PyTorch support"
        );
        assert!(content.contains("numpy"), "Should mention NumPy support");
        assert!(content.contains("matlab"), "Should mention MATLAB support");

        // Check for integration examples
        assert!(
            content.contains("MLflow"),
            "Should have MLflow integration example"
        );
        assert!(
            content.contains("Weights & Biases"),
            "Should have W&B integration example"
        );

        // Check for error handling documentation
        assert!(
            content.contains("DiffaiError"),
            "Should document error handling"
        );
        assert!(
            content.contains("BinaryNotFoundError"),
            "Should document binary errors"
        );

        // Check for license
        assert!(content.contains("MIT License"), "Should mention license");
    }
}

#[test]
fn test_python_integration_test_completeness() {
    let test_path = Path::new("diffai-python/test_integration.py");

    if test_path.exists() {
        let content =
            std::fs::read_to_string(test_path).expect("Should be able to read test_integration.py");

        // Check for test class structure
        assert!(
            content.contains("class TestDiffaiIntegration"),
            "Should have integration test class"
        );
        assert!(
            content.contains("class TestDiffaiWithoutBinary"),
            "Should have no-binary test class"
        );

        // Check for essential test methods
        assert!(
            content.contains("test_installation_verification"),
            "Should test installation"
        );
        assert!(
            content.contains("test_basic_diff"),
            "Should test basic diff"
        );
        assert!(
            content.contains("test_json_output"),
            "Should test JSON output"
        );
        assert!(content.contains("test_diff_options"), "Should test options");
        assert!(
            content.contains("test_string_comparison"),
            "Should test string comparison"
        );
        assert!(
            content.contains("test_error_handling"),
            "Should test error handling"
        );
        assert!(
            content.contains("test_ml_analysis_options"),
            "Should test ML options"
        );
        assert!(
            content.contains("test_backward_compatibility"),
            "Should test compatibility"
        );

        // Check for unittest usage
        assert!(
            content.contains("import unittest"),
            "Should use unittest framework"
        );
        assert!(
            content.contains("def setUp(self)"),
            "Should have setup method"
        );
        assert!(
            content.contains("def tearDown(self)"),
            "Should have teardown method"
        );

        // Check for proper error handling in tests
        assert!(
            content.contains("self.assertRaises"),
            "Should test exceptions"
        );
        assert!(
            content.contains("BinaryNotFoundError"),
            "Should handle binary errors"
        );
        assert!(
            content.contains("self.skipTest"),
            "Should skip when binary unavailable"
        );
    }
}

#[test]
#[ignore] // Requires Python environment
fn test_python_package_syntax() {
    // This test would verify Python syntax is valid
    // Requires Python to be available in test environment

    let python_files = [
        "diffai-python/src/diffai/__init__.py",
        "diffai-python/src/diffai/diffai.py",
        "diffai-python/src/diffai/compat.py",
        "diffai-python/src/diffai/installer.py",
        "diffai-python/test_integration.py",
    ];

    for file_path in &python_files {
        let path = Path::new(file_path);
        if path.exists() {
            // Would run: python -m py_compile <file>
            // This verifies Python syntax without executing code
        }
    }
}

#[test]
#[ignore] // Requires Python environment
fn test_python_package_imports() {
    // This test would verify all imports work correctly
    // Requires Python environment with package installed

    // Would test:
    // python -c "import diffai; print(diffai.__version__)"
    // python -c "from diffai import diff, DiffOptions, OutputFormat"
    // python -c "import diffai.compat"

    // For now, just verify the Python files exist
    let init_file = Path::new("diffai-python/src/diffai/__init__.py");
    if init_file.exists() {
        let content = std::fs::read_to_string(init_file).expect("Should read __init__.py");
        assert!(content.contains("__version__"));
    }
}

#[test]
fn test_python_package_type_annotations() {
    let diffai_path = Path::new("diffai-python/src/diffai/diffai.py");

    if diffai_path.exists() {
        let content =
            std::fs::read_to_string(diffai_path).expect("Should be able to read diffai.py");

        // Check for proper type annotations
        assert!(
            content.contains("from typing import"),
            "Should import typing"
        );
        assert!(
            content.contains("-> DiffResult"),
            "Should annotate return types"
        );
        assert!(content.contains(": Optional["), "Should use Optional types");
        assert!(content.contains(": List["), "Should use List types");
        assert!(content.contains(": Dict["), "Should use Dict types");
        assert!(content.contains(": Union["), "Should use Union types");

        // Check for dataclass usage
        assert!(
            content.contains("@dataclass"),
            "Should use dataclass decorator"
        );
        assert!(
            content.contains("from dataclasses import"),
            "Should import dataclasses"
        );
    }
}
