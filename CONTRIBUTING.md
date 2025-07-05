# Contributing to diffai

Thank you for your interest in contributing to diffai! We welcome contributions from the AI/ML community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [AI/ML Specific Contributions](#aiml-specific-contributions)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- **Rust 1.70+**: Install from [rustup.rs](https://rustup.rs/)
- **Git**: For version control
- **Python 3.8+** (optional): For creating test ML models

### First-time Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/diffai.git
   cd diffai
   ```

2. **Build the project**
   ```bash
   cargo build
   ```

3. **Run tests**
   ```bash
   cargo test
   ```

4. **Install for local testing**
   ```bash
   cargo install --path diffai-cli
   ```

## Development Setup

### Recommended Tools

- **IDE**: VS Code with rust-analyzer extension
- **Formatter**: `cargo fmt` (auto-formatting)
- **Linter**: `cargo clippy` (additional checks)
- **Benchmark**: `cargo bench` (performance testing)

### Environment Configuration

```bash
# Set up git hooks (optional)
cp scripts/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Install cargo-watch for auto-rebuild (optional)
cargo install cargo-watch
cargo watch -x test
```

## Project Structure

```
diffai/
├── diffai-core/          # Core library (the heart of diffai)
│   ├── src/lib.rs       # Main library code
│   └── benches/         # Performance benchmarks
├── diffai-cli/          # CLI wrapper
│   └── src/main.rs      # CLI application
├── tests/               # Test suite
│   ├── fixtures/        # Test data files
│   ├── integration/     # CLI integration tests
│   └── unit/           # Core library unit tests
├── scripts/             # Development utilities
├── docs/               # Documentation
└── examples/           # Usage examples
```

### Key Components

#### 1. Core Library (`diffai-core`)
- **Format parsers**: JSON, YAML, TOML, XML, INI, CSV
- **ML parsers**: PyTorch (.pt/.pth), Safetensors (.safetensors)
- **Diff engine**: Semantic comparison logic
- **ML analyzer**: Tensor statistics calculation
- **Output formatters**: CLI, JSON, YAML outputs

#### 2. CLI Application (`diffai-cli`)
- **Argument parsing**: Using `clap`
- **File I/O**: Reading various formats
- **Output formatting**: Human-readable display
- **Error handling**: User-friendly error messages

## Making Changes

### 1. Choose an Issue

Look for issues labeled:
- `good first issue`: Great for new contributors
- `help wanted`: Community input needed
- `ml-feature`: AI/ML specific features
- `bug`: Bug fixes
- `enhancement`: New features

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Coding Standards

#### Rust Style Guide
```rust
// Use descriptive names
fn parse_tensor_statistics(data: &[f32]) -> TensorStats {
    // Implementation
}

// Document public APIs
/// Calculates tensor statistics from raw data
/// 
/// # Arguments
/// * `data` - Raw tensor data as f32 slice
/// * `shape` - Tensor dimensions
/// 
/// # Returns
/// TensorStats containing mean, std, min, max
pub fn calculate_stats(data: &[f32], shape: &[usize]) -> TensorStats {
    // Implementation
}
```

#### Error Handling
```rust
use anyhow::{Result, Context};

fn parse_model_file(path: &Path) -> Result<ModelData> {
    let data = fs::read(path)
        .context(format!("Failed to read model file: {}", path.display()))?;
    
    parse_safetensors(&data)
        .context("Failed to parse safetensors format")
}
```

#### Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_stats_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = calculate_stats(&data, &[5]);
        
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.total_params, 5);
    }
}
```

### 4. Code Formatting

Always run before committing:
```bash
cargo fmt
cargo clippy
```

## Testing

### Running Tests

```bash
# All tests
cargo test

# Unit tests only
cargo test --lib

# Integration tests only
cargo test --test '*'

# Specific test
cargo test test_tensor_stats

# With output
cargo test -- --nocapture
```

### Test Categories

#### 1. Unit Tests (`tests/unit/`)
- **Core functionality**: diff algorithms, parsers
- **ML functions**: tensor statistics, model parsing
- **Utilities**: helper functions

#### 2. Integration Tests (`tests/integration/`)
- **CLI behavior**: command-line interface testing
- **File I/O**: reading various formats
- **End-to-end**: complete workflows

#### 3. Performance Tests (`diffai-core/benches/`)
- **Benchmarks**: performance comparison
- **Memory usage**: large file handling
- **Optimization**: identifying bottlenecks

### Writing Tests

#### For New ML Features
```rust
#[test]
fn test_new_ml_feature() {
    // Create test data
    let model1 = create_test_model("basic");
    let model2 = create_test_model("modified");
    
    // Test the feature
    let result = new_ml_function(&model1, &model2);
    
    // Assert expected behavior
    assert!(result.is_ok());
    assert_eq!(result.unwrap().changes.len(), 3);
}
```

#### For CLI Features
```rust
#[test]
fn test_new_cli_option() -> Result<(), Box<dyn std::error::Error>> {
    let mut cmd = Command::cargo_bin("diffai")?;
    cmd.arg("--new-option")
       .arg("file1.safetensors")
       .arg("file2.safetensors");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("expected output"));
    
    Ok(())
}
```

## Submitting Changes

### 1. Commit Guidelines

Use conventional commit format:
```
type(scope): description

feat(ml): add tensor shape comparison
fix(cli): handle missing file error gracefully
docs(readme): update installation instructions
test(core): add tests for epsilon tolerance
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement

### 2. Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new functionality
3. **Run full test suite**: `cargo test`
4. **Check formatting**: `cargo fmt --check`
5. **Run linter**: `cargo clippy`
6. **Update CHANGELOG.md** if significant

### 3. Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing completed

## ML-Specific (if applicable)
- [ ] Tested with real model files
- [ ] Performance impact measured
- [ ] Memory usage validated

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if needed)
```

## AI/ML Specific Contributions

### High-Priority Areas

#### 1. Additional ML Framework Support
```rust
// Example: Adding TensorFlow support
pub fn parse_tensorflow_model(path: &Path) -> Result<HashMap<String, TensorStats>> {
    // Implementation needed
}
```

#### 2. Advanced Statistical Analysis
```rust
// Example: Adding distribution comparison
pub fn compare_tensor_distributions(
    stats1: &TensorStats, 
    stats2: &TensorStats
) -> DistributionComparison {
    // Implementation needed
}
```

#### 3. MLOps Integration
```rust
// Example: MLflow integration
pub fn export_to_mlflow(diff_results: &[DiffResult]) -> Result<MLflowArtifact> {
    // Implementation needed
}
```

### Testing ML Features

#### Real Model Testing
```bash
# Download test models (large files - use git LFS)
wget https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
```

#### Performance Testing
```rust
#[bench]
fn bench_large_model_comparison(b: &mut Bencher) {
    let model1 = load_test_model("large_model_1gb.safetensors");
    let model2 = load_test_model("large_model_1gb_modified.safetensors");
    
    b.iter(|| {
        diff_ml_models(&model1, &model2, None)
    });
}
```

### Documentation for ML Features

Always include:
1. **Use case**: Why this feature is needed
2. **Examples**: Real-world usage scenarios
3. **Performance**: Memory/time complexity
4. **Limitations**: Current constraints

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions or share ideas
- **Documentation**: Check [docs/](docs/) for detailed guides

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to join the maintainer team (for significant contributions)

---

Thank you for contributing to diffai! Together, we're building better tools for the AI/ML community. 🚀