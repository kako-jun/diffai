# diffai Examples

This directory contains practical examples demonstrating diffai's capabilities for AI/ML workflows.

## 📁 Directory Structure

```
examples/
├── ml-models/           # ML model comparison examples
│   ├── simple_comparison.rs
│   ├── finetuning_analysis.rs
│   └── quantization_impact.rs
├── integration/         # MLOps and CI/CD integration examples
│   ├── mlflow_integration.py
│   ├── ci_cd_pipeline.yml
│   └── pre_commit_hook.sh
└── README.md           # This file
```

## 🚀 Quick Examples

### Basic Model Comparison

```bash
# Compare two PyTorch models
diffai models/base.safetensors models/finetuned.safetensors

# Output tensor statistics changes
diffai checkpoint_1.pt checkpoint_2.pt --output json
```

### Advanced ML Analysis

```bash
# Fine-tuning impact analysis
diffai pretrained.safetensors finetuned.safetensors --epsilon 1e-6

# Quantization comparison
diffai model_fp32.safetensors model_int8.safetensors --epsilon 0.1
```

## 📊 ML Model Examples

### 1. Simple Comparison (`ml-models/simple_comparison.rs`)
- Basic tensor statistics comparison
- Shape change detection
- Parameter count differences

### 2. Fine-tuning Analysis (`ml-models/finetuning_analysis.rs`)
- Pre-trained vs fine-tuned model comparison
- Layer-wise change analysis
- Statistical significance testing

### 3. Quantization Impact (`ml-models/quantization_impact.rs`)
- FP32 vs INT8 model comparison
- Precision loss assessment
- Memory usage optimization analysis

## 🔧 Integration Examples

### 1. MLflow Integration (`integration/mlflow_integration.py`)
- Automatic model comparison logging
- Experiment tracking integration
- Metric change detection

### 2. CI/CD Pipeline (`integration/ci_cd_pipeline.yml`)
- GitHub Actions workflow
- Automated model validation
- Deployment gate based on changes

### 3. Pre-commit Hook (`integration/pre_commit_hook.sh`)
- Git hook for model change validation
- Interactive approval workflow
- Change summary generation

## 🏃‍♂️ Running Examples

### Rust Examples

```bash
# Navigate to project root
cd diffai

# Run a specific example
cargo run --example simple_comparison

# Build all examples
cargo build --examples
```

### Integration Scripts

```bash
# Make scripts executable
chmod +x examples/integration/*.sh

# Run integration examples
./examples/integration/pre_commit_hook.sh
```

## 📚 Real-World Use Cases

### Research & Development
- **Model evolution tracking**: Compare models across training epochs
- **Architecture experiments**: Analyze structural differences between models
- **Hyperparameter impact**: Measure parameter changes effects

### Production & MLOps
- **Deployment validation**: Ensure model changes meet quality gates
- **A/B testing**: Compare candidate models for production
- **Monitoring**: Track model drift in production environments

### Collaboration
- **Code reviews**: Visualize model changes in pull requests
- **Documentation**: Generate automatic change summaries
- **Team communication**: Share meaningful model differences

## 🛠️ Creating Custom Examples

### Example Template

```rust
use anyhow::Result;
use diffai_core::{diff_ml_models, DiffResult};
use std::path::Path;

fn main() -> Result<()> {
    println!("🤖 diffai Example: Your Use Case");
    
    // Load models
    let model1_path = Path::new("path/to/model1.safetensors");
    let model2_path = Path::new("path/to/model2.safetensors");
    
    // Compare models
    let differences = diff_ml_models(model1_path, model2_path, Some(1e-6))?;
    
    // Process results
    for diff in differences {
        match diff {
            DiffResult::TensorStatsChanged(name, stats1, stats2) => {
                println!("📊 {}: mean {:.4} → {:.4}", name, stats1.mean, stats2.mean);
            }
            DiffResult::TensorShapeChanged(name, shape1, shape2) => {
                println!("⬚ {}: {:?} → {:?}", name, shape1, shape2);
            }
            _ => {} // Handle other cases
        }
    }
    
    Ok(())
}
```

## 📋 Prerequisites

### For ML Examples
- Access to model files (PyTorch .pt/.pth or Safetensors .safetensors)
- Basic understanding of ML model structure
- Rust development environment

### For Integration Examples
- Python 3.8+ (for MLflow integration)
- Git (for hooks and CI/CD)
- CI/CD platform access (GitHub Actions, GitLab CI, etc.)

## 🤝 Contributing Examples

We welcome new examples! Please:

1. **Follow the template structure**
2. **Include clear documentation**
3. **Test with real data when possible**
4. **Add to this README**

### Example Contribution Areas
- New ML frameworks (TensorFlow, ONNX, JAX)
- Additional MLOps tools (Weights & Biases, Neptune, etc.)
- Different deployment scenarios (Docker, Kubernetes, etc.)
- Advanced statistical analyses

## 📞 Support

If you have questions about these examples:
- Check the [main documentation](../docs/)
- Open an issue on GitHub
- Join the discussion in GitHub Discussions

---

**Happy diffing! 🚀**