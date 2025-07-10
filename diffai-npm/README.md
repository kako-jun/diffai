# diffai - AI/ML Specialized Diff Tool (npm package)

[![npm version](https://badge.fury.io/js/diffai.svg)](https://badge.fury.io/js/diffai)
[![Downloads](https://img.shields.io/npm/dm/diffai.svg)](https://npmjs.org/package/diffai)

AI/ML specialized data diff tool for deep tensor comparison and analysis. This npm package provides a convenient way to install and use diffai through Node.js.

## 🚀 Quick Start

### Installation

```bash
# Global installation
npm install -g diffai

# Project-specific installation
npm install diffai

# One-time usage
npx diffai model1.safetensors model2.safetensors --stats
```

### Usage

```bash
# Compare ML models with detailed statistics
diffai model_v1.safetensors model_v2.safetensors --stats

# Analyze architecture changes
diffai baseline.safetensors modified.safetensors --architecture-comparison

# Memory analysis for deployment optimization
diffai model_fp32.safetensors model_quantized.safetensors --memory-analysis

# Comprehensive ML analysis (Phase 3 features)
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  --architecture-comparison \
  --memory-analysis \
  --anomaly-detection \
  --convergence-analysis

# JSON output for MLOps integration
diffai model1.safetensors model2.safetensors --stats --output json

# Compare scientific data
diffai experiment_v1.npy experiment_v2.npy --stats
diffai simulation_v1.mat simulation_v2.mat --stats

# Standard structured data comparison
diffai config_v1.json config_v2.json
diffai data_v1.yaml data_v2.yaml
```

## 📦 Supported File Formats

### AI/ML Formats (Specialized Analysis)
- **Safetensors** (.safetensors) - PyTorch model format with ML analysis
- **PyTorch** (.pt, .pth) - Native PyTorch models with tensor statistics
- **NumPy** (.npy, .npz) - Scientific computing arrays with statistical analysis
- **MATLAB** (.mat) - Engineering/scientific data with numerical analysis

### Structured Data Formats (Universal)
- **JSON** (.json) - API configurations, model metadata
- **YAML** (.yaml, .yml) - Configuration files, CI/CD pipelines
- **TOML** (.toml) - Rust configs, Python pyproject.toml
- **XML** (.xml) - Legacy configurations, model definitions
- **CSV** (.csv) - Datasets, experiment results
- **INI** (.ini) - Legacy configuration files

## 🔬 35 ML Analysis Functions

diffai provides 35 specialized analysis functions for AI/ML workflows:

### Core Analysis (Always Available)
- `--stats` - Detailed tensor statistics (mean, std, min, max, shape, dtype)
- `--quantization-analysis` - Quantization effect analysis
- `--sort-by-change-magnitude` - Priority-sorted differences
- `--show-layer-impact` - Layer-by-layer impact analysis

### Phase 3 Advanced Analysis (v0.2.6+)
- `--architecture-comparison` - Model architecture and structural changes
- `--memory-analysis` - Memory usage and optimization opportunities  
- `--anomaly-detection` - Numerical anomalies and training issues
- `--change-summary` - Detailed change summaries and patterns
- `--convergence-analysis` - Training convergence patterns
- `--gradient-analysis` - Gradient flow health assessment
- `--similarity-matrix` - Inter-layer similarity analysis

## 💡 Usage Examples

### Research & Development
```bash
# Monitor training progress
diffai epoch_100.safetensors epoch_101.safetensors --stats --convergence-analysis

# Analyze fine-tuning effects
diffai base_model.safetensors finetuned_model.safetensors \
  --architecture-comparison --show-layer-impact

# Debug training issues
diffai stable_checkpoint.safetensors problematic_checkpoint.safetensors \
  --anomaly-detection --gradient-analysis
```

### MLOps & Production
```bash
# Pre-deployment validation
diffai current_prod.safetensors candidate.safetensors \
  --memory-analysis --quantization-analysis --change-summary

# CI/CD integration with JSON output
diffai baseline.safetensors modified.safetensors --stats --output json | jq .

# Performance impact assessment
diffai model_v1.safetensors model_v2.safetensors \
  --memory-analysis --architecture-comparison
```

### Scientific Computing
```bash
# Compare experimental results
diffai control_group.npy treatment_group.npy --stats

# Engineering simulation analysis
diffai simulation_baseline.mat simulation_optimized.mat --stats

# Dataset version comparison
diffai dataset_v1.npz dataset_v2.npz --stats --sort-by-change-magnitude
```

## 🔧 Integration Examples

### Node.js Integration
```javascript
const { spawn } = require('child_process');

function compareTensors(model1, model2, options = []) {
  return new Promise((resolve, reject) => {
    const args = [model1, model2, ...options];
    const child = spawn('diffai', args);
    
    let output = '';
    child.stdout.on('data', (data) => output += data);
    child.on('close', (code) => {
      if (code === 0) resolve(output);
      else reject(new Error(`diffai failed with code ${code}`));
    });
  });
}

// Usage
compareTensors('model1.safetensors', 'model2.safetensors', ['--stats', '--output', 'json'])
  .then(result => console.log(JSON.parse(result)))
  .catch(console.error);
```

### MLflow Integration
```javascript
const fs = require('fs');
const { execSync } = require('child_process');

function logModelDiff(model1Path, model2Path, runId) {
  const output = execSync(`diffai "${model1Path}" "${model2Path}" --stats --output json`, 
                          { encoding: 'utf8' });
  const diffData = JSON.parse(output);
  
  // Save comparison results
  fs.writeFileSync(`mlruns/${runId}/artifacts/model_comparison.json`, 
                   JSON.stringify(diffData, null, 2));
  
  console.log(`Model comparison logged for run ${runId}`);
}
```

## 🏗️ Platform Support

This npm package automatically downloads the appropriate binary for your platform:

- **Linux** (x86_64, ARM64)
- **macOS** (Intel x86_64, Apple Silicon ARM64)  
- **Windows** (x86_64)

The binary is downloaded during `npm install` and cached locally. If download fails, the package falls back to using `diffai` from your system PATH.

## 🔗 Related Projects

- **[diffx](https://www.npmjs.com/package/diffx-js)** - General-purpose structured data diff tool
- **[diffai (PyPI)](https://pypi.org/project/diffai-python/)** - Python package for diffai
- **[diffai (GitHub)](https://github.com/diffai-team/diffai)** - Main repository

## 📚 Documentation

- [CLI Reference](https://github.com/diffai-team/diffai/blob/main/docs/reference/cli-reference.md)
- [ML Analysis Guide](https://github.com/diffai-team/diffai/blob/main/docs/reference/ml-analysis.md)
- [User Guide](https://github.com/diffai-team/diffai/blob/main/docs/user-guide/)

## 📄 License

MIT License - see [LICENSE](https://github.com/diffai-team/diffai/blob/main/LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/diffai-team/diffai/blob/main/CONTRIBUTING.md) for guidelines.

---

**diffai** - Making AI/ML data differences visible, measurable, and actionable. 🚀