# diffai - AI/ML Specialized Diff Tool (npm package)

[![npm version](https://badge.fury.io/js/diffai-js.svg)](https://badge.fury.io/js/diffai-js)
[![Downloads](https://img.shields.io/npm/dm/diffai-js.svg)](https://npmjs.org/package/diffai-js)

AI/ML specialized data diff tool for deep tensor comparison and analysis. This npm package provides a convenient way to install and use diffai through Node.js.

## 🚀 Quick Start

### Installation

```bash
# Global installation
npm install -g diffai-js

# Project-specific installation
npm install diffai-js

# One-time usage (30+ ML analysis features automatic)
npx diffai model1.safetensors model2.safetensors
```

### Usage

```bash
# Compare ML models (30+ analysis features automatic)
diffai model_v1.safetensors model_v2.safetensors

# All ML analysis features run automatically:
# - Architecture comparison, memory analysis, anomaly detection
# - Convergence analysis, gradient analysis, quantization analysis
# - Deployment readiness, regression testing, and 22+ more features

# JSON output for MLOps integration (all features included)
diffai model1.safetensors model2.safetensors --output json

# Verbose output with debugging information
diffai model1.safetensors model2.safetensors --verbose

# Compare scientific data
diffai experiment_v1.npy experiment_v2.npy
diffai simulation_v1.mat simulation_v2.mat

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

## 🔬 30+ ML Analysis Functions (Automatic)

diffai provides 30+ specialized analysis functions that run automatically for AI/ML models:

### Automatic Comprehensive Analysis
For PyTorch and Safetensors files, all analysis features run automatically:
- **Statistical Analysis** - Detailed tensor statistics (mean, std, min, max, shape, dtype)
- **Quantization Analysis** - Quantization effect analysis
- **Architecture Comparison** - Model architecture and structural changes
- **Memory Analysis** - Memory usage and optimization opportunities  
- **Anomaly Detection** - Numerical anomalies and training issues
- **Convergence Analysis** - Training convergence patterns
- **Gradient Analysis** - Gradient flow health assessment
- **Deployment Readiness** - Production deployment assessment
- **Plus 22+ additional specialized features**

## 💡 Usage Examples

### Research & Development
```bash
# Monitor training progress (all analysis automatic)
diffai epoch_100.safetensors epoch_101.safetensors

# Analyze fine-tuning effects (comprehensive analysis automatic)
diffai base_model.safetensors finetuned_model.safetensors

# Debug training issues (full analysis automatic)
diffai stable_checkpoint.safetensors problematic_checkpoint.safetensors
```

### MLOps & Production
```bash
# Pre-deployment validation (all analysis automatic)
diffai current_prod.safetensors candidate.safetensors

# CI/CD integration with JSON output
diffai baseline.safetensors modified.safetensors --output json | jq .

# Performance impact assessment (comprehensive analysis automatic)
diffai model_v1.safetensors model_v2.safetensors
```

### Scientific Computing
```bash
# Compare experimental results
diffai control_group.npy treatment_group.npy

# Engineering simulation analysis
diffai simulation_baseline.mat simulation_optimized.mat

# Dataset version comparison
diffai dataset_v1.npz dataset_v2.npz --sort-by-change-magnitude
```

## 🔧 Integration Examples

### JavaScript API (Recommended)
```javascript
const { diff, diffString, inspect, isDiffaiAvailable, getVersion, DiffaiError } = require('diffai');

// Basic model comparison
async function compareModels() {
  try {
    const result = await diff('model1.safetensors', 'model2.safetensors', {
      output: 'json'
    });
    
    console.log(`Found ${result.length} differences`);
    result.forEach(diff => {
      console.log(`${diff.type}: ${diff.path}`);
    });
  } catch (error) {
    if (error instanceof DiffaiError) {
      console.error(`diffai failed: ${error.message}`);
    }
  }
}

// String comparison
async function compareStrings() {
  const model1Data = JSON.stringify({name: "bert-base", layers: 12});
  const model2Data = JSON.stringify({name: "bert-large", layers: 24});
  
  const result = await diffString(model1Data, model2Data, 'json', {
    output: 'json'
  });
  
  return result;
}

// Check availability
async function checkDiffai() {
  if (await isDiffaiAvailable()) {
    const version = await getVersion();
    console.log(`diffai ${version} is available`);
  } else {
    console.error('diffai is not available');
  }
}
```

### Node.js CLI Integration (Legacy)
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
compareTensors('model1.safetensors', 'model2.safetensors', ['--output', 'json'])
  .then(result => console.log(JSON.parse(result)))
  .catch(console.error);
```

### MLflow Integration
```javascript
const fs = require('fs');
const { diff } = require('diffai');

async function logModelDiff(model1Path, model2Path, runId) {
  try {
    const diffData = await diff(model1Path, model2Path, {
      output: 'json'
    });
    
    // Save comparison results
    fs.writeFileSync(`mlruns/${runId}/artifacts/model_comparison.json`, 
                     JSON.stringify(diffData, null, 2));
    
    console.log(`Model comparison logged for run ${runId}`);
    return diffData;
  } catch (error) {
    console.error(`MLflow integration failed: ${error.message}`);
    throw error;
  }
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
- **[diffai (GitHub)](https://github.com/kako-jun/diffai)** - Main repository

## 📚 Documentation

- [CLI Reference](https://github.com/kako-jun/diffai/blob/main/docs/reference/cli-reference.md)
- [ML Analysis Guide](https://github.com/kako-jun/diffai/blob/main/docs/reference/ml-analysis.md)
- [User Guide](https://github.com/kako-jun/diffai/blob/main/docs/user-guide/)

## 📄 License

MIT License - see [LICENSE](https://github.com/diffai-team/diffai/blob/main/LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://github.com/diffai-team/diffai/blob/main/CONTRIBUTING.md) for guidelines.

---

**diffai** - Making AI/ML data differences visible, measurable, and actionable. 🚀