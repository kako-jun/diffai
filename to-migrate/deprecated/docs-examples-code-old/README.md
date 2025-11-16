# diffai Code Examples

This directory contains practical code examples demonstrating diffai's AI/ML specialized capabilities with automatic comprehensive analysis. Examples are organized by usage type for clarity.

## 📁 Directory Structure

```
examples-code/
├── cli-usage/            # CLI tool usage examples
│   ├── mlflow_integration.py    - MLflow integration via CLI
│   ├── ci_cd_pipeline.yml       - GitHub Actions CI/CD pipeline
│   └── pre_commit_hook.sh       - Git pre-commit hook
├── rust-core/            # Rust library usage examples
│   ├── simple_comparison.rs     - Basic model comparison
│   ├── finetuning_analysis.rs   - Fine-tuning analysis
│   └── quantization_impact.rs   - Quantization impact analysis
├── python-pip/           # Python package usage examples
│   ├── model_comparison.py      - Direct Python API usage
│   └── batch_analysis.py        - Batch processing example
├── typescript-npm/       # TypeScript/npm package examples
│   ├── model-comparison.ts      - Node.js API usage with TypeScript
│   └── web-integration.ts       - Express.js web integration
└── README.md            # This file
```

## 🚀 Quick Start by Usage Type

### CLI Tool Usage

```bash
# Install diffai CLI
cargo install diffai

# Basic model comparison - 11 ML analyses run automatically
diffai models/base.safetensors models/finetuned.safetensors

# JSON output for integration
diffai checkpoint_1.pt checkpoint_2.pt --output json
```

### Python Package Usage

```bash
# Install Python package
pip install diffai-python

# Use in Python code
python docs/examples-code/python-pip/model_comparison.py model1.pt model2.pt
```

### TypeScript/npm Package Usage

```bash
# Install npm package
npm install diffai-js

# Install TypeScript tooling
npm install -g tsx

# Use in TypeScript/Node.js code
npx tsx docs/examples-code/typescript-npm/model-comparison.ts model1.pt model2.pt
```

### Rust Library Usage

```bash
# Add to Cargo.toml
diffai-core = "0.3.16"

# Build and run examples
cargo run --example simple_comparison
```

## 📊 Usage Examples by Category

### 1. CLI Usage (`cli-usage/`)

Examples that demonstrate using diffai as a command-line tool via subprocess or shell commands.

#### MLflow Integration (`mlflow_integration.py`)
- **Usage Type**: CLI via subprocess
- **Features**: Comprehensive ML experiment tracking
- **Installation**: `pip install mlflow`
- **Run**: `python mlflow_integration.py model1.safetensors model2.safetensors`

#### CI/CD Pipeline (`ci_cd_pipeline.yml`)
- **Usage Type**: CLI in GitHub Actions
- **Features**: Automated model validation in CI/CD
- **Platform**: GitHub Actions
- **Usage**: Copy to `.github/workflows/`

#### Pre-commit Hook (`pre_commit_hook.sh`)
- **Usage Type**: CLI in Git hooks
- **Features**: Automated model validation before commits
- **Installation**: Copy to `.git/hooks/pre-commit`
- **Run**: Automatic on `git commit`

### 2. Rust Core Library (`rust-core/`)

Examples that demonstrate using diffai as a Rust library directly.

#### Simple Comparison (`simple_comparison.rs`)
- **Usage Type**: Rust library API
- **Features**: Basic model comparison with all 11 ML analyses
- **Dependencies**: `diffai-core = "0.3.16"`
- **Run**: `cargo run --example simple_comparison`

#### Fine-tuning Analysis (`finetuning_analysis.rs`)
- **Usage Type**: Rust library API
- **Features**: Training dynamics and learning rate analysis
- **Run**: `cargo run --example finetuning_analysis`

#### Quantization Impact (`quantization_impact.rs`)
- **Usage Type**: Rust library API
- **Features**: Precision and compression analysis
- **Run**: `cargo run --example quantization_impact`

### 3. Python Package (`python-package/`)

Examples that demonstrate using diffai-python package directly in Python code.

#### Model Comparison (`model_comparison.py`)
- **Usage Type**: Python package API (import diffai_python)
- **Features**: Direct Python integration with structured data
- **Installation**: `pip install diffai-python`
- **Run**: `python model_comparison.py model1.pt model2.pt`

#### Batch Analysis (`batch_analysis.py`)
- **Usage Type**: Python package API for batch processing
- **Features**: Parallel processing of multiple model pairs
- **Run**: `python batch_analysis.py baseline_dir/ finetuned_dir/ --output results/`

### 4. JavaScript/npm Package (`javascript-npm/`)

Examples that demonstrate using diffai npm package in Node.js applications.

#### Model Comparison (`model-comparison.js`)
- **Usage Type**: JavaScript/npm package API
- **Features**: Promise-based API with async/await
- **Installation**: `npm install diffai`
- **Run**: `node model-comparison.js model1.pt model2.pt`

#### Web Integration (`web-integration.js`)
- **Usage Type**: JavaScript/npm in Express.js web app
- **Features**: REST API, WebSocket real-time updates, file uploads
- **Installation**: `npm install diffai express multer socket.io uuid`
- **Run**: `node web-integration.js` (server on http://localhost:3000)

## 🔧 Installation Guide by Usage Type

### CLI Tool
```bash
# Install Rust and Cargo (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install diffai CLI
cargo install diffai
```

### Python Package
```bash
# Install Python package
pip install diffai-python

# For advanced examples, install additional dependencies
pip install mlflow pandas
```

### JavaScript/npm Package
```bash
# Install npm package
npm install diffai

# For web integration example
npm install express multer socket.io uuid
```

### Rust Library
```toml
# Add to Cargo.toml
[dependencies]
diffai-core = "0.3.16"
anyhow = "1.0"
serde_json = "1.0"
```

## 📚 Real-World Use Cases by Type

### CLI Tool Integration
- **DevOps/MLOps**: CI/CD pipelines, monitoring scripts
- **Shell Scripts**: Automation, batch processing
- **Git Hooks**: Pre-commit validation, automated checks
- **Containerized Workflows**: Docker, Kubernetes jobs

### Python Package Integration  
- **Data Science**: Jupyter notebooks, research workflows
- **ML Pipelines**: Training validation, model comparison
- **Web Applications**: Flask/Django model analysis endpoints
- **Batch Processing**: Large-scale model comparison jobs

### JavaScript/npm Integration
- **Web Applications**: Model comparison APIs, dashboards
- **Node.js Services**: Microservices, real-time analysis
- **Electron Apps**: Desktop applications for ML teams
- **Serverless Functions**: AWS Lambda, Vercel functions

### Rust Library Integration
- **High-Performance**: Large model analysis, real-time processing
- **Systems Programming**: Low-level ML tooling
- **WASM**: Browser-based model analysis
- **CLI Tools**: Custom analysis applications

## 🛠️ API Comparison by Usage Type

### CLI Tool (Subprocess)
```bash
# Simple command-line usage
diffai model1.pt model2.pt --output json
```

### Python Package (Direct Import)
```python
import diffai_python
from diffai_python import diff_models, DiffOptions

options = DiffOptions(ml_analysis_enabled=True)
result = diff_models("model1.pt", "model2.pt", options)
```

### JavaScript/npm (Promise-based)
```javascript
const diffai = require('diffai');

const options = { mlAnalysisEnabled: true };
const result = await diffai.compareModels('model1.pt', 'model2.pt', options);
```

### Rust Library (Direct API)
```rust
use diffai_core::{diff, DiffOptions};

let options = DiffOptions {
    ml_analysis_enabled: Some(true),
    ..Default::default()
};
let differences = diff(&old_model, &new_model, Some(&options))?;
```

## 🎯 Automatic ML Analysis Features

All usage types provide the same comprehensive ML analysis:

### 11 Automatic Analysis Functions
1. **📈 Learning Rate Analysis** - Training dynamics tracking
2. **⚙️ Optimizer Comparison** - State and momentum analysis
3. **📉 Loss Tracking** - Convergence pattern analysis
4. **🎯 Accuracy Tracking** - Performance metrics monitoring
5. **🏷️ Model Version Analysis** - Checkpoint evolution tracking
6. **🌊 Gradient Analysis** - Flow and stability analysis
7. **🔢 Quantization Analysis** - Precision detection
8. **📊 Convergence Analysis** - Learning curve analysis
9. **⚡ Activation Analysis** - Function usage analysis
10. **👁️ Attention Analysis** - Transformer mechanism analysis
11. **🤝 Ensemble Analysis** - Multi-model structure detection

### Convention over Configuration
- **Zero Setup**: Automatic analysis for ML files
- **Format Detection**: Automatic selection based on file type
- **Comprehensive Coverage**: All 11 functions run automatically

## 🤝 Contributing Examples

We welcome new examples that demonstrate:

### By Usage Type
- **CLI**: New integration patterns, shell automation
- **Python**: Advanced data science workflows, ML pipelines  
- **JavaScript**: Web applications, real-time analysis
- **Rust**: High-performance analysis, systems integration

### By Use Case
- **MLOps**: Deployment pipelines, monitoring workflows
- **Research**: Academic workflows, experiment tracking
- **Production**: A/B testing, model validation
- **Education**: Teaching materials, tutorials

## 📞 Support

If you have questions about these examples:
- Check the [main documentation](../quick-start.md)
- Review [ML Analysis details](../ml-analysis.md)  
- Browse examples for your usage type
- Open an issue on GitHub
- Join GitHub Discussions

---

**Choose your preferred integration method and experience automatic ML analysis with diffai! 🚀**