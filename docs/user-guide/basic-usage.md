# Basic Usage

Learn the fundamental operations of diffai.

## 🚀 Quick Start

### Basic File Comparison

```bash
# Compare two files
diffai file1.txt file2.txt

# Detailed output
diffai file1.txt file2.txt --verbose

# Specify output format
diffai file1.txt file2.txt --format json
```

### Directory Comparison

```bash
# Compare entire directories
diffai dir1/ dir2/ --recursive

# Filter by file extensions
diffai dir1/ dir2/ --include "*.py" --include "*.json"
```

## 🤖 AI/ML Specialized Features

### PyTorch Model Comparison

```bash
# Compare PyTorch model files
diffai model1.pth model2.pth

# Show detailed structure information
diffai model1.pth model2.pth --show-structure

# Show only differences
diffai model1.pth model2.pth --diff-only
```

**Example Output:**
```
=== PyTorch Model Comparison ===

📊 Model Structure:
  ├─ model1.pth: ResNet-18 (11.7M params)
  └─ model2.pth: ResNet-34 (21.8M params)

🔍 Layer Differences:
  + model2.pth: layer4.1.conv2 (512x512x3x3)
  + model2.pth: layer4.1.bn2 (512 features)
  - model1.pth: Only has 2 blocks in layer4

📈 Parameter Count:
  model1.pth: 11,689,512 parameters
  model2.pth: 21,797,672 parameters
  Difference: +10,108,160 parameters (+86.4%)
```

### Safetensors File Comparison

```bash
# Compare Safetensors files
diffai model1.safetensors model2.safetensors

# Show tensor details
diffai model1.safetensors model2.safetensors --tensor-details
```

### Dataset Comparison

```bash
# Compare CSV datasets
diffai train.csv test.csv --format csv

# Compare JSON datasets
diffai dataset1.json dataset2.json --format json

# Show statistics
diffai train.csv test.csv --stats
```

## 📋 Command Options

### Basic Options

| Option | Description | Example |
|--------|-------------|---------|
| `--format` | Specify output format | `--format json` |
| `--verbose` | Detailed output | `--verbose` |
| `--quiet` | Minimal output | `--quiet` |
| `--color` | Control color output | `--color always` |

### File Processing Options

| Option | Description | Example |
|--------|-------------|---------|
| `--recursive` | Process directories recursively | `--recursive` |
| `--include` | Include file patterns | `--include "*.py"` |
| `--exclude` | Exclude file patterns | `--exclude "*.pyc"` |
| `--follow-symlinks` | Follow symbolic links | `--follow-symlinks` |

### AI/ML Specific Options

| Option | Description | Example |
|--------|-------------|---------|
| `--show-structure` | Show model structure | `--show-structure` |
| `--tensor-details` | Show tensor details | `--tensor-details` |
| `--diff-only` | Show only differences | `--diff-only` |
| `--stats` | Show statistics | `--stats` |

## 🎨 Output Formats

### Default Format

Standard diff format output:

```
--- model1.pth
+++ model2.pth
@@ -1,3 +1,4 @@
 layer1.conv1: Conv2d(3, 64, kernel_size=(7, 7))
 layer1.bn1: BatchNorm2d(64, eps=1e-05)
+layer1.relu: ReLU(inplace=True)
 layer1.maxpool: MaxPool2d(kernel_size=3, stride=2)
```

### JSON Format

```bash
diffai model1.pth model2.pth --format json
```

```json
{
  "comparison": {
    "file1": "model1.pth",
    "file2": "model2.pth",
    "type": "pytorch",
    "differences": [
      {
        "type": "added",
        "layer": "layer1.relu",
        "details": "ReLU(inplace=True)"
      }
    ]
  }
}
```

### Custom Format

```bash
# Use custom template
diffai model1.pth model2.pth --template custom.jinja2
```

## 🔧 Configuration

### Global Configuration

`~/.config/diffai/config.toml`:

```toml
[defaults]
format = "default"
color = "auto"
verbose = false

[pytorch]
show_structure = true
tensor_details = false

[output]
pager = "less"
max_lines = 1000
```

### Project Configuration

`.diffai.toml`:

```toml
[project]
name = "my-ml-project"

[include]
patterns = ["*.py", "*.pth", "*.safetensors"]

[exclude]
patterns = ["*.pyc", "__pycache__/*"]

[pytorch]
show_structure = true
```

## 🎯 Practical Examples

### Experiment Comparison

```bash
# Compare two experiment results
diffai experiment_v1/ experiment_v2/ --recursive --include "*.json"

# Compare model checkpoints
diffai checkpoints/epoch_10.pth checkpoints/epoch_20.pth --show-structure
```

### CI/CD Usage

```yaml
- name: Compare models
  run: |
    diffai baseline/model.pth new/model.pth --format json > model_diff.json
    
- name: Check significant changes
  run: |
    if diffai baseline/model.pth new/model.pth --diff-only --quiet; then
      echo "No significant model changes"
    else
      echo "Model has changed - review required"
      exit 1
    fi
```

## 📚 Next Steps

- [ML/AI Workflows](ml-workflows.md) - Integration with ML development
- [Configuration](configuration.md) - Advanced configuration
- [API Reference](../api/cli.md) - Complete command reference

## 🌐 Language Support

- **English**: Current documentation
- **日本語**: [Japanese version](basic-usage_ja.md)