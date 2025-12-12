# diffai CLI Specification

## Overview

`diffai` is an AI/ML specialized diff tool for comparing PyTorch, Safetensors, NumPy, and MATLAB files.

## Usage

```
diffai [OPTIONS] <FILE1> <FILE2>
```

## Arguments

| Argument | Description |
|----------|-------------|
| `FILE1` | First input file (model, array, etc.) |
| `FILE2` | Second input file |

## Options

### Input/Output

| Option | Short | Description |
|--------|-------|-------------|
| `--format <FORMAT>` | `-f` | Input file format (auto-detected if not specified) |
| `--output <OUTPUT>` | `-o` | Output format: `json`, `yaml`, `text` (default: text) |

### Comparison Options

| Option | Description |
|--------|-------------|
| `--path <PATH>` | Filter by path (show only differences containing this string) |
| `--ignore-keys-regex <RE>` | Ignore keys matching regex pattern |
| `--epsilon <N>` | Numerical comparison tolerance for floating point |
| `--array-id-key <KEY>` | Compare arrays by this field instead of index |

### Output Control

| Option | Short | Description |
|--------|-------|-------------|
| `--quiet` | `-q` | Suppress output; return only exit status |
| `--brief` | | Report only whether files differ |
| `--verbose` | `-v` | Show verbose processing information |
| `--no-color` | | Disable colored output |

### Help

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Print help |
| `--version` | `-V` | Print version |

## Supported Formats

| Format | Extensions | Auto-detect |
|--------|------------|-------------|
| PyTorch | `.pt`, `.pth` | ✅ |
| Safetensors | `.safetensors` | ✅ |
| NumPy | `.npy`, `.npz` | ✅ |
| MATLAB | `.mat` | ✅ |

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Files are identical (no differences) |
| 1 | Files differ |
| 2 | Error (file not found, parse error, etc.) |

## Output Symbols

| Symbol | Meaning |
|--------|---------|
| `+` | Added tensor/parameter |
| `-` | Removed tensor/parameter |
| `~` | Modified tensor/parameter |

## ML Analysis (Automatic)

When comparing PyTorch/Safetensors files, diffai automatically performs:

1. **Learning Rate Analysis** - Training dynamics tracking
2. **Optimizer Comparison** - Adam/SGD state analysis
3. **Loss Tracking** - Convergence pattern detection
4. **Accuracy Tracking** - Performance metric evolution
5. **Model Version Analysis** - Checkpoint progression
6. **Gradient Analysis** - Flow health, vanishing/exploding detection
7. **Quantization Analysis** - Mixed precision detection (FP32/FP16/INT8)
8. **Convergence Analysis** - Learning curves, plateau detection
9. **Activation Analysis** - ReLU/GELU/Tanh distribution
10. **Attention Analysis** - Transformer mechanism detection
11. **Ensemble Analysis** - Multi-model structure detection

## Examples

### Basic Comparison
```bash
diffai model_v1.pt model_v2.pt
```

### JSON Output for Automation
```bash
diffai model1.safetensors model2.safetensors --output json
```

### With Tolerance
```bash
diffai weights1.npy weights2.npy --epsilon 0.001
```

### Quiet Mode (CI/CD)
```bash
if ! diffai old.pt new.pt --quiet; then
  echo "Model changed"
fi
```
