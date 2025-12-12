# diffai

[日本語](README.ja.md)

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/diffai.svg)](https://crates.io/crates/diffai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Semantic diff tool for AI/ML models (PyTorch, Safetensors, NumPy, MATLAB). Provides tensor statistics, parameter comparisons, and automatic ML analysis.

## Why diffai?

Traditional `diff` doesn't understand binary ML files:

```bash
$ diff model_v1.pt model_v2.pt
Binary files model_v1.pt and model_v2.pt differ
```

`diffai` shows meaningful analysis:

```bash
$ diffai model_v1.safetensors model_v2.safetensors
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
gradient_analysis: flow_health=healthy, norm=0.021
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

## Installation

```bash
# As CLI tool
cargo install diffai

# As library (Cargo.toml)
[dependencies]
diffai-core = "0.3"
```

## Usage

```bash
# Basic
diffai model1.pt model2.pt

# JSON output for automation
diffai model1.safetensors model2.safetensors --output json

# With numerical tolerance
diffai weights1.npy weights2.npy --epsilon 0.001
```

## Supported Formats

- **PyTorch** (.pt, .pth) - Full ML analysis + tensor statistics
- **Safetensors** (.safetensors) - Full ML analysis + tensor statistics
- **NumPy** (.npy, .npz) - Tensor statistics
- **MATLAB** (.mat) - Tensor statistics

## Main Options

```bash
--format <FORMAT>       # Force input format (pytorch, safetensors, numpy, matlab)
--output <FORMAT>       # Output format: json, yaml, text (default: text)
--epsilon <N>           # Float comparison tolerance
--ignore-keys-regex RE  # Ignore keys matching regex
--quiet                 # Return only exit code (0: same, 1: diff found)
--verbose               # Show detailed analysis
```

## Output Symbols

- `+` Added tensor/parameter
- `-` Removed tensor/parameter
- `~` Modified tensor/parameter

## Automatic ML Analysis

When comparing PyTorch/Safetensors files, diffai automatically runs 11 specialized analyses:

1. Learning Rate Analysis
2. Optimizer Comparison
3. Loss Tracking
4. Accuracy Tracking
5. Model Version Analysis
6. Gradient Analysis
7. Quantization Analysis
8. Convergence Analysis
9. Activation Analysis
10. Attention Analysis
11. Ensemble Analysis

## CI/CD Usage

```bash
# Detect model changes
if ! diffai production.pt candidate.pt --quiet; then
  echo "Model has changed"
  diffai production.pt candidate.pt --output json > changes.json
fi
```

## Examples

See [diffai-cli/tests/cmd/](diffai-cli/tests/cmd/) for executable examples:

- [Basic comparison](diffai-cli/tests/cmd/basic.md)
- [Supported formats](diffai-cli/tests/cmd/formats.md)
- [Output formats](diffai-cli/tests/cmd/output.md)
- [Options](diffai-cli/tests/cmd/options.md)
- [ML Analysis](diffai-cli/tests/cmd/ml-analysis.md)

## Documentation

- [CLI Specification](docs/specs/cli.md)
- [Core API Specification](docs/specs/core.md)

## Related Projects

- **[diffx](https://github.com/kako-jun/diffx)** - Structured data diff (JSON, YAML, CSV, XML)
- **[lawkit](https://github.com/kako-jun/lawkit)** - Statistical law analysis toolkit

## License

MIT
