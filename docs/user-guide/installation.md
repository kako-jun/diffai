# Installation

Setup guide for diffai in various environments.

## Package Manager Installation

### Cargo (Rust)

```bash
# Install from official Rust package manager
cargo install diffai

# Install development version (latest features)
cargo install --git https://github.com/kako-jun/diffai
```

### Homebrew (macOS/Linux)

```bash
# Coming soon
# brew install diffai
```

## Build from Source

### Prerequisites
- Rust 1.70 or higher
- Git

### Build Steps

```bash
# Clone repository
git clone https://github.com/kako-jun/diffai.git
cd diffai

# Release build
cargo build --release

# Install
cargo install --path .

# Verify installation
diffai --version
```

## Using Docker

```bash
# Build Docker image
docker build -t diffai .

# Usage example
docker run -v $(pwd):/workspace diffai /workspace/model1.pth /workspace/model2.pth
```

## Development Environment Setup

### Recommended Environment
- VS Code + rust-analyzer
- Python 3.8+ (for PyTorch support)
- Node.js (for documentation generation)

### Development Installation

```bash
# Clone with development dependencies
git clone https://github.com/kako-jun/diffai.git
cd diffai

# Development setup
cargo build
cargo test

# Setup pre-commit hooks
pre-commit install
```

## Verification

After installation, verify with the following commands:

```bash
# Check version
diffai --version

# Show help
diffai --help

# Test with sample files
diffai examples/models/sample1.json examples/models/sample2.json
```

## Troubleshooting

### Common Issues

#### Outdated Rust Version
```bash
# Update Rust to latest
rustup update
```

#### PyTorch Dependency Errors
```bash
# Install PyTorch separately
pip install torch torchvision
```

#### Permission Errors
```bash
# Install to user directory
cargo install --user diffai
```

### Support

If issues persist, get support at:

- [GitHub Issues](https://github.com/kako-jun/diffai/issues)
- [Discussion](https://github.com/kako-jun/diffai/discussions)

## Next Steps

After installation, see [Basic Usage](basic-usage.md) to get started with diffai.

## Language Support

- **English**: Current documentation
- **日本語**: [Japanese version](installation_ja.md)