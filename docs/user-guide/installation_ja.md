# インストール

各種環境でのdiffaiの完全インストールガイド。

## 前提条件

- **Rust 1.75+**: [rustup.rs](https://rustup.rs/)からインストール
- **オペレーティングシステム**: Linux、macOS、またはWindows
- **メモリ**: 大型モデルファイルには4GB+推奨

## インストール方法

### 方法1: crates.ioから（推奨）

```bash
cargo install diffai
```

**注意**: この方法はdiffaiがcrates.ioに公開され次第利用可能になります。

### 方法2: ソースから（現在）

```bash
# リポジトリをクローン
git clone https://github.com/kako-jun/diffai.git
cd diffai

# ビルドとインストール
cargo install --path diffai-cli

# インストールの確認
diffai --version
```

### 方法3: GitHubリリースから

[GitHubリリースページ](https://github.com/kako-jun/diffai/releases)からビルド済みバイナリをダウンロード：

- **Linux (x86_64)**: `diffai-linux-x86_64.tar.gz`
- **macOS (x86_64)**: `diffai-macos-x86_64.tar.gz`
- **macOS (ARM64)**: `diffai-macos-aarch64.tar.gz`
- **Windows (x86_64)**: `diffai-windows-x86_64.zip`

```bash
# Extract and move to PATH
tar -xzf diffai-linux-x86_64.tar.gz
sudo mv diffai /usr/local/bin/
```

## Platform-Specific Instructions

### Linux

#### Ubuntu/Debian
```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install diffai
cargo install diffai
```

#### Arch Linux
```bash
# Using AUR (when available)
yay -S diffai

# Or from source
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo install --path diffai-cli
```

#### CentOS/RHEL/Fedora
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install diffai
cargo install diffai
```

### macOS

#### Using Homebrew (Planned)
```bash
# This will be available in the future
brew install diffai
```

#### Manual Installation
```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install diffai
cargo install diffai
```

### Windows

#### Using Cargo
```powershell
# Install Rust from https://rustup.rs/
# Then install diffai
cargo install diffai
```

#### Using Scoop (Planned)
```powershell
# This will be available in the future
scoop install diffai
```

## Container Installation

### Docker

```bash
# Pull the image (when available)
docker pull ghcr.io/kako-jun/diffai:latest

# Run diffai in a container
docker run --rm -v $(pwd):/workspace ghcr.io/kako-jun/diffai:latest \
  model1.safetensors model2.safetensors
```

### Building Docker Image

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai

# Build the Docker image
docker build -t diffai .

# Run
docker run --rm -v $(pwd):/workspace diffai \
  model1.safetensors model2.safetensors
```

## Verification

After installation, verify that diffai is working correctly:

```bash
# Check version
diffai --version

# Run help
diffai --help

# サンプルAI/MLファイルでテスト（利用可能な場合）
# diffai model1.safetensors model2.safetensors

# 汎用構造化データフォーマットにはdiffxを使用:
# echo '{"a": 1}' > test1.json
# echo '{"a": 2}' > test2.json  
# diffx test1.json test2.json
```

## Development Installation

For development work, you'll need additional tools:

```bash
# Clone the repository
git clone https://github.com/kako-jun/diffai.git
cd diffai

# Install development dependencies
cargo install cargo-watch
cargo install criterion

# Build in development mode
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench

# Install locally for testing
cargo install --path diffai-cli
```

## Troubleshooting

### Common Issues

#### 1. Rust Not Found
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### 2. Compilation Errors
```bash
# Update Rust to latest version
rustup update

# Clean and rebuild
cargo clean
cargo build
```

#### 3. Permission Denied (Linux/macOS)
```bash
# Install to user directory instead
cargo install --path diffai-cli --root ~/.local

# Add to PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Large Model Files
For very large model files (>1GB), ensure you have sufficient memory:

```bash
# Check available memory
free -h  # Linux
vm_stat  # macOS

# For large files, consider using streaming mode (future feature)
diffai --stream large_model1.safetensors large_model2.safetensors
```

## Performance Considerations

### Memory Requirements

| Model Size | Recommended RAM |
|------------|----------------|
| < 100MB    | 1GB            |
| 100MB-1GB  | 4GB            |
| 1GB-10GB   | 16GB           |
| > 10GB     | 32GB+          |

### Optimization Tips

1. **Use SSD storage** for faster file I/O
2. **Close other applications** when comparing large models
3. **Use epsilon tolerance** to ignore minor floating-point differences
4. **Filter results** using `--path` or `--ignore-keys-regex` for focused analysis

## Updating

### From crates.io
```bash
cargo install diffai --force
```

### From Source
```bash
cd diffai
git pull origin main
cargo install --path diffai-cli --force
```

## Uninstallation

```bash
# Remove binary
cargo uninstall diffai

# Or manually remove
rm $(which diffai)
```

## Getting Help

If you encounter installation issues:

1. Check the [GitHub Issues](https://github.com/kako-jun/diffai/issues)
2. Join the [GitHub Discussions](https://github.com/kako-jun/diffai/discussions)
3. Review the [Contributing Guide](../../CONTRIBUTING.md) for development setup

## Next Steps

After installation, see [Basic Usage](basic-usage_ja.md) to get started with diffai.