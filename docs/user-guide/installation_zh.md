# 安装指南

在各种环境中安装 diffai 的完整指南。

## 先决条件

- **Rust 1.75+**: 从 [rustup.rs](https://rustup.rs/) 安装
- **操作系统**: Linux、macOS 或 Windows
- **内存**: 推荐 4GB+ 用于大型模型文件

## 安装方法

### 方法 1: 从 crates.io 安装（推荐）

```bash
cargo install diffai
```

**注意**: 一旦 diffai 发布到 crates.io，此方法将可用。

### 方法 2: 从源码构建（当前）

```bash
# 克隆仓库
git clone https://github.com/kako-jun/diffai.git
cd diffai

# 构建并安装
cargo install --path diffai-cli

# 验证安装
diffai --version
```

### 方法 3: 从 GitHub 发布页面

从 [GitHub 发布页面](https://github.com/kako-jun/diffai/releases) 下载预构建的二进制文件：

- **Linux (x86_64)**: `diffai-linux-x86_64.tar.gz`
- **macOS (x86_64)**: `diffai-macos-x86_64.tar.gz`
- **macOS (ARM64)**: `diffai-macos-aarch64.tar.gz`
- **Windows (x86_64)**: `diffai-windows-x86_64.zip`

```bash
# 解压并移动到 PATH
tar -xzf diffai-linux-x86_64.tar.gz
sudo mv diffai /usr/local/bin/
```

## 平台特定说明

### Linux

#### Ubuntu/Debian
```bash
# 如果尚未安装 Rust，请先安装
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 安装 diffai
cargo install diffai
```

#### Arch Linux
```bash
# 使用 AUR（当可用时）
yay -S diffai

# 或从源码构建
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo install --path diffai-cli
```

#### CentOS/RHEL/Fedora
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 安装 diffai
cargo install diffai
```

### macOS

#### 使用 Homebrew（计划中）
```bash
# 这将在将来可用
brew install diffai
```

#### 手动安装
```bash
# 如果需要，请安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# 安装 diffai
cargo install diffai
```

### Windows

#### 使用 Cargo
```powershell
# 从 https://rustup.rs/ 安装 Rust
# 然后安装 diffai
cargo install diffai
```

#### 使用 Scoop（计划中）
```powershell
# 这将在将来可用
scoop install diffai
```

## 容器安装

### Docker

```bash
# 拉取镜像（当可用时）
docker pull ghcr.io/kako-jun/diffai:latest

# 在容器中运行 diffai
docker run --rm -v $(pwd):/workspace ghcr.io/kako-jun/diffai:latest \
  model1.safetensors model2.safetensors
```

### 构建 Docker 镜像

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai

# 构建 Docker 镜像
docker build -t diffai .

# 运行
docker run --rm -v $(pwd):/workspace diffai \
  model1.safetensors model2.safetensors
```

## 验证

安装后，验证 diffai 是否正常工作：

```bash
# 检查版本
diffai --version

# 运行帮助
diffai --help

# 使用示例文件测试
echo '{"a": 1}' > test1.json
echo '{"a": 2}' > test2.json
diffai test1.json test2.json

# 预期输出：
# ~ a: 1 -> 2

# 清理
rm test1.json test2.json
```

## 开发安装

对于开发工作，您需要额外的工具：

```bash
# 克隆仓库
git clone https://github.com/kako-jun/diffai.git
cd diffai

# 安装开发依赖
cargo install cargo-watch
cargo install criterion

# 开发模式构建
cargo build

# 运行测试
cargo test

# 运行基准测试
cargo bench

# 本地安装以进行测试
cargo install --path diffai-cli
```

## 故障排除

### 常见问题

#### 1. 找不到 Rust
```bash
# 安装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### 2. 编译错误
```bash
# 更新 Rust 到最新版本
rustup update

# 清理并重新构建
cargo clean
cargo build
```

#### 3. 权限被拒绝（Linux/macOS）
```bash
# 改为安装到用户目录
cargo install --path diffai-cli --root ~/.local

# 添加到 PATH
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 4. 大型模型文件
对于非常大的模型文件（>1GB），请确保有足够的内存：

```bash
# 检查可用内存
free -h  # Linux
vm_stat  # macOS

# 对于大文件，考虑使用流式模式（未来功能）
diffai --stream large_model1.safetensors large_model2.safetensors
```

## 性能考虑

### 内存需求

| 模型大小 | 推荐内存 |
|----------|----------|
| < 100MB  | 1GB      |
| 100MB-1GB | 4GB     |
| 1GB-10GB | 16GB     |
| > 10GB   | 32GB+    |

### 优化提示

1. **使用 SSD 存储** 以获得更快的文件 I/O
2. **比较大型模型时关闭其他应用程序**
3. **使用 epsilon 容差** 忽略微小的浮点差异
4. **使用 `--path` 或 `--ignore-keys-regex` 过滤结果** 进行集中分析

## 更新

### 从 crates.io 更新
```bash
cargo install diffai --force
```

### 从源码更新
```bash
cd diffai
git pull origin main
cargo install --path diffai-cli --force
```

## 卸载

```bash
# 移除二进制文件
cargo uninstall diffai

# 或手动移除
rm $(which diffai)
```

## 获取帮助

如果您遇到安装问题：

1. 查看 [GitHub Issues](https://github.com/kako-jun/diffai/issues)
2. 加入 [GitHub Discussions](https://github.com/kako-jun/diffai/discussions)
3. 查看 [贡献指南](../../CONTRIBUTING.md) 了解开发设置

## 下一步

安装后，请参阅 [基本用法](basic-usage_zh.md) 开始使用 diffai。