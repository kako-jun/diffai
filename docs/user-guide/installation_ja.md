# インストールガイド

様々なプラットフォームへのdiffaiのインストール方法を解説します。

## 前提条件

- **Rust 1.75以上**: [rustup.rs](https://rustup.rs/)からインストール
- **対応OS**: Linux、macOS、Windows
- **メモリ**: 大規模モデルファイルの処理には4GB以上を推奨

## インストール方法

### 方法1: crates.ioから（推奨）

```bash
cargo install diffai
```

**注意**: この方法はdiffaiがcrates.ioに公開され次第利用可能になります。

### 方法2: ソースコードから（現在の方法）

```bash
# リポジトリをクローン
git clone https://github.com/kako-jun/diffai.git
cd diffai

# ビルドしてインストール
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
# 解凍してPATHに移動
tar -xzf diffai-linux-x86_64.tar.gz
sudo mv diffai /usr/local/bin/
```

## プラットフォーム別の手順

### Linux

#### Ubuntu/Debian
```bash
# Rustがインストールされていない場合はインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# diffaiをインストール
cargo install diffai
```

#### Arch Linux
```bash
# AURから（将来提供予定）
yay -S diffai

# またはソースから
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo install --path diffai-cli
```

#### CentOS/RHEL/Fedora
```bash
# Rustをインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# diffaiをインストール
cargo install diffai
```

### macOS

#### Homebrewを使用（計画中）
```bash
# 将来的に利用可能になります
brew install diffai
```

#### 手動インストール
```bash
# 必要に応じてRustをインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# diffaiをインストール
cargo install diffai
```

### Windows

#### Cargoを使用
```powershell
# https://rustup.rs/ からRustをインストール
# その後diffaiをインストール
cargo install diffai
```

#### Scoopを使用（計画中）
```powershell
# 将来的に利用可能になります
scoop install diffai
```

## コンテナでのインストール

### Docker

```bash
# イメージをプル（提供開始後）
docker pull ghcr.io/kako-jun/diffai:latest

# コンテナでdiffaiを実行
docker run --rm -v $(pwd):/workspace ghcr.io/kako-jun/diffai:latest \
  model1.safetensors model2.safetensors
```

### Dockerイメージのビルド

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai

# Dockerイメージをビルド
docker build -t diffai .

# 実行
docker run --rm -v $(pwd):/workspace diffai \
  model1.safetensors model2.safetensors
```

## 動作確認

インストール後、diffaiが正しく動作することを確認：

```bash
# バージョン確認
diffai --version

# ヘルプを表示
diffai --help

# サンプルファイルでテスト
echo '{"a": 1}' > test1.json
echo '{"a": 2}' > test2.json
diffai test1.json test2.json

# Expected output:
# ~ a: 1 -> 2

# クリーンアップ
rm test1.json test2.json
```

## 開発環境のセットアップ

開発作業には追加ツールが必要です：

```bash
# リポジトリをクローン
git clone https://github.com/kako-jun/diffai.git
cd diffai

# 開発用依存関係をインストール
cargo install cargo-watch
cargo install criterion

# 開発モードでビルド
cargo build

# テストを実行
cargo test

# ベンチマークを実行
cargo bench

# テスト用にローカルインストール
cargo install --path diffai-cli
```

## トラブルシューティング

### よくある問題

#### 1. Rustが見つからない
```bash
# Rustをインストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

#### 2. コンパイルエラー
```bash
# Rustを最新版に更新
rustup update

# クリーンして再ビルド
cargo clean
cargo build
```

#### 3. 権限エラー（Linux/macOS）
```bash
# ユーザーディレクトリにインストール
cargo install --path diffai-cli --root ~/.local

# PATHに追加
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 4. 大規模モデルファイル
非常に大きなモデルファイル（1GB超）の場合、十分なメモリを確保してください：

```bash
# 利用可能メモリを確認
free -h  # Linux
vm_stat  # macOS

# 大規模ファイルにはストリーミングモードを検討（将来機能）
diffai --stream large_model1.safetensors large_model2.safetensors
```

## パフォーマンスに関する考慮事項

### メモリ要件

| モデルサイズ | 推奨メモリ |
|------------|----------|
| 100MB未満   | 1GB      |
| 100MB-1GB  | 4GB      |
| 1GB-10GB   | 16GB     |
| 10GB超     | 32GB以上  |

### 最適化のヒント

1. **SSDストレージを使用** - 高速なファイルI/Oを実現
2. **他のアプリケーションを終了** - 大規模モデル比較時
3. **イプシロン許容誤差を活用** - 微小な浮動小数点差を無視
4. **結果をフィルタリング** - `--path`や`--ignore-keys-regex`で分析対象を絞り込み

## アップデート

### crates.ioから
```bash
cargo install diffai --force
```

### ソースコードから
```bash
cd diffai
git pull origin main
cargo install --path diffai-cli --force
```

## アンインストール

```bash
# バイナリを削除
cargo uninstall diffai

# または手動で削除
rm $(which diffai)
```

## ヘルプ

インストールで問題が発生した場合：

1. [GitHub Issues](https://github.com/kako-jun/diffai/issues)を確認
2. [GitHub Discussions](https://github.com/kako-jun/diffai/discussions)に参加
3. 開発環境のセットアップは[貢献ガイド](../../CONTRIBUTING_ja.md)を参照

## 次のステップ

インストール完了後は、[基本的な使い方](basic-usage_ja.md)でdiffaiの使用を開始しましょう。