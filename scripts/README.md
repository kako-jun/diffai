# Scripts Directory

## 📁 Directory Structure

### 🛠️ utils/
ユーティリティスクリプト

- **`create-github-shared-symlink.sh`** - GitHub共有ディレクトリのシンボリックリンク作成

### 📦 archive/
アーカイブされたスクリプト

- **`demo/`** - 古いデモ生成・テスト出力用のスクリプト（非推奨）

## 🚀 リリース関連スクリプト

リリース関連のスクリプトは `.github/rust-cli-kiln/release-guide.md` を参照してください。

## 🧪 テスト関連スクリプト

テスト関連のスクリプトも `.github/rust-cli-kiln/` 配下に移動しています。

## 📋 日常開発での使用

```bash
# GitHub共有ディレクトリのシンボリックリンク作成
./scripts/utils/create-github-shared-symlink.sh
```

## 📚 現在のドキュメント生成

デモやテスト出力の生成は以下の方法を推奨：

```bash
# 実際のテストを実行してドキュメントを生成
cargo test --test docs_examples

# 個別のサンプルを実行
cargo run --example simple_comparison

# 最新の例は docs/examples-code/ を参照
ls docs/examples-code/
```