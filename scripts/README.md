# Scripts Directory

## 📁 Directory Structure

### 🎬 demo/
デモ生成・テスト出力用のスクリプト

- **`generate-comprehensive-demo.sh`** - 包括的デモ生成
- **`generate-test-outputs.sh`** - テスト出力生成

### 🛠️ utils/
ユーティリティスクリプト

- **`check-docs-consistency.sh`** - ドキュメント整合性チェック
- **`create-rust-cli-kiln-symlink.sh`** - rust-cli-kilnシンボリックリンク作成
- **`setup-github-workflow.sh`** - GitHubワークフロー設定

## 🚀 リリース関連スクリプト

リリース関連のスクリプトは `mnt/rust-cli-kiln/release-guide.md` を参照してください。

## 🧪 テスト関連スクリプト

テスト関連のスクリプトも `mnt/rust-cli-kiln/` 配下に移動しています。

## 📋 日常開発での使用

```bash
# ドキュメント整合性チェック
./scripts/utils/check-docs-consistency.sh

# rust-cli-kilnシンボリックリンク作成
./scripts/utils/create-rust-cli-kiln-symlink.sh

# GitHubワークフロー設定
./scripts/utils/setup-github-workflow.sh

# デモ生成
./scripts/demo/generate-comprehensive-demo.sh
```