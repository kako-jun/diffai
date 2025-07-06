# diffai Documentation

AI/ML特化の diff ツール diffai の包括的なドキュメント

## 🌐 Language Support

- **日本語**: 現在のドキュメント
- **English**: [English documentation](index_ja.md)

## 📖 目次

### 🚀 ユーザーガイド
- [**インストール**](user-guide/installation_ja.md) - 各環境での導入方法
- [**基本的な使い方**](user-guide/basic-usage_ja.md) - 基本コマンドと操作
- [**ML/AI ワークフロー**](user-guide/ml-workflows_ja.md) - ML開発での活用法
- [**設定**](user-guide/configuration_ja.md) - 設定ファイルとカスタマイズ

### 🤖 AI/ML特化機能
- [**PyTorch モデル比較**](examples/pytorch-models_ja.md) - モデル構造の差分確認
- [**Safetensors 対応**](examples/safetensors_ja.md) - 安全なテンソル形式のサポート
- [**データセット比較**](examples/datasets_ja.md) - データセット形式の差分分析
- [**実験管理**](examples/experiments_ja.md) - MLflow との連携例

### 🏗️ アーキテクチャ
- [**設計原則**](architecture/design-principles_ja.md) - diffai の設計思想
- [**コア機能**](architecture/core-features_ja.md) - 主要機能の詳細
- [**拡張性**](architecture/extensibility_ja.md) - プラグイン機能とカスタマイズ

### 📚 API リファレンス
- [**CLI API**](api/cli_ja.md) - コマンドラインインターフェース
- [**Rust API**](api/rust_ja.md) - Rust ライブラリとしての使用
- [**設定オプション**](api/config_ja.md) - 全設定項目の詳細

## 🎯 Quick Start

```bash
# PyTorchモデルの比較
diffai model1.pth model2.pth

# Safetensorsファイルの比較
diffai model1.safetensors model2.safetensors

# データセットの比較
diffai dataset1.csv dataset2.csv --format csv

# 実験結果の比較
diffai experiment1/results experiment2/results --recursive
```

## 🔗 関連リンク

- [GitHub Repository](https://github.com/kako-jun/diffai)
- [crates.io](https://crates.io/crates/diffai)
- [Issues & Support](https://github.com/kako-jun/diffai/issues)
- [Contributing Guide](../CONTRIBUTING_ja.md)