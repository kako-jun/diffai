# diffai

> **PyTorch、Safetensors、NumPy、MATLABファイル対応のAI/ML特化差分ツール**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/diffai.svg)](https://crates.io/crates/diffai)
[![Documentation](https://img.shields.io/badge/docs-GitHub-blue)](https://github.com/kako-jun/diffai/tree/main/docs/index_ja.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**AI/ML・科学計算ワークフロー**に特化した次世代差分ツール。モデル構造、テンソル統計、数値データを理解し、単なるテキスト変更ではなく意味のある差分を表示します。PyTorch、Safetensors、NumPy配列、MATLABファイル、構造化データをネイティブサポート。

```bash
# 従来のdiffはバイナリモデルファイルでは失敗
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffaiは意味のあるモデル変更を完全分析付きで表示
$ diffai model_v1.safetensors model_v2.safetensors
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  gradient_analysis: flow_health=healthy, norm=0.015000, ratio=1.0500
  deployment_readiness: readiness=0.92, strategy=blue_green, risk=low
  quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%

[WARNING]
• Memory usage increased moderately (+250MB). Monitor resource consumption.
• Inference speed moderately affected (1.3x slower). Consider optimization opportunities.
```

## 主な機能

- **AI/MLネイティブ**: PyTorch (.pt/.pth)、Safetensors (.safetensors)、NumPy (.npy/.npz)、MATLAB (.mat) ファイルを直接サポート
- **テンソル分析**: テンソル統計の自動計算（平均、標準偏差、最小値、最大値、形状、メモリ使用量）
- **包括的ML分析**: PyTorch/Safetensorsファイルに対して30+の分析機能が自動実行（量子化、アーキテクチャ、メモリ、収束、異常検出、デプロイ準備度など） - すべてデフォルトで有効
- **科学データサポート**: 複素数対応のNumPy配列とMATLAB行列
- **純粋Rust実装**: システム依存なし、Windows/Linux/macOSで追加インストール不要
- **複数出力形式**: 色付きCLI、MLOps統合用JSON、人間可読YAML
- **高速・メモリ効率**: 大型モデルファイルを効率的に処理するRust実装

## なぜdiffaiなのか？

従来のdiffツールはAI/MLワークフローには不適切です：

| 課題 | 従来ツール | diffai |
|------|------------|---------|
| **バイナリモデルファイル** | "Binary files differ" | 統計付きテンソルレベル分析 |
| **大型ファイル (GB+)** | メモリ問題や失敗 | 効率的なストリーミング・チャンク処理 |
| **統計的変化** | 意味理解なし | 有意性のある平均/標準偏差/形状比較 |
| **ML特化フォーマット** | サポートなし | ネイティブPyTorch/Safetensors/NumPy/MATLAB |
| **科学計算ワークフロー** | テキスト比較のみ | 数値配列分析と可視化 |

### diffai vs MLOpsツール

diffaiは既存のMLOpsツールを**構造的比較**に焦点を当てることで補完します：

| 観点 | diffai | MLflow / DVC / ModelDB |
|------|--------|------------------------|
| **主眼** | 「比較不能なものを比較可能に」 | 体系化・再現性・CI/CDの一環 |
| **データの前提** | 出自が不明なファイル／ブラックボックス生成物 | きちんと記録されている前提 |
| **操作性** | 差分を構造的・視覚的に比較しやすく | バージョン管理や実験トラッキングに特化 |
| **適用範囲** | JSON・YAML・モデルファイルなど"曖昧な構造"を含めて可視化 | 実験メタデータ・バージョン管理・再現性 |

## インストール

### crates.io から（推奨）

```bash
cargo install diffai
```

### ソースから

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build --release
```

## クイックスタート

### 基本的なモデル比較

```bash
# PyTorchモデル比較（デフォルトで完全分析）
diffai model_old.pt model_new.pt

# Safetensors比較（30+のML分析機能を含む包括的分析）
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# NumPy配列比較
diffai data_v1.npy data_v2.npy

# MATLABファイル比較
diffai experiment_v1.mat experiment_v2.mat
```

### 自動ML分析

```bash
# PyTorch/Safetensorsファイルでは完全なML分析が自動実行
diffai baseline.safetensors finetuned.safetensors
# 出力: 量子化、アーキテクチャ、メモリ分析など30+種類の分析

# 自動化用JSON出力
diffai model_v1.safetensors model_v2.safetensors --output json

# 詳細な診断情報付きでのverboseモード
diffai model_v1.safetensors model_v2.safetensors --verbose

# 人間可読レポート用YAML出力
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 📚 ドキュメント

- **[実動例とデモンストレーション](docs/examples/)** - 実際の出力でdiffaiを確認
- **[APIドキュメント](https://docs.rs/diffai-core)** - Rustライブラリドキュメント
- **[ユーザーガイド](docs/user-guide.md)** - 包括的使用ガイド
- **[ML分析ガイド](docs/ml-analysis-guide.md)** - ML特化機能の詳細ガイド

## 対応ファイル形式

### MLモデル形式
- **Safetensors** (.safetensors) - HuggingFace標準形式
- **PyTorch** (.pt, .pth) - Candle統合PyTorchモデルファイル

### 科学データ形式
- **NumPy** (.npy, .npz) - 完全統計解析付きNumPy配列
- **MATLAB** (.mat) - 複素数対応MATLAB行列

### 構造化データ形式
- **JSON** (.json) - JavaScript Object Notation
- **YAML** (.yaml, .yml) - YAML Ain't Markup Language
- **TOML** (.toml) - Tom's Obvious Minimal Language
- **XML** (.xml) - Extensible Markup Language
- **INI** (.ini) - 設定ファイル
- **CSV** (.csv) - カンマ区切り値

## ML分析機能

### 自動包括的分析 (v0.3.4)
PyTorchまたはSafetensorsファイルを比較する際、diffaiは30+のML分析機能を自動実行します：

**自動機能に含まれるもの：**
- **統計分析**: 詳細なテンソル統計（平均、標準偏差、最小値、最大値、形状、メモリ）
- **量子化分析**: 量子化効果と効率の分析
- **アーキテクチャ比較**: モデルアーキテクチャと構造変更の比較
- **メモリ分析**: メモリ使用量と最適化機会の分析
- **異常検知**: モデルパラメータの数値異常を検出
- **収束分析**: モデルパラメータの収束パターンを分析
- **勾配分析**: 利用可能な場合の勾配情報を分析
- **デプロイ準備度**: 本番デプロイの準備状況を評価
- **回帰テスト**: 自動的な性能劣化検出
- **その他20+の専門機能**

### 将来の拡張
- TensorFlow形式サポート (.pb, .h5, SavedModel)
- ONNX形式サポート
- 高度な可視化とチャート機能

### 設計思想
diffaiはMLモデルに対してデフォルトで包括的な分析を提供し、選択の迷いを排除します。ユーザーは数十の分析フラグを覚えたり指定したりする必要なく、すべての関連する洞察を得られます。

## デバッグと診断

### 詳細モード（`--verbose` / `-v`）
デバッグとパフォーマンス分析のための包括的な診断情報を取得：

```bash
# 基本的な詳細出力（ML分析機能自動実行）
diffai model1.safetensors model2.safetensors --verbose

# 構造化データの詳細出力
diffai data1.json data2.json --verbose --epsilon 0.001 --ignore-keys-regex "^id$"
```

**詳細出力に含まれる情報：**
- **設定診断**: アクティブなML機能、フォーマット設定、フィルター
- **ファイル解析**: パス、サイズ、検出されたフォーマット、処理コンテキスト
- **パフォーマンス指標**: 処理時間、差分カウント、最適化状況
- **ディレクトリ統計**: ファイル数、比較サマリー（`--recursive`使用時）

**詳細出力例：**
```
=== diffai verbose mode enabled ===
Configuration:
  Input format: Safetensors
  Output format: Cli
  ML analysis: Full analysis enabled (all 30 features)
  Epsilon tolerance: 0.001

File analysis:
  Input 1: model1.safetensors
  Input 2: model2.safetensors
  Detected format: Safetensors
  File 1 size: 1048576 bytes
  File 2 size: 1048576 bytes

Processing results:
  Total processing time: 1.234ms
  Differences found: 15
  ML/Scientific data analysis completed
```

📚 **詳細については[詳細出力ガイド](docs/user-guide/verbose-output_ja.md)をご覧ください**


## 出力形式

### CLI出力（デフォルト）
直感的な記号付き色付き人間可読出力：
- `~` 統計比較付き変更テンソル/配列
- `+` メタデータ付き追加テンソル/配列
- `-` メタデータ付き削除テンソル/配列

### JSON出力
MLOps統合・自動化用構造化出力：
```bash
diffai model1.safetensors model2.safetensors --output json | jq .
```

### YAML出力
文書化用人間可読構造化出力：
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## 実用例

### 研究開発
```bash
# ファインチューニング前後のモデル比較（完全分析が自動）
diffai pretrained_model.safetensors finetuned_model.safetensors
# 出力: 学習進捗、収束分析、パラメータ統計、その他27+の分析

# 開発中のアーキテクチャ変更分析
diffai baseline_architecture.pt improved_architecture.pt
# 出力: アーキテクチャ比較、パラメータ効率分析、完全なML分析
```

### MLOps・CI/CD
```bash
# CI/CDでの自動モデル検証（包括的分析）
diffai production_model.safetensors candidate_model.safetensors
# 出力: デプロイ準備度、回帰テスト、リスク評価、その他27+の分析

# 自動化用の性能影響評価（JSON出力）
diffai original_model.pt optimized_model.pt --output json
# 出力: 量子化分析、メモリ分析、性能影響推定など
```

### 科学計算
```bash
# NumPy実験結果比較
diffai baseline_results.npy new_results.npy

# MATLABシミュレーションデータ分析
diffai simulation_v1.mat simulation_v2.mat

# 圧縮NumPyアーカイブ比較
diffai dataset_v1.npz dataset_v2.npz
```

### 実験追跡
```bash
# 包括的レポート生成
diffai experiment_baseline.safetensors experiment_improved.safetensors \
  --generate-report --markdown-output --review-friendly

# A/Bテスト分析
diffai model_a.safetensors model_b.safetensors \
  --statistical-significance --hyperparameter-comparison
```

## コマンドラインオプション

### 基本オプション
- `-f, --format <FORMAT>` - 入力ファイル形式指定
- `-o, --output <OUTPUT>` - 出力形式選択（cli, json, yaml）
- `-r, --recursive` - ディレクトリ再帰比較

**注意:** MLモデル（PyTorch/Safetensors）では、統計を含む包括的分析が自動的に実行されます

### 高度オプション
- `--path <PATH>` - 特定パスでの差分フィルタ
- `--ignore-keys-regex <REGEX>` - 正規表現パターンに一致するキーを無視
- `--epsilon <FLOAT>` - 浮動小数点比較の許容誤差設定
- `--array-id-key <KEY>` - 配列要素識別用キー指定
- `--sort-by-change-magnitude` - 変更量でソート

## 使用例

### 基本テンソル比較（自動）
```bash
$ diffai simple_model_v1.safetensors simple_model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.bias: mean=-0.0076->-0.0257, std=0.0661->0.0973
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  ~ fc3.bias: mean=-0.0074->-0.0130, std=0.1031->0.1093
  ~ fc3.weight: mean=-0.0035->-0.0010, std=0.0990->0.1113
```

### 自動化用JSON出力
```bash
$ diffai baseline.safetensors improved.safetensors --output json
{
  "anomaly_detection": {"type": "none", "severity": "none"},
  "architecture_comparison": {"type1": "feedforward", "type2": "feedforward"},
  "deployment_readiness": {"readiness": 0.92, "strategy": "blue_green"},
  "quantization_analysis": {"compression": "0.0%", "speedup": "1.8x"},
  "regression_test": {"passed": true, "degradation": "-2.5%"}
  // ... その他25+の分析機能
}
```

### 科学データ分析
```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLABファイル比較
```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

## 性能

diffaiは大型ファイルと科学計算ワークフローに最適化：

- **メモリ効率**: GB+ファイルのストリーミング処理
- **高速**: 最適化されたテンソル演算のRust実装
- **スケーラブル**: 数百万/数十億パラメータモデル対応
- **クロスプラットフォーム**: 依存関係なしでWindows、Linux、macOSで動作

## 貢献

貢献を歓迎します！ガイドラインは [CONTRIBUTING](CONTRIBUTING.md) をご覧ください。

### 開発環境セットアップ

```bash
git clone https://github.com/kako-jun/diffai.git
cd diffai
cargo build
cargo test
```

### テスト実行

```bash
# 全テスト実行
cargo test

# 特定テストカテゴリ実行
cargo test --test integration
cargo test --test ml_analysis
```

## ライセンス

このプロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 関連プロジェクト

- **[diffx](https://github.com/kako-jun/diffx)** - 汎用構造化データ差分ツール（diffaiの兄弟プロジェクト）
- **[safetensors](https://github.com/huggingface/safetensors)** - テンソル保存・配布のためのシンプルで安全な方法
- **[PyTorch](https://pytorch.org/)** - 機械学習フレームワーク
- **[NumPy](https://numpy.org/)** - Pythonでの科学計算のための基盤パッケージ
