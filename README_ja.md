# diffai

> **PyTorch、Safetensors、NumPy、MATLABファイル専用のAI/ML特化diffツール**

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io CLI](https://img.shields.io/crates/v/diffai.svg?label=diffai-cli)](https://crates.io/crates/diffai)
[![Docs.rs Core](https://docs.rs/diffai-core/badge.svg)](https://docs.rs/diffai-core)
[![npm](https://img.shields.io/npm/v/diffai-js.svg?label=diffai-js)](https://www.npmjs.com/package/diffai-js)
[![PyPI](https://img.shields.io/pypi/v/diffai-python.svg?label=diffai-python)](https://pypi.org/project/diffai-python/)
[![Documentation](https://img.shields.io/badge/📚%20User%20Guide-Documentation-green)](https://github.com/kako-jun/diffai/tree/main/docs/index_ja.md)
[![API Reference](https://img.shields.io/badge/🔧%20API%20Reference-docs.rs-blue)](https://docs.rs/diffai-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

モデル構造、テンソル統計、数値データを理解する**AI/ML・科学計算ワークフロー**専用の次世代diffツール。単なるテキスト変更ではなく、PyTorch、Safetensors、NumPy配列、MATLABファイル、構造化データをネイティブサポート。

```bash
# Traditional diff fails with binary model files
$ diff model_v1.safetensors model_v2.safetensors
Binary files model_v1.safetensors and model_v2.safetensors differ

# diffai shows meaningful model changes with full analysis
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

- **AI/MLネイティブ対応**: PyTorch（.pt/.pth）、Safetensors（.safetensors）、NumPy（.npy/.npz）、MATLAB（.mat）ファイルの直接サポート
- **テンソル解析**: テンソル統計（平均、標準偏差、最小値、最大値、形状、メモリ使用量）の自動計算
- **包括的ML解析**: 量子化、アーキテクチャ、メモリ、収束、異常検出、デプロイメント準備状況を含む30以上の解析機能 - すべてデフォルトで有効
- **科学データサポート**: 複素数対応のNumPy配列とMATLAB行列
- **Pure Rust実装**: システム依存関係なし、Windows/Linux/macOSで追加インストール不要
- **複数出力形式**: 色付きCLI、MLOps統合用JSON、人間が読みやすいYAMLレポート
- **高速・メモリ効率**: 大型モデルファイルの効率的処理を可能にするRust製

## なぜdiffaiなのか？

従来のdiffツールはAI/MLワークフローには不適切です：

| 課題 | 従来ツール | diffai |
|------|------------|--------|
| **バイナリモデルファイル** | "Binary files differ" | 統計付きテンソルレベル解析 |
| **大容量ファイル（GB+）** | メモリ不足や処理失敗 | 効率的ストリーミング・チャンク処理 |
| **統計的変化** | 意味理解なし | 統計的有意性付き平均/標準偏差/形状比較 |
| **ML特化形式** | サポートなし | PyTorch/Safetensors/NumPy/MATLABネイティブ対応 |
| **科学計算ワークフロー** | テキストのみ比較 | 数値配列解析・可視化 |

### diffai vs MLOpsツール

diffaiは実験管理ではなく**構造比較**に焦点を当てることで、既存のMLOpsツールを補完します：

| 観点 | diffai | MLflow / DVC / ModelDB |
|------|--------|------------------------|
| **焦点** | 「比較不可能なものを比較可能にする」 | 体系化、再現性、CI/CD統合 |
| **データ前提** | 出自不明ファイル / ブラックボックス生成成果物 | 適切に文書化・追跡されたデータ |
| **操作** | 構造・視覚比較の最適化 | バージョン管理・実験追跡の専門化 |
| **範囲** | JSON/YAML/モデルファイルを含む「曖昧な構造」の可視化 | 実験メタデータ、バージョン管理、再現性 |

## インストール

### crates.ioから（推奨）

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
# PyTorchモデルを完全解析付きで比較（デフォルト）
diffai model_old.pt model_new.pt

# Safetensorsを完全ML解析付きで比較
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# NumPy配列を比較
diffai data_v1.npy data_v2.npy

# MATLABファイルを比較
diffai experiment_v1.mat experiment_v2.mat
```

### ML解析機能

```bash
# PyTorch/Safetensorsでは完全ML解析が自動実行
diffai baseline.safetensors finetuned.safetensors
# 出力: 量子化、アーキテクチャ、メモリ等の30以上の解析種別

# 自動化用JSON出力
diffai model_v1.safetensors model_v2.safetensors --output json

# 詳細な診断情報を表示する詳細モード
diffai model_v1.safetensors model_v2.safetensors --verbose

# 人間が読みやすいレポート用YAML出力
diffai model_v1.safetensors model_v2.safetensors --output yaml
```

## 📚 ドキュメント

- **[実用例・デモンストレーション](docs/examples/)** - 実際の出力付きdiffaiの動作確認
- **[APIドキュメント](https://docs.rs/diffai-core)** - Rustライブラリドキュメント
- **[ユーザーガイド](docs/user-guide/getting-started_ja.md)** - 包括的な使用ガイド
- **[ML解析ガイド](docs/reference/ml-analysis_ja.md)** - ML特化機能の詳細解説

## サポートされるファイル形式

### MLモデル形式
- **Safetensors** (.safetensors) - HuggingFace標準形式
- **PyTorch** (.pt, .pth) - Candle統合付きPyTorchモデルファイル

### 科学データ形式  
- **NumPy** (.npy, .npz) - 完全統計解析付きNumPy配列
- **MATLAB** (.mat) - 複素数サポート付きMATLAB行列

### 構造化データ形式
- **JSON** (.json) - JavaScript Object Notation
- **YAML** (.yaml, .yml) - YAML Ain't Markup Language
- **TOML** (.toml) - Tom's Obvious Minimal Language  
- **XML** (.xml) - Extensible Markup Language
- **INI** (.ini) - 設定ファイル
- **CSV** (.csv) - カンマ区切り値

## ML解析機能

### 自動包括解析（v0.3.4）
PyTorchまたはSafetensorsファイルを比較する際、diffaiは30以上のML解析機能を自動実行します：

**自動機能には以下が含まれます：**
- **統計解析**: 詳細なテンソル統計（平均、標準偏差、最小値、最大値、形状、メモリ）
- **量子化解析**: 量子化効果と効率性を解析
- **アーキテクチャ比較**: モデルアーキテクチャと構造変化を比較
- **メモリ解析**: メモリ使用量と最適化機会を解析
- **異常検出**: モデルパラメータの数値異常を検出
- **収束解析**: モデルパラメータの収束パターンを解析
- **勾配解析**: 利用可能な勾配情報を解析
- **デプロイメント準備状況**: 本番デプロイメントの準備状況を評価
- **リグレッションテスト**: パフォーマンス劣化の自動検出
- **さらに20以上の専門機能**

### 将来の機能強化
- TensorFlow形式サポート（.pb, .h5, SavedModel）
- ONNX形式サポート
- 高度な可視化・チャート機能

### 設計哲学
diffaiはMLモデルに対してデフォルトで包括的解析を提供し、選択の麻痺を解消します。ユーザーは数十の解析フラグを覚えたり指定したりする必要なく、関連するすべての洞察を得られます。

## Debugging and Diagnostics

### Verbose Mode (`--verbose` / `-v`)
Get comprehensive diagnostic information for debugging and performance analysis:

```bash
# Basic verbose output
diffai model1.safetensors model2.safetensors --verbose

# Verbose with structured data filtering
diffai data1.json data2.json --verbose --epsilon 0.001 --ignore-keys-regex "^id$"
```

**Verbose output includes:**
- **Configuration diagnostics**: Format settings, filters, analysis modes
- **File analysis**: Paths, sizes, detected formats, processing context
- **Performance metrics**: Processing time, difference counts, optimization status
- **Directory statistics**: File counts, comparison summaries (ディレクトリ自動処理)

**Example verbose output:**
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

📚 **See [Verbose Output Guide](docs/user-guide/verbose-output_ja.md) for detailed usage**

## Output Formats

### CLI Output (Default)
Colored, human-readable output with intuitive symbols:
- `~` Changed tensors/arrays with statistical comparison
- `+` Added tensors/arrays with metadata
- `-` Removed tensors/arrays with metadata

### JSON Output
Structured output for MLOps integration and automation:
```bash
diffai model1.safetensors model2.safetensors --output json | jq .
```

### YAML Output  
Human-readable structured output for documentation:
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## Real-World Use Cases

### Research & Development
```bash
# Compare model before and after fine-tuning (full analysis automatic)
diffai pretrained_model.safetensors finetuned_model.safetensors
# Outputs: learning_progress, convergence_analysis, parameter stats, and 27 more analyses

# Analyze architectural changes during development
diffai baseline_architecture.pt improved_architecture.pt
# Outputs: architecture_comparison, param_efficiency_analysis, and full ML analysis
```

### MLOps & CI/CD
```bash
# Automated model validation in CI/CD (comprehensive analysis)
diffai production_model.safetensors candidate_model.safetensors
# Outputs: deployment_readiness, regression_test, risk_assessment, and 27 more analyses

# Performance impact assessment with JSON output for automation
diffai original_model.pt optimized_model.pt --output json
# Outputs: quantization_analysis, memory_analysis, performance_impact_estimate, etc.
```

### Scientific Computing
```bash
# Compare NumPy experiment results
diffai baseline_results.npy new_results.npy

# Analyze MATLAB simulation data
diffai simulation_v1.mat simulation_v2.mat

# Compare compressed NumPy archives
diffai dataset_v1.npz dataset_v2.npz
```

### Experiment Tracking
```bash
# Generate comprehensive reports
diffai experiment_baseline.safetensors experiment_improved.safetensors \
  --generate-report --markdown-output --review-friendly

# A/B test analysis
diffai model_a.safetensors model_b.safetensors \
  --statistical-significance --hyperparameter-comparison
```

## Command-Line Options

### Basic Options
- `-f, --format <FORMAT>` - 入力ファイル形式を指定
- `-o, --output <OUTPUT>` - 出力形式を選択 (cli, json, yaml)
- **ディレクトリ比較** - ディレクトリが提供された場合、自動的に再帰処理

**Note:** For ML models (PyTorch/Safetensors), comprehensive analysis including statistics runs automatically

### Advanced Options
- `--path <PATH>` - Filter differences by specific path
- `--ignore-keys-regex <REGEX>` - Ignore keys matching regex pattern
- `--epsilon <FLOAT>` - Set tolerance for float comparisons
- `--array-id-key <KEY>` - Specify key for array element identification
- `--sort-by-change-magnitude` - Sort by change magnitude

## Examples

### Basic Tensor Comparison (Automatic)
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

### JSON Output for Automation
```bash
$ diffai baseline.safetensors improved.safetensors --output json
{
  "anomaly_detection": {"type": "none", "severity": "none"},
  "architecture_comparison": {"type1": "feedforward", "type2": "feedforward"},
  "deployment_readiness": {"readiness": 0.92, "strategy": "blue_green"},
  "quantization_analysis": {"compression": "0.0%", "speedup": "1.8x"},
  "regression_test": {"passed": true, "degradation": "-2.5%"}
  // ... plus 25+ additional analysis features
}
```

### Scientific Data Analysis
```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLAB File Comparison
```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

## パフォーマンス

diffaiは大容量ファイルと科学計算ワークフロー用に最適化されています：

- **メモリ効率**: GB+ファイルのストリーミング処理
- **高速**: 最適化されたテンソル操作を伴うRust実装
- **スケーラブル**: 数百万/数十億パラメータのモデルに対応
- **クロスプラットフォーム**: 依存関係なしでWindows、Linux、macOSで動作

## コントリビューション

コントリビューションを歓迎します！ガイドラインは[CONTRIBUTING](CONTRIBUTING.md)をご覧ください。

### 開発環境の設定

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

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。

## 関連プロジェクト

- **[diffx](https://github.com/kako-jun/diffx)** - 汎用構造化データdiffツール（diffaiの姉妹プロジェクト）
- **[safetensors](https://github.com/huggingface/safetensors)** - テンソルを保存・配布するシンプルで安全な方法
- **[PyTorch](https://pytorch.org/)** - 機械学習フレームワーク
- **[NumPy](https://numpy.org/)** - Python科学計算の基礎パッケージ

