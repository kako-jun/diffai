# 基本的な使い方

diffai - AI/ML特化差分ツールの基本操作を学習

## クイックスタート

### 基本的なファイル比較

```bash
# 2つのモデルファイルを比較（包括的分析が自動実行）
diffai model1.safetensors model2.safetensors

# JSON形式で出力
diffai model1.safetensors model2.safetensors --output json

# YAML形式で出力
diffai model1.safetensors model2.safetensors --output yaml
```

### ディレクトリ比較

```bash
# ディレクトリ全体を再帰的に比較
diffai dir1/ dir2/ --recursive

# 特定のファイル形式で比較
diffai models_v1/ models_v2/ --format safetensors --recursive
```

## AI/ML特化機能

### PyTorchモデル比較

```bash
# PyTorchモデルファイルを比較（完全分析が自動実行）
diffai model1.pt model2.pt

# 訓練チェックポイントを比較
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt

# ベースラインモデルと改良モデルを比較
diffai baseline_model.pt improved_model.pt
```

**出力例（完全分析）**:
```
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.25, speedup=1.8x, precision_loss=minimal
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### Safetensorsファイル比較

```bash
# Safetensorsファイル比較（包括的分析が自動実行）
diffai model1.safetensors model2.safetensors

# 本番デプロイ検証
diffai baseline.safetensors candidate.safetensors
```

### 科学データ比較

```bash
# NumPy配列比較（自動統計分析）
diffai data_v1.npy data_v2.npy

# MATLABファイル比較（自動統計分析）
diffai simulation_v1.mat simulation_v2.mat

# 圧縮NumPyアーカイブ比較（自動統計分析）
diffai dataset_v1.npz dataset_v2.npz
```

## コマンドオプション

### 基本オプション

| オプション | 説明 | 例 |
|-----------|-------------|---------|
| `-f, --format` | 入力ファイル形式を指定 | `--format safetensors` |
| `-o, --output` | 出力形式を選択 | `--output json` |
| `-r, --recursive` | ディレクトリを再帰的に比較 | `--recursive` |
| `-v, --verbose` | 詳細な処理情報を表示 | `--verbose` |

### 高度オプション

| オプション | 説明 | 例 |
|-----------|-------------|---------|
| `--path` | 特定のパスでフィルタ | `--path "config.model"` |
| `--ignore-keys-regex` | 正規表現に一致するキーを無視 | `--ignore-keys-regex "^id$"` |
| `--epsilon` | 浮動小数点比較の許容誤差 | `--epsilon 0.001` |
| `--array-id-key` | 配列要素識別 | `--array-id-key "id"` |
| `--sort-by-change-magnitude` | 変更量でソート | `--sort-by-change-magnitude` |


## 出力形式

### CLI出力（デフォルト - 完全分析）

包括的分析を含む人間可読な色付き出力：

```bash
$ diffai model_v1.safetensors model_v2.safetensors
anomaly_detection: type=none, severity=none, action="continue_training"
architecture_comparison: type1=feedforward, type2=feedforward, deployment_readiness=ready
convergence_analysis: status=converging, stability=0.92
gradient_analysis: flow_health=healthy, norm=0.021069
memory_analysis: delta=+0.0MB, efficiency=1.000000
quantization_analysis: compression=0.0%, speedup=1.8x, precision_loss=1.5%
regression_test: passed=true, degradation=-2.5%, severity=low
deployment_readiness: readiness=0.92, risk=low
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### JSON出力

```bash
diffai model1.safetensors model2.safetensors --output json
```

```json
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518, "shape": [64], "dtype": "f32"},
      {"mean": 0.0017, "std": 0.0647, "shape": [64], "dtype": "f32"}
    ]
  }
]
```

### YAML出力

```bash
diffai model1.safetensors model2.safetensors --output yaml
```

```yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
    shape: [64]
    dtype: f32
  - mean: 0.0017
    std: 0.0647
    shape: [64]
    dtype: f32
```

## 環境変数

```bash
# ログレベル設定
export DIFFAI_LOG_LEVEL="info"

# 最大メモリ使用量設定
export DIFFAI_MAX_MEMORY="1024"
```

## 実用例

### 実験比較

```bash
# 2つの実験結果を比較
diffai experiment_v1/ experiment_v2/ --recursive

# モデルチェックポイント比較（自動学習分析）
diffai checkpoints/epoch_10.safetensors checkpoints/epoch_20.safetensors
```

### CI/CD使用

```yaml
- name: モデル比較
  run: |
    diffai baseline/model.safetensors new/model.safetensors --output json > model_diff.json
    
- name: デプロイ準備チェック（分析に含まれる）
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors
```

### 科学データ分析

```bash
# NumPy実験結果比較（自動統計分析）
diffai baseline_results.npy new_results.npy

# MATLABシミュレーションデータ比較
diffai simulation_v1.mat simulation_v2.mat
```

## 対応ファイル形式

### MLモデル形式
- **Safetensors** (.safetensors) - HuggingFace標準形式
- **PyTorch** (.pt, .pth) - PyTorchモデルファイル

### 科学データ形式
- **NumPy** (.npy, .npz) - 統計分析付きNumPy配列
- **MATLAB** (.mat) - 複素数対応MATLAB行列

### 構造化データ形式
- **JSON** (.json), **YAML** (.yaml, .yml), **TOML** (.toml)
- **XML** (.xml), **INI** (.ini), **CSV** (.csv)

## 次のステップ

- [MLモデル比較](ml-model-comparison_ja.md) - 高度MLモデル分析
- [科学データ分析](scientific-data_ja.md) - NumPyとMATLABファイル比較
- [CLIリファレンス](../reference/cli-reference_ja.md) - 完全なコマンドリファレンス

