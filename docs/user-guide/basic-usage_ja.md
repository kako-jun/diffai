# 基本的な使い方

diffai - AI/ML特化差分ツールの基本操作を学習

## クイックスタート

### 基本的なファイル比較

```bash
# 2つのモデルファイルを比較
diffai model1.safetensors model2.safetensors

# 詳細なテンソル統計を表示
diffai model1.safetensors model2.safetensors --stats

# JSON形式で出力
diffai model1.safetensors model2.safetensors --output json
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
# PyTorchモデルファイルを比較
diffai model1.pt model2.pt --stats

# 学習進捗分析付き
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt --learning-progress

# アーキテクチャ比較付き
diffai baseline_model.pt improved_model.pt --architecture-comparison
```

**出力例**:
```
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
  learning_progress: trend=improving, magnitude=0.0234, speed=0.0156
```

### Safetensorsファイル比較

```bash
# 統計付きSafetensorsファイル比較
diffai model1.safetensors model2.safetensors --stats

# デプロイ準備分析付き
diffai baseline.safetensors candidate.safetensors --deployment-readiness
```

### 科学データ比較

```bash
# NumPy配列比較
diffai data_v1.npy data_v2.npy --stats

# MATLABファイル比較
diffai simulation_v1.mat simulation_v2.mat --stats

# 圧縮NumPyアーカイブ比較
diffai dataset_v1.npz dataset_v2.npz --stats
```

## コマンドオプション

### 基本オプション

| オプション | 説明 | 例 |
|-----------|-------------|---------|
| `-f, --format` | 入力ファイル形式を指定 | `--format safetensors` |
| `-o, --output` | 出力形式を選択 | `--output json` |
| `-r, --recursive` | ディレクトリを再帰的に比較 | `--recursive` |
| `--stats` | 詳細統計を表示 | `--stats` |

### 高度オプション

| オプション | 説明 | 例 |
|-----------|-------------|---------|
| `--path` | 特定のパスでフィルタ | `--path "config.model"` |
| `--ignore-keys-regex` | 正規表現に一致するキーを無視 | `--ignore-keys-regex "^id$"` |
| `--epsilon` | 浮動小数点比較の許容誤差 | `--epsilon 0.001` |
| `--array-id-key` | 配列要素識別 | `--array-id-key "id"` |
| `--sort-by-change-magnitude` | 変更量でソート | `--sort-by-change-magnitude` |

### ML分析オプション

| オプション | 説明 | 例 |
|-----------|-------------|---------|
| `--learning-progress` | 学習進捗追跡 | `--learning-progress` |
| `--convergence-analysis` | 収束分析 | `--convergence-analysis` |
| `--architecture-comparison` | アーキテクチャ比較 | `--architecture-comparison` |
| `--deployment-readiness` | デプロイ準備評価 | `--deployment-readiness` |
| `--quantization-analysis` | 量子化分析 | `--quantization-analysis` |

## 出力形式

### CLI出力（デフォルト）

記号付きの人間可読な色付き出力：

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
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

## 設定

### グローバル設定

`~/.config/diffx/config.toml`:

```toml
[diffai]
default_output = "cli"
default_format = "auto"
epsilon = 0.001
sort_by_magnitude = false

[ml_analysis]
enable_all = false
learning_progress = true
convergence_analysis = true
memory_analysis = true
```

### 環境変数

```bash
# 設定ファイルパス設定
export DIFFAI_CONFIG="/path/to/config.toml"

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

# 学習分析付きモデルチェックポイント比較
diffai checkpoints/epoch_10.safetensors checkpoints/epoch_20.safetensors --learning-progress
```

### CI/CD使用

```yaml
- name: モデル比較
  run: |
    diffai baseline/model.safetensors new/model.safetensors --output json > model_diff.json
    
- name: デプロイ準備チェック
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors --deployment-readiness
```

### 科学データ分析

```bash
# NumPy実験結果比較
diffai baseline_results.npy new_results.npy --stats

# MATLABシミュレーションデータ比較
diffai simulation_v1.mat simulation_v2.mat --stats
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

