# CLIリファレンス

diffai v0.3.4の完全なコマンドラインリファレンス - 自動包括分析機能を持つAI/ML専用差分ツール。

## 概要

```
diffai <INPUT1> <INPUT2>
```

## 説明

diffaiは、モデル構造、テンソル統計、科学データの包括的な分析を自動的に提供するAI/MLワークフロー専用の差分ツールです。複雑なオプションを必要とせず、インテリジェントな自動分析でPyTorchモデル、Safetensorsファイル、NumPy配列、MATLABマトリックスを比較します。

**主な機能：**
- **自動分析**：デフォルトでML固有の包括的分析
- **ゼロ設定**：詳細な洞察に設定は不要
- **AI/ML重視**：モデル比較ワークフローに最適化

## 引数

### 必須引数

#### `<INPUT1>`
比較する最初の入力ファイルまたはディレクトリ。

- **型**：ファイルパスまたはディレクトリパス
- **形式**：PyTorch（.pt/.pth）、Safetensors（.safetensors）、NumPy（.npy/.npz）、MATLAB（.mat）
- **特別**：標準入力には`-`を使用

#### `<INPUT2>`
比較する2番目の入力ファイルまたはディレクトリ。

- **型**：ファイルパスまたはディレクトリパス
- **形式**：INPUT1と同じ
- **特別**：標準入力には`-`を使用

**注意**：AI/MLファイルはバイナリ形式のため、標準入力はサポートされません。ファイルパスのみを使用してください。

**例**：
```bash
# 基本ファイル比較
diffai model1.safetensors model2.safetensors
diffai data_v1.npy data_v2.npy
diffai experiment_v1.mat experiment_v2.mat
# 一般的な構造化データには diffx を使用：
# diffx config.json config_new.json

# ディレクトリ比較（自動再帰）
diffai dir1/ dir2/

# バイナリAI/MLファイルでは標準入力はサポートされません
# 一般データ比較には diffx を使用：
# cat config.json | diffx - config_new.json
# echo '{"old": "data"}
# {"new": "data"}' | diffx - -
```

## オプション

### 基本オプション

#### `-h, --help`
ヘルプ情報を表示。

#### `-V, --version`
バージョン情報を表示。

#### `--no-color`
スクリプトや自動化環境との互換性を向上させるためカラー出力を無効化。

- **例**：`diffai model1.safetensors model2.safetensors --no-color`
- **用途**：カラーフォーマットなしのプレーンテキスト出力

## 自動分析

### 包括的AI/ML分析

**diffaiは、オプションを必要とせずに11のML分析機能すべてを自動的に実行します：**

#### ✅ 完全実装機能（すべて現在利用可能）

**高優先度機能：**
1. **テンソル統計**：完全な統計分析（平均、標準偏差、最小/最大、形状、データ型）
2. **モデルアーキテクチャ**：レイヤー検出、パラメータ計数、構造変更
3. **重み変化**：設定可能しきい値による重要なパラメータ変更検出
4. **メモリ分析**：メモリ使用量分析と最適化推奨

**中優先度機能：**
5. **学習率**：オプティマイザ状態と訓練メタデータからの学習率検出
6. **収束分析**：モデル変更からの訓練収束パターン分析
7. **勾配分析**：パラメータ更新から推定された勾配フロー分析

**高度機能：**
8. **注意分析**：トランスフォーマー注意機構の分析とパターン
9. **アンサンブル分析**：マルチモデルアンサンブル構成と投票戦略分析
10. **量子化分析**：モデル量子化検出と精度分析

#### 形式認識自動機能選択

- **PyTorch（.pt/.pth）**：11機能すべてが完全に活動
- **Safetensors（.safetensors）**：10機能が活動（アンサンブル分析は制限）
- **NumPy（.npy/.npz）**：4つのコア機能が活動（テンソル統計、アーキテクチャ基本、重み、メモリ）
- **MATLAB（.mat）**：基本量子化サポート付き4つのコア機能が活動

**🎯 設定は不要** - 各形式に最適な分析が自動選択されます。

**例**：`diffai model1.pt model2.pt`を実行するだけで、適用可能な分析機能すべてを取得できます。

## 出力例

### CLI出力（デフォルト - 完全分析）

```bash
$ diffai model_v1.pt model_v2.pt
TensorStatsChanged: fc1.weight
  Old: mean=-0.0002, std=0.0514, shape=[128, 256], dtype=float32
  New: mean=-0.0001, std=0.0716, shape=[128, 256], dtype=float32

ModelArchitectureChanged: model
  Old: {layers: 12, parameters: 124439808, types: [conv, linear, norm]}
  New: {layers: 12, parameters: 124440064, types: [conv, linear, norm, attention]}

WeightSignificantChange: transformer.attention.query.weight
  Change Magnitude: 0.0234 (above threshold: 0.01)

MemoryAnalysis: memory_change
  Old: 487.2MB (tensors: 485.1MB, metadata: 2.1MB)
  New: memory_change: +12.5MB, breakdown: tensors: +12.3MB, metadata: +0.2MB

LearningRateChanged: optimizer.learning_rate
  Old: 0.001, New: 0.0005 (scheduler: step_decay, epoch: 10)

ConvergenceAnalysis: convergence_patterns
  Old: evaluating
  New: loss: improving (trend: decreasing), stability: gradient_norm: stable, epoch: 10 → 11

GradientAnalysis: gradient_magnitudes
  Old: norm: 0.018456, max: 0.145234, var: 0.000234
  New: total_norm: 0.021234 (+14.8%, increasing), max_gradient: 0.156789 (+8.0%)

AttentionAnalysis: attention_heads
  Old: heads: 8, dim: 64, patterns: 4
  New: num_heads: 8 → 12, head_dim: 64 → 48, patterns: +query, +value

QuantizationAnalysis: quantization_precision
  Old: 32bit float32, layers: 0, mixed: false
  New: bit_width: 32 → 8, data_type: float32 → int8, quantized_layers: 8 (+8)
```

### 包括的分析の利点

- **11のML分析機能すべて**が自動実行
- **形式認識機能選択** - 各ファイル型に最適な分析
- **設定不要** - デフォルトで最大の洞察
- **本番レディ分析** - 包括的なモデル評価

### 科学データ分析（自動）

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLABファイル比較（自動）

```bash
$ diffai simulation_v1.mat simulation_v2.mat
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### JSON出力

```bash
$ diffai model_v1.safetensors model_v2.safetensors --output json
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
$ diffai model_v1.safetensors model_v2.safetensors --output yaml
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

## 終了コード

- **0**：成功 - 差異が見つかったか、差異なし
- **1**：エラー - 無効な引数またはファイルアクセス問題
- **2**：致命的エラー - 内部処理失敗

## 環境変数

diffaiは設定に環境変数を使用しません。すべての設定はコマンドラインオプションで制御されます。

## パフォーマンスの考慮事項

- **大きなファイル**：diffaiはGB+ファイルにストリーミング処理を使用
- **メモリ使用量**：大きなファイルの自動メモリ最適化
- **並列処理**：複数ファイル比較の自動並列化
- **キャッシング**：繰り返し比較のインテリジェントキャッシング

## トラブルシューティング

### よくある問題

1. **「Binary files differ」メッセージ**：ファイル型を指定するために`--format`を使用
2. **メモリ不足**：大きなファイルのメモリ最適化は自動
3. **処理が遅い**：大きなモデルの分析は自動最適化
4. **依存関係の欠如**：Rustツールチェーンが適切にインストールされていることを確認

### デバッグモード

`--verbose`オプションでデバッグ出力を有効化：
```bash
diffai model1.safetensors model2.safetensors --verbose
```

## 関連項目

- [基本使用ガイド](../user-guide/basic-usage_ja.md)
- [MLモデル比較ガイド](../user-guide/ml-model-comparison_ja.md)
- [科学データ分析ガイド](../user-guide/scientific-data_ja.md)
- [出力形式リファレンス](output-formats_ja.md)
- [サポート形式リファレンス](formats_ja.md)