# MLモデル比較ガイド

PyTorchやSafetensorsファイルなど、機械学習モデルを比較するためのdiffaiの特別な機能について説明します。

## 概要

diffaiはAI/MLモデル形式をネイティブサポートし、モデルをバイナリファイルとしてではなくテンソルレベルで比較できます。これにより、学習、ファインチューニング、量子化、デプロイメント中のモデル変化の意味のある分析が可能になります。

## サポートされているML形式

### PyTorchモデル
- **`.pt` ファイル**: PyTorchモデルファイル（Candle統合付きpickle形式）
- **`.pth` ファイル**: PyTorchモデルファイル（代替拡張子）

### Safetensorsモデル
- **`.safetensors` ファイル**: HuggingFace Safetensors形式（推奨）

### 将来のサポート（フェーズ3）
- **`.onnx` ファイル**: ONNX形式
- **`.h5` ファイル**: Keras/TensorFlow HDF5形式
- **`.pb` ファイル**: TensorFlow Protocol Buffer形式

## diffaiが分析する内容

### テンソル統計
モデル内の各テンソルについて、diffaiは以下を計算・比較します：

- **平均値**: 全パラメータの平均値
- **標準偏差**: パラメータ分散の指標
- **最小値**: 最小パラメータ値
- **最大値**: 最大パラメータ値
- **形状**: テンソルの次元と大きさ
- **データ型**: 精度レベル（f16、f32、f64など）

### 構造的変化
- **レイヤー追加/削除**: アーキテクチャの変更
- **形状変更**: テンソル次元の変更
- **名前変更**: パラメータの名前変更

### 学習進捗分析
- **パラメータドリフト**: 時間経過によるパラメータ変化
- **収束指標**: 学習の安定性指標
- **勾配流**: 勾配の健全性評価

## 基本的な使用法

### シンプルなモデル比較

```bash
# 2つのSafetensorsファイルを比較（包括的分析が自動実行）
diffai model1.safetensors model2.safetensors
```

**出力例（完全分析）**:
```
anomaldy_detection: type=none, severity=none, action="continue_training"
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

### PyTorchモデル比較

```bash
# PyTorchモデルファイルを比較（完全分析が自動実行）
diffai model1.pt model2.pt

# 学習チェックポイントの比較
diffai checkpoint_epoch_1.pt checkpoint_epoch_10.pt
```

### 出力形式

#### JSON出力
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

#### YAML出力
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

## 高度な分析機能

### 1. テンソル統計分析（自動実行）

学習進捗の詳細監視：

```bash
# テンソル統計は自動的に含まれる
diffai checkpoint_1.safetensors checkpoint_2.safetensors
```

### 2. 量子化分析

量子化効果の分析：

```bash
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis
```

**出力**:
```
quantization_analysis: compression=0.25, speedup=1.8x, precision_loss=minimal
```

### 3. 変化量ソート

最大の変化を優先表示：

```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

### 4. レイヤー影響分析

レイヤー別変化の分析：

```bash
diffai baseline.safetensors modified.safetensors --show-layer-impact
```

### 5. アーキテクチャ比較

モデル構造の分析：

```bash
diffai model1.safetensors model2.safetensors --architecture-comparison
```

**出力**:
```
architecture_comparison: transformer->transformer, complexity=similar_complexity, migration=easy
```

### 6. メモリ分析

メモリ使用量の分析：

```bash
diffai model1.safetensors model2.safetensors --memory-analysis
```

**出力**:
```
memory_analysis: delta=+12.5MB, peak=156.3MB, efficiency=0.85, recommendation=optimal
```

### 7. 異常検出

数値異常の検出：

```bash
diffai model1.safetensors model2.safetensors --anomaly-detection
```

**出力**:
```
anomaldy_detection: type=none, severity=none, affected_layers=[], confidence=0.95
```

## 実践的なワークフロー

### 学習監視

```bash
# 各エポック後のモデル変化を監視
diffai epoch_10.safetensors epoch_11.safetensors

# 最大の変化に焦点を当てる
diffai epoch_10.safetensors epoch_11.safetensors --sort-by-change-magnitude
```

### ファインチューニング分析

```bash
# ベースモデルとファインチューニング済みモデルを比較
diffai base_model.safetensors finetuned_model.safetensors --show-layer-impact
```

### 量子化検証

```bash
# 量子化前後の品質評価
diffai original.safetensors quantized.safetensors --quantization-analysis
```

### デプロイメント検証

```bash
# 本番環境への展開前検証
diffai current_prod.safetensors candidate.safetensors
```

## 学習中の使用

### 過学習検出

```bash
# バリデーション損失が上昇し始めた時点でのモデル比較
diffai best_val_model.safetensors current_model.safetensors
```

### 収束分析

```bash
# 連続するエポック間の変化を分析
diffai epoch_95.safetensors epoch_100.safetensors --convergence-analysis
```

### 勾配健全性チェック

```bash
# 勾配爆発/消失の検出
diffai prev_checkpoint.safetensors current_checkpoint.safetensors --gradient-analysis
```

## MLOpsとの統合

### CI/CDパイプライン

```yaml
- name: Model regression test
  run: |
    diffai baseline/model.safetensors candidate/model.safetensors --output json > model_diff.json
    
- name: Quantization validation
  run: |
    diffai fp32/model.safetensors quantized/model.safetensors --quantization-analysis
```

### 実験管理

```bash
# 実験結果の比較
diffai experiments/baseline.safetensors experiments/variant_a.safetensors

# 複数の実験候補の比較
for model in experiments/*.safetensors; do
  diffai baseline.safetensors "$model" --output json >> comparison_results.jsonl
done
```

### A/Bテスト

```bash
# 本番モデルと候補モデルの比較
diffai production_v1.safetensors candidate_v2.safetensors
```

## トラブルシューティング

### よくある問題

#### テンソル形状の不一致
```bash
# 構造変更の詳細分析
diffai old_model.safetensors new_model.safetensors --architecture-comparison
```

#### 数値不安定性
```bash
# NaN/Inf値の検出
diffai stable_model.safetensors unstable_model.safetensors --anomaly-detection
```

#### メモリ使用量の問題
```bash
# メモリ効率の分析
diffai small_model.safetensors large_model.safetensors --memory-analysis
```

## 最適化ヒント

### パフォーマンス
- 大きなモデルの場合、`--format safetensors`を明示的に指定
- JSON出力は自動化処理に最適
- レイヤー影響分析は詳細だが時間がかかる場合がある

### 精度
- 浮動小数点比較には`--epsilon`を調整
- 量子化モデルには適切な許容誤差を設定

## 関連項目

- [基本使用ガイド](basic-usage_ja.md) - diffaiの基本操作
- [科学データ分析](scientific-data_ja.md) - NumPyとMATLABファイル比較
- [CLIリファレンス](../reference/cli-reference_ja.md) - 完全なコマンドリファレンス
- [ML分析機能](../reference/ml-analysis_ja.md) - 詳細な分析機能説明