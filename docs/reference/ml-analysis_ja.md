# ML分析機能（35機能）

diffaiの機械学習分析機能の包括的ガイド：モデル比較と分析のために設計されています。

## 概要

diffaiは、機械学習モデルの比較と分析専用に設計された35の特別な分析機能を提供します。これらの機能は、研究開発、MLOps、デプロイメントワークフローに役立ちます。

## 自動包括分析（v0.3.4+）

### オールインワンML分析
diffaiはPyTorchとSafetensorsファイルに対して自動的に包括的分析を提供します。フラグは不要で、30以上の分析機能がデフォルトで実行されます。

### 1. テンソル統計分析
モデル比較のための詳細なテンソル統計を提供します。

**使用法**:
```bash
diffai model1.safetensors model2.safetensors
```

**出力**:
```
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

**分析フィールド**:
- **mean**: パラメータの平均値
- **std**: パラメータの標準偏差
- **min/max**: パラメータ値の範囲
- **shape**: テンソルの次元
- **dtype**: データ型の精度

**用途**:
- 訓練中のパラメータ変化を監視
- モデル重みの統計的変化を検出
- モデルの一貫性を検証

### 2. `--quantization-analysis` 量子化分析
量子化効果と効率を分析します。

**使用法**:
```bash
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis
```

**出力**:
```
quantization_analysis: compression=0.25, precision_loss=minimal
```

**分析フィールド**:
- **compression**: モデルサイズ削減比率
- **precision_loss**: 精度への影響評価
- **efficiency**: 性能と品質のトレードオフ

**用途**:
- 量子化品質の検証
- デプロイメントサイズの最適化
- 圧縮技術の比較

### 3. `--sort-by-change-magnitude` 変化量ソート
優先順位付けのために差分を変化量でソートします。

**使用法**:
```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

**出力**: 結果は最大の変化を最初に表示してソートされます

**用途**:
- 最も重要な変化に焦点を当てる
- デバッグ作業の優先順位付け
- 重要なパラメータ変化の特定

### 4. `--show-layer-impact` レイヤー影響分析
変化のレイヤー別影響を分析します。

**使用法**:
```bash
diffai baseline.safetensors modified.safetensors --show-layer-impact
```

**出力**: レイヤー別変化分析

**用途**:
- どのレイヤーが最も変化したかを理解
- ファインチューニング戦略のガイド
- アーキテクチャ修正の分析

# 包括的モデル分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors

# 自動化のためのJSON出力
diffai model1.safetensors model2.safetensors --output json

## 機能選択ガイド

### 5. `--architecture-comparison` アーキテクチャ比較
アーキテクチャの違いを分析します。

### 6. `--memory-analysis` メモリ分析
メモリ使用量を分析します。

### 7. `--anomaly-detection` 異常検出
異常なパターンを検出します。

### 8. `--change-summary` 変更サマリー
詳細な変更要約を生成します。

### 9. `--convergence-analysis` 収束分析
収束パターンを分析します。

### 10. `--gradient-analysis` 勾配分析
勾配情報を分析します。

### 11. `--similarity-matrix` 類似性マトリクス
類似性マトリクスを生成します。

## フェーズ3機能（現在利用可能）

上記の7つの新機能（5-11）は、現在完全に実装され利用可能なフェーズ3機能です。

## 設計哲学

diffaiはUNIX哲学に従います：1つのことを上手に行うシンプルで構成可能なツールです。

## 統合例

### MLflow統合
MLflowとの統合例を示します。

### CI/CDパイプライン
CI/CDパイプラインでの使用例を示します。

## 関連項目

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドリファレンス
- [基本使用ガイド](../user-guide/basic-usage_ja.md) - diffaiの始め方
- [MLモデル比較ガイド](../user-guide/ml-model-comparison_ja.md) - 高度なモデル比較テクニック

