# ML分析機能（35機能）

diffaiの機械学習分析機能の包括的ガイド：モデル比較と分析のために設計されています。

## 概要

diffaiは、機械学習モデルの比較と分析専用に設計された35の特別な分析機能を提供します。これらの機能は、研究開発、MLOps、デプロイメントワークフローに役立ちます。

## 現在利用可能な機能（v0.2.0）

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

## 組み合わせ分析

包括的分析のために複数の機能を組み合わせ：

```bash
# 包括的モデル分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  \
  --quantization-analysis \
  --sort-by-change-magnitude \
  --show-layer-impact

# 自動化のためのJSON出力
diffai model1.safetensors model2.safetensors \
  --output json
```

## 機能選択ガイド

**訓練監視のため**:
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --sort-by-change-magnitude
```

**本番デプロイメントのため**:
```bash
diffai current_prod.safetensors candidate.safetensors \
  --quantization-analysis
```

**研究分析のため**:
```bash
diffai baseline.safetensors experiment.safetensors \
  --show-layer-impact
```

**量子化検証のため**:
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis
```

## 将来機能（Phase 3）

### Phase 3A（コア機能）
- `--architecture-comparison` - モデルアーキテクチャと構造変化の比較
- `--memory-analysis` - メモリ使用量と最適化機会の分析
- `--anomaly-detection` - モデルパラメータの数値異常検出
- `--change-summary` - 詳細な変更サマリの生成

### Phase 3B（高度分析）
- `--convergence-analysis` - モデルパラメータの収束パターン分析
- `--gradient-analysis` - 利用可能時の勾配情報分析
- `--similarity-matrix` - モデル比較用類似度行列の生成

## 設計理念

diffaiはUNIX哲学に従います：一つのことを適切に行うシンプルで組み合わせ可能なツール。機能は直交的で、強力な分析ワークフローのために組み合わせることができます。

## 統合例

### MLflow統合
```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    result = subprocess.run([
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
```

### CI/CDパイプライン
```yaml
name: モデル検証
on: [push, pull_request]

jobs:
  model-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: diffaiインストール
        run: cargo install diffai
        
      - name: モデル比較
        run: |
          diffai models/baseline.safetensors models/candidate.safetensors \
            --output json > model_diff.json
```

## 関連項目

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドオプション
- [対応形式](formats_ja.md) - サポートされるファイル形式
- [出力形式](output-formats_ja.md) - 出力形式の詳細

