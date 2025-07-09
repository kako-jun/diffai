# MLモデル比較ガイド

このガイドでは、PyTorch や Safetensors ファイルを含む機械学習モデルの比較における diffai の特化機能について説明します。

## 概要

diffai は AI/ML モデル形式をネイティブサポートし、単なるバイナリファイルとしてではなく、テンソルレベルでモデルを比較できます。これにより、学習、ファインチューニング、量子化、デプロイメント中のモデル変更を意味のある方法で分析できます。

## サポートされているML形式

### PyTorchモデル
- **`.pt` ファイル**: PyTorch モデルファイル（Candle統合によるpickle形式）
- **`.pth` ファイル**: PyTorch モデルファイル（代替拡張子）

### Safetensors モデル
- **`.safetensors` ファイル**: HuggingFace Safetensors 形式（推奨）

### 将来サポート予定（Phase 3）
- **`.onnx` ファイル**: ONNX 形式
- **`.h5` ファイル**: Keras/TensorFlow HDF5 形式
- **`.pb` ファイル**: TensorFlow Protocol Buffer 形式

## diffai が分析する内容

### テンソル統計
モデル内の各テンソルについて、diffai は以下を計算・比較します：

- **平均値**: すべてのパラメータの平均値
- **標準偏差**: パラメータの分散の指標
- **最小値**: 最も小さいパラメータ値
- **最大値**: 最も大きいパラメータ値
- **形状**: テンソルの次元
- **データ型**: パラメータの精度（f32、f64など）
- **総パラメータ数**: テンソル内のパラメータ数

### モデルアーキテクチャ
- **パラメータ数の変更**: モデル全体のパラメータ数
- **レイヤーの追加/削除**: 新規または削除されたレイヤー
- **形状の変更**: 変更されたレイヤーの次元

## 基本的なモデル比較

### シンプルな比較

```bash
# 2つのPyTorchモデルを比較
diffai model1.pt model2.pt --stats

# Safetensorsモデルを比較（推奨）
diffai model1.safetensors model2.safetensors --stats

# 自動形式検出
diffai pretrained.safetensors finetuned.safetensors --stats
```

### 出力例

```bash
$ diffai model_v1.safetensors model_v2.safetensors --stats
  ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
  ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
  ~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

### 出力記号

| 記号 | 意味 | 説明 |
|------|------|------|
| `~` | 統計値が変更 | テンソル値が変更されたが形状は同じ |
| `+` | 追加 | 新しいテンソル/レイヤーが追加された |
| `-` | 削除 | テンソル/レイヤーが削除された |

## 高度なオプション

### イプシロン許容値

イプシロンを使用して小さな浮動小数点の差を無視します：

```bash
# 1e-6より小さい差を無視
diffai model1.safetensors model2.safetensors --epsilon 1e-6

# 量子化分析に有用
diffai fp32_model.safetensors int8_model.safetensors --epsilon 0.1
```

### 出力形式

```bash
# 自動化用のJSON出力
diffai model1.pt model2.pt --output json

# 可読性のためのYAML出力
diffai model1.pt model2.pt --output yaml

# 処理のためファイルにパイプ
diffai model1.pt model2.pt --output json > changes.json
```

### 結果のフィルタリング

```bash
# 特定のレイヤーに注目
diffai model1.safetensors model2.safetensors --path "classifier"

# タイムスタンプやメタデータを無視
diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(timestamp|_metadata)"
```

## 一般的な使用例

### 1. ファインチューニング分析

事前学習済みモデルとファインチューニング版を比較：

```bash
diffai pretrained_bert.safetensors finetuned_bert.safetensors --stats

# 期待される出力: アテンション層の統計的変化
# ~ bert.encoder.layer.11.attention.self.query.weight: mean=-0.0001→0.0023
# ~ classifier.weight: mean=0.0000→0.0145, std=0.0200→0.0890
```

**分析**:
- 初期レイヤーでの小さな変化（特徴抽出は似たまま）
- 最終レイヤーでの大きな変化（タスク特化の適応）

### 2. 量子化影響評価

FP32と量子化モデルを比較：

```bash
diffai model_fp32.safetensors model_int8.safetensors --epsilon 0.1

# 期待される出力: 制御された精度損失
# ~ conv1.weight: mean=0.0045→0.0043, std=0.2341→0.2298
# 違いが見つかりません（イプシロン許容値内）
```

**分析**:
- 小さな統計的変化は量子化の成功を示す
- 大きな変化は品質損失を示唆する可能性

### 3. 訓練進捗の追跡

訓練中のチェックポイントを比較：

```bash
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt --stats

# 期待される出力: 収束パターン
# ~ layers.0.weight: mean=-0.0012→0.0034, std=1.2341→0.8907
# ~ layers.1.bias: mean=0.1234→0.0567, std=0.4567→0.3210
```

**分析**:
- 標準偏差の減少は収束を示唆
- 平均値の変化は学習方向を示す

### 4. アーキテクチャ比較

異なるモデルアーキテクチャを比較：

```bash
diffai resnet50.safetensors efficientnet_b0.safetensors --stats

# 期待される出力: 構造的違い
# ~ features.conv1.weight: shape=[64, 3, 7, 7] -> [32, 3, 3, 3]
# + features.mbconv.expand_conv.weight: shape=[96, 32, 1, 1]
# - features.layer4.2.downsample.0.weight: shape=[2048, 1024, 1, 1]
```

**分析**:
- 形状変化は異なるレイヤーサイズを示す
- 追加/削除されたテンソルはアーキテクチャの革新を示す

## パフォーマンス最適化

### メモリ考慮事項

大きなモデル（>1GB）の場合：

```bash
# ストリーミングモード使用（将来機能）
diffai --stream huge_model1.safetensors huge_model2.safetensors

# 特定部分に分析を集中
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# より高速な比較のためより大きなイプシロンを使用
diffai model1.safetensors model2.safetensors --epsilon 1e-3
```

### 速度最適化

```bash
# 並列処理（将来機能）
diffai --threads 8 model1.safetensors model2.safetensors

# 形状のみの分析で統計計算をスキップ
diffai --shape-only model1.safetensors model2.safetensors
```

## 統合例

### MLflow統合

```python
import subprocess
import json
import mlflow

def log_model_diff(model1_path, model2_path):
    # diffai比較を実行
    result = subprocess.run([
        'diffai', model1_path, model2_path, '--output', 'json'
    ], capture_output=True, text=True)
    
    diff_data = json.loads(result.stdout)
    
    # MLflowにログ
    with mlflow.start_run():
        mlflow.log_dict(diff_data, "model_comparison.json")
        mlflow.log_metric("total_changes", len(diff_data))
        
        # 変更タイプをカウント
        stats_changes = len([d for d in diff_data if 'TensorStatsChanged' in d])
        shape_changes = len([d for d in diff_data if 'TensorShapeChanged' in d])
        
        mlflow.log_metric("stats_changes", stats_changes)
        mlflow.log_metric("shape_changes", shape_changes)
```

### CI/CDパイプライン

```yaml
name: Model Validation
on: [push, pull_request]

jobs:
  model-diff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install diffai
        run: cargo install diffai
        
      - name: Compare models
        run: |
          diffai models/baseline.safetensors models/candidate.safetensors \
            --output json > model_diff.json
            
      - name: Analyze changes
        run: |
          # 重要なレイヤーが変更された場合は失敗
          if jq -e '.[] | select(.TensorShapeChanged and (.TensorShapeChanged[0] | contains("classifier")))' model_diff.json; then
            echo "CRITICAL: Critical layer shape changes detected"
            exit 1
          fi
          
          # 多数のパラメータが変更された場合は警告
          changes=$(jq length model_diff.json)
          if [ "$changes" -gt 10 ]; then
            echo "WARNING: Many parameter changes detected: $changes"
          fi
```

### Git Pre-commitフック

```bash
#!/bin/bash
# .git/hooks/pre-commit

model_files=$(git diff --cached --name-only | grep -E '\.(pt|pth|safetensors)$')

for file in $model_files; do
    if [ -f "$file" ]; then
        echo "Analyzing model changes in $file"
        
        # 前のバージョンと比較
        git show HEAD:"$file" > /tmp/old_model
        
        diffai /tmp/old_model "$file" --output json > /tmp/model_diff.json
        
        # 重要な変更をチェック
        shape_changes=$(jq '[.[] | select(.TensorShapeChanged)] | length' /tmp/model_diff.json)
        
        if [ "$shape_changes" -gt 0 ]; then
            echo "WARNING: Architecture changes detected in $file"
            diffai /tmp/old_model "$file"
            
            read -p "Continue with commit? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        rm -f /tmp/old_model /tmp/model_diff.json
    fi
done
```

## トラブルシューティング

### 一般的な問題

#### 1. "Failed to parse" エラー

```bash
# ファイル形式をチェック
file model.safetensors

# ファイルの整合性を確認
diffai --check model.safetensors

# 明示的な形式指定で試行
diffai --format safetensors model1.safetensors model2.safetensors
```

#### 2. 大きなモデルでのメモリ問題

```bash
# 精度を下げるためより大きなイプシロンを使用
diffai --epsilon 1e-3 large1.safetensors large2.safetensors

# 特定レイヤーに注目
diffai --path "tensor.classifier" large1.safetensors large2.safetensors
```

#### 3. バイナリファイルエラー

```bash
# ファイルが実際のモデルファイルで破損していないことを確認
ls -la model*.safetensors

# ファイルが圧縮されているかチェック
file model.safetensors

# 圧縮されている場合は展開を試行
gunzip model.safetensors.gz
```

## ベストプラクティス

### 1. イプシロン値の選択

| 使用例 | 推奨イプシロン | 理由 |
|--------|---------------|------|
| 厳密な比較 | イプシロンなし | すべての変更を検出 |
| 訓練進捗 | 1e-6 | 数値ノイズを無視 |
| 量子化分析 | 0.01-0.1 | 精度損失を考慮 |
| アーキテクチャチェック | 1e-3 | 構造的変化に注目 |

### 2. 出力形式の選択

- **CLI**: 人間によるレビューとデバッグ
- **JSON**: 自動化とスクリプト
- **YAML**: 設定ファイルとドキュメント

### 3. パフォーマンスのヒント

- `--path` を使用して関連レイヤーに分析を集中
- ノイズを避けるため適切なイプシロン値を設定
- 比較戦略を選択する際にモデルサイズを考慮

## 高度なML分析機能

diffai は包括的なモデル評価のために設計された28の高度な機械学習分析機能を提供します：

### 1. 学習進捗分析（`--learning-progress`）

モデルチェックポイント間の訓練進捗を分析：

```bash
# 訓練チェックポイントを比較
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress

# 出力例:
# + learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

**分析情報:**
- **trend**: `improving`、`degrading`、または `stable`
- **magnitude**: 変化の量（0.0-1.0）
- **speed**: 収束速度（0.0-1.0）

**使用例:**
- 訓練進捗の監視
- 学習停滞の検出
- 訓練スケジュールの最適化

### 2. 収束分析（`--convergence-analysis`）

モデルの安定性と収束状態を評価：

```bash
# チェックポイント間の収束を分析
diffai model_before.safetensors model_after.safetensors --convergence-analysis

# 出力例:
# + convergence_analysis: status=stable, stability=0.0234, action="Continue training"
```

**分析情報:**
- **status**: `converged`、`diverging`、`oscillating`、`stable`
- **stability**: パラメータ変化の分散（低い = より安定）
- **action**: 推奨される次のステップ

**使用例:**
- 訓練を停止するタイミングの決定
- 訓練不安定性の検出
- ハイパーパラメータの最適化

### 3. 異常検知（`--anomaly-detection`）

モデル重みの異常なパターンを検出：

```bash
# 訓練異常を検出
diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection

# 出力例:
# anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

**検出される異常:**
- **勾配爆発**: 極端に大きな重み値
- **勾配消失**: ゼロに近い勾配
- **重み分布シフト**: 異常な統計パターン
- **NaN/Inf値**: 無効な数値

**使用例:**
- 訓練問題のデバッグ
- モデルの健全性検証
- 破損したモデルのデプロイ防止

### 4. メモリ分析（`--memory-analysis`）

メモリ使用量とモデル効率を分析：

```bash
# モデルのメモリフットプリントを比較
diffai small_model.safetensors large_model.safetensors --memory-analysis

# 出力例:
# memory_analysis: delta=+2.7MB, gpu_est=4.5MB, efficiency=0.25
```

**分析情報:**
- **delta**: モデル間のメモリ差
- **gpu_est**: 推定GPU メモリ要件
- **efficiency**: MB あたりのパラメータ比

**使用例:**
- デプロイ制約の最適化
- アーキテクチャ効率の比較
- ハードウェア要件の計画

### 5. アーキテクチャ比較（`--architecture-comparison`）

モデルアーキテクチャと構造を比較：

```bash
# 異なるアーキテクチャを比較
diffai resnet.safetensors transformer.safetensors --architecture-comparison

# 出力例:
# architecture_comparison: type1=cnn, type2=transformer, differences=15
```

**分析情報:**
- **アーキテクチャタイプ**: CNN、RNN、Transformer、MLP など
- **レイヤー差**: 追加、削除、または変更されたレイヤー
- **パラメータ分布**: パラメータの配分方法

**使用例:**
- アーキテクチャ変更の評価
- モデルファミリーの比較
- 設計決定の検証

### 6. 複数機能分析

包括的分析のため複数機能を組み合わせ：

```bash
# 包括的な訓練分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection \
  --memory-analysis \
  --stats

# 出力例:
# + learning_progress: trend=improving, magnitude=0.0432, speed=0.75
# + convergence_analysis: status=stable, stability=0.0156
# memory_analysis: delta=+0.1MB, efficiency=0.89
# テンソル統計と詳細分析...
```

### 7. 本番デプロイ機能

本番環境に不可欠な機能：

```bash
# 本番準備チェック
diffai production.safetensors candidate.safetensors \
  --anomaly-detection \
  --memory-analysis \
  --deployment-readiness

# 回帰テスト
diffai baseline.safetensors new_version.safetensors \
  --regression-test \
  --alert-on-degradation
```

### 8. 研究開発機能

研究ワークフロー向けの高度分析：

```bash
# ハイパーパラメータ影響分析
diffai model_lr_001.safetensors model_lr_0001.safetensors \
  --hyperparameter-impact \
  --learning-rate-analysis

# アーキテクチャ効率分析
diffai efficient_model.safetensors baseline_model.safetensors \
  --param-efficiency-analysis \
  --architecture-comparison
```

## 機能選択ガイド

**訓練監視用:**
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress --convergence-analysis --anomaly-detection
```

**本番デプロイ用:**
```bash
diffai current_prod.safetensors candidate.safetensors \
  --anomaly-detection --memory-analysis --deployment-readiness
```

**研究分析用:**
```bash
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison --hyperparameter-impact --stats
```

**量子化検証用:**
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --memory-analysis --performance-impact-estimate
```

## 全28の高度機能

### 学習・収束分析（4機能）
- `--learning-progress` - チェックポイント間の学習進捗を追跡
- `--convergence-analysis` - 収束の安定性とパターンを分析
- `--anomaly-detection` - 訓練異常を検出（勾配爆発、消失勾配）
- `--gradient-analysis` - 勾配の特性と流れを分析

### アーキテクチャ・パフォーマンス分析（4機能）
- `--architecture-comparison` - モデルアーキテクチャと構造変化を比較
- `--param-efficiency-analysis` - モデル間のパラメータ効率を分析
- `--memory-analysis` - メモリ使用量と最適化機会を分析
- `--inference-speed-estimate` - 推論速度とパフォーマンス特性を推定

### MLOps・デプロイ支援（7機能）
- `--deployment-readiness` - デプロイ準備と互換性を評価
- `--regression-test` - 自動回帰テストを実行
- `--risk-assessment` - デプロイリスクと安定性を評価
- `--hyperparameter-impact` - モデル変更へのハイパーパラメータ影響を分析
- `--learning-rate-analysis` - 学習率の効果と最適化を分析
- `--alert-on-degradation` - 閾値を超えた性能劣化でアラート
- `--performance-impact-estimate` - 変更の性能影響を推定

### 実験・文書化支援（4機能）
- `--generate-report` - 包括的な分析レポートを生成
- `--markdown-output` - ドキュメント用のMarkdown形式で結果を出力
- `--include-charts` - 出力にチャートと視覚化を含める
- `--review-friendly` - 人間のレビュー担当者向けの出力を生成

### 高度分析機能（6機能）
- `--embedding-analysis` - 埋め込み層の変化と意味的ドリフトを分析
- `--similarity-matrix` - モデル比較用の類似度行列を生成
- `--clustering-change` - モデル表現のクラスタリング変化を分析
- `--attention-analysis` - アテンション機構パターンを分析（Transformerモデル）
- `--head-importance` - アテンションヘッドの重要度と特化を分析
- `--attention-pattern-diff` - モデル間のアテンションパターンを比較

### 追加分析機能（3機能）
- `--quantization-analysis` - 量子化効果と効率を分析
- `--sort-by-change-magnitude` - 優先順位付けのため変化の大きさでソート
- `--change-summary` - 詳細な変更要約を生成

## 次のステップ

- [基本的な使用法](basic-usage_ja.md) - 基本的な操作を学習
- [科学データ分析](scientific-data_ja.md) - NumPyとMATLABファイルの比較
- [CLIリファレンス](../reference/cli-reference_ja.md) - 完全なコマンドリファレンス