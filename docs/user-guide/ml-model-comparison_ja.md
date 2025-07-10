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

diffai は現在以下の機械学習分析機能を提供します（v0.2.0）：

### 現在利用可能な機能（v0.2.0）

diffai は現在以下の実装済み分析機能を提供します：

### 1. 統計分析（`--stats`）

モデル比較のための詳細なテンソル統計を提供：

```bash
# 詳細統計付きで訓練チェックポイントを比較
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --stats

# 出力例:
# ~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
# ~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
```

**分析情報:**
- **mean**: パラメータの平均値
- **std**: パラメータの標準偏差
- **min/max**: パラメータ値の範囲
- **shape**: テンソルの次元
- **dtype**: データ型の精度

**使用例:**
- 訓練中のパラメータ変化の監視
- モデル重みの統計的変化の検出
- モデル一貫性の検証

### 2. 量子化分析（`--quantization-analysis`）

量子化効果と効率を分析：

```bash
# 量子化モデルと全精度モデルを比較
diffai fp32_model.safetensors quantized_model.safetensors --quantization-analysis

# 出力例:
# quantization_analysis: compression=0.25, precision_loss=minimal
```

**分析情報:**
- **compression**: モデルサイズ削減比率
- **precision_loss**: 精度への影響評価
- **efficiency**: パフォーマンス対品質のトレードオフ

**使用例:**
- 量子化品質の検証
- デプロイサイズの最適化
- 圧縮技術の比較

### 3. 変更量ソート（`--sort-by-change-magnitude`）

優先順位付けのため差異を変更量でソート：

```bash
# 重要度順で変更をソート
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude --stats

# 出力は最大の変化から順に表示
```

**使用例:**
- 最も重要な変更に焦点を当てる
- デバッグ作業の優先順位付け
- 重要なパラメータ変化の特定

### 4. レイヤー影響分析（`--show-layer-impact`）

レイヤー別の変更影響を分析：

```bash
# モデルレイヤー全体の影響を分析
diffai baseline.safetensors modified.safetensors --show-layer-impact

# レイヤー別変更分析の出力
```

**使用例:**
- どのレイヤーが最も変化したかを理解
- ファインチューニング戦略のガイド
- アーキテクチャ変更の分析

### 5. 複合分析

包括的分析のため複数機能を組み合わせ：

```bash
# 包括的なモデル分析
diffai checkpoint_v1.safetensors checkpoint_v2.safetensors \
  --stats \
  --quantization-analysis \
  --sort-by-change-magnitude \
  --show-layer-impact

# 自動化用のJSON出力
diffai model1.safetensors model2.safetensors \
  --stats --output json
```

## 機能選択ガイド

**訓練監視用:**
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --stats --sort-by-change-magnitude
```

**本番デプロイ用:**
```bash
diffai current_prod.safetensors candidate.safetensors \
  --stats --quantization-analysis
```

**研究分析用:**
```bash
diffai baseline.safetensors experiment.safetensors \
  --stats --show-layer-impact
```

**量子化検証用:**
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis --stats
```

## 将来機能（Phase 3）

### Phase 3A（コア機能）
- `--architecture-comparison` - モデルアーキテクチャと構造変化の比較
- `--memory-analysis` - メモリ使用量と最適化機会の分析
- `--anomaly-detection` - モデルパラメータの数値異常検出
- `--change-summary` - 詳細な変更要約の生成

### Phase 3B（高度分析）
- `--convergence-analysis` - モデルパラメータの収束パターン分析
- `--gradient-analysis` - 利用可能時の勾配情報分析
- `--similarity-matrix` - モデル比較用類似度行列の生成

### 設計哲学
diffai はUNIX哲学に従います：シンプルで組み合わせ可能な、一つのことを適切に行うツール。機能は直交的で、強力な分析ワークフローのために組み合わせることができます。

## 次のステップ

- [基本的な使用法](basic-usage_ja.md) - 基本的な操作を学習
- [科学データ分析](scientific-data_ja.md) - NumPyとMATLABファイルの比較
- [CLIリファレンス](../reference/cli-reference_ja.md) - 完全なコマンドリファレンス