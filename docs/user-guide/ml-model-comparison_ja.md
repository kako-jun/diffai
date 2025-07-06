# MLモデル比較ガイド

このガイドでは、PyTorch や Safetensors ファイルを含む機械学習モデルの比較における diffai の特化機能について説明します。

## 🧠 概要

diffai は AI/ML モデル形式をネイティブサポートし、単なるバイナリファイルとしてではなく、テンソルレベルでモデルを比較できます。これにより、学習、ファインチューニング、量子化、デプロイメント中のモデル変更を意味のある方法で分析できます。

## 📊 サポートされているML形式

### PyTorchモデル
- **`.pt` ファイル**: PyTorch モデルファイル（pickle 形式）
- **`.pth` ファイル**: PyTorch モデルファイル（別の拡張子）

### Safetensors モデル
- **`.safetensors` ファイル**: Hugging Face Safetensors 形式（推奨）

### 将来サポート予定
- **`.onnx` ファイル**: ONNX 形式
- **`.h5` ファイル**: Keras/TensorFlow HDF5 形式
- **`.pb` ファイル**: TensorFlow Protocol Buffer 形式

## 🔍 diffai が分析する内容

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

## 🚀 基本的なモデル比較

### シンプルな比較

```bash
# 2つのPyTorchモデルを比較
diffai model1.pt model2.pt

# Safetensorsモデルを比較
diffai model1.safetensors model2.safetensors

# 自動形式検出
diffai pretrained.safetensors finetuned.safetensors
```

### 出力例

```
📊 tensor.transformer.h.0.attn.weight: mean=0.0023→0.0156, std=0.0891→0.1234
⬚ tensor.classifier.weight: [768, 1000] -> [768, 10]
+ tensor.new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- tensor.old_layer.bias: shape=[256], dtype=f32, params=256
```

### 出力記号

| 記号 | 意味 | 説明 |
|------|------|------|
| `📊` | 統計値が変更 | テンソル値が変更されたが形状は同じ |
| `⬚` | 形状が変更 | テンソルの次元が変更された |
| `+` | 追加 | 新しいテンソル/レイヤーが追加された |
| `-` | 削除 | テンソル/レイヤーが削除された |
| `~` | 変更 | 一般的な変更 |
| `!` | 型が変更 | データ型が変更された |

## ⚙️ 高度なオプション

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
# 特定のレイヤーに焦点を当てる
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# タイムスタンプやメタデータを無視
diffai model1.safetensors model2.safetensors --ignore-keys-regex "^(timestamp|_metadata)"
```

## 🎯 一般的な使用例

### 1. ファインチューニング分析

事前学習モデルとファインチューニング版を比較：

```bash
diffai pretrained_bert.safetensors finetuned_bert.safetensors

# 期待される出力: アテンション層の統計的変化
# 📊 tensor.bert.encoder.layer.11.attention.self.query.weight: mean=-0.0001→0.0023
# 📊 tensor.classifier.weight: mean=0.0000→0.0145, std=0.0200→0.0890
```

**分析**: 
- 初期レイヤーの小さな変化（特徴抽出は類似を保つ）
- 最終レイヤーの大きな変化（タスク固有の適応）

### 2. 量子化の影響評価

FP32 と量子化モデルを比較：

```bash
diffai model_fp32.safetensors model_int8.safetensors --epsilon 0.1

# 期待される出力: 制御された精度損失
# 📊 tensor.conv1.weight: mean=0.0045→0.0043, std=0.2341→0.2298
# 差異が見つかりませんでした（イプシロン許容値内）
```

**分析**:
- 小さな統計的変化は成功した量子化を示す
- 大きな変化は品質損失を示唆する可能性

### 3. 学習進捗の追跡

学習中のチェックポイントを比較：

```bash
diffai checkpoint_epoch_10.pt checkpoint_epoch_50.pt

# 期待される出力: 収束パターン
# 📊 tensor.layers.0.weight: mean=-0.0012→0.0034, std=1.2341→0.8907
# 📊 tensor.layers.1.bias: mean=0.1234→0.0567, std=0.4567→0.3210
```

**分析**:
- 標準偏差の減少は収束を示唆
- 平均値のシフトは学習方向を示す

### 4. アーキテクチャ比較

異なるモデルアーキテクチャを比較：

```bash
diffai resnet50.safetensors efficientnet_b0.safetensors

# 期待される出力: 構造的差異
# ⬚ tensor.features.conv1.weight: [64, 3, 7, 7] -> [32, 3, 3, 3]
# + tensor.features.mbconv.expand_conv.weight: shape=[96, 32, 1, 1]
# - tensor.features.layer4.2.downsample.0.weight: shape=[2048, 1024, 1, 1]
```

**分析**:
- 形状変化は異なるレイヤーサイズを示す
- 追加/削除されたテンソルはアーキテクチャ革新を示す

## 📈 パフォーマンス最適化

### メモリ考慮事項

大きなモデル（1GB以上）の場合：

```bash
# ストリーミングモードを使用（将来の機能）
diffai --stream huge_model1.safetensors huge_model2.safetensors

# 特定の部分に分析を集中
diffai model1.safetensors model2.safetensors --path "tensor.classifier"

# より高速な比較のため高いイプシロンを使用
diffai model1.safetensors model2.safetensors --epsilon 1e-3
```

### 速度最適化

```bash
# 並列処理（将来の機能）
diffai --threads 8 model1.safetensors model2.safetensors

# 形状のみの分析で統計計算をスキップ
diffai --shape-only model1.safetensors model2.safetensors
```

## 🔧 統合例

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
            echo "❌ 重要なレイヤーの形状変化が検出されました"
            exit 1
          fi
          
          # 多くのパラメータが変更された場合は警告
          changes=$(jq length model_diff.json)
          if [ "$changes" -gt 10 ]; then
            echo "⚠️ 多くのパラメータ変更が検出されました: $changes"
          fi
```

### Git プリコミットフック

```bash
#!/bin/bash
# .git/hooks/pre-commit

model_files=$(git diff --cached --name-only | grep -E '\.(pt|pth|safetensors)$')

for file in $model_files; do
    if [ -f "$file" ]; then
        echo "🔍 $file のモデル変更を分析中"
        
        # 前のバージョンと比較
        git show HEAD:"$file" > /tmp/old_model
        
        diffai /tmp/old_model "$file" --output json > /tmp/model_diff.json
        
        # 重要な変更をチェック
        shape_changes=$(jq '[.[] | select(.TensorShapeChanged)] | length' /tmp/model_diff.json)
        
        if [ "$shape_changes" -gt 0 ]; then
            echo "⚠️ $file でアーキテクチャ変更が検出されました"
            diffai /tmp/old_model "$file"
            
            read -p "コミットを続行しますか？ (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
        
        rm -f /tmp/old_model /tmp/model_diff.json
    fi
done
```

## 🚨 トラブルシューティング

### よくある問題

#### 1. "解析に失敗" エラー

```bash
# ファイル形式をチェック
file model.safetensors

# ファイル整合性を確認
diffai --check model.safetensors

# 明示的な形式指定で試行
diffai --format safetensors model1.safetensors model2.safetensors
```

#### 2. 大きなモデルでのメモリ問題

```bash
# 精度を下げる高いイプシロンを使用
diffai --epsilon 1e-3 large1.safetensors large2.safetensors

# 特定のレイヤーに焦点を当てる
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

## 📊 ベストプラクティス

### 1. イプシロン値の選択

| 使用例 | 推奨イプシロン | 理由 |
|--------|---------------|------|
| 厳密な比較 | イプシロンなし | すべての変更を検出 |
| 学習進捗 | 1e-6 | 数値ノイズを無視 |
| 量子化分析 | 0.01-0.1 | 精度損失を考慮 |
| アーキテクチャチェック | 1e-3 | 構造的変更に焦点 |

### 2. 出力形式の選択

- **CLI**: 人間によるレビューとデバッグ
- **JSON**: 自動化とスクリプト作成
- **YAML**: 設定ファイルとドキュメント

### 3. パフォーマンスのコツ

- 関連するレイヤーに分析を集中するため `--path` を使用
- ノイズを避けるため適切なイプシロン値を設定
- 比較戦略を選ぶ際はモデルサイズを考慮

---

**次へ**: [Examples](examples.md) - 実世界での使用シナリオを探索