# 出力形式

diffaiがサポートする出力形式とその仕様について説明します。

## 概要

diffaiは、用途に応じて4つの異なる出力形式をサポートしています：

1. **CLI** - 人間による読み取り向け（デフォルト）
2. **JSON** - 機械処理・自動化向け
3. **YAML** - 設定ファイル・人間可読向け
4. **Unified** - Git統合・従来のdiff形式

## CLI出力形式

### 概要
人間が読みやすい色付きの出力形式です。デフォルトで使用されます。

### 特徴
- **色付き表示**: 変更の種類に応じた色分け
- **記号表示**: 直感的な記号（`+`, `-`, `~`, `□`）
- **階層表示**: ネストした構造の表現
- **ML特化記号**: 機械学習分析結果の専用表示

### 使用例
```bash
diffai model1.safetensors model2.safetensors --output cli
# または
diffai model1.safetensors model2.safetensors  # デフォルト
```

### 出力例
```
~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
+ new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- old_layer.bias: shape=[256], dtype=f32, params=256
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

### 記号の意味

#### 基本記号
| 記号 | 意味 | 色 | 説明 |
|------|------|-----|------|
| `~` | 変更 | 青 | 値が変更されたが構造は同じ |
| `+` | 追加 | 緑 | 新しい要素が追加された |
| `-` | 削除 | 赤 | 要素が削除された |
| `□` | 形状変更 | 黄 | テンソルの形状が変更された |

#### ML分析記号
| 記号 | 意味 | 色 | 説明 |
|------|------|-----|------|
| `◦` | 分析結果 | シアン | ML分析機能の結果 |
| `[CRITICAL]` | 重要アラート | 赤 | 重要な問題の検出 |
| `[WARNING]` | 警告 | 黄 | 注意が必要な状況 |
| `[ALERT]` | アラート | 赤 | 閾値超過の警告 |
| `[HIGH]` | 高優先度 | 赤 | 高優先度の問題 |
| `[HOLD]` | 保留 | 赤 | デプロイ保留推奨 |
| `[GRADUAL]` | 段階的 | 黄 | 段階的デプロイ推奨 |

### 色の意味
- **赤**: 削除、エラー、重要な警告
- **緑**: 追加、成功、良好な状態
- **青**: 変更、一般的な情報
- **黄**: 警告、注意が必要
- **シアン**: 分析結果、補助情報
- **紫**: 高度な分析結果

## JSON出力形式

### 概要
機械処理と自動化に適した構造化データ形式です。

### 特徴
- **構造化データ**: プログラムから簡単に解析可能
- **型情報**: 完全な型情報を保持
- **API統合**: RESTful APIとの統合が容易
- **自動化**: CI/CDやスクリプトでの使用に最適

### 使用例
```bash
diffai model1.safetensors model2.safetensors --output json
```

### 出力例
```json
[
  {
    "TensorStatsChanged": [
      "fc1.bias",
      {"mean": 0.0018, "std": 0.0518, "shape": [64], "dtype": "f32", "total_params": 64},
      {"mean": 0.0017, "std": 0.0647, "shape": [64], "dtype": "f32", "total_params": 64}
    ]
  },
  {
    "TensorAdded": [
      "new_layer.weight",
      {"mean": 0.0123, "std": 0.0456, "shape": [64, 64], "dtype": "f32", "total_params": 4096}
    ]
  },
  {
    "AnomalyDetection": [
      "anomaly_detection",
      {
        "anomaly_type": "gradient_explosion",
        "severity": "critical",
        "affected_layers": ["layer1", "layer2"],
        "recommended_action": "Reduce learning rate"
      }
    ]
  }
]
```

### データ構造

#### 基本変更タイプ
- **Added**: `["key", value]` - 追加された要素
- **Removed**: `["key", value]` - 削除された要素
- **Modified**: `["key", old_value, new_value]` - 変更された要素
- **TypeChanged**: `["key", old_type, new_type]` - 型が変更された要素

#### ML固有タイプ
- **TensorStatsChanged**: テンソル統計の変更
- **TensorShapeChanged**: テンソル形状の変更
- **TensorAdded**: テンソルの追加
- **TensorRemoved**: テンソルの削除
- **LearningProgress**: 学習進捗分析
- **ConvergenceAnalysis**: 収束分析
- **AnomalyDetection**: 異常検知
- **MemoryAnalysis**: メモリ分析
- **DeploymentReadiness**: デプロイ準備度
- **RiskAssessment**: リスク評価

#### 科学データタイプ
- **NumpyArrayChanged**: NumPy配列の変更
- **NumpyArrayAdded**: NumPy配列の追加
- **NumpyArrayRemoved**: NumPy配列の削除
- **MatlabArrayChanged**: MATLAB配列の変更
- **MatlabArrayAdded**: MATLAB配列の追加
- **MatlabArrayRemoved**: MATLAB配列の削除

## YAML出力形式

### 概要
人間が読みやすい構造化データ形式です。設定ファイルやドキュメント作成に適しています。

### 特徴
- **可読性**: 人間が読みやすい形式
- **コメント**: コメントの追加が可能
- **階層構造**: 明確な階層表現
- **設定ファイル**: 設定ファイルとして使用可能

### 使用例
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

### 出力例
```yaml
- TensorStatsChanged:
  - fc1.bias
  - mean: 0.0018
    std: 0.0518
    shape: [64]
    dtype: f32
    total_params: 64
  - mean: 0.0017
    std: 0.0647
    shape: [64]
    dtype: f32
    total_params: 64

- TensorAdded:
  - new_layer.weight
  - mean: 0.0123
    std: 0.0456
    shape: [64, 64]
    dtype: f32
    total_params: 4096

- AnomalyDetection:
  - anomaly_detection
  - anomaly_type: gradient_explosion
    severity: critical
    affected_layers:
      - layer1
      - layer2
    recommended_action: Reduce learning rate
```

## Unified出力形式

### 概要
Git統合や従来のdiffツールとの互換性を提供する形式です。

### 特徴
- **Git統合**: git diffとの互換性
- **マージツール**: 3-way mergeツールとの連携
- **従来形式**: 既存のdiffツールとの互換性
- **パッチ適用**: git applyでの適用が可能

### 使用例
```bash
diffai config1.json config2.json --output unified
```

### 出力例
```diff
--- config1.json
+++ config2.json
@@ -1,5 +1,6 @@
 {
   "model": {
-    "layers": 12,
+    "layers": 24,
     "hidden_size": 768
   },
+  "optimizer": "adam"
 }
```

## 出力形式の選択指針

### 用途別推奨形式

| 用途 | 推奨形式 | 理由 |
|------|----------|------|
| 人間によるレビュー | CLI | 色付き、直感的記号 |
| 自動化・スクリプト | JSON | 機械処理しやすい |
| 設定ファイル | YAML | 可読性、コメント対応 |
| ドキュメント作成 | YAML | 人間が読みやすい |
| Git統合 | Unified | 既存ツールとの互換性 |
| API連携 | JSON | 標準的なデータ交換形式 |
| レポート生成 | JSON | 構造化データ処理 |

### 環境別推奨形式

| 環境 | 推奨形式 | 理由 |
|------|----------|------|
| 対話的コマンドライン | CLI | 即座の理解と判断 |
| CI/CD パイプライン | JSON | 自動化されたチェック |
| 開発環境 | CLI | 迅速なデバッグ |
| 本番環境 | JSON | ログ記録と監視 |
| 研究・実験 | YAML | 結果の文書化 |

## 高度な使用例

### パイプライン処理
```bash
# JSONで出力してjqで加工
diffai model1.safetensors model2.safetensors --output json | \
  jq '.[] | select(.TensorStatsChanged)'

# YAMLで出力してファイルに保存
diffai config1.yaml config2.yaml --output yaml > changes.yaml
```

### 条件分岐
```bash
# 変更があるかチェック
if diffai model1.safetensors model2.safetensors --output json | jq -e 'length > 0'; then
  echo "変更が検出されました"
fi
```

### 複数形式同時出力
```bash
# 人間用とスクリプト用を同時生成
diffai model1.safetensors model2.safetensors > human_readable.txt
diffai model1.safetensors model2.safetensors --output json > machine_readable.json
```

## 設定とカスタマイズ

### 設定ファイル
```toml
[output]
default = "cli"
json_pretty = true
yaml_flow = false
cli_colors = true
unified_context = 3

[colors]
added = "green"
removed = "red"
modified = "blue"
warning = "yellow"
error = "red"
```

### 環境変数
```bash
export DIFFAI_OUTPUT_FORMAT="json"
export DIFFAI_CLI_COLORS="true"
export DIFFAI_JSON_PRETTY="true"
export DIFFAI_YAML_FLOW="false"
```

## パフォーマンス考慮事項

### 形式別のパフォーマンス

| 形式 | 処理速度 | メモリ使用量 | ファイルサイズ |
|------|----------|-------------|--------------|
| CLI | 最速 | 最小 | 最大 |
| JSON | 速い | 小 | 中 |
| YAML | 中 | 中 | 中 |
| Unified | 中 | 中 | 小 |

### 大容量データでの最適化
```bash
# 大きなモデルでは簡潔な形式を選択
diffai huge_model1.safetensors huge_model2.safetensors --output json | \
  jq -c '.'  # コンパクトJSON

# 必要な情報のみフィルタリング
diffai model1.safetensors model2.safetensors --output json | \
  jq '[.[] | select(.TensorStatsChanged)]'
```

## トラブルシューティング

### 一般的な問題

#### 文字化け
```bash
# 文字エンコーディングを確認
export LANG=ja_JP.UTF-8
diffai model1.safetensors model2.safetensors
```

#### 色が表示されない
```bash
# 色付きを強制
diffai model1.safetensors model2.safetensors --output cli --force-colors

# 色を無効化
diffai model1.safetensors model2.safetensors --output cli --no-colors
```

#### JSON解析エラー
```bash
# 有効なJSONか確認
diffai model1.safetensors model2.safetensors --output json | jq empty

# Pretty print無効で確認
diffai model1.safetensors model2.safetensors --output json --no-pretty
```

#### YAML解析エラー
```bash
# YAML構文チェック
diffai model1.safetensors model2.safetensors --output yaml | yamllint

# フロー形式で出力
diffai model1.safetensors model2.safetensors --output yaml --flow
```

## 関連項目

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドオプション
- [対応形式](formats_ja.md) - 入力ファイル形式
- [ML分析機能](ml-analysis_ja.md) - 機械学習分析機能

