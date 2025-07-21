# 出力形式

diffaiがサポートする出力形式とその仕様。

## 概要

diffaiは、さまざまな用途に適した4つの異なる出力形式をサポートします：

1. **CLI** - 人間可読形式（デフォルト）
2. **JSON** - 機械処理と自動化
3. **YAML** - 設定ファイルと人間可読構造化データ
4. **Unified** - Git統合と従来のdiff形式

## CLI出力形式

### 概要
デフォルトで使用される人間可読なカラー出力形式。

### 機能
- **カラー表示**: 変更タイプ別の色分け
- **シンボル表示**: 直感的なシンボル（`+`, `-`, `~`, `□`）
- **階層表示**: ネスト構造の表現
- **ML専用シンボル**: ML分析結果の専用表示

### 使用方法
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

### シンボル意味

#### 基本シンボル
| シンボル | 意味 | 色 | 説明 |
|----------|------|----|----- |
| `~` | 変更 | 青 | 値は変更されたが構造は同じ |
| `+` | 追加 | 緑 | 新しい要素が追加 |
| `-` | 削除 | 赤 | 要素が削除 |
| `□` | 形状変更 | 黄 | テンソル形状が変更 |

#### ML分析シンボル
| シンボル | 意味 | 色 | 説明 |
|----------|------|----|----- |
| `◦` | 分析結果 | シアン | ML分析機能結果 |
| `[CRITICAL]` | 重大アラート | 赤 | 重大な問題を検出 |
| `[WARNING]` | 警告 | 黄 | 注意が必要 |
| `[ALERT]` | アラート | 赤 | 閾値を超過 |
| `[HIGH]` | 高優先度 | 赤 | 高優先度の問題 |
| `[HOLD]` | デプロイ保留 | 赤 | デプロイ保留を推奨 |
| `[GRADUAL]` | 段階的デプロイ | 黄 | 段階的デプロイを推奨 |

## JSON出力形式

### 概要
機械処理と自動化に適した構造化データ形式。

### 機能
- **構造化データ**: プログラムによる解析が容易
- **型情報**: 完全な型情報を保持
- **API統合**: RESTful APIとの統合が容易
- **自動化**: CI/CDとスクリプトに最適

### 使用方法
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

### データ構造タイプ

#### 基本変更タイプ
- **Added**: `["key", value]` - 追加された要素
- **Removed**: `["key", value]` - 削除された要素
- **Modified**: `["key", old_value, new_value]` - 変更された要素
- **TypeChanged**: `["key", old_type, new_type]` - 型が変更

#### ML専用タイプ
- **TensorStatsChanged**: テンソル統計変更
- **TensorShapeChanged**: テンソル形状変更
- **TensorAdded**: テンソル追加
- **TensorRemoved**: テンソル削除
- **LearningProgress**: 学習進捗分析
- **ConvergenceAnalysis**: 収束分析
- **AnomalyDetection**: 異常検出
- **MemoryAnalysis**: メモリ分析
- **DeploymentReadiness**: デプロイ準備度
- **RiskAssessment**: リスク評価

## YAML出力形式

### 概要
設定ファイルとドキュメント化に適した人間可読構造化データ形式。

### 機能
- **可読性**: 人間に優しい形式
- **コメント**: コメントサポート
- **階層構造**: 明確な階層表現
- **設定ファイル**: 設定ファイルとして使用可能

### 使用方法
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
Git統合と従来のdiffツールとの互換性。

### 機能
- **Git統合**: git diffとの互換性
- **マージツール**: 3-wayマージツールとの連携
- **従来形式**: 既存のdiffツールとの互換性
- **パッチ適用**: git applyで適用可能

### 使用方法
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

## 出力形式選択ガイドライン

### 用途別

| 用途 | 推奨形式 | 理由 |
|------|----------|------|
| 人間によるレビュー | CLI | カラー、直感的シンボル |
| 自動化/スクリプト | JSON | 機械処理可能 |
| 設定ファイル | YAML | 可読性、コメントサポート |
| ドキュメント化 | YAML | 人間に優しい |
| Git統合 | Unified | 既存ツールとの互換性 |
| API統合 | JSON | 標準データ交換形式 |
| レポート生成 | JSON | 構造化データ処理 |

### 環境別

| 環境 | 推奨形式 | 理由 |
|------|----------|------|
| 対話的コマンドライン | CLI | 即座の理解 |
| CI/CDパイプライン | JSON | 自動チェック |
| 開発環境 | CLI | 迅速なデバッグ |
| 本番環境 | JSON | ログとモニタリング |
| 研究/実験 | YAML | 結果ドキュメント化 |

## 高度な使用例

### パイプライン処理
```bash
# JSONを出力してjqで処理
diffai model1.safetensors model2.safetensors --output json | \
  jq '.[] | select(.TensorStatsChanged)'

# YAMLを出力してファイル保存
diffai config1.yaml config2.yaml --output yaml > changes.yaml
```

### 条件分岐
```bash
# 変更が存在するかチェック
if diffai model1.safetensors model2.safetensors --output json | jq -e 'length > 0'; then
  echo "Changes detected"
fi
```

### 複数形式生成
```bash
# 人間用と機械用の両方の形式を生成
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

## 関連ドキュメント

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドラインオプション
- [サポート形式](formats_ja.md) - 入力ファイル形式
- [ML分析機能](ml-analysis_ja.md) - 機械学習分析機能