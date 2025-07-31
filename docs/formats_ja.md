# ファイル形式と出力オプション

diffaiはAI/MLと科学計算のファイル形式に特化しており、PyTorchとSafetensorsファイルの自動分析を提供します。

## サポートされている入力形式

### AI/MLモデル形式（自動ML分析）

#### PyTorchモデル（.pt、.pth）
- **自動ML分析**：11の機能すべてが自動的に実行
- **サポート**：モデル状態辞書、チェックポイント、完全なモデル
- **統合**：PyTorch形式解析にCandleを使用
- **メモリ**：大規模モデル（GB+ファイル）の効率的な処理

```bash
# 自動包括分析
diffai baseline_model.pt finetuned_model.pt
# 出力：11のML分析機能すべて + テンソル統計
```

#### Safetensors（.safetensors）
- **自動ML分析**：11の機能すべてが自動的に実行  
- **サポート**：HuggingFace標準形式
- **パフォーマンス**：高速読み込みとメモリ効率的な解析
- **安全性**：安全なテンソル保存形式

```bash
# 自動包括分析
diffai model_v1.safetensors model_v2.safetensors
# 出力：11のML分析機能すべて + テンソル統計
```

### 科学データ形式（基本分析）

#### NumPy配列（.npy、.npz）
- **分析**：テンソル統計のみ（ML固有の分析なし）
- **サポート**：単一配列（.npy）とアーカイブ（.npz）
- **データ型**：複素数を含むすべてのNumPyデータ型
- **統計**：形状、平均、標準偏差、最小、最大、メモリ使用量

```bash
# 基本テンソル分析
diffai experiment_data_v1.npy experiment_data_v2.npy
# 出力：形状、統計、データ型の変化
```

#### MATLABファイル（.mat）
- **分析**：テンソル統計のみ（ML固有の分析なし）
- **サポート**：変数検出付きMATLABマトリックスファイル
- **データ型**：double、single、complex、論理配列
- **変数**：多変数ファイルサポート

```bash
# 基本テンソル分析
diffai simulation_v1.mat simulation_v2.mat
# 出力：変数の変化、マトリックス統計
```

### 形式検出
diffaiは以下に基づいてファイル形式を自動検出します：
1. **ファイル拡張子**（.pt、.safetensors、.npy、.mat）
2. **マジックバイト**とファイルヘッダー
3. **コンテンツ構造**分析

## 出力形式

### CLI出力（デフォルト）
カラーコーディングと直感的なシンボルを使用した人間が読みやすい形式。

#### 基本シンボル
| シンボル | 意味 | 色 | 説明 |
|----------|------|-----|-----|
| `~` | 変更 | 青 | 値は変更されたが構造は同じ |
| `+` | 追加 | 緑 | 新しい要素が追加された |
| `-` | 削除 | 赤 | 要素が削除された |
| `□` | 形状変更 | 黄 | テンソル形状が変更された |

#### ML分析シンボル
| シンボル | 意味 | 色 | 説明 |
|----------|------|-----|-----|
| `◦` | 分析結果 | シアン | ML分析機能の結果 |
| `[CRITICAL]` | 重要アラート | 赤 | 重要な問題が検出された |
| `[WARNING]` | 警告 | 黄 | 注意が必要 |

#### CLI出力例
```bash
diffai model1.safetensors model2.safetensors

learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
~ fc1.bias: mean=0.0018->0.0017, std=0.0518->0.0647
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
+ new_layer.weight: shape=[64, 64], dtype=f32, params=4096
- old_layer.bias: shape=[256], dtype=f32, params=256
```

### JSON出力
自動化とMLOps統合のための構造化形式。

```bash
diffai model1.safetensors model2.safetensors --output json
```

**JSON構造例**：
```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing"
  },
  "tensor_changes": [
    {
      "path": "fc1.weight",
      "change_type": "modified",
      "old_stats": {"mean": -0.0002, "std": 0.0514},
      "new_stats": {"mean": -0.0001, "std": 0.0716}
    }
  ]
}
```

### YAML出力
レポートとドキュメント用の人間が読みやすい構造化形式。

```bash
diffai model1.safetensors model2.safetensors --output yaml
```

**YAML構造例**：
```yaml
learning_rate_analysis:
  old: 0.001
  new: 0.0015
  change: "+50.0%"
  trend: "increasing"

gradient_analysis:
  flow_health: "healthy"
  gradient_norm: 0.021069
  variance_change: "+15.3%"
```

### 詳細モード
デバッグと分析のための包括的な診断情報。

```bash
diffai model1.safetensors model2.safetensors --verbose
```

**詳細出力に含まれるもの**：
- 設定診断
- ファイル分析詳細
- パフォーマンス指標
- 処理コンテキスト
- メモリ使用量情報

## 形式固有の機能

### PyTorch統合
- **状態辞書**：モデル状態の自動解析
- **オプティマイザ状態**：オプティマイザパラメータの分析
- **チェックポイント**：完全なチェックポイント比較サポート
- **カスタムオブジェクト**：カスタムPyTorchオブジェクトの処理

### Safetensorsの利点
- **セキュリティ**：任意コード実行リスクなし
- **パフォーマンス**：高速メモリマップ読み込み
- **クロスプラットフォーム**：言語間で一貫性
- **メタデータ**：豊富なテンソルメタデータサポート

### NumPy機能
- **配列**：単一配列ファイル（.npy）
- **アーカイブ**：複数配列ファイル（.npz）
- **複素数**：完全な複素データサポート
- **メモリビュー**：効率的な大型配列処理

### MATLABサポート
- **変数**：多変数MATLABファイル
- **データ型**：すべてのMATLAB数値型
- **セル配列**：基本的なセル配列サポート
- **スパース行列**：スパース行列検出

## パフォーマンスの考慮事項

### メモリ使用量
- **ストリーミング**：大きなファイルをチャンクで処理
- **メモリマッピング**：Safetensorsの効率的なメモリ使用
- **ガベージコレクション**：自動メモリクリーンアップ
- **バッチ処理**：設定可能なバッチサイズ

### ファイルサイズ制限
- **制限なし**：任意のサイズのファイルを処理
- **自動最適化**：大きなファイルのメモリ最適化
- **進捗報告**：大きなファイルの進捗表示
- **チャンク処理**：大きな比較を小さなチャンクに分割

### 速度最適化
- **早期終了**：パターンが検出されない場合は停止
- **並列処理**：有益な場合のマルチスレッド
- **キャッシング**：中間結果のキャッシュ
- **効率的なアルゴリズム**：lawkit増分統計

## エラーハンドリング

### サポートされていない形式
diffaiはAI/ML形式に焦点を当てています。汎用構造化データには：
```bash
# JSON、XML、CSV等の場合 - 代わりにdiffxを使用
diffx data1.json data2.json
```

### 破損ファイル
- **緩やかな劣化**：可能な場合は部分分析
- **エラー回復**：利用可能なデータで継続
- **明確なメッセージ**：説明的なエラーメッセージ
- **検証**：処理前のファイル形式検証

### 大きなファイルの処理
- **メモリ監視**：自動メモリ使用量追跡
- **最適化**：自動パフォーマンス最適化
- **進捗**：長時間の操作の進捗報告
- **中断**：ユーザー中断の適切な処理

## 統合例

### Python統合
```python
import diffai
import json

# PyTorchモデルの比較
result = diffai.diff_files("model1.pt", "model2.pt", output_format="json")
analysis = json.loads(result)

# ML分析結果へのアクセス
lr_analysis = analysis["learning_rate_analysis"]
gradient_info = analysis["gradient_analysis"]
```

### JavaScript統合
```javascript
const diffai = require('diffai-js');

// Safetensorsモデルの比較
const result = diffai.diffFiles("model1.safetensors", "model2.safetensors", {
    outputFormat: "json"
});

const analysis = JSON.parse(result);
console.log(analysis.quantization_analysis);
```

### コマンドライン自動化
```bash
#!/bin/bash
# CI/CDモデル検証スクリプト

RESULT=$(diffai baseline.safetensors candidate.safetensors --output json)
ACCURACY_CHANGE=$(echo "$RESULT" | jq -r '.accuracy_tracking.accuracy_delta')

if [[ "$ACCURACY_CHANGE" == +* ]]; then
    echo "モデル改善: $ACCURACY_CHANGE"
    exit 0
else
    echo "モデル劣化: $ACCURACY_CHANGE"
    exit 1
fi
```

## 関連項目

- **[クイックスタート](quick-start_ja.md)** - すぐに始める
- **[ML分析](ml-analysis_ja.md)** - 11の分析機能を理解
- **[APIリファレンス](reference/api-reference_ja.md)** - プログラミングインターフェース
- **[使用例](examples/)** - 実際の使用例