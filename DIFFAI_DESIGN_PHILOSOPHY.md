# diffai 設計哲学・方針 - 記憶継続用ドキュメント

## 🎯 根本設計思想

### diffaiの本質
diffaiは**AI/ML専用diffツール**であり、各ファイル形式で**技術的に可能な最大限の分析を標準提供**する。

### オプション概念の排除
- **旧思考**: オプション → 発動 → 機能実行
- **新思考**: フォーマット → 標準機能 → 最大限出力

**重要**: diffaiにとって、ML分析機能は「オプション」ではなく「そのフォーマットで当然提供されるべき標準機能」

## 📋 機能名の扱い方針

### 3つの用途での機能名使用
1. **内部的**: 実装・デバッグ・タスク管理の識別子
2. **出力的**: 結果の構造化・可読性のための区分
3. **ドキュメント的**: 機能説明・理解促進のための名称

### 具体例
```
learning_rate_tracking    → Learning Rate Tracking
optimizer_comparison      → Optimizer Comparison  
activation_analysis       → Activation Analysis
accuracy_tracking        → Accuracy Tracking
loss_tracking            → Loss Tracking
```

## 🔄 フォーマット適応型標準機能

### PyTorch (.pt/.pth) - フル機能
```
標準提供機能:
- Tensor Statistics
- Learning Rate Tracking  
- Optimizer Comparison
- Activation Analysis
- Accuracy Tracking
- Loss Tracking
- Model Version Check
```

### Safetensors (.safetensors) - 部分機能
```
標準提供機能:
- Tensor Statistics
- Learning Rate Tracking
- Activation Analysis  
- Loss Tracking
- Model Version Check

除外機能（技術的制約）:
- Optimizer Comparison (状態情報なし)
- Accuracy Tracking (評価情報なし)
```

### NumPy (.npy/.npz) - 基本機能
```
標準提供機能:
- Tensor Statistics
- Model Version Check

除外機能（意味なし）:
- 全学習関連機能
```

### MATLAB (.mat) - 基本機能
```
標準提供機能:
- Tensor Statistics  
- Model Version Check

除外機能（意味なし）:
- 全学習関連機能
```

## 💬 ユーザーコミュニケーション方針

### ドキュメントでの表現
❌ 悪い例: "Use --learning-rate-tracking to enable..."
✅ 良い例: "For PyTorch files, diffai automatically provides Learning Rate Tracking..."

### 出力での構造化
```json
{
  "tensor_statistics": { ... },
  "learning_rate_tracking": { ... },
  "optimizer_comparison": { ... }
}
```

```
=== Tensor Statistics ===
[結果]

=== Learning Rate Tracking ===  
[結果]

=== Optimizer Comparison ===
[結果]
```

## 🚨 重要な禁止事項

### 絶対に避けるべき表現
- "Enable this option to..."
- "This feature can be activated by..."
- "Optional ML analysis..."

### 推奨表現
- "diffai automatically provides..."
- "For [format] files, the following analysis is performed..."
- "[Feature] is standard for [format] files..."

## 🔧 実装指針

### コード内コメント
```rust
// Learning rate tracking - standard feature for PyTorch/Safetensors
if format_supports_learning_metadata(format) {
    analyze_learning_rate_changes(old, new, results);
}
```

### 設定関数命名
```rust
// ❌ enable_learning_rate_tracking()
// ✅ configure_standard_features_for_format()
fn build_format_aware_diffai_options(format: FileFormat) -> DiffaiSpecificOptions
```

## 📝 タスク管理での記述方針

### 機能実装タスクの書き方
❌ "learning_rate_trackingオプションの実装"
✅ "Learning Rate Tracking機能の実装（PyTorch/Safetensors標準機能）"

### 進捗報告での表現
❌ "ML分析オプションの有効化完了"
✅ "フォーマット適応型ML分析標準機能の実装完了"

---
**このドキュメントは記憶継続のための設計指針です。セッション開始時に必ず参照してください。**

最終更新: Claude Code セッション - diffai設計哲学確立時点