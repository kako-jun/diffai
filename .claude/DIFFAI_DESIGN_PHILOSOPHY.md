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

### 具体例（11個の完全ML分析機能）
```
learning_rate_analysis    → Learning Rate Analysis - 学習率変化追跡
optimizer_comparison      → Optimizer Comparison - Adam/SGD状態分析
loss_tracking            → Loss Tracking - 損失関数収束パターン
accuracy_tracking        → Accuracy Tracking - 性能指標進化
model_version_analysis    → Model Version Analysis - チェックポイント比較
gradient_analysis        → Gradient Analysis - 勾配健全性・消失/爆発検出
quantization_analysis    → Quantization Analysis - 混合精度（FP32/FP16/INT8/INT4）
convergence_analysis     → Convergence Analysis - 学習曲線・プラトー検出
activation_analysis      → Activation Analysis - ReLU/GELU/Tanh分布
attention_analysis       → Attention Analysis - Transformerメカニズム検出
ensemble_analysis        → Ensemble Analysis - マルチモデル構造検出
```

## 🔄 フォーマット適応型標準機能

### PyTorch (.pt/.pth) - フル機能（11個）
```
標準提供機能:
1. Learning Rate Analysis - 学習率変化追跡
2. Optimizer Comparison - Adam/SGD状態分析
3. Loss Tracking - 損失関数収束パターン
4. Accuracy Tracking - 性能指標進化
5. Model Version Analysis - チェックポイント比較
6. Gradient Analysis - 勾配健全性・消失/爆発検出
7. Quantization Analysis - 混合精度（FP32/FP16/INT8/INT4）
8. Convergence Analysis - 学習曲線・プラトー検出
9. Activation Analysis - ReLU/GELU/Tanh分布
10. Attention Analysis - Transformerメカニズム検出
11. Ensemble Analysis - マルチモデル構造検出
```

### Safetensors (.safetensors) - フル機能（11個）
```
標準提供機能: PyTorchと同等の11個すべて
- 技術的制約は克服済み（diffx-core統合により）
- HuggingFace標準フォーマットの完全サポート
```

### NumPy (.npy/.npz) - 基本統計機能
```
標準提供機能:
- Tensor Statistics（形状、平均、標準差、データ型）
- 配列比較（要素差分）
- メモリ使用量分析

除外機能（ML機能は対象外）:
- 学習関連の11個のML分析機能
```

### MATLAB (.mat) - 基本統計機能
```
標準提供機能:
- Tensor Statistics（行列統計）
- 変数比較（マルチ変数対応）
- メモリ使用量分析

除外機能（ML機能は対象外）:
- 学習関連の11個のML分析機能
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

最終更新: Claude Code セッション - Phase F完了・翻訳版README作成完了時点