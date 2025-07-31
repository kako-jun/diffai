# ML分析機能 - 技術リファレンス

PyTorch（.pt/.pth）またはSafetensors（.safetensors）ファイルを比較する際にdiffaiが自動実行する11の専門的なML分析機能の完全な技術ドキュメント。

## 概要

diffaiは**設定より規約**の原則に従います：AI/MLファイルが検出されると、手動設定を必要とせずに包括的な洞察を提供する11のML分析機能すべてが自動実行されます。lawkitのメモリ効率パターンとdiffx-core最適化技術を使用して構築されています。

**自動トリガー条件：**
- **PyTorchファイル（.pt/.pth）**：11の分析すべてが実行
- **Safetensorsファイル（.safetensors）**：11の分析すべてが実行  
- **NumPy/MATLABファイル**：基本的なテンソル統計のみ
- **その他の形式**：diffx-coreによる標準的な構造比較

## 1. 学習率分析

**関数**：`analyze_learning_rate_changes()`  
**目的**：学習率の変化と訓練動態を追跡

### 検出ロジック
```rust
// モデルデータ内の学習率フィールドを自動検索
let lr_fields = ["learning_rate", "lr", "step_size", "base_lr", "current_lr"];
// オプティマイザ学習率パラメータの変化を分析
```

### 出力形式
```bash
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
```

### JSON出力
```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing",
    "significance": "moderate"
  }
}
```

### 技術実装
- **アルゴリズム**：パーセンテージ計算を伴う直接値比較
- **メモリ効率**：lawkit増分処理パターン
- **しきい値**：5%を超える変化を重要と判定
- **エラーハンドリング**：LRフィールドが見つからない場合の適切なフォールバック

## 2. オプティマイザ比較

**関数**：`analyze_optimizer_comparison()`  
**目的**：オプティマイザの状態とモメンタム情報を比較

### 検出ロジック
```rust
// オプティマイザ状態辞書を検索
let optimizer_fields = ["optimizer", "optimizer_state_dict", "optim", "momentum", "adam"];
// モメンタム、ベータパラメータ、状態進化を分析
```

### 出力形式
```bash
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
```

### JSON出力
```json
{
  "optimizer_comparison": {
    "type": "Adam",
    "momentum_change": "+2.1%",
    "state_evolution": "stable",
    "beta1": 0.9,
    "beta2": 0.999
  }
}
```

### 技術実装
- **状態追跡**：モメンタムバッファとオプティマイザパラメータを比較
- **メモリ最適化**：大きなオプティマイザ状態のストリーミング比較
- **サポートオプティマイザ**：Adam、SGD、AdamW、RMSprop（自動検出）

## 3. 損失追跡

**関数**：`analyze_loss_tracking()`  
**目的**：損失関数の進化と収束パターンを分析

### 検出ロジック
```rust
// 損失関連フィールドを自動検出
let loss_fields = ["loss", "train_loss", "val_loss", "epoch_loss", "step_loss"];
// 損失トレンドと収束指標を分析
```

### 出力形式
```bash
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
```

### JSON出力
```json
{
  "loss_tracking": {
    "loss_trend": "decreasing",
    "improvement_rate": "15.2%",
    "convergence_score": 0.89,
    "stability": "high"
  }
}
```

### 技術実装
- **トレンド分析**：損失値の変化方向を計算
- **収束スコア**：0.0-1.0のスケールで安定性を評価
- **改善率**：損失減少のパーセンテージ変化

## 4. 精度追跡

**関数**：`analyze_accuracy_tracking()`  
**目的**：精度の変化とパフォーマンス指標を監視

### 出力形式
```bash
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
```

### 技術実装
- **指標サポート**：精度、F1、適合率、再現率
- **トレンド分析**：パフォーマンスの方向性評価
- **多指標処理**：複数のパフォーマンス指標の同時追跡

## 5. モデルバージョン分析

**関数**：`analyze_model_version()`  
**目的**：モデルのバージョニングとチェックポイント情報を特定

### 出力形式
```bash
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
```

### 技術実装
- **バージョン検出**：セマンティックバージョニングパターンの認識
- **チェックポイント分析**：エポック/イテレーション進行の追跡
- **進化パターン**：増分 vs 主要変更の分類

## 6. 勾配分析

**関数**：`analyze_gradient_flow()`  
**目的**：勾配フロー、勾配消失/爆発、安定性を分析

### 出力形式
```bash
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
```

### JSON出力
```json
{
  "gradient_analysis": {
    "flow_health": "healthy",
    "gradient_norm": 0.021069,
    "variance_change": "+15.3%",
    "vanishing_risk": "low",
    "exploding_risk": "low"
  }
}
```

### 技術実装
- **勾配ノルム**：L2ノルムによる勾配大きさ計算
- **健全性評価**：healthy/warning/critical分類
- **しきい値**：消失（< 1e-7）、爆発（> 100）
- **lawkit統計**：メモリ効率的な増分計算

## 7. 量子化分析

**関数**：`analyze_quantization()`  
**目的**：混合精度（FP32/FP16/INT8/INT4）と圧縮効果を検出

### 出力形式
```bash
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
```

### JSON出力
```json
{
  "quantization_analysis": {
    "mixed_precision": "FP16+FP32",
    "compression": "12.5%",
    "precision_loss": "1.2%",
    "quantized_layers": 8,
    "bit_widths": [16, 32]
  }
}
```

### 技術実装
- **精度検出**：テンソルデータ型の自動分析
- **圧縮率**：メモリ使用量の比較
- **精度損失**：数値精度の変化推定

## 8. 収束分析

**関数**：`analyze_convergence()`  
**目的**：学習曲線分析、プラトー検出、最適化軌道

### 出力形式
```bash
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
```

### 技術実装
- **収束状態**：converging/converged/diverging分類
- **安定性スコア**：0.0-1.0スケール
- **プラトー検出**：学習停滞の識別

## 9. 活性化分析

**関数**：`analyze_activations()`  
**目的**：活性化関数の使用と分布を分析

### 出力形式
```bash
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
```

### 技術実装
- **関数タイプ**：ReLU、GELU、Tanh、Sigmoid、Swish検出
- **使用分布**：レイヤー間の活性化関数分析
- **飽和リスク**：死んだニューロンの検出

## 10. 注意機構分析

**関数**：`analyze_attention()`  
**目的**：トランスフォーマーと注意機構を分析

### 出力形式
```bash
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
```

### 技術実装
- **アーキテクチャ検出**：BERT/GPT/T5パターンの認識
- **ヘッド分析**：マルチヘッド注意構造
- **効率スコア**：注意機構のパフォーマンス評価

## 11. アンサンブル分析

**関数**：`analyze_ensemble()`  
**目的**：アンサンブルモデル構造を検出・分析

### 出力形式
```bash
ensemble_analysis: ensemble_detected=false, model_type=feedforward
```

### 技術実装
- **アンサンブル検出**：複数モデル構造の識別
- **手法分析**：バギング、ブースティング、スタッキング
- **多様性スコア**：モデル間の差異評価

## メモリ最適化と性能

### lawkitパターン統合
- **増分統計**：Welfordのアルゴリズム
- **ストリーミング処理**：大規模モデル対応
- **メモリマッピング**：効率的なファイルアクセス

### 並列処理
- **マルチスレッド**：独立分析の並列実行
- **バッチ処理**：設定可能なバッチサイズ
- **早期終了**：不適用分析のスキップ

## エラーハンドリング

### 堅牢性パターン
- **適切な劣化**：部分分析継続
- **フォールバック値**：分析失敗時のデフォルト
- **検証**：入力データの健全性チェック

### デバッグサポート
- **詳細ログ**：`--verbose`オプション
- **エラー分類**：具体的なエラーメッセージ
- **パフォーマンス追跡**：実行時間とメモリ使用量

## カスタマイゼーション

### 設定オプション
```rust
pub struct AnalysisOptions {
    pub weight_threshold: f64,      // デフォルト: 0.01
    pub gradient_threshold: f64,    // デフォルト: 1e-7
    pub convergence_window: usize,  // デフォルト: 10
    pub enable_all: bool,           // デフォルト: true
}
```

### 機能制御
- 個別機能の有効/無効化
- カスタムしきい値設定
- 出力形式のカスタマイゼーション

## 関連項目

- **[ML分析概要](../ml-analysis_ja.md)** - ユーザー向け概要
- **[APIリファレンス](api-reference_ja.md)** - プログラミングインターフェース
- **[CLIリファレンス](cli-reference_ja.md)** - コマンドライン使用法