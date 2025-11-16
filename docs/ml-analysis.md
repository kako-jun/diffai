# ML分析機能

diffaiは、PyTorch（.pt/.pth）またはSafetensors（.safetensors）ファイルを比較する際に、**11種類の専門的なML分析機能**を自動的に実行します。設定は不要で、設定より規約の原則に従います。

## 自動実行

### ML分析が実行される場合
- **PyTorchファイル（.pt/.pth）**：11の分析すべてが自動的に実行
- **Safetensorsファイル（.safetensors）**：11の分析すべてが自動的に実行
- **NumPy/MATLABファイル**：基本的なテンソル統計のみ
- **その他の形式**：diffx-coreによる標準的な構造比較

### ゼロ設定
```bash
# 11のML分析機能すべてが自動的に実行される
diffai baseline.safetensors finetuned.safetensors

# フラグは不要 - diffaiがAI/MLファイルを検出し、包括的な分析を実行
```

## 11のML分析機能

### 1. 学習率分析
**目的**：学習率の変化と訓練動態を追跡

**出力例**：
```bash
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%, trend=increasing
```

**検出内容**：
- 学習率パラメータの変化
- 訓練スケジュールの調整
- 適応的学習率の変更
- トレンド分析（増加/減少/安定）

### 2. オプティマイザ比較
**目的**：オプティマイザの状態とモメンタム情報を比較

**出力例**：
```bash
optimizer_comparison: type=Adam, momentum_change=+2.1%, state_evolution=stable
```

**検出内容**：
- オプティマイザタイプ（Adam、SGD、AdamW、RMSprop）
- モメンタムバッファの変化
- ベータパラメータの進化
- オプティマイザ状態の一貫性

### 3. 損失追跡
**目的**：損失関数の進化と収束パターンを分析

**出力例**：
```bash
loss_tracking: loss_trend=decreasing, improvement_rate=15.2%, convergence_score=0.89
```

**検出内容**：
- 損失トレンドの方向
- 改善率
- 収束インジケータ
- 訓練の安定性

### 4. 精度追跡
**目的**：精度の変化とパフォーマンス指標を監視

**出力例**：
```bash
accuracy_tracking: accuracy_delta=+3.2%, performance_trend=improving
```

**検出内容**：
- 精度/F1/適合率/再現率の変化
- パフォーマンストレンド分析
- 指標改善率
- 複数指標サポート

### 5. モデルバージョン分析
**目的**：モデルのバージョニングとチェックポイント情報を特定

**出力例**：
```bash
model_version_analysis: version_change=1.0->1.1, checkpoint_evolution=incremental
```

**検出内容**：
- バージョン番号の変化
- チェックポイントの進行
- エポック/イテレーション追跡
- セマンティック vs 数値バージョニング

### 6. 勾配分析
**目的**：勾配フロー、勾配消失/爆発、安定性を分析

**出力例**：
```bash
gradient_analysis: flow_health=healthy, norm=0.021069, variance_change=+15.3%
```

**検出内容**：
- 勾配フローの健全性（healthy/warning/critical）
- 勾配消失の検出（< 1e-7）
- 勾配爆発の検出（> 100）
- 勾配の分散と安定性
- lawkitのメモリ効率的な増分統計を使用

### 7. 量子化分析
**目的**：混合精度（FP32/FP16/INT8/INT4）と圧縮効果を検出

**出力例**：
```bash
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%
```

**検出内容**：
- 混合精度の使用（FP32、FP16、INT8、INT4）
- 圧縮率
- 精度損失の推定
- モデル全体の量子化カバレッジ
- メモリ効率の向上

### 8. 収束分析
**目的**：学習曲線分析、プラトー検出、最適化軌道

**出力例**：
```bash
convergence_analysis: status=converging, stability=0.92, plateau_detected=false
```

**検出内容**：
- 収束ステータス（converging/converged/diverging）
- 学習曲線パターン
- 訓練中のプラトー検出
- 安定性スコア（0.0-1.0）
- 最適化軌道の健全性

### 9. 活性化分析
**目的**：活性化関数の使用と分布を分析

**出力例**：
```bash
activation_analysis: relu_usage=45%, gelu_usage=55%, distribution=healthy
```

**検出内容**：
- 活性化関数タイプ（ReLU、GELU、Tanh、Sigmoid、Swish）
- レイヤー間の使用分布
- 飽和リスク評価
- 死んだニューロンの検出
- 現代的な活性化サポート

### 10. 注意機構分析
**目的**：トランスフォーマーと注意機構を分析

**出力例**：
```bash
attention_analysis: head_count=12, attention_patterns=stable, efficiency=0.87
```

**検出内容**：
- マルチヘッド注意構造
- 注意パターンの安定性
- トランスフォーマーコンポーネントの識別
- 注意効率スコア
- BERT/GPT/T5アーキテクチャの認識

### 11. アンサンブル分析
**目的**：アンサンブルモデル構造を検出・分析

**出力例**：
```bash
ensemble_analysis: ensemble_detected=false, model_type=feedforward
```

**検出内容**：
- アンサンブルモデルの検出
- コンポーネントモデルの計数
- アンサンブル手法（バギング、ブースティング、スタッキング）
- モデル多様性スコア
- 単一 vs 複数モデル分類

## 出力形式

### CLI出力（デフォルト）
カラーコーディングと直感的なシンボルを使用した人間が読みやすい形式。

### JSON出力（MLOps統合）
```bash
diffai model1.safetensors model2.safetensors --output json
```

```json
{
  "learning_rate_analysis": {
    "old": 0.001,
    "new": 0.0015,
    "change": "+50.0%",
    "trend": "increasing"
  },
  "gradient_analysis": {
    "flow_health": "healthy",
    "gradient_norm": 0.021069,
    "variance_change": "+15.3%"
  }
  // ... 11の分析すべてが含まれる
}
```

### YAML出力（レポート）
```bash
diffai model1.safetensors model2.safetensors --output yaml
```

## 技術実装

### メモリ効率
- **lawkitパターン**：Welfordのアルゴリズムを使用した増分統計
- **ストリーミング処理**：大規模モデル分析用
- **diffx-core基盤**：実証済みのdiffエンジンの信頼性

### エラーハンドリング
- **緩やかな劣化**：特定のパターンが見つからない場合でも継続
- **堅牢な解析**：様々なモデルファイル構造に対応
- **フォールバック機構**：分析が完了できない場合のデフォルト値

### パフォーマンス最適化
- **早期終了**：データパターンが検出されない場合は分析をスキップ
- **バッチ処理**：大規模モデルパラメータの効率的な処理
- **メモリ制限**：大きなファイルの自動最適化

## ユースケース

### 研究開発
訓練進捗の監視、収束問題の検出、アーキテクチャ変更の分析。

### MLOps & CI/CD
自動モデル検証、回帰検出、パフォーマンス監視。

### モデル最適化
量子化分析、メモリ使用量追跡、圧縮評価。

### 実験追跡  
モデル変種の比較、ハイパーパラメータ効果の追跡、改善の検証。

## 関連項目

- **[クイックスタート](quick-start_ja.md)** - 5分で始める
- **[APIリファレンス](reference/api-reference_ja.md)** - コードで使用
- **[使用例](examples/)** - 実際の使用例と出力
- **[技術詳細](reference/ml-analysis-detailed_ja.md)** - 実装の詳細