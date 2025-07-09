# ML分析機能

diffaiの28の高度な機械学習分析機能について詳しく説明します。

## 概要

diffaiは、機械学習モデルの比較と分析のために特別に設計された28の専門機能を提供します。これらの機能は、研究開発からMLOps、デプロイメントまで幅広い用途をカバーします。

## 学習・収束分析 (4機能)

### 1. `--learning-progress` 学習進捗追跡
モデルのチェックポイント間での学習進捗を追跡・分析します。

**使用例**:
```bash
diffai checkpoint_epoch_10.safetensors checkpoint_epoch_20.safetensors --learning-progress
```

**出力例**:
```
+ learning_progress: trend=improving, magnitude=0.0543, speed=0.80
```

**分析項目**:
- **trend**: `improving`, `degrading`, `stable`
- **magnitude**: 変化の大きさ (0.0-1.0)
- **speed**: 収束速度 (0.0-1.0)

### 2. `--convergence-analysis` 収束分析
モデルの安定性と収束状況を評価します。

**使用例**:
```bash
diffai model_before.safetensors model_after.safetensors --convergence-analysis
```

**出力例**:
```
+ convergence_analysis: status=stable, stability=0.0234, action="Continue training"
```

**分析項目**:
- **status**: `converged`, `diverging`, `oscillating`, `stable`
- **stability**: パラメータ変化の分散 (低い=安定)
- **action**: 推奨される次のステップ

### 3. `--anomaly-detection` 異常検知
学習中の異常パターンを検出します。

**使用例**:
```bash
diffai normal_model.safetensors anomalous_model.safetensors --anomaly-detection
```

**出力例**:
```
[CRITICAL] anomaly_detection: type=gradient_explosion, severity=critical, affected=2 layers
```

**検出される異常**:
- **gradient_explosion**: 勾配爆発
- **gradient_vanishing**: 勾配消失
- **weight_distribution_shift**: 重み分布の異常
- **nan_inf_values**: NaN/Inf値の検出

### 4. `--gradient-analysis` 勾配分析
勾配の特性と流れを分析します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --gradient-analysis
```

**出力例**:
```
gradient_analysis: flow_health=healthy, norm=0.000156, ratio=0.8456
```

**分析項目**:
- **flow_health**: `healthy`, `exploding`, `dead`, `diminishing`
- **norm**: 勾配ノルムの推定
- **ratio**: 勾配比率

## アーキテクチャ・性能分析 (4機能)

### 5. `--architecture-comparison` アーキテクチャ比較
モデル構造と設計の比較を行います。

**使用例**:
```bash
diffai resnet.safetensors transformer.safetensors --architecture-comparison
```

**出力例**:
```
architecture_comparison: type1=cnn, type2=transformer, depth=50→24, differences=15
```

**分析項目**:
- **Architecture types**: CNN, RNN, Transformer, MLP等
- **Layer differences**: 追加、削除、変更されたレイヤー
- **Parameter distribution**: パラメータの配分

### 6. `--param-efficiency-analysis` パラメータ効率分析
パラメータの効率性を分析します。

**使用例**:
```bash
diffai baseline.safetensors optimized.safetensors --param-efficiency-analysis
```

**出力例**:
```
param_efficiency_analysis: efficiency_ratio=0.8456, utilization=0.92, pruning_potential=0.15
```

**分析項目**:
- **efficiency_ratio**: 効率比率
- **utilization**: パラメータ使用率
- **pruning_potential**: プルーニング可能性

### 7. `--memory-analysis` メモリ分析
メモリ使用量と最適化機会を分析します。

**使用例**:
```bash
diffai small_model.safetensors large_model.safetensors --memory-analysis
```

**出力例**:
```
memory_analysis: delta=+2.7MB, gpu_est=4.5MB, efficiency=0.25
```

**分析項目**:
- **delta**: メモリ差異
- **gpu_est**: 推定GPU メモリ要件
- **efficiency**: パラメータ/MB比率

### 8. `--inference-speed-estimate` 推論速度推定
推論速度と性能特性を推定します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --inference-speed-estimate
```

**出力例**:
```
inference_speed_estimate: speed_ratio=1.25x, flops_ratio=1.15x, bottlenecks=2
```

**分析項目**:
- **speed_ratio**: 速度比率
- **flops_ratio**: FLOPS比率
- **bottlenecks**: ボトルネックレイヤー数

## MLOps・デプロイ支援 (7機能)

### 9. `--deployment-readiness` デプロイ準備評価
デプロイの準備状況を評価します。

**使用例**:
```bash
diffai production.safetensors candidate.safetensors --deployment-readiness
```

**出力例**:
```
[GRADUAL] deployment_readiness: readiness=0.75, strategy=gradual, risk=medium
```

**評価項目**:
- **readiness**: 準備度スコア (0.0-1.0)
- **strategy**: `full`, `gradual`, `hold`
- **risk**: `low`, `medium`, `high`

### 10. `--regression-test` 回帰テスト
自動回帰テストを実行します。

**使用例**:
```bash
diffai baseline.safetensors new_version.safetensors --regression-test
```

**出力例**:
```
[HIGH] regression_test: passed=false, degradation=15.2%, severity=high, failed=3 checks
```

**テスト項目**:
- **passed**: テスト通過状況
- **degradation**: 性能劣化率
- **severity**: 重要度レベル
- **failed_checks**: 失敗したチェック数

### 11. `--risk-assessment` リスク評価
デプロイリスクを評価します。

**使用例**:
```bash
diffai current.safetensors candidate.safetensors --risk-assessment
```

**出力例**:
```
[CRITICAL] risk_assessment: risk=critical, readiness=0.45, factors=5, rollback=difficult
```

**評価項目**:
- **risk**: `low`, `medium`, `high`, `critical`
- **readiness**: デプロイ準備度
- **factors**: リスク要因数
- **rollback**: ロールバック難易度

### 12. `--hyperparameter-impact` ハイパーパラメータ影響分析
ハイパーパラメータ変更の影響を分析します。

**使用例**:
```bash
diffai model_lr_001.safetensors model_lr_0001.safetensors --hyperparameter-impact
```

**出力例**:
```
hyperparameter_impact: lr_impact=0.0234, batch_impact=0.0156, convergence=0.0543
```

**分析項目**:
- **lr_impact**: 学習率の影響
- **batch_impact**: バッチサイズの影響
- **convergence**: 収束への影響

### 13. `--learning-rate-analysis` 学習率分析
学習率の効果と最適化を分析します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --learning-rate-analysis
```

**出力例**:
```
learning_rate_analysis: current_lr=0.000100, schedule=exponential, effectiveness=0.8456
```

**分析項目**:
- **current_lr**: 現在の学習率
- **schedule**: スケジュール種別
- **effectiveness**: 効果指標

### 14. `--alert-on-degradation` 性能劣化アラート
性能劣化の検出とアラートを発生させます。

**使用例**:
```bash
diffai baseline.safetensors new_model.safetensors --alert-on-degradation
```

**出力例**:
```
[ALERT] alert_on_degradation: triggered=true, type=performance, threshold_exceeded=2.10x
```

**アラート種別**:
- **performance**: 性能劣化
- **memory**: メモリ使用量増加
- **stability**: 安定性低下

### 15. `--performance-impact-estimate` 性能影響推定
変更による性能への影響を推定します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --performance-impact-estimate
```

**出力例**:
```
performance_impact_estimate: latency_change=+5.2%, throughput_change=-2.1%, confidence=0.85
```

**推定項目**:
- **latency_change**: レイテンシ変化
- **throughput_change**: スループット変化
- **confidence**: 信頼度

## 実験・文書化支援 (4機能)

### 16. `--generate-report` 実験レポート自動生成
包括的な分析レポートを自動生成します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --generate-report
```

**出力例**:
```
generate_report: type="comprehensive", findings=12, recommendations=5, confidence=0.92
```

**レポート種別**:
- **comprehensive**: 包括的レポート
- **summary**: 要約レポート
- **technical**: 技術レポート

### 17. `--markdown-output` Markdown形式出力
Markdown形式でレポートを出力します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --markdown-output
```

**出力例**:
```
markdown_output: sections=8, tables=3, charts=2, length=2456 chars
```

**出力要素**:
- **sections**: セクション数
- **tables**: テーブル数
- **charts**: チャート数

### 18. `--include-charts` チャート・可視化組み込み
可視化チャートを含めます。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --include-charts
```

**出力例**:
```
include_charts: types=["histogram", "scatter"], data_points=1024, complexity=medium
```

**チャート種別**:
- **histogram**: ヒストグラム
- **scatter**: 散布図
- **line**: 線グラフ
- **heatmap**: ヒートマップ

### 19. `--review-friendly` レビュー向け出力
人間のレビューに最適化された出力を生成します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --review-friendly
```

**出力例**:
```
review_friendly: impact=medium, approval=recommended, key_changes=5
```

**レビュー項目**:
- **impact**: 影響度評価
- **approval**: 承認推奨
- **key_changes**: 重要な変更数

## 高度分析機能 (6機能)

### 20. `--embedding-analysis` 埋め込み層変化解析
埋め込み層の変化と意味ドリフトを分析します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --embedding-analysis
```

**出力例**:
```
embedding_analysis: dim_change=512→768, semantic_drift=0.0234, similarity_preservation=0.8456
```

**分析項目**:
- **dim_change**: 次元変化
- **semantic_drift**: 意味ドリフト
- **similarity_preservation**: 類似度保持

### 21. `--similarity-matrix` 類似度行列生成
モデル比較の類似度行列を生成します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --similarity-matrix
```

**出力例**:
```
similarity_matrix: matrix_dims=768x768, clustering_coeff=0.8456, sparsity=0.1234
```

**分析項目**:
- **matrix_dims**: 行列次元
- **clustering_coeff**: クラスタリング係数
- **sparsity**: スパース性

### 22. `--clustering-change` クラスタリング変化解析
クラスタリングの変化を分析します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --clustering-change
```

**出力例**:
```
clustering_change: clusters=8→12, stability=0.7654, migrated=25, new=4
```

**分析項目**:
- **clusters**: クラスタ数変化
- **stability**: クラスタ安定性
- **migrated**: 移動した要素数
- **new**: 新しいクラスタ数

### 23. `--attention-analysis` アテンション機構分析
Transformerモデルのアテンション機構を分析します。

**使用例**:
```bash
diffai transformer1.safetensors transformer2.safetensors --attention-analysis
```

**出力例**:
```
attention_analysis: layers=12, pattern_changes=5, focus_shift=0.1234
```

**分析項目**:
- **layers**: アテンション層数
- **pattern_changes**: パターン変化数
- **focus_shift**: 焦点シフト

### 24. `--head-importance` アテンションヘッド重要度分析
アテンションヘッドの重要度を分析します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --head-importance
```

**出力例**:
```
head_importance: important_heads=8, prunable_heads=4, specializations=12
```

**分析項目**:
- **important_heads**: 重要ヘッド数
- **prunable_heads**: プルーニング可能ヘッド数
- **specializations**: 特化パターン数

### 25. `--attention-pattern-diff` アテンションパターン比較
アテンションパターンの違いを比較します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --attention-pattern-diff
```

**出力例**:
```
attention_pattern_diff: pattern=local→global, similarity=0.7654, span_change=0.1234
```

**分析項目**:
- **pattern**: パターン進化
- **similarity**: パターン類似度
- **span_change**: スパン変化

## その他の分析機能 (3機能)

### 26. `--quantization-analysis` 量子化分析
量子化の効果と効率性を分析します。

**使用例**:
```bash
diffai fp32.safetensors quantized.safetensors --quantization-analysis
```

**出力例**:
```
quantization_analysis: compression=75.0%, speedup=2.5x, precision_loss=2.0%, suitability=good
```

**分析項目**:
- **compression**: 圧縮率
- **speedup**: 速度向上
- **precision_loss**: 精度損失
- **suitability**: デプロイ適性

### 27. `--sort-by-change-magnitude` 変化量ソート
変化の大きさでソートして優先順位を付けます。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --sort-by-change-magnitude
```

**効果**:
- 最も大きな変化を最初に表示
- 重要な変更の特定が容易
- ノイズの削減

### 28. `--change-summary` 変更詳細サマリー
変更の詳細なサマリーを生成します。

**使用例**:
```bash
diffai model1.safetensors model2.safetensors --change-summary
```

**出力例**:
```
change_summary: layers_changed=15, magnitude=0.1234, patterns=3, most_changed=5
```

**サマリー項目**:
- **layers_changed**: 変更レイヤー数
- **magnitude**: 全体的な変化量
- **patterns**: 変化パターン数
- **most_changed**: 最大変更レイヤー数

## 機能の組み合わせ

### 訓練監視向け組み合わせ
```bash
diffai checkpoint_old.safetensors checkpoint_new.safetensors \
  --learning-progress \
  --convergence-analysis \
  --anomaly-detection
```

### 本番デプロイ向け組み合わせ
```bash
diffai current_prod.safetensors candidate.safetensors \
  --deployment-readiness \
  --risk-assessment \
  --regression-test
```

### 研究分析向け組み合わせ
```bash
diffai baseline.safetensors experiment.safetensors \
  --architecture-comparison \
  --embedding-analysis \
  --generate-report
```

### 量子化検証向け組み合わせ
```bash
diffai fp32.safetensors quantized.safetensors \
  --quantization-analysis \
  --memory-analysis \
  --performance-impact-estimate
```

## 設定とカスタマイズ

### 設定ファイル例
```toml
[ml_analysis]
enable_all = false
learning_progress = true
convergence_analysis = true
anomaly_detection = true

[thresholds]
critical_degradation = 0.1
warning_degradation = 0.05
memory_limit_mb = 1024
```

### 環境変数
```bash
export DIFFAI_ML_ANALYSIS_ALL=true
export DIFFAI_ANOMALY_THRESHOLD=0.01
export DIFFAI_CONVERGENCE_PATIENCE=10
```

## 関連項目

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドオプション
- [対応形式](formats_ja.md) - サポートされるファイル形式
- [出力形式](output-formats_ja.md) - 出力形式の詳細

