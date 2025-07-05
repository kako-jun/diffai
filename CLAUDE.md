# diffai の思想（Philosophy）
「AI/MLデータの差分を、誰でも、どこでも、簡単に」

diffaiは、AI・機械学習分野でのデータ差分抽出に特化したツールです。
従来の diff はテキストベースで、構造を理解できない。
diffai は JSON/YAML/TOMLなどの構造化データに特化し、AI/MLのデータセット、設定ファイル、モデルパラメータの変更を明確に可視化する。

## 名前「diffai」に込めた意味
diff + ai の「ai」は：

- **AI/ML focused** AI・機械学習に特化
- **intelligent** 賢い差分抽出
- **automated** 自動化された分析
- **accessible** 誰でもアクセス可能
- **advanced** 高度な機能

## diffaiとdiffxの関係
diffaiは、diffxプロジェクトの兄弟プロジェクトです。
diffxは汎用的な構造化データ差分ツールであり、diffaiはAI/ML分野に特化したバージョンです。

### 分離の理由
- **対象分野の特化**: diffxは汎用、diffaiはAI/ML特化
- **機能の最適化**: AI/MLデータに特化した機能追加
- **開発の独立性**: それぞれの分野に最適化した開発

## 技術仕様
diffxの技術基盤をそのまま継承し、以下の追加機能を予定：

- **モデルパラメータ差分**: 深層学習モデルの重み変更追跡
- **データセット差分**: 訓練データの変更検出
- **実験設定差分**: ハイパーパラメータの変更追跡
- **評価指標差分**: モデル性能の変化可視化

## 対応フォーマット
- JSON（設定ファイル、APIレスポンス）
- YAML（MLOps設定、Kubernetes設定）
- TOML（プロジェクト設定）
- XML（レガシーシステム連携）
- INI（設定ファイル）
- CSV（データセット、実験結果）

## 将来的な展望
- **MLOps統合**: CI/CDパイプラインでの自動検証
- **実験追跡**: MLflow、Weights & Biasesとの連携
- **モデルバージョニング**: Git LFS、DVC統合
- **可視化**: Jupyter Notebook、Tensorboard連携

---

# TODO: AI/ML特化機能の実装計画

## 🎯 フェーズ1: モデルファイル比較 (優先度: 高)

### 対象ファイル形式
- **PyTorch**: `.pth`, `.pt` ファイル
- **TensorFlow**: `.pb`, `.h5`, `.keras` ファイル  
- **ONNX**: `.onnx` ファイル
- **Safetensors**: `.safetensors` ファイル
- **Pickle**: `.pkl` ファイル（scikit-learn等）

### 実装すべき機能
1. **テンソル形状比較**: モデル層の構造変更検出
2. **重み統計比較**: 平均値、標準偏差、分布の差分
3. **アーキテクチャ差分**: レイヤー追加・削除・変更の検出
4. **メモリ使用量比較**: モデルサイズの変化追跡

### ニーズ・使用場面
- **モデル最適化**: 量子化・プルーニング前後の比較
- **ファインチューニング**: 事前学習モデルからの変化量測定
- **A/Bテスト**: 異なるアーキテクチャの性能比較
- **デバッグ**: 意図しない重み変更の検出

## 🎯 フェーズ2: 実験結果比較 (優先度: 高)

### 対象データ
- **メトリクス**: accuracy, loss, F1-score, AUC等
- **ハイパーパラメータ**: learning_rate, batch_size, epochs等  
- **学習履歴**: training_loss, validation_loss の時系列
- **実験メタデータ**: 実行時間、使用GPU、データセット版数

### 実装すべき機能
1. **統計的有意差判定**: メトリクス改善の信頼性評価
2. **学習曲線比較**: 収束性・過学習の可視化
3. **ハイパーパラメータ影響度**: 性能への寄与度分析
4. **実験再現性チェック**: 同条件での結果一致性確認

### ニーズ・使用場面
- **論文執筆**: 実験結果の客観的比較
- **プロダクション判定**: 新モデルのデプロイ可否決定
- **チューニング効率化**: 有効なハイパーパラメータ特定
- **異常検知**: 想定外の性能劣化の早期発見

## 🎯 フェーズ3: データセット比較 (優先度: 中)

### 対象データ形式
- **NumPy**: `.npy`, `.npz` 配列ファイル
- **Pandas**: `.parquet`, `.feather` データフレーム
- **HuggingFace**: datasets ライブラリ形式
- **画像データ**: メタデータ・統計量比較
- **テキストデータ**: トークン分布・語彙変化

### 実装すべき機能
1. **データドリフト検出**: 分布変化の統計的測定
2. **欠損値パターン**: 欠損データの変化追跡
3. **スキーマ変更**: カラム追加・削除・型変更
4. **サンプルサイズ影響**: データ量変化の性能への影響

### ニーズ・使用場面
- **データ品質管理**: 継続的なデータ品質監視
- **モデル再学習判定**: データ変化によるモデル更新要否
- **特徴量エンジニアリング**: 特徴量変更の効果測定
- **データバージョン管理**: データセット変更履歴の追跡

## 🎯 フェーズ4: MLOps統合 (優先度: 中)

### 統合対象
- **MLflow**: 実験管理プラットフォーム
- **Weights & Biases**: 実験追跡・可視化
- **Kubeflow**: Kubernetes上のML Pipeline
- **DVC**: データバージョン管理
- **Git LFS**: 大容量ファイル管理

### 実装すべき機能
1. **CI/CD統合**: 自動テスト・デプロイでの差分チェック
2. **アラート機能**: 閾値超過時の自動通知
3. **レポート生成**: 差分結果の自動レポート作成
4. **API連携**: 他ツールからのプログラマティック呼び出し

### ニーズ・使用場面
- **自動品質保証**: デプロイ前の自動検証
- **チーム協業**: 変更内容の共有・レビュー
- **監査対応**: 変更履歴の透明性確保
- **運用効率化**: 手動チェック作業の自動化

## 🚀 実装戦略

### 段階的アプローチ
1. **Phase 1**: PyTorchファイル対応（最もニーズが高い）
2. **Phase 2**: 実験結果JSON/YAML対応（既存技術で実装可能）
3. **Phase 3**: TensorFlow・ONNX対応（ライブラリ依存が複雑）
4. **Phase 4**: データセット対応（統計ライブラリ追加）

### 技術的課題
- **ライブラリ依存**: PyTorch/TensorFlow等の重い依存関係
- **メモリ効率**: 大容量モデルファイルの処理最適化
- **精度問題**: 浮動小数点数の差分判定基準
- **バイナリ対応**: テキストベースでないファイル形式

---

# 将来的な拡張案（メモ）

## diffaiが独立したツールとして必要な理由

### 1. AI/ML特化の引数を自由に追加できる
- `--tolerance-per-layer`: レイヤーごとの許容誤差設定
- `--ignore-optimizer-state`: オプティマイザの状態を無視
- `--compare-gradients`: 勾配情報も比較対象に
- `--quantization-aware`: 量子化を考慮した比較
- `--model-type`: モデルタイプ（CNN、Transformer等）に応じた比較ロジック

### 2. MLワークフロー専用機能
- 実験結果の統計的有意差判定（t検定、ウィルコクソン検定等）
- ハイパーパラメータの影響度分析（SHAP値、重要度ランキング）
- データドリフト検出（KLダイバージェンス、コルモゴロフ・スミルノフ検定）
- 学習曲線の比較（収束速度、過学習の検出）

### 3. MLOpsツールとの深い統合
- 実験管理ツールとの連携（実験間の差分を自動記録）
- モデルレジストリとの統合（バージョン間の差分追跡）
- CI/CDパイプラインでのモデル検証自動化
- アラート機能（性能劣化の自動検知）

### 4. パフォーマンス最適化
- 大規模モデル（数GB〜数十GB）の効率的な処理
- GPUメモリを考慮した比較アルゴリズム
- 分散処理対応（複数GPU/ノードでの並列比較）
- ストリーミング処理（メモリに収まらないモデルの比較）

### 5. 研究・開発の自由度
- diffxの汎用性を損なわずに実験的機能を追加
- AI/ML分野の急速な進化に迅速に対応
- 研究者向けの高度な分析機能
- プラグインシステム（カスタム比較ロジックの追加）

---

# 🚨 diffxレベル完成度到達のためのTODO

## 現状分析（2025-01-05時点）
diffxと比較して、diffaiは以下の要素が不足しており、crates.ioリリース可能なレベルに到達していない：

## 📋 必須対応項目（優先度順）

### 1. **README.md の充実** 🔥🔥🔥
**現状**: ほぼ空っぽ（1行のみ）
**必要な内容**:
- プロジェクト概要とキャッチフレーズ
- バッジ（CI、crates.io、license等）
- AI/ML特化の価値提案
- クイックスタート例
- インストール方法
- 基本的な使用例（ML比較の実例）
- アーキテクチャ図
- 将来計画
- ライセンス情報

### 2. **docsディレクトリの作成** 🔥🔥🔥
**現状**: 存在しない
**必要な構造**:
```
docs/
├── user-guide/
│   ├── installation.md
│   ├── getting-started.md
│   ├── ml-model-comparison.md    # 🆕 AI/ML特化
│   └── examples.md
├── reference/
│   ├── cli-reference.md
│   ├── api-reference.md
│   └── ml-formats.md             # 🆕 サポート形式詳細
├── guides/
│   ├── integrations.md
│   └── mlops-workflow.md         # 🆕 MLOps統合
└── index.md
```

### 3. **LICENSEファイルの追加** 🔥🔥🔥
**現状**: 存在しない
**対応**: MIT LicenseまたはApache-2.0を選択して追加
**重要性**: crates.ioリリースの必須要件

### 4. **CONTRIBUTING.mdの作成** 🔥🔥
**現状**: 存在しない
**必要な内容**:
- 開発環境セットアップ
- テスト実行方法
- コード規約
- プルリクエストガイドライン
- AI/ML機能の拡張方法

### 5. **ML機能のテストケース追加** 🔥🔥
**現状**: PyTorch/Safetensors機能のテストが存在しない
**必要なテスト**:
- `parse_pytorch_model` 関数のテスト
- `parse_safetensors_model` 関数のテスト
- `diff_ml_models` 関数のテスト
- CLI経由でのML比較テスト
- エラーケース（不正ファイル等）のテスト

### 6. **examplesディレクトリの作成** 🔥🔥
**現状**: 存在しない
**必要な内容**:
```
examples/
├── ml-models/
│   ├── simple_comparison.rs      # 🆕 基本的なモデル比較
│   ├── finetuning_analysis.rs    # 🆕 ファインチューニング分析
│   └── quantization_impact.rs    # 🆕 量子化影響評価
├── integration/
│   ├── mlflow_integration.rs     # 🆕 MLflow連携例
│   └── ci_cd_pipeline.rs         # 🆕 CI/CD統合例
└── README.md
```

### 7. **crates.ioリリース準備** 🔥🔥
**現状**: メタデータが不完全
**必要な対応**:
- `diffai-core/Cargo.toml` の description 最適化
- keywords の AI/ML関連追加（"ai", "ml", "pytorch", "safetensors"）
- categories の追加（"machine-learning", "development-tools"）
- homepage/repository/documentation URLの設定
- README.md の充実後に `readme = "../README.md"` 設定

## 📚 追加で必要な項目（中優先度）

### 8. **CHANGELOG.mdの作成** 🔥
- バージョン履歴の記録
- マイグレーションガイド
- 破壊的変更の明示

### 9. **CI/CDワークフローの改善** 🔥
- GitHub Actions でのテスト自動化
- リリース自動化
- マルチプラットフォームビルド

### 10. **パフォーマンステストの追加** 🔥
- 大容量モデルファイルでのベンチマーク
- メモリ使用量測定
- 他ツールとの比較

## 🎯 作業スケジュール提案

### Week 1: 基本リリース準備
- [x] README.md の充実
- [x] LICENSE ファイル追加
- [x] ML機能のテストケース追加

### Week 2: ドキュメント整備
- [ ] docs ディレクトリ作成
- [x] CONTRIBUTING.md 作成
- [x] examples ディレクトリ作成

### Week 3: リリース準備完了
- [ ] crates.io メタデータ最適化
- [ ] CHANGELOG.md 作成
- [ ] CI/CD ワークフロー改善

## 💡 diffaiの独自価値（実装完了後の強み）

1. **AI/ML特化**: PyTorch/Safetensors直接サポート
2. **統計分析**: テンソル統計の自動計算・比較
3. **MLOps統合**: 実験管理ツールとの連携前提設計
4. **拡張性**: AI分野の急速な進化に対応できる設計
5. **研究支援**: 学術研究でのモデル比較に最適化

これらの項目を順次実装することで、diffaiはdiffxと同等以上の完成度を持つ、AI/ML特化の差分ツールとして確立される。

---

# 具体的な使用場面と便利機能の提案

## 🔍 モデル開発フェーズでの活用

### 1. ファインチューニング効果の可視化
```bash
# 事前学習モデルとファインチューニング後の比較
diffai pretrained_model.safetensors finetuned_model.safetensors \
  --show-layer-impact --sort-by-change-magnitude
```
**提案機能**: 
- `--show-layer-impact`: レイヤーごとの変更量をヒートマップで表示
- `--sort-by-change-magnitude`: 変更量の大きい順にソート

### 2. 量子化・プルーニング前後の比較
```bash
# 量子化による精度への影響を分析
diffai model_fp32.safetensors model_int8.safetensors \
  --quantization-analysis --precision-loss-threshold 0.01
```
**提案機能**:
- `--quantization-analysis`: 量子化による精度劣化の詳細分析
- `--precision-loss-threshold`: 許容できる精度低下の閾値設定

### 3. 学習途中のチェックポイント比較
```bash
# エポックごとの学習進捗を追跡
diffai checkpoint_epoch_10.pt checkpoint_epoch_20.pt \
  --learning-progress --convergence-analysis
```
**提案機能**:
- `--learning-progress`: 学習の進捗状況を可視化
- `--convergence-analysis`: 収束状態の分析

## 🚀 MLOpsでの活用

### 4. A/Bテストでのモデル比較
```bash
# 本番環境でのA/Bテスト用モデル比較
diffai model_a.safetensors model_b.safetensors \
  --deployment-readiness --performance-impact-estimate
```
**提案機能**:
- `--deployment-readiness`: デプロイメント準備状況のチェック
- `--performance-impact-estimate`: 性能影響の推定

### 5. 継続的インテグレーション
```bash
# CI/CDパイプラインでの自動検証
diffai production_model.safetensors candidate_model.safetensors \
  --regression-test --alert-on-degradation \
  --output-format junit-xml
```
**提案機能**:
- `--regression-test`: 回帰テスト実行
- `--alert-on-degradation`: 性能劣化時のアラート
- `--output-format junit-xml`: CI/CDツールとの統合

## 📊 研究・実験での活用

### 6. アーキテクチャ比較実験
```bash
# 異なるアーキテクチャの比較（ResNet vs EfficientNet）
diffai resnet_model.safetensors efficientnet_model.safetensors \
  --architecture-comparison --param-efficiency-analysis
```
**提案機能**:
- `--architecture-comparison`: アーキテクチャの構造的差異分析
- `--param-efficiency-analysis`: パラメータ効率の比較

### 7. ハイパーパラメータチューニング
```bash
# 異なるハイパーパラメータでの学習結果比較
diffai model_lr_0.01.safetensors model_lr_0.001.safetensors \
  --hyperparameter-impact --learning-rate-analysis
```
**提案機能**:
- `--hyperparameter-impact`: ハイパーパラメータの影響度分析
- `--learning-rate-analysis`: 学習率の影響を詳細分析

## 🛠️ デバッグ・トラブルシューティング

### 8. 異常検知
```bash
# 学習中の異常（gradient explosion, vanishing gradient）検出
diffai model_before_anomaly.pt model_after_anomaly.pt \
  --anomaly-detection --gradient-analysis
```
**提案機能**:
- `--anomaly-detection`: 異常パターンの自動検出
- `--gradient-analysis`: 勾配の分析（爆発・消失の検出）

### 9. メモリ使用量の最適化
```bash
# モデルサイズとメモリ使用量の比較
diffai lightweight_model.safetensors heavy_model.safetensors \
  --memory-analysis --inference-speed-estimate
```
**提案機能**:
- `--memory-analysis`: メモリ使用量の詳細分析
- `--inference-speed-estimate`: 推論速度の推定

## 🎯 チーム開発での活用

### 10. コードレビュー支援
```bash
# プルリクエストでのモデル変更レビュー
diffai main_branch_model.safetensors feature_branch_model.safetensors \
  --review-friendly --change-summary --risk-assessment
```
**提案機能**:
- `--review-friendly`: レビュアーに優しい出力形式
- `--change-summary`: 変更内容の要約
- `--risk-assessment`: 変更に伴うリスク評価

### 11. 実験記録・文書化
```bash
# 実験結果の自動文書化
diffai baseline_model.safetensors improved_model.safetensors \
  --generate-report --markdown-output --include-charts
```
**提案機能**:
- `--generate-report`: 実験レポートの自動生成
- `--markdown-output`: Markdown形式での出力
- `--include-charts`: グラフ・チャートの生成

## 🔬 高度な分析機能

### 12. 分散表現の比較
```bash
# 埋め込み層の分散表現比較（類似度、クラスタリング）
diffai model_v1.safetensors model_v2.safetensors \
  --embedding-analysis --similarity-matrix --clustering-change
```
**提案機能**:
- `--embedding-analysis`: 埋め込み層の詳細分析
- `--similarity-matrix`: 類似度行列の可視化
- `--clustering-change`: クラスタリング結果の変化

### 13. 注意機構の比較（Transformer系）
```bash
# アテンションヘッドの比較
diffai transformer_v1.safetensors transformer_v2.safetensors \
  --attention-analysis --head-importance --attention-pattern-diff
```
**提案機能**:
- `--attention-analysis`: アテンション機構の分析
- `--head-importance`: アテンションヘッドの重要度
- `--attention-pattern-diff`: アテンションパターンの差分

### 競合分析
- **既存ツール不足**: AI/ML特化の差分ツールは少ない
- **ニッチ市場**: 特定分野への特化で差別化可能
- **オープンソース優位**: 研究・教育分野での採用促進
- **企業ニーズ**: MLOpsの成熟に伴う需要増加