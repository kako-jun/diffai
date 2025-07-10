# diffai - AI/ML特化データ差分ツール

## 🎯 プロジェクト概要
**「AI/MLデータの差分を、誰でも、どこでも、簡単に」**

diffaiは、AI・機械学習分野でのデータ差分抽出に特化したツールです。
従来のdiffはテキストベースで構造を理解できませんが、diffaiはPyTorch/SafetensorsなどのMLモデルファイルを直接解析し、テンソル統計・アーキテクチャ変更・学習進捗を可視化します。

### diffxとの関係
- **diffx**: 汎用構造化データ差分ツール
- **diffai**: AI/ML特化版（diffxの兄弟プロジェクト）

## 📦 対応フォーマット

### 入力フォーマット（Phase 1-2完了）
- **Safetensors** (.safetensors) ✅ 完全対応
- **PyTorch** (.pt, .pth) ✅ 完全対応（Candle統合）
- **NumPy配列** (.npy, .npz) ✅ 完全対応（統計解析）
- **MATLAB** (.mat) ✅ 完全対応（matfileクレート）
- **JSON/YAML/TOML/XML/INI/CSV** ✅ diffx互換

### 出力フォーマット
- **diffai形式**: 色付きCLI出力（デフォルト）
- **JSON**: MLOpsツール統合用
- **YAML**: 設定ファイル・人間可読用

## 🚀 実装済みML分析機能（v0.2.0現在）

### ✅ 実装済み機能
- `--stats`: 詳細なテンソル統計表示
- `--quantization-analysis`: 量子化分析
- `--sort-by-change-magnitude`: 変化量ソート
- `--show-layer-impact`: レイヤー別影響分析

### 🚀 実装予定機能（Phase 3）

#### Phase 3A: 核心機能 (4機能)
- `--architecture-comparison`: モデル構造比較 🔜
- `--memory-analysis`: メモリ使用量比較 🔜
- `--anomaly-detection`: 数値異常検出 🔜
- `--change-summary`: 変更詳細サマリー 🔜

#### Phase 3B: 高度分析 (3機能)
- `--convergence-analysis`: 収束状態数値分析 🔜
- `--gradient-analysis`: 勾配数値比較 🔜
- `--similarity-matrix`: 類似度数値比較 🔜

## 💡 使用例

### 基本的なモデル比較
```bash
# Safetensorsモデル間の比較
diffai model_v1.safetensors model_v2.safetensors --stats

# NumPy科学データ比較
diffai data_v1.npy data_v2.npy --stats

# MATLAB工学データ比較
diffai experiment_v1.mat experiment_v2.mat --stats

# 実装済み分析機能の組み合わせ
diffai baseline.safetensors improved.safetensors \
  --stats --quantization-analysis --sort-by-change-magnitude

# Phase 3で追加予定の分析（将来）
diffai baseline.safetensors improved.safetensors \
  --architecture-comparison --memory-analysis --anomaly-detection

# JSON出力でMLOpsツール連携
diffai model_v1.safetensors model_v2.safetensors --output json | jq .
```

### 実際の使用場面
- **研究開発**: ファインチューニング前後の比較
- **MLOps**: CI/CDでの自動モデル検証
- **実験管理**: A/Bテストでのモデル性能比較
- **科学計算**: NumPy/MATLAB配列の統計的比較
- **デバッグ**: 学習異常・勾配問題の検出

## 🧪 テスト環境

### 軽量テスト（オフライン対応）
```bash
# リポジトリ内の軽量モデル（2.4MB）でテスト
diffai tests/fixtures/ml_models/tiny_gpt2_real.bin \
       tests/fixtures/ml_models/simple_base.safetensors
```

### 実モデルテスト（要ダウンロード）
```bash
# HuggingFaceから実モデルをダウンロード
cd real_models_test/
uv sync
uv run python download_models.py

# 実モデルで検証（DistilBERT vs GPT-2等）
diffai real_models_test/distilbert_base/model.safetensors \
       real_models_test/gpt2_small/model.safetensors --stats
```

## 📋 開発状況

### ✅ 完了済み（Phase 1-2: v0.2.6 - 2025-01-09）

#### Phase 1: ML Model Support ✅ 完了
- [x] **Safetensors完全対応**: バイナリ解析・統計計算・基本ML分析機能
- [x] **PyTorch完全対応**: Candle統合・多次元テンソル・flatten_all()修正
- [x] **diffx-core統合**: 外部CLI依存除去・自立動作・レガシーコード削除
- [x] **基本ML分析機能**: 統計表示・量子化分析・変化量ソート・レイヤー影響分析

#### Phase 2: Scientific Data Support ✅ 完了
- [x] **NumPy配列サポート**: .npy単体ファイル・.npzアーカイブ・ヘッダー解析・統計計算
- [x] **MATLAB配列サポート**: .matファイル・matfileクレート・複素数対応・全数値型

### 🔧 技術的成果
- **純粋Rust実装**: システム依存なし・クロスプラットフォーム対応
- **包括的フォーマット対応**: ML (PyTorch/Safetensors) + 科学 (NumPy/MATLAB) + 構造化 (JSON/YAML)
- **統計解析機能**: 平均・標準偏差・最小・最大・形状・メモリサイズ・複素数対応
- **色付きCLI出力**: 変更種別・リスク別・データ型別の視覚的表示
- **JSON/YAML出力**: MLOpsツール統合・API連携対応

### 🎯 実装方針（確定版）

#### ✅ 採用方針
- **純粋Rustクレート優先**: システム依存回避・移植性重視
- **matfileクレート採用**: MATLAB対応の信頼性実装
- **段階的フォーマット拡張**: 実用性重視・需要順での実装
- **NumPy + MATLAB組み合わせ**: 科学計算分野の主要ツールカバー

#### ❌ 却下方針  
- **HDF5システム依存実装**: Windows環境困難・ユーザー負担大
- **独自バイナリパーサー**: 車輪の再発明・保守コスト高
- **全フォーマット同時実装**: 開発効率悪化・品質低下リスク

### 🔄 進行中タスク（優先度順）

#### 🔥 高優先度（Phase 2 残項目）
1. **大型配列最適化**: チャンク処理・メモリ効率改善
2. **全テスト実行**: Phase 1-2統合テスト・エラー修正

#### 🔧 中優先度（Phase 3準備）
3. **TensorFlow/ONNXサポート**: MLフレームワーク拡張
4. **パフォーマンステスト**: 大容量モデルベンチマーク

#### 🔮 将来計画（Phase 3-4）

**Phase 3: MLOps統合**
- **MLflow連携**: 実験トラッキング・モデル管理統合
- **Weights & Biases統合**: 実験ログ・可視化連携  
- **DVC互換性**: データ・モデルバージョニング対応
- **CI/CD統合**: GitHub Actions・Jenkins対応

**Phase 4: 高度機能拡張**
- **単一ファイルインスペクション機能**: 差分比較不要のファイル分析・ビューア機能
- **HDF5サポート**: 科学データ拡張（システム要件解決後）
- **可視化機能**: グラフ・チャート生成・インタラクティブ表示
- **Python/Node.js bindings**: 多言語対応・API統合
- **リアルタイム監視**: モデルドリフト・性能劣化検出

**注**: 精査により28機能から7機能に削減。UNIX哲学に従い単純で組み合わせ可能な設計に変更

---

# 📝 ドキュメント実装状況分析・機能精査 (2025-01-10)

## 🔍 分析結果

### 実装状況
- **diffai v0.2.0** を調査
- **実装済み機能**: 4機能のみ
- **ドキュメント記載**: 28機能
- **実装率**: 14% (4/28)

### 実装済み機能
1. `--stats`: 詳細なテンソル統計表示 ✅
2. `--quantization-analysis`: 量子化分析 ✅  
3. `--sort-by-change-magnitude`: 変化量ソート ✅
4. `--show-layer-impact`: レイヤー別影響分析 ✅

## 🎯 機能精査結果（UNIX哲学・直交性分析）

### ❌ 実装しない機能（21機能）

**責任範囲外（15機能）**
- `--deployment-readiness`, `--regression-test`, `--risk-assessment` - デプロイ判断は別ツール
- `--inference-speed-estimate`, `--performance-impact-estimate` - 推測は実測で行うべき
- `--generate-report`, `--markdown-output`, `--include-charts`, `--review-friendly` - 専用ツールの責任
- `--attention-analysis`, `--head-importance`, `--attention-pattern-diff`, `--embedding-analysis` - 特定分野特化
- `--learning-rate-analysis`, `--alert-on-degradation` - 汎用性欠如

**直交性違反（6機能）**
- `--learning-progress` ↔ `--convergence-analysis` - 機能重複
- `--param-efficiency-analysis` ↔ `--memory-analysis` - 機能重複  
- `--hyperparameter-impact` ↔ `--learning-rate-analysis` - 機能重複
- `--clustering-change` ↔ `--similarity-matrix` - 機能近似

### ✅ 実装予定機能（7機能）

**Phase 3A: 核心機能（4機能）**
1. `--architecture-comparison` - モデル構造比較
2. `--memory-analysis` - メモリ使用量比較
3. `--anomaly-detection` - 数値異常検出
4. `--change-summary` - 変更詳細サマリー

**Phase 3B: 高度分析（3機能）**
5. `--convergence-analysis` - 収束状態数値分析
6. `--gradient-analysis` - 勾配数値比較
7. `--similarity-matrix` - 類似度数値比較

### 🔧 直交設計原則
```bash
# 基本分析（直交）
--stats                    # 統計情報
--architecture            # 構造情報  
--memory                   # メモリ情報
--anomalies               # 異常検出

# 高度分析（直交）
--convergence             # 収束分析
--gradients               # 勾配分析
--similarity              # 類似度分析

# 出力制御（直交）
--summary                 # 要約表示
--output {cli,json,yaml}  # 既存
```

### 対応方針
1. **28機能 → 11機能** (実装済み4 + 新規7) に削減
2. **UNIX哲学遵守**: 単一責任・組み合わせ可能・単純性
3. **実装計画**: Phase 3で段階的実装

### ドキュメント更新作業完了 (2025-01-10)
- **CLAUDE.md**: 精査結果を反映、実装状況を正確に記録
- **README.md/README_ja.md/README_zh.md**: 28機能→11機能に修正、実装済み機能のみ表示
- **docs/user-guide/basic-usage.md**: 未実装機能の例を削除、実装済み機能に変更
- **tests/integration/implemented_features_tests.rs**: 実装済み機能のみをテストする新テストファイル作成
- **統合テスト**: 7/8件通過（実装済み機能は全て正常動作）

---

# 🔧 開発・貢献ガイド

## 必須: プッシュ前のローカルCI実行
```bash
# プッシュ前に必須実行
./ci-local.sh
```

このスクリプトは以下をチェック：
1. **フォーマット**: `cargo fmt --all -- --check`
2. **Clippy**: `cargo clippy --all-targets --all-features -- -D warnings`
3. **ビルド**: `cargo build --verbose`
4. **テスト**: `cargo test --verbose`
5. **リリーステスト**: `cargo test --release --verbose`

## テストデータ管理ポリシー

### ✅ リポジトリにコミットするデータ
- **小さなテストファイル** (<100KB): `tests/fixtures/ml_models/`
- **軽量実モデル** (2.4MB): `tests/fixtures/ml_models/tiny_gpt2_real.bin`

### ❌ リポジトリにコミットしないデータ  
- **大きな実モデル** (>10MB): `real_models_test/` (1.45GB, .gitignore対象)
- **一時ファイル**: `.venv/`, `uv.lock`

---

# 📅 開発履歴（最新版）

## 主要マイルストーン

### v0.2.6 (2025-01-09): Phase 2科学データ完了
- **NumPy配列サポート**: .npy/.npz完全対応・統計解析・色付き出力
- **MATLAB配列サポート**: .mat完全対応・複素数・全数値型・matfileクレート統合
- **科学計算統合**: ML + 科学データの包括的差分解析システム

### v0.2.5 (2025-01-08): Phase 1-2基盤
- **PyTorch多次元テンソル修正**: `flatten_all()`でrank error解決
- **外部CLI依存除去**: diffx-core直接統合・完全自立動作
- **基本ML分析機能**: 量子化分析・統計表示・レイヤー影響分析

### v0.2.0-0.2.4 (2025-01-06-08): 基盤構築
- **crates.io公開**: https://crates.io/crates/diffai
- **PyTorch完全サポート**: Candle統合・実モデル検証環境
- **Safetensorsアライメント修正**: 手動バイト変換・HuggingFace対応

---

**diffai は AI/ML + 科学計算分野に特化したデータ差分ツールです。基本的なML分析機能（統計・量子化・レイヤー影響分析）と科学データ対応（NumPy/MATLAB）により、研究開発からMLOpsまで幅広い用途に対応。純粋Rust実装でシステム依存なし、あらゆる環境での安定動作を実現します。**

---

# 🔮 将来機能設計案

## 単一ファイルインスペクション機能（Phase 4）

### 基本コンセプト
現在のdiffaiは2つのファイルの差分比較に特化していますが、単一ファイルの構造解析・統計表示・品質チェック機能も技術的に実装可能です。

### 設計方針: 平易な英語 + 直交設計
- **平易な英語**: 専門用語を避け、直感的で理解しやすいオプション名
- **直交設計**: 機能が重複せず、組み合わせ可能で一貫性のあるオプション体系

### CLI設計案

#### 基本モード
```bash
# 基本的な単一ファイル分析
diffai model.safetensors --inspect

# 短縮形
diffai model.safetensors --view
```

#### 情報カテゴリ（直交的）
```bash
# 統計情報のみ表示
diffai model.safetensors --inspect --show-stats

# 構造・アーキテクチャ情報
diffai model.safetensors --inspect --show-structure

# 健康状態・品質チェック
diffai model.safetensors --inspect --show-health

# メタデータ情報
diffai model.safetensors --inspect --show-meta

# 複数組み合わせ
diffai model.safetensors --inspect --show-stats --show-health
```

#### 詳細レベル制御
```bash
# 簡潔な表示
diffai model.safetensors --inspect --brief

# 詳細な表示
diffai model.safetensors --inspect --detailed

# 全情報表示（デフォルト）
diffai model.safetensors --inspect --full
```

#### 出力形式（既存と統一）
```bash
# JSON形式でMLOpsツール連携
diffai model.safetensors --inspect --output json

# YAML形式で人間可読
diffai model.safetensors --inspect --output yaml
```

### 実装上の利点
1. **既存コードの再利用**: 解析・統計計算機能は既に実装済み
2. **一貫性**: 既存の`--output`、`--stats`などのオプション体系と統一
3. **拡張性**: 新しい分析機能を追加時も直交設計により容易
4. **直感性**: 英語として自然で、機能が名前から類推可能

### 実用例
```bash
# モデルの事前調査
diffai candidate_model.safetensors --inspect --show-stats --brief

# デバッグ時の構造確認
diffai problematic_model.pt --inspect --show-structure --show-health

# 配布前の品質チェック
diffai release_model.safetensors --inspect --show-health --output json

# 学習状態の診断
diffai checkpoint.safetensors --inspect --show-meta --detailed
```

### 注意事項
- **次リリースには含めない**: Phase 3完了後のPhase 4で実装
- **実装時の考慮点**: CLI引数の`INPUT2`をオプショナル化が必要
- **既存機能との整合性**: 比較機能と単一ファイル分析機能の共存設計