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

## 🚀 高度なML分析機能（28機能実装済み）

### 🔬 学習・収束分析 (4機能)
- `--learning-progress`: 学習進捗追跡
- `--convergence-analysis`: 収束状態分析
- `--anomaly-detection`: 異常検知（勾配爆発・消失）
- `--gradient-analysis`: 勾配分析

### 🏗️ アーキテクチャ・性能分析 (4機能)
- `--architecture-comparison`: モデル構造比較
- `--param-efficiency-analysis`: パラメータ効率分析
- `--memory-analysis`: メモリ使用量分析
- `--inference-speed-estimate`: 推論速度推定

### 🔧 MLOps・デプロイ支援 (7機能)
- `--deployment-readiness`: デプロイ準備評価
- `--regression-test`: 回帰テスト
- `--risk-assessment`: リスク評価
- `--hyperparameter-impact`: ハイパーパラメータ影響分析
- `--learning-rate-analysis`: 学習率効果分析
- `--alert-on-degradation`: 性能劣化アラート
- `--performance-impact-estimate`: 性能影響推定

### 📊 実験・文書化支援 (4機能)
- `--generate-report`: 実験レポート自動生成
- `--markdown-output`: Markdown形式出力
- `--include-charts`: チャート・可視化組み込み
- `--review-friendly`: レビュー向け出力

### 🧠 高度分析機能 (6機能)
- `--embedding-analysis`: 埋め込み層変化解析
- `--similarity-matrix`: 類似度行列生成
- `--clustering-change`: クラスタリング変化解析
- `--attention-analysis`: アテンション機構分析（Transformer）
- `--head-importance`: アテンションヘッド重要度分析
- `--attention-pattern-diff`: アテンションパターン比較

### ⚡ その他の分析機能 (3機能)
- `--quantization-analysis`: 量子化分析
- `--sort-by-change-magnitude`: 変化量ソート
- `--change-summary`: 変更詳細サマリー

## 💡 使用例

### 基本的なモデル比較
```bash
# Safetensorsモデル間の比較
diffai model_v1.safetensors model_v2.safetensors --stats

# NumPy科学データ比較
diffai data_v1.npy data_v2.npy --stats

# MATLAB工学データ比較
diffai experiment_v1.mat experiment_v2.mat --stats

# 高度な分析
diffai baseline.safetensors improved.safetensors \
  --learning-progress --convergence-analysis --deployment-readiness

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
- [x] **Safetensors完全対応**: バイナリ解析・統計計算・28ML分析機能
- [x] **PyTorch完全対応**: Candle統合・多次元テンソル・flatten_all()修正
- [x] **diffx-core統合**: 外部CLI依存除去・自立動作・レガシーコード削除
- [x] **28のML分析機能**: 学習分析・アーキテクチャ比較・MLOps支援・文書化支援

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
- **HDF5サポート**: 科学データ拡張（システム要件解決後）
- **可視化機能**: グラフ・チャート生成・インタラクティブ表示
- **Python/Node.js bindings**: 多言語対応・API統合
- **リアルタイム監視**: モデルドリフト・性能劣化検出

**注**: 基本的な勾配解析・アテンション解析・埋め込み解析は28のML分析機能として既に実装済み

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
- **28のML分析機能**: 量子化・転移学習・実験再現性・アンサンブル分析

### v0.2.0-0.2.4 (2025-01-06-08): 基盤構築
- **crates.io公開**: https://crates.io/crates/diffai
- **PyTorch完全サポート**: Candle統合・実モデル検証環境
- **Safetensorsアライメント修正**: 手動バイト変換・HuggingFace対応

---

**diffai は AI/ML + 科学計算分野で最も包括的なデータ差分ツールです。28のML分析機能と科学データ対応（NumPy/MATLAB）により、研究開発からMLOpsまで幅広い用途に対応。純粋Rust実装でシステム依存なし、あらゆる環境での安定動作を実現します。**