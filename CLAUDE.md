# diffai - AI/ML特化データ差分ツール

## 🎯 プロジェクト概要
**「AI/MLデータの差分を、誰でも、どこでも、簡単に」**

diffaiは、AI・機械学習分野でのデータ差分抽出に特化したツールです。
従来のdiffはテキストベースで構造を理解できませんが、diffaiはPyTorch/SafetensorsなどのMLモデルファイルを直接解析し、テンソル統計・アーキテクチャ変更・学習進捗を可視化します。

### diffxとの関係
- **diffx**: 汎用構造化データ差分ツール
- **diffai**: AI/ML特化版（diffxの兄弟プロジェクト）

## 📦 対応フォーマット

### 入力フォーマット
- **Safetensors** (.safetensors) ✅ 完全対応
- **PyTorch** (.pt, .pth) ✅ 完全対応（Candle統合）
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

### ✅ 完了済み（v0.2.5 - 2025-01-08）
- [x] **Phase 1完全動作確認**: PyTorch/Safetensorsパーシング、テンソル統計比較、CLI記号出力
- [x] **PyTorch多次元テンソル修正**: `flatten_all()`でrank error解決
- [x] **Phase 2完全実装**: 実験分析3機能（ハイパーパラメータ比較、学習曲線分析、統計的有意性検定）
- [x] **ROADMAPベース実装**: README仕様通りの段階的開発完了

### ✅ 技術的成果
- **Phase 1**: PyTorch (.pt/.pth) + Safetensors (.safetensors) 完全対応
- **Phase 2**: 実験分析3機能で科学的根拠のある比較
- **diffx形式スーパーセット**: 基本記号(`+`, `-`, `~`, `!`) + 拡張ML分析(`◦`)
- **色付き出力**: リスク・重要度別の視覚的表示
- **28のML分析機能**: 包括的モデル評価システム

### 🔮 将来計画（Phase 3以降）
- **Phase 3**: MLOps統合（MLflow、Weights & Biases、DVC）
- **Phase 4**: 高度分析（勾配解析、アテンション解析、埋め込み空間分析）
- **TensorFlow/ONNX対応**: 他フレームワークサポート
- **Python/Node.js bindings**: 多言語対応

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

# 📅 開発履歴（簡略版）

## 主要マイルストーン

### v0.2.5 (2025-01-08): Phase 1-2完了
- **Phase 1**: PyTorch/Safetensors完全対応、`flatten_all()`でrank error解決
- **Phase 2**: 実験分析3機能（ハイパーパラメータ比較、学習曲線分析、統計的有意性検定）
- **ROADMAPベース実装**: README仕様通りの段階的開発

### v0.2.3-0.2.4 (2025-01-08): アーキテクチャ改善
- **外部CLI依存除去**: diffx-core直接統合、完全自立動作
- **28のML分析機能**: 量子化、転移学習、実験再現性、アンサンブル分析
- **PyTorch多次元テンソル対応**: 全テンソル形状サポート

### v0.2.1-0.2.2 (2025-01-07): 基盤強化
- **PyTorch完全サポート**: Candleライブラリ統合
- **実モデル検証環境**: HuggingFaceモデル対応
- **Safetensorsアライメント修正**: 手動バイト変換

### v0.2.0 (2025-01-06): 初回リリース
- **crates.io公開**: https://crates.io/crates/diffai
- **13の基本ML分析機能**: 実装・テスト完了
- **diffx-core統合**: 効率的な基本差分処理

---

# 📋 次のタスク（優先度順）

## 🔥 高優先度（Phase 3準備）
1. **MLOps統合**: MLflow、Weights & Biases連携
2. **パフォーマンステスト**: 大容量モデルでのベンチマーク
3. **DVC互換性**: データ・モデルバージョニング対応

## 🔧 中優先度（Phase 4準備）
4. **勾配解析**: 訓練デバッグ機能
5. **アテンション解析**: Transformerモデル対応
6. **埋め込み空間分析**: 表現学習評価

## 🔮 長期計画
7. **TensorFlow/ONNX対応**: 他フレームワークサポート
8. **Python bindings**: PyO3でのPython統合
9. **可視化機能**: グラフ・チャート生成

---

**diffai は AI/ML分野で最も包括的なモデル比較ツールとして、28の分析軸で360度モデル評価を提供します。外部CLI依存を完全除去した自立動作アーキテクチャにより、あらゆる環境での安定動作を実現。PyTorch多次元テンソル対応・テスト完全通過・diffx-core統合により、研究開発からMLOpsまで幅広い用途に対応します。**