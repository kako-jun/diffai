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
- **PyTorch** (.pt, .pth) 🔄 部分対応（エラーメッセージ改善済み）
- **JSON/YAML/TOML/XML/INI/CSV** ✅ diffx互換

### 出力フォーマット
- **diffai形式**: 色付きCLI出力（デフォルト）
- **JSON**: MLOpsツール統合用
- **YAML**: 設定ファイル・人間可読用

## 🚀 高度なML分析機能（13機能実装済み）

### 🔬 学習・収束分析
- `--learning-progress`: 学習進捗追跡
- `--convergence-analysis`: 収束状態分析
- `--anomaly-detection`: 異常検知（勾配爆発・消失）
- `--gradient-analysis`: 勾配分析

### 🏗️ アーキテクチャ・性能分析  
- `--architecture-comparison`: モデル構造比較
- `--param-efficiency-analysis`: パラメータ効率分析
- `--memory-analysis`: メモリ使用量分析
- `--inference-speed-estimate`: 推論速度推定

### 🔧 MLOps・デプロイ支援
- `--deployment-readiness`: デプロイ準備評価
- `--regression-test`: 回帰テスト
- `--risk-assessment`: リスク評価
- `--hyperparameter-impact`: ハイパーパラメータ影響分析

### 📊 実験・文書化支援
- `--generate-report`: 実験レポート自動生成

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

### ✅ 完了済み（v0.2.0）
- [x] **crates.io正式リリース**: https://crates.io/crates/diffai
- [x] **13の高度ML分析機能**: 全機能実装・テスト完了
- [x] **実モデル検証**: 5種類のHuggingFaceモデルで動作確認
- [x] **Safetensorsアライメント修正**: bytemuck → 手動変換
- [x] **uvベースダウンロード環境**: SSL対応・依存関係管理

### 🔄 進行中
- [ ] **パフォーマンステスト**: 大容量モデルでのベンチマーク
- [ ] **PyTorch完全サポート**: Candleライブラリ統合

### 🔮 将来計画
- **TensorFlow/ONNX対応**: 他フレームワークサポート
- **Python/Node.js bindings**: 多言語対応
- **可視化機能**: グラフ・チャート生成

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

# 📅 最新の改善履歴

## 2025-01-07: 実モデル検証環境構築とSafetensors修正 ✅

### 🎯 課題解決
1. **実HuggingFaceモデルでの動作検証**
2. **Safetensorsアライメントエラー修正**  
3. **オフライン環境での軽量テスト対応**

### ✅ 実装内容
- **uvベースダウンロード環境**: 5種類の実モデル自動取得
- **アライメント修正**: `bytemuck::cast_slice` → 手動バイト変換
- **軽量モデル追加**: Tiny GPT-2 (2.4MB) をリポジトリに追加
- **SSL対応**: 証明書問題回避でHuggingFaceアクセス

### 🎉 検証結果
```bash
# 実モデル間の比較成功例
./target/release/diffai real_models_test/distilbert_base/model.safetensors \
                         real_models_test/distilgpt2/model.safetensors \
                         --architecture-comparison --deployment-readiness

# 出力: 
🏗️ architecture_comparison: type1=transformer, type2=transformer, depth=105→82, differences=5
✅ deployment_readiness: readiness=1.00, strategy=full, risk=low
```

- ✅ **全13機能**: 実モデルで動作確認完了
- ✅ **アーキテクチャ比較**: BERT vs GPT構造差異検出  
- ✅ **統計分析**: 実テンソル統計計算成功

---

# 📋 次のタスク（優先度順）

## 🔥 高優先度
1. **パフォーマンステスト**: 大容量モデルでのメモリ効率・速度測定
2. **PyTorch完全サポート**: Candleライブラリでの.pt/.pth読み込み
3. **ドキュメント拡充**: 実用例・ベストプラクティス追加

## 🔧 中優先度  
4. **バイナリサイズ最適化**: 不要依存関係削除
5. **CI/CD改善**: リリース自動化・マルチプラットフォーム対応
6. **Python bindings**: PyO3でのPython統合

---

**diffai は AI/ML分野で最も包括的なモデル比較ツールとして、13の分析軸で360度モデル評価を提供します。**