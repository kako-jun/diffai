# diffai タスク管理（リブート後）

## 🎯 現在の状況

**バージョン**: v0.3.17
**状態**: Rust コア完成、ML分析機能実装完了
**テスト**: 全67テスト通過（0 ignored）

## 🚀 リブート進捗

### ✅ Phase 0: 構造の簡素化（完了 2025-11-16）
- [x] モノレポから脱却（diffai-js, diffai-python を to-migrate/ に移動）
- [x] Cargo.toml をRustコアのみに簡素化
- [x] リブート計画書作成（.claude/reboot/）

### ✅ Phase 1: diffx-core 依存パスの更新（完了）
- [x] diffx-core を crates.io から取得（v0.6）
- [x] ビルド・テスト確認

### ✅ Phase 2: ML分析機能の実装（完了 2025-12-12）

**実装した11種類の DiffResult:**
- [x] TensorShapeChanged - テンソル形状変更
- [x] TensorStatsChanged - テンソル統計変更（mean, std, min, max）
- [x] TensorDataChanged - テンソルデータ変更
- [x] ModelArchitectureChanged - モデルアーキテクチャ変更
- [x] WeightSignificantChange - 重み有意変更
- [x] ActivationFunctionChanged - 活性化関数変更
- [x] LearningRateChanged - 学習率変更
- [x] OptimizerChanged - オプティマイザ変更
- [x] LossChange - 損失値変更
- [x] AccuracyChange - 精度変更
- [x] ModelVersionChanged - モデルバージョン変更

**テスト結果:**
- diffai-core: 29テスト通過
- fixtures: 4テスト通過
- diffai CLI（ユニット）: 3テスト通過
- diffai CLI（trycmd）: 8テスト通過

**修正したバグ:**
- TensorStatsChanged の重複出力（data_summary と別に stats 行が出る問題）
- 各種フォーマット対応（PyTorch, Safetensors, NumPy, MATLAB）

### 📝 Phase 3: ドキュメント更新（未開始）
- [ ] docs/specs/core.md の実装との整合性確認
- [ ] README.md の更新

### 🔧 Phase 4: CI/CD の簡素化（未開始）
- [ ] github-shared への依存を削除
- [ ] diffai 専用のシンプルな CI/CD に変更

## 📊 アーキテクチャ

### コンポーネント構成

**Rust コア（このリポジトリ）**:
- `diffai-core/` - コアライブラリ + ML分析
- `diffai-cli/` - CLI ツール

### ML分析モジュール構造

```
diffai-core/src/ml_analysis/
├── mod.rs           # モジュールエクスポート
├── activation.rs    # 活性化関数分析
├── batch_norm.rs    # バッチ正規化分析
├── complexity.rs    # モデル複雑性分析
├── gradient.rs      # 勾配分析
├── learning_rate.rs # 学習率分析
├── metrics.rs       # 訓練メトリクス（loss, accuracy, version, optimizer）
└── weight.rs        # 重み分析
```

## 📝 次のアクション

1. ドキュメント更新（Phase 3）
2. CI/CD 簡素化（Phase 4）
3. crates.io 公開準備

---

**最終更新**: 2025-12-12 Phase 2 完了
**メンテナ**: Claude Code
