# diffai タスク管理（リブート後）

## 🎯 現在の状況

**バージョン**: v0.3.16
**状態**: Rust コア（diffai-core + diffai-cli）は完成済み
**課題**: 複雑性の増大により、保守が困難になっている

## 🚀 リブート進捗

### ✅ Phase 0: 構造の簡素化（完了 2025-11-16）
- [x] モノレポから脱却（diffai-js, diffai-python を to-migrate/ に移動）
- [x] Cargo.toml をRustコアのみに簡素化
- [x] リブート計画書作成（.claude/reboot/）

### 🔄 Phase 1: diffx-core 依存パスの更新（待機中）
**ブロッカー**: diffx のリブート完了待ち

**待機理由**:
- diffai-core は diffx-core に依存（コード重複削減のため）
- diffx が現在リブート中で、パスが一時的に解決できない
- diffx-core が crates.io に公開されるか、新しいパスが確定するまで待機

**準備完了**:
- diffai-core/Cargo.toml:36 に依存定義済み
- 依存関係は継続（削除しない）

### 📝 Phase 2: ドキュメントの簡素化（進行中）
**目的**: 完璧主義による麻痺を解消

**実行中のタスク**:
- [x] tasks.md を簡潔なリストに書き換え（414行 → 約100行）
- [ ] 実装と乖離したドキュメントを特定
- [ ] 乖離ドキュメントを修正または to-migrate/deprecated/ に移動

**方針**:
- 「動く80%」で十分（完璧を求めない）
- 実装を検証してから記述（理想を書かない）
- diffx リブート戦略に準拠

### 🔧 Phase 3: CI/CD の簡素化（未開始）
**目的**: 複雑な共有システムからの脱却

**予定タスク**:
- [ ] github-shared への依存を削除
- [ ] diffai 専用のシンプルな CI/CD に変更
- [ ] 言語ごとに独立した GitHub Actions

## 📊 アーキテクチャ

### コンポーネント構成

**Rust コア（このリポジトリ）**:
- `diffai-core/` - コアライブラリ
- `diffai-cli/` - CLI ツール

**別リポジトリ（準備中）**:
- `to-migrate/diffai-js/` → https://github.com/kako-jun/diffai-js
- `to-migrate/diffai-python/` → https://github.com/kako-jun/diffai-python

### 依存関係

```
diffai-core
 ├─ diffx-core (path = "../../diffx/diffx-core")
 │   └─ 基本機能を再利用（コード重複削減）
 └─ AI/ML ライブラリ
     ├─ candle-core, candle-nn
     ├─ safetensors
     ├─ ndarray
     └─ matfile
```

## 🚨 既知の問題

### ビルドエラー（Phase 1 待ち）
```
error: failed to load manifest for dependency `diffx-core`
```
**原因**: diffx リブート中でパスが解決できない
**対応**: diffx リブート完了を待つ

### 過剰なドキュメント（Phase 2 対応中）
- 旧 tasks.md: 414行（to-migrate/deprecated/ に移動済み）
- 実装と乖離した理想的な記述が多数存在

## 📝 次のアクション

**即座に実行可能**:
1. docs/ ディレクトリ内の乖離ドキュメントを特定
2. 修正または deprecated に移動
3. Phase 2 完了後コミット

**diffx リブート完了後**:
1. diffx-core の新しいパスまたは crates.io バージョンを確認
2. diffai-core/Cargo.toml を更新
3. ビルド・テスト確認
4. Phase 1 完了

**その後**:
1. Phase 3（CI/CD 簡素化）に着手

## 📚 参考資料

- **リブート計画**: `.claude/reboot/reboot-plan.md`
- **移行ステータス**: `.claude/reboot/migration-status.md`
- **旧タスク管理**: `to-migrate/deprecated/tasks-old-414-lines.md`
- **diffx リブート**: https://github.com/kako-jun/diffx/tree/main/.claude/reboot

## 💡 リブート哲学

**diffx から学んだ原則**:
1. **完璧主義の放棄** - 動くCI/CDで十分、80%の品質で良い
2. **既存ファイルを信じない** - 実装を検証してから仕様を決定
3. **段階的アプローチ** - 小さな改善サイクルで進める
4. **独立性の確保** - 言語ごとに独立したリポジトリとワークフロー

---

**最終更新**: 2025-11-16 Phase 2 進行中
**メンテナ**: Claude Code（リブート作業中）
