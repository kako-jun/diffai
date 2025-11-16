# diffai リブート計画

## 🎯 目的
モノレポの複雑性を排除し、Rust コア（diffai-core + diffai-cli）に集中する

## 📋 背景
diffai、diffx、lawkit の3つは兄弟プロジェクトであり、フォルダ構成を似せていたが、その複雑性により3つとも現在失敗中。diffx がリブート中なので、diffai もリブートする。

## 🚨 主要な問題点

### 1. モノレポの複雑性
- diffai-js と diffai-python が同じリポジトリに含まれ、管理が複雑化
- ワークスペースメンバーが多すぎる（4つ: core, cli, js, python）

### 2. diffx-core への依存（**設計上正しい依存関係**）
- diffai-core が diffx-core に依存している（`path = "../../diffx/diffx-core"`）
- **目的**: コード重複削減のため、diffxの基本機能を再利用
- **使用機能**: value_type_name, estimate_memory_usage, would_exceed_memory_limit, format_diff_output, base_diff, BaseDiffOptions, BaseOutputFormat
- **現状**: diffx がリブート中のため、パスが一時的に解決できない可能性

### 3. 過剰なドキュメント
- tasks.md が400行超で、完璧主義による麻痺を引き起こしている
- 実装と乖離したドキュメントが存在

## 📅 リブート戦略

### Phase 0: 構造の簡素化 ✅
**完了日**: 2025-11-16
**作業内容**:
- [x] `to-migrate/` ディレクトリ作成
- [x] `diffai-js/` を `to-migrate/diffai-js/` に移動
- [x] `diffai-python/` を `to-migrate/diffai-python/` に移動
- [x] `Cargo.toml` のワークスペースメンバーを diffai-core と diffai-cli のみに更新
- [x] `.gitignore` に `to-migrate/` を追加

### Phase 1: diffx-core 依存パスの更新 🔄
**予定**: diffx リブート完了後
**目的**: diffx-core への依存を**維持しながら**、正しいパスに更新
**作業内容**:
- [ ] diffx のリブートが完了し、diffx-core のパスが確定するのを待つ
- [ ] diffx-core が crates.io に公開された場合: crates.io バージョンに変更
- [ ] ローカルパスのまま継続する場合: 新しいパスに更新
- [ ] **重要**: diffx-core への依存は継続（コード重複削減のため）

### Phase 2: ドキュメントの簡素化 📝
**予定**: Phase 1 完了後
**作業内容**:
- [ ] tasks.md を簡潔な TODO リストに変更
- [ ] 実装と乖離したドキュメントを削除または修正
- [ ] リブート後の現実を反映したドキュメントに更新

### Phase 3: CI/CD の簡素化 🔧
**予定**: Phase 2 完了後
**作業内容**:
- [ ] 複雑な共有スクリプトへの依存を削除
- [ ] diffai 専用のシンプルな CI/CD に変更
- [ ] 言語ごとに独立した GitHub Actions ワークフロー

## 🔍 既知の問題

### ビルドエラー
```
error: failed to load manifest for dependency `diffx-core`
Caused by:
  failed to read `/home/user/diffx/diffx-core/Cargo.toml`
```

**原因**: diffai-core/Cargo.toml:35 で diffx-core への相対パス依存
**解決策**: diffx のリブートが完了し、diffx-core が crates.io に公開されるまで待つ

### 移動したディレクトリ
以下のディレクトリは `to-migrate/` に移動され、将来的に別リポジトリとして独立予定:
- `to-migrate/diffai-js/` - JavaScript/npm バインディング
- `to-migrate/diffai-python/` - Python/PyPI バインディング
- `to-migrate/deprecated/` - 削除候補ファイル

## 🎯 リブートの哲学

**diffx から学んだ原則**:
1. **完璧主義の放棄**: 動くCI/CDで十分、80%の品質で良い
2. **既存ファイルを信じない**: 実装を検証してから仕様を決定
3. **段階的アプローチ**: 小さな改善サイクルで進める
4. **独立性の確保**: 言語ごとに独立したリポジトリとワークフロー

## 📊 進捗状況

- **Phase 0**: ✅ 完了 (2025-11-16)
- **Phase 1**: 🔄 diffx リブート待ち
- **Phase 2**: ⏸️ 未開始
- **Phase 3**: ⏸️ 未開始

## 🔗 関連リソース

- [diffx リブート計画](https://github.com/kako-jun/diffx/tree/main/.claude/reboot)
- [diffai-js 独立リポジトリ](https://github.com/kako-jun/diffai-js) ※準備中
- [diffai-python 独立リポジトリ](https://github.com/kako-jun/diffai-python) ※準備中
