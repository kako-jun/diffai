# diffai 移行ステータス

## ✅ Phase 0 完了: 構造の簡素化

**実施日**: 2025-11-16

### 実施内容

#### 1. リポジトリ構造の簡素化
**モノレポから Rust コア専用リポジトリへ**

```
変更前:
diffai/
├── diffai-core/
├── diffai-cli/
├── diffai-js/        ← 削除
├── diffai-python/    ← 削除
└── Cargo.toml       (4メンバー)

変更後:
diffai/
├── diffai-core/
├── diffai-cli/
├── to-migrate/
│   ├── diffai-js/        ← 移動（将来的に独立リポジトリへ）
│   ├── diffai-python/    ← 移動（将来的に独立リポジトリへ）
│   └── deprecated/       ← 削除候補
└── Cargo.toml       (2メンバー)
```

#### 2. Cargo.toml の更新
**ワークスペースメンバーを Rust コアのみに限定**

```toml
[workspace]
members = [
    "diffai-core",
    "diffai-cli"
    # "diffai-python",  ← 削除
    # "diffai-js"       ← 削除
]
```

#### 3. .gitignore の更新
```gitignore
# === Migration directories ===
to-migrate/
```

### 移動されたコンポーネント

#### to-migrate/diffai-js/
- **説明**: JavaScript/TypeScript バインディング（NAPI-RS）
- **移動先リポジトリ**: https://github.com/kako-jun/diffai-js (準備中)
- **依存関係**: diffai-core (crates.io 公開後に更新予定)

#### to-migrate/diffai-python/
- **説明**: Python バインディング（PyO3）
- **移動先リポジトリ**: https://github.com/kako-jun/diffai-python (準備中)
- **依存関係**: diffai-core (crates.io 公開後に更新予定)

## 🔄 Phase 1 待機中: diffx-core 依存の解決

### 現在の問題

**ビルドエラー**:
```
error: failed to load manifest for dependency `diffx-core`
Caused by:
  failed to read `/home/user/diffx/diffx-core/Cargo.toml`
```

**原因**:
- diffai-core が diffx-core に依存 (`path = "../../diffx/diffx-core"`)
- diffx が現在リブート中で、構造が変更される可能性

**影響範囲**:
- diffai-core/Cargo.toml:35
- diffai-core/src/lib.rs:16-24 (7つの関数/型を再エクスポート)
  - value_type_name
  - estimate_memory_usage
  - would_exceed_memory_limit
  - format_diff_output
  - base_diff
  - BaseDiffOptions
  - BaseOutputFormat

### 解決待ち

diffx のリブートが完了し、以下のいずれかが実現するまで待機:
1. diffx-core が crates.io に公開される → 依存をcrates.ioバージョンに変更
2. diffx-core の安定版パスが確定する → パスを更新
3. 必要な関数を diffai-core 内に実装する → diffx-core への依存を削除

## 📊 現在の状態

### リポジトリ構造
```
diffai/
├── .claude/
│   └── reboot/
│       ├── reboot-plan.md          ← リブート計画
│       └── migration-status.md     ← このファイル
├── diffai-core/                    ← Rust コアライブラリ
├── diffai-cli/                     ← CLI ツール
├── to-migrate/                     ← 移行待ちファイル (gitignore済み)
│   ├── diffai-js/
│   ├── diffai-python/
│   └── deprecated/
├── docs/                           ← ドキュメント（要整理）
├── tests/                          ← テスト（要整理）
├── scripts/                        ← スクリプト（要整理）
└── Cargo.toml                      ← ワークスペース設定（簡素化済み）
```

### ビルド状態
- ✅ Cargo.toml の構文は正しい
- ❌ diffx-core への依存が解決されないため、ビルドは通らない
- ⏸️ テストは実行できない（ビルドが必要）

### 次のステップ
1. **即座に実施可能**: 変更をコミット＆プッシュ（Phase 0 完了として記録）
2. **diffx リブート待ち**: diffx-core 依存の解決（Phase 1）
3. **Phase 1 後**: ドキュメント簡素化（Phase 2）
4. **Phase 2 後**: CI/CD 簡素化（Phase 3）

## 🎯 成果

### 達成したこと
- ✅ モノレポ構造を Rust 専用に簡素化
- ✅ JavaScript/Python バインディングを分離準備
- ✅ リブート計画とドキュメントを作成
- ✅ 既知の問題を明確に記録

### 残っている課題
- ❌ diffx-core への依存（ビルド不可）
- ❌ 過剰なドキュメント（tasks.md等）
- ❌ 複雑な CI/CD 設定
- ❌ github-shared への依存

## 📝 メモ

このリブート作業は、diffx のリブート進捗に追従する形で進めています。
完全な独立を目指すのではなく、段階的に複雑性を削減し、
将来的に独立したリポジトリとして機能できる基盤を整えることが目標です。

**参考**: [diffx リブート計画](https://github.com/kako-jun/diffx/tree/main/.claude/reboot)
