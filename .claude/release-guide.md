# 内部リリースガイド（開発者専用）

## リリース時チェックリスト

### 1. ブランチとマージ状況確認
```bash
# 現在のブランチがmainであることを確認
git branch --show-current

# リモートからの最新情報を取得
git fetch origin

# mainブランチが最新であることを確認
git status
git log --oneline -10  # 最新10コミットを確認

# 作業ブランチがmainにマージ済みかを確認
git branch --merged main | grep -v main  # mainにマージ済みのブランチ一覧
git branch --no-merged main  # まだマージされていないブランチ一覧
```

### 2. Issue・PR状況確認
```bash
# GitHubのIssue・PR状況を確認
gh issue list --state open
gh pr list --state open

# 古いIssue・PRがないかWebで確認
# https://github.com/kako-jun/diffai/issues
# https://github.com/kako-jun/diffai/pulls
```

### 3. ローカルビルド・テスト確認
```bash
# 必須: プッシュ前チェック実行
./scripts/testing/ci-local.sh

# 手動確認（必要に応じて）
cargo test --release
cargo build --release
```

### 4. バージョン更新
```bash
# CHANGELOG.mdの更新（手動）
# バージョン番号はscripts/release/で自動更新されるため手動変更不要

# コミット
git add CHANGELOG.md
git commit -m "chore: Update CHANGELOG for vX.Y.Z"
```

### 5. リリース実行
```bash
# リリーススクリプト実行（バージョン自動判定・更新）
./scripts/release/release.sh

# または特定バージョン指定
./scripts/release/release.sh 0.3.0
```

### 6. リリース後確認
```bash
# GitHub Releasesで確認
open https://github.com/kako-jun/diffai/releases

# パッケージ公開確認
open https://crates.io/crates/diffai
open https://www.npmjs.com/package/diffai-js
open https://pypi.org/project/diffai-python/

# 動作テスト
./scripts/testing/test-published-packages.sh
```

## AI向けリリース指示

Claude/AI に以下のように指示すると自動でリリースが実行されます:

```
今からdiffaiをリリースしてください。
現在のCHANGELOG.mdを確認し、まだ更新されていない変更があれば追記してください。
その後、scripts/release/release.shを実行してリリースプロセスを開始してください。
```

## トラブルシューティング

### GitHub Actions失敗時
```bash
# Act1（Rust build）失敗時
./scripts/testing/ci-local.sh  # ローカルで問題特定

# Act2（npm/PyPI）失敗時
gh run list  # 失敗した実行を特定
gh run view <run-id>  # ログ確認
```

### パッケージ公開失敗時
```bash
# 手動再実行
gh workflow run release-act2.yml

# トークン確認
gh auth status
```

---

**注意**: このガイドはdiffaiプロジェクト専用です。プロセスはdiffxと同様ですが、リポジトリ名・パッケージ名が異なります。
EOF < /dev/null