# diffai

[English](README.md)

[![CI](https://github.com/kako-jun/diffai/actions/workflows/ci.yml/badge.svg)](https://github.com/kako-jun/diffai/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/diffai.svg)](https://crates.io/crates/diffai)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

AI/MLモデル（PyTorch, Safetensors, NumPy, MATLAB）の意味的差分ツール。テンソル統計、パラメータ比較、自動ML分析を提供。

## なぜ diffai？

従来の `diff` はバイナリのMLファイルを理解しません：

```bash
$ diff model_v1.pt model_v2.pt
Binary files model_v1.pt and model_v2.pt differ
```

`diffai` は意味のある分析を表示：

```bash
$ diffai model_v1.safetensors model_v2.safetensors
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
gradient_analysis: flow_health=healthy, norm=0.021
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883
```

## インストール

```bash
# CLIツールとして
cargo install diffai

# ライブラリとして（Cargo.toml）
[dependencies]
diffai-core = "0.4"
```

## 使い方

```bash
# 基本
diffai model1.pt model2.pt

# JSON出力（自動化用）
diffai model1.safetensors model2.safetensors --output json

# 数値許容誤差付き
diffai weights1.npy weights2.npy --epsilon 0.001
```

## 対応フォーマット

- **PyTorch** (.pt, .pth) - フルML分析 + テンソル統計
- **Safetensors** (.safetensors) - フルML分析 + テンソル統計
- **NumPy** (.npy, .npz) - テンソル統計
- **MATLAB** (.mat) - テンソル統計

## 主なオプション

```bash
--format <FORMAT>       # 入力形式を強制（pytorch, safetensors, numpy, matlab）
--output <FORMAT>       # 出力形式: json, yaml, text（デフォルト: text）
--epsilon <N>           # 浮動小数点の許容誤差
--ignore-keys-regex RE  # 正規表現にマッチするキーを無視
--quiet                 # 終了コードのみ返す（0:同一, 1:差分あり）
--verbose               # 詳細分析を表示
```

## 出力記号

- `+` 追加されたテンソル/パラメータ
- `-` 削除されたテンソル/パラメータ
- `~` 変更されたテンソル/パラメータ

## 自動ML分析

PyTorch/Safetensorsファイル比較時、diffaiは11種類の専門分析を自動実行：

1. 学習率分析
2. オプティマイザ比較
3. 損失追跡
4. 精度追跡
5. モデルバージョン分析
6. 勾配分析
7. 量子化分析
8. 収束分析
9. 活性化関数分析
10. アテンション分析
11. アンサンブル分析

## CI/CDでの活用

```bash
# モデル変更検知
if ! diffai production.pt candidate.pt --quiet; then
  echo "モデルが変更されています"
  diffai production.pt candidate.pt --output json > changes.json
fi
```

## 実行例

[diffai-cli/tests/cmd/](diffai-cli/tests/cmd/) に詳細な例があります：

- [基本的な比較](diffai-cli/tests/cmd/basic.md)
- [対応フォーマット](diffai-cli/tests/cmd/formats.md)
- [出力形式](diffai-cli/tests/cmd/output.md)
- [オプション](diffai-cli/tests/cmd/options.md)

## ドキュメント

- [CLI仕様書](docs/specs/cli.md)
- [Core API仕様書](docs/specs/core.md)

## 関連プロジェクト

- **[diffx](https://github.com/kako-jun/diffx)** - 構造化データ差分（JSON, YAML, CSV, XML）
- **[lawkit](https://github.com/kako-jun/lawkit)** - 統計法則分析ツールキット

## ライセンス

MIT
