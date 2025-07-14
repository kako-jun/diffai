# 詳細出力ガイド

diffaiの`--verbose`フラグは、処理の詳細を理解し、問題をデバッグし、パフォーマンスを分析するための包括的な診断情報を提供します。このガイドでは、詳細モードを効果的に使用する方法について説明します。

## 概要

詳細モードでは以下の詳細情報が表示されます：
- 設定とアクティブな機能
- ファイル解析とフォーマット検出
- 処理時間とパフォーマンス指標
- ML固有の解析状況
- ディレクトリ比較統計

## 基本的な使用方法

### 詳細モードの有効化

```bash
# 基本的な詳細出力
diffai file1.json file2.json --verbose

# 短縮形
diffai file1.json file2.json -v
```

### 出力例

```
=== diffai verbose mode enabled ===
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false

File analysis:
  Input 1: file1.json
  Input 2: file2.json
  Detected format: Json
  File 1 size: 156 bytes
  File 2 size: 162 bytes

Processing results:
  Total processing time: 234.567µs
  Differences found: 3
  Format-specific analysis: Json
```

## 設定診断

詳細モードでは、すべてのアクティブな設定オプションが表示されます：

### 基本設定
- **入力フォーマット**: 明示的に設定されたか自動検出されたフォーマット
- **出力フォーマット**: CLI、JSON、YAML、または統一diff
- **再帰モード**: ディレクトリ比較が有効かどうか

### 高度なオプション
```bash
diffai file1.json file2.json --verbose \
  --epsilon 0.001 \
  --ignore-keys-regex "^id$" \
  --path "config.users"
```

出力には以下が含まれます：
```
Configuration:
  Input format: None
  Output format: Cli
  Recursive mode: false
  Epsilon tolerance: 0.001
  Ignore keys regex: ^id$
  Path filter: config.users
```

### ML解析機能
ML解析オプションが有効な場合、詳細モードではどの機能がアクティブかが表示されます：

```bash
diffai model1.safetensors model2.safetensors --verbose \
  --stats \
  --architecture-comparison \
  --memory-analysis \
  --anomaly-detection
```

出力：
```
Configuration:
  ML analysis features: statistics, architecture_comparison, memory_analysis, anomaly_detection
```

## ファイル解析情報

### ファイルメタデータ
詳細モードでは詳細なファイル情報が提供されます：

- **ファイルパス**: 入力ファイルのフルパス
- **ファイルサイズ**: 各ファイルの正確なバイト数
- **フォーマット検出**: diffaiがファイルフォーマットを識別した方法

例：
```
File analysis:
  Input 1: /path/to/model1.safetensors
  Input 2: /path/to/model2.safetensors
  Detected format: Safetensors
  File 1 size: 1048576 bytes
  File 2 size: 1048576 bytes
```

### フォーマット検出プロセス
詳細モードでは、ファイルフォーマットがどのように決定されたかが説明されます：

1. **明示的フォーマット**: `--format`が指定された場合
2. **自動検出**: ファイル拡張子に基づく
3. **フォールバックロジック**: フォーマットが推測できない場合

## パフォーマンス指標

### 処理時間
詳細モードでは処理時間をマイクロ秒精度で測定・報告します：

```
Processing results:
  Total processing time: 1.234567ms
  Differences found: 15
```

### ML/科学データ解析
MLおよび科学データファイルの場合、詳細モードでは完了状況が示されます：

```
Processing results:
  Format-specific analysis: Safetensors
  ML/Scientific data analysis completed
```

## ディレクトリ比較

ディレクトリ比較で`--recursive`を使用する際、詳細モードでは追加の統計が提供されます：

```bash
diffai dir1/ dir2/ --verbose --recursive
```

出力例：
```
Configuration:
  Recursive mode: true

Directory scan results:
  Files in /path/to/dir1: 12
  Files in /path/to/dir2: 14
  Total files to compare: 16

Directory comparison summary:
  Files compared: 10
  Files only in one directory: 6
  Total files processed: 16
```

## 使用例

### 処理問題のデバッグ

diffaiが予期しない動作をする場合、詳細モードで以下を特定できます：

1. **フォーマット検出の問題**: 正しいフォーマットが検出されたかチェック
2. **設定の競合**: すべてのオプションが正しく適用されているか確認
3. **パフォーマンスボトルネック**: 遅い処理ステップの特定
4. **ファイルアクセス問題**: ファイルが読み取り可能で期待されるサイズかを確認

デバッグセッション例：
```bash
# フォーマット検出が正しいかチェック
diffai problematic_file1.dat problematic_file2.dat --verbose

# ML解析機能確認（MLモデルでは自動実行）
diffai model1.pt model2.pt --verbose

# ディレクトリ比較動作を解析
diffai dir1/ dir2/ --verbose --recursive
```

### パフォーマンス分析

詳細モードを使用して処理パフォーマンスを理解：

```bash
# 異なるフォーマットの処理時間を測定
diffai large_model1.safetensors large_model2.safetensors --verbose

# 異なるオプションでのパフォーマンスを比較
diffai data1.json data2.json --verbose --epsilon 0.0001
```

### 設定検証

複雑な設定が正しく適用されているかを確認：

```bash
# 複数のフィルターとオプションをチェック
diffai config1.yaml config2.yaml --verbose \
  --ignore-keys-regex "^(id|timestamp)$" \
  --path "application.settings" \
  --epsilon 0.01 \
  --output json
```

## ヒントとベストプラクティス

### 1. デバッグには常に詳細モードを使用
予期しない動作に遭遇した場合は、diffaiが何をしているかを理解するために常に`--verbose`を追加してください。

### 2. パフォーマンス監視
大きなファイルの処理時間を監視し、それに応じて最適化するために詳細モードを使用してください。

### 3. 設定確認
バッチ操作を実行する前に、単一ファイルで詳細モードを使用して設定を確認してください。

### 4. ファイルフォーマットの学習
詳細モードを使用して、diffaiが異なるファイルフォーマットをどのように検出・処理するかを理解してください。

### 5. ML解析の最適化
複数のML解析機能を使用する場合、詳細モードでどの機能がアクティブで、処理時間への影響を特定できます。

## 出力リダイレクト

### 詳細出力と結果の分離
詳細情報はstderrに送信されるため、結果から分離できます：

```bash
# 結果をファイルに保存、詳細出力を画面に表示
diffai file1.json file2.json --verbose --output json > results.json

# 結果と詳細出力の両方を保存
diffai file1.json file2.json --verbose > results.txt 2> verbose.log

# 詳細情報のみを表示
diffai file1.json file2.json --verbose 2>&1 >/dev/null
```

## 他のツールとの統合

### CI/CDパイプライン
CI/CDでデバッグのために詳細モードを使用：

```bash
# GitHub Actionsなどで
- name: 詳細出力でモデル比較
  run: diffai baseline.safetensors new_model.safetensors --verbose --stats
```

### スクリプトと自動化
自動化のために詳細出力を解析：

```bash
#!/bin/bash
output=$(diffai file1.json file2.json --verbose 2>&1)
if echo "$output" | grep -q "Differences found: 0"; then
    echo "ファイルは同一です"
else
    echo "ファイルが異なります"
fi
```

## 関連コマンド

- [`--help`](basic-usage.md#help): 利用可能なすべてのオプションを表示
- [`--output`](output-formats.md): 出力フォーマットの制御
- [`--stats`](ml-analysis.md#statistics): ML統計解析の有効化
- [`--recursive`](directory-comparison.md): ディレクトリ比較の有効化