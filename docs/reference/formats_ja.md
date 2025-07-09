# 対応形式

diffaiがサポートするファイル形式とその仕様について説明します。

## 機械学習モデル形式

### PyTorchモデル
- **拡張子**: `.pt`, `.pth`
- **形式**: Pickle形式（Candleライブラリ統合）
- **サポート**: テンソル統計、形状変更、パラメータ追加/削除
- **データ型**: F32, F64, F16, BF16, I64, U32, U8

**例**:
```bash
diffai model1.pt model2.pt --stats
```

### Safetensorsモデル
- **拡張子**: `.safetensors`
- **形式**: HuggingFace Safetensors形式（推奨）
- **サポート**: 高速読み込み、メモリ効率、セキュリティ
- **データ型**: F32, F64, F16, BF16, I64, U32, U8

**例**:
```bash
diffai model1.safetensors model2.safetensors --stats
```

## 科学データ形式

### NumPy配列
- **拡張子**: `.npy`, `.npz`
- **形式**: NumPy配列形式
- **サポート**: 統計分析、形状比較、データ型検証
- **データ型**: すべてのNumPyデータ型

**例**:
```bash
diffai data1.npy data2.npy --stats
diffai archive1.npz archive2.npz --stats
```

### MATLAB行列
- **拡張子**: `.mat`
- **形式**: MATLAB行列ファイル
- **サポート**: 複素数、変数名、多次元配列
- **データ型**: double, single, int8-64, uint8-64, logical

**例**:
```bash
diffai simulation1.mat simulation2.mat --stats
```

## 構造化データ形式

### JSON
- **拡張子**: `.json`
- **形式**: JavaScript Object Notation
- **サポート**: ネストされたオブジェクト、配列、基本データ型
- **用途**: 設定ファイル、API応答、実験結果

**例**:
```bash
diffai config1.json config2.json
```

### YAML
- **拡張子**: `.yaml`, `.yml`
- **形式**: YAML Ain't Markup Language
- **サポート**: 階層構造、コメント、複数ドキュメント
- **用途**: 設定ファイル、CI/CD、Kubernetes

**例**:
```bash
diffai config1.yaml config2.yaml
```

### TOML
- **拡張子**: `.toml`
- **形式**: Tom's Obvious, Minimal Language
- **サポート**: 型安全、設定向け構造
- **用途**: Rust設定、プロジェクト設定

**例**:
```bash
diffai Cargo1.toml Cargo2.toml
```

### XML
- **拡張子**: `.xml`
- **形式**: eXtensible Markup Language
- **サポート**: 属性、名前空間、階層構造
- **用途**: 設定ファイル、データ交換

**例**:
```bash
diffai config1.xml config2.xml
```

### INI
- **拡張子**: `.ini`
- **形式**: Initialization file
- **サポート**: セクション、キー値ペア
- **用途**: 設定ファイル、レガシーアプリケーション

**例**:
```bash
diffai config1.ini config2.ini
```

### CSV
- **拡張子**: `.csv`
- **形式**: Comma-Separated Values
- **サポート**: 表形式データ、ヘッダー行
- **用途**: データ分析、スプレッドシート

**例**:
```bash
diffai data1.csv data2.csv
```

## 形式の自動検出

diffaiは以下の優先順位で形式を検出します：

1. **--format** オプション（明示的指定）
2. **ファイル拡張子**による自動判定
3. **設定ファイル**のデフォルト形式
4. **エラー**（判定不可の場合）

```bash
# 明示的な形式指定
diffai file1.bin file2.bin --format safetensors

# 自動検出
diffai model1.safetensors model2.safetensors
```

## 形式別の特徴

### 機械学習モデル形式の比較

| 形式 | 速度 | セキュリティ | メモリ効率 | 推奨度 |
|------|------|-------------|-----------|--------|
| Safetensors | 高 | 高 | 高 | 推奨 |
| PyTorch | 中 | 低 | 中 | 利用可能 |

### 科学データ形式の比較

| 形式 | 精度 | 圧縮 | 複素数 | 推奨用途 |
|------|------|------|--------|----------|
| NumPy | 高 | 中 | 対応 | Python科学計算 |
| MATLAB | 高 | 低 | 対応 | MATLAB環境 |

### 構造化データ形式の比較

| 形式 | 可読性 | 型安全性 | コメント | 推奨用途 |
|------|--------|----------|----------|----------|
| JSON | 中 | 低 | 不可 | API、設定 |
| YAML | 高 | 低 | 可能 | 設定、CI/CD |
| TOML | 高 | 中 | 可能 | プロジェクト設定 |
| XML | 低 | 中 | 可能 | レガシー、データ交換 |
| INI | 高 | 低 | 可能 | 簡単な設定 |
| CSV | 中 | 低 | 不可 | 表形式データ |

## 実装状況

### Phase 1-2 (完了)
- ✅ PyTorch (.pt, .pth)
- ✅ Safetensors (.safetensors)
- ✅ NumPy (.npy, .npz)
- ✅ MATLAB (.mat)
- ✅ JSON (.json)
- ✅ YAML (.yaml, .yml)
- ✅ TOML (.toml)
- ✅ XML (.xml)
- ✅ INI (.ini)
- ✅ CSV (.csv)

### Phase 3 (予定)
- ⏳ TensorFlow (.pb, .h5, SavedModel)
- ⏳ ONNX (.onnx)
- ⏳ HDF5 (.h5, .hdf5)

## 制限事項

### ファイルサイズ
- **推奨最大**: 10GB
- **メモリ使用量**: ファイルサイズの約2倍
- **対策**: --pathフィルタリング、--epsilonの使用

### データ型
- **サポート**: 数値、文字列、ブール、配列、オブジェクト
- **非サポート**: 関数、クラス、循環参照

### エンコーディング
- **テキスト形式**: UTF-8
- **バイナリ形式**: ネイティブエンディアン

## 設定例

### 設定ファイル (diffx.toml)
```toml
[format]
default = "auto"
numpy_precision = "float64"
matlab_version = "v7.3"

[output]
default = "cli"
json_pretty = true
yaml_flow = false
```

### 環境変数
```bash
export DIFFAI_DEFAULT_FORMAT="safetensors"
export DIFFAI_MAX_MEMORY="8192"
export DIFFAI_NUMPY_PRECISION="float32"
```

## トラブルシューティング

### 形式検出エラー
```bash
# 明示的に形式を指定
diffai file1.bin file2.bin --format safetensors

# ファイルの内容を確認
file mysterious_file.bin
```

### メモリ不足
```bash
# メモリ制限を設定
DIFFAI_MAX_MEMORY=4096 diffai large1.safetensors large2.safetensors

# 特定部分のみ比較
diffai large1.safetensors large2.safetensors --path "classifier"
```

### 精度問題
```bash
# 適切なイプシロンを設定
diffai model1.safetensors model2.safetensors --epsilon 1e-6

# データ型を確認
diffai data1.npy data2.npy --stats
```

## 関連項目

- [CLIリファレンス](cli-reference_ja.md) - 完全なコマンドオプション
- [ML分析機能](ml-analysis_ja.md) - 機械学習特化機能
- [出力形式](output-formats_ja.md) - 出力形式の詳細

## 言語サポート

- **日本語**: 現在のドキュメント
- **English**: [English version](formats.md)