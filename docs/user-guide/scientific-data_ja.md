# 科学データ分析ガイド

NumPy配列やMATLAB行列などの科学データ形式を分析するdiffaiの機能について説明します。

## 概要

diffaiは機械学習モデルを超えて、研究や計算科学で一般的に使用される科学データ形式をサポートしています。これにより、数値配列、実験結果、シミュレーションデータの意味のある比較が可能になります。

## サポートされている科学データ形式

### NumPy配列
- **`.npy` ファイル**: 完全な統計分析を含む単一のNumPy配列
- **`.npz` ファイル**: 複数の配列を含む圧縮NumPyアーカイブ

### MATLAB行列
- **`.mat` ファイル**: 複素数サポート付きのMATLAB行列ファイル

## diffaiが分析する内容

### 配列統計
データ内の各配列について、diffaiは以下を計算・比較します：

- **平均値**: 全要素の平均値
- **標準偏差**: データの分散の指標
- **最小値**: 配列内の最小値
- **最大値**: 配列内の最大値
- **形状**: 配列の次元
- **データ型**: 要素の精度（float64、int32など）
- **要素数**: 総要素数

### MATLAB固有の機能
- **複素数サポート**: 実部と虚部を別々に分析
- **変数名**: MATLAB変数名を保持
- **多次元配列**: N次元行列の完全サポート
- **混合データ型**: 単一の.matファイル内の異なるデータ型

## 基本的なデータ比較

### NumPy配列比較

```bash
# 単一のNumPy配列を比較
diffai data_v1.npy data_v2.npy --stats

# 圧縮NumPyアーカイブを比較
diffai dataset_v1.npz dataset_v2.npz --stats

# 特定の出力形式で比較
diffai experiment_baseline.npy experiment_result.npy --output json
```

### MATLABファイル比較

```bash
# MATLABファイルを比較
diffai simulation_v1.mat simulation_v2.mat --stats

# 特定の変数に焦点を当てる
diffai results_v1.mat results_v2.mat --path "experiment_data"

# ドキュメント用にYAMLで出力
diffai analysis_v1.mat analysis_v2.mat --output yaml
```

## 出力例

### NumPy配列の変化

```bash
$ diffai experiment_data_v1.npy experiment_data_v2.npy --stats
  ~ data: shape=[1000, 256], mean=0.1234->0.1456, std=0.9876->0.9654, dtype=float64
```

### MATLABファイルの変化

```bash
$ diffai simulation_v1.mat simulation_v2.mat --stats
  ~ results: var=results, shape=[500, 100], mean=2.3456->2.4567, std=1.2345->1.3456, dtype=double
  + new_variable: var=new_variable, shape=[100], dtype=single, elements=100, size=0.39KB
```

### 圧縮アーカイブの比較

```bash
$ diffai dataset_v1.npz dataset_v2.npz --stats
  ~ train_data: shape=[60000, 784], mean=0.1307->0.1309, std=0.3081->0.3082, dtype=float32
  ~ test_data: shape=[10000, 784], mean=0.1325->0.1327, std=0.3105->0.3106, dtype=float32
  + validation_data: shape=[5000, 784], mean=0.1315, std=0.3095, dtype=float32
```

## 高度なオプション

### 数値データのイプシロン許容値

```bash
# 小さな数値差を無視
diffai experiment_v1.npy experiment_v2.npy --epsilon 1e-6

# シミュレーション結果の比較に有用
diffai simulation_v1.mat simulation_v2.mat --epsilon 1e-8
```

### 結果のフィルタリング

```bash
# MATLABファイルの特定の変数に焦点を当てる
diffai results_v1.mat results_v2.mat --path "experimental_data"

# メタデータ変数を無視
diffai data_v1.mat data_v2.mat --ignore-keys-regex "^(metadata|timestamp)"
```

## 一般的な使用例

### 1. 実験データの検証

異なる条件下での実験結果を比較：

```bash
diffai baseline_experiment.npy treated_experiment.npy --stats

# 期待される出力: 変化の統計的有意性
# ~ data: shape=[1000, 50], mean=0.4567->0.5123, std=0.1234->0.1456, dtype=float64
```

**分析**:
- 平均値の変化は治療効果を示す
- 標準偏差の変化は分散への影響を示す
- 形状の一貫性はデータの整合性を確認

### 2. シミュレーション結果の比較

パラメータセット間でのシミュレーション出力を比較：

```bash
diffai simulation_param_1.mat simulation_param_2.mat --stats

# 期待される出力: パラメータ感度分析
# ~ velocity_field: var=velocity_field, shape=[100, 100, 50], mean=1.234->1.567
# ~ pressure_field: var=pressure_field, shape=[100, 100, 50], mean=101.3->102.1
```

**分析**:
- 速度場の変化は流れの違いを示す
- 圧力の変動はシステムの応答を示す
- 一貫した形状はメッシュの安定性を確認

### 3. データ処理パイプラインの検証

異なる処理段階でのデータを比較：

```bash
diffai raw_data.npz processed_data.npz --stats

# 期待される出力: 処理の影響評価
# ~ features: shape=[10000, 512], mean=0.0->0.5, std=1.0->0.25, dtype=float32
# ~ labels: shape=[10000], mean=4.5->4.5, std=2.87->2.87, dtype=int64
```

**分析**:
- 特徴の正規化が成功（平均～0.5、標準偏差～0.25）
- ラベルが変更されていない（処理で分類が保持）
- 一貫した形状でデータ損失がないことを確認

### 4. 時系列分析

異なる時間期間での時系列データを比較：

```bash
diffai timeseries_q1.npy timeseries_q2.npy --stats

# 期待される出力: 時間パターンの変化
# ~ data: shape=[2160, 24], mean=23.45->25.67, std=5.67->6.23, dtype=float32
```

**分析**:
- 平均値の増加は季節トレンドを示す
- 標準偏差の増加は変動性の増加を示す
- 形状の一貫性はデータ構造を確認

## パフォーマンスの最適化

### 大きな配列の処理

非常に大きな配列（>1GB）の場合：

```bash
# より高速な比較のため、より大きなイプシロンを使用
diffai large_array_v1.npy large_array_v2.npy --epsilon 1e-3

# 特定のセクションに焦点を当てる
diffai large_sim_v1.mat large_sim_v2.mat --path "summary_stats"
```

### メモリの考慮事項

```bash
# 大きなファイルのメモリ制限を設定
DIFFAI_MAX_MEMORY=2048 diffai huge_dataset_v1.npz huge_dataset_v2.npz
```

## 統合例

### Pythonデータサイエンスワークフロー

```python
import numpy as np
import subprocess
import json

def compare_arrays(array1_path, array2_path, epsilon=1e-6):
    """diffaiを使用して2つのNumPy配列を比較"""
    result = subprocess.run([
        'diffai', array1_path, array2_path, 
        '--output', 'json', '--epsilon', str(epsilon)
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise RuntimeError(f"比較に失敗しました: {result.stderr}")

# 使用例
changes = compare_arrays('experiment_v1.npy', 'experiment_v2.npy')
for change in changes:
    if 'NumpyArrayChanged' in change:
        array_name, old_stats, new_stats = change['NumpyArrayChanged']
        print(f"配列 {array_name}: 平均値が {old_stats['mean']:.4f} から {new_stats['mean']:.4f} に変化")
```

### MATLAB統合

```matlab
function compare_mat_files(file1, file2)
    % diffaiを使用してMATLABファイルを比較
    command = sprintf('diffai %s %s --output json', file1, file2);
    [status, result] = system(command);
    
    if status == 0
        changes = jsondecode(result);
        for i = 1:length(changes)
            if isfield(changes(i), 'MatlabArrayChanged')
                change = changes(i).MatlabArrayChanged;
                fprintf('変数 %s: 平均値 %.4f -> %.4f\n', ...
                    change{1}, change{2}.mean, change{3}.mean);
            end
        end
    else
        error('比較に失敗しました: %s', result);
    end
end
```

### R統計分析

```r
library(jsonlite)

compare_data <- function(file1, file2, epsilon = 1e-6) {
  # diffaiを使用してデータファイルを比較
  command <- sprintf("diffai %s %s --output json --epsilon %.2e", 
                     file1, file2, epsilon)
  result <- system(command, intern = TRUE)
  
  if (length(result) > 0) {
    changes <- fromJSON(paste(result, collapse = ""))
    return(changes)
  } else {
    return(NULL)
  }
}

# 使用例
changes <- compare_data("analysis_v1.mat", "analysis_v2.mat")
if (!is.null(changes)) {
  for (change in changes) {
    if ("MatlabArrayChanged" %in% names(change)) {
      cat(sprintf("変数 %s が変更されました\n", change$MatlabArrayChanged[[1]]))
    }
  }
}
```

## ベストプラクティス

### 1. イプシロン値の選択

| データ型 | 推奨イプシロン | 理由 |
|----------|---------------|------|
| 実験測定 | 1e-6 to 1e-8 | 測定精度を考慮 |
| シミュレーション結果 | 1e-8 to 1e-10 | 数値計算の精度 |
| 画像データ | 1e-3 to 1e-6 | ピクセル値の精度 |
| 時系列 | 1e-4 to 1e-6 | 時間解像度 |

### 2. 出力形式の選択

- **CLI**: 人間によるレビューと迅速な比較
- **JSON**: 自動化された分析とスクリプト
- **YAML**: ドキュメント化とレポート作成

### 3. パフォーマンスのヒント

- ノイズを避けるため適切なイプシロン値を使用
- 比較戦略を選択する際はデータサイズを考慮
- 大きな多変数ファイルにはパスフィルタリングを使用
- 非常に大きなデータセットではメモリ使用量を監視

## データ形式の仕様

### NumPy配列サポート

diffaiは全てのNumPyデータ型をサポートします：
- **整数型**: int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **浮動小数点型**: float16, float32, float64
- **複素数型**: complex64, complex128
- **ブール型**: bool

### MATLAB行列サポート

diffaiはMATLABデータ型をサポートします：
- **数値型**: double, single, int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **複素数型**: 複素数倍精度と単精度
- **論理型**: logical（ブール）
- **文字配列**: メタデータの基本サポート

## トラブルシューティング

### 一般的な問題

#### 1. ファイル形式エラー

```bash
# ファイル形式を確認
file data.npy

# ファイルの整合性をチェック
python -c "import numpy as np; print(np.load('data.npy').shape)"

# MATLABファイルの場合
python -c "import scipy.io; print(scipy.io.loadmat('data.mat').keys())"
```

#### 2. メモリ問題

```bash
# ファイルサイズを確認
ls -lh large_data.npy

# ストリーミングモードを使用（利用可能な場合）
diffai --stream large_data_v1.npy large_data_v2.npy
```

#### 3. 精度問題

```bash
# データ精度を確認
python -c "import numpy as np; print(np.load('data.npy').dtype)"

# それに応じてイプシロンを調整
diffai data_v1.npy data_v2.npy --epsilon 1e-8
```

## 次のステップ

- [基本的な使い方](basic-usage_ja.md) - 基本操作を学習
- [MLモデル比較](ml-model-comparison_ja.md) - PyTorchとSafetensors分析
- [CLIリファレンス](../reference/cli-reference_ja.md) - 完全なコマンドリファレンス

## 言語サポート

- **日本語**: 現在のドキュメント
- **English**: [English version](scientific-data.md)