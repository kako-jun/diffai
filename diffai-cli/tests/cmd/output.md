# Output Formats

## Text (Default)

Human-readable output with ML analysis.

```console
$ diffai tests/fixtures/model_v1.safetensors tests/fixtures/model_v2.safetensors
? 1
  ~ tensors.fc1.weight.data_summary.max: 0.09999847412109375 -> 0.1999977082014084
  ~ tensors.fc1.weight.data_summary.mean: -7.629394644971532e-7 -> 0.049998855736234304
  ~ tensors.fc1.weight.data_summary.min: -0.10000000149011612 -> -0.10000000894069672
  ~ tensors.fc1.weight.data_summary.std: 0.057735027919685454 -> 0.0866025439171912
  ~ tensors.fc2.weight.data_summary.max: 0.14999085664749146 -> 0.16998779773712158
  ~ tensors.fc2.weight.data_summary.mean: -4.57763690064894e-6 -> -0.03000610376952295
  ~ tensors.fc2.weight.data_summary.min: -0.15000000596046448 -> -0.23000000417232513
  ~ tensors.fc2.weight.data_summary.std: 0.08660254389937629 -> 0.11547005545920479


```

## Quiet Mode

Exit code only, no output.

```console
$ diffai tests/fixtures/model_v1.safetensors tests/fixtures/model_v2.safetensors --quiet
? 1

```

## Version

```console
$ diffai --version
diffai 0.4.0

```
