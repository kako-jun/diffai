# Basic Model Comparison

Compare two ML model files and show semantic differences.

```console
$ diffai model_v1.pt model_v2.pt
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716
~ fc2.weight: mean=-0.0008->-0.0018, std=0.0719->0.0883

```

## No Differences

When models are identical, no output is shown and exit code is 0.

```console
$ diffai model.pt model.pt

```
