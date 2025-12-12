# Output Formats

## Text (Default)

Human-readable output with ML analysis.

```console
$ diffai model1.pt model2.pt
learning_rate_analysis: old=0.001, new=0.0015, change=+50.0%
~ fc1.weight: mean=-0.0002->-0.0001, std=0.0514->0.0716

```

## JSON

Machine-readable JSON for automation.

```console
$ diffai model1.pt model2.pt --output json
[
  {
    "Modified": [
      "fc1.weight",
      {"mean": -0.0002, "std": 0.0514},
      {"mean": -0.0001, "std": 0.0716}
    ]
  }
]

```

## YAML

```console
$ diffai model1.pt model2.pt --output yaml
- Modified:
  - fc1.weight
  - mean: -0.0002
    std: 0.0514
  - mean: -0.0001
    std: 0.0716

```

## Quiet Mode

Exit code only, no output.

```console
$ diffai model1.pt model2.pt --quiet
? 1

```

## Version

```console
$ diffai --version
diffai 0.3.16

```
