# Comparison Options

## Numerical Tolerance (epsilon)

Treat numbers as equal if difference is within tolerance.

```console
$ diffai model1.pt model2.pt --epsilon 0.001
# Small weight changes within tolerance are ignored

```

## Ignore Keys by Regex

Skip keys matching a pattern.

```console
$ diffai model1.pt model2.pt --ignore-keys-regex "^optimizer"
# Optimizer state changes are ignored

```

## Filter by Path

Show only differences in paths containing this string.

```console
$ diffai model1.pt model2.pt --path "conv1"
~ conv1.weight: mean=0.01->0.02
~ conv1.bias: mean=0.0->0.001

```

## Verbose Mode

Show detailed analysis including all ML metrics.

```console
$ diffai model1.pt model2.pt --verbose
learning_rate_analysis:
  old: 0.001
  new: 0.0015
  change: +50.0%
  trend: increasing
gradient_analysis:
  flow_health: healthy
  norm: 0.021
  variance_change: +15.3%
...

```

## No Color

Disable ANSI color output.

```console
$ diffai model1.pt model2.pt --no-color
...

```
