# ML Analysis Features

diffai automatically runs specialized ML analyses when comparing PyTorch/Safetensors files.

## Learning Rate Analysis

```console
$ diffai checkpoint_epoch1.pt checkpoint_epoch10.pt
learning_rate_analysis: old=0.001, new=0.0001, change=-90.0%, trend=decreasing

```

## Gradient Analysis

```console
$ diffai model_before.pt model_after.pt
gradient_analysis: flow_health=healthy, norm=0.021, variance_change=+15.3%

```

## Quantization Analysis

```console
$ diffai full_precision.pt quantized.pt
quantization_analysis: mixed_precision=FP16+FP32, compression=12.5%, precision_loss=1.2%

```

## Convergence Analysis

```console
$ diffai checkpoint_early.pt checkpoint_final.pt
convergence_analysis: status=converged, stability=0.98, plateau_detected=false

```

## Optimizer Comparison

```console
$ diffai model_sgd.pt model_adam.pt
optimizer_comparison: type=SGD->Adam, momentum_change=+2.1%, state_evolution=changed

```
