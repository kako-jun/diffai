# Supported Formats

diffai supports multiple AI/ML file formats.

## PyTorch (.pt, .pth)

```console
$ diffai model_v1.pt model_v2.pt
learning_rate_analysis: old=0.001, new=0.0015
~ conv1.weight: shape=[64,3,7,7], mean=0.01->0.02
...

```

## Safetensors

```console
$ diffai model_v1.safetensors model_v2.safetensors
~ embed_tokens.weight: dtype=F32->F16
~ layers.0.self_attn.q_proj.weight: mean=0.001->0.002
...

```

## NumPy (.npy, .npz)

```console
$ diffai data_v1.npy data_v2.npy
~ array: shape=[1000,784], mean=0.5->0.6
...

```

## MATLAB (.mat)

```console
$ diffai network_v1.mat network_v2.mat
~ weights: shape=[100,50], std=0.1->0.15
...

```
