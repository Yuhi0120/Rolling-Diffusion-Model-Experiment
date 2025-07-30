# Rolling-Diffusion-Model-Experiment
Train and sample a rolling‑diffusion model on the BAIR Robot Pushing (small) dataset.

Based on: https://arxiv.org/html/2402.09470v3

## Environment

- NVIDIA GPU with CUDA support (8–12 GB VRAM recommended)
- Docker
- NVIDIA drivers
- NVIDIA Container Toolkit

To check it, please run this command

```
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

## How to run
1. Download the dataset
~~~
make tfds
~~~

2. Build Docker image
~~~
make build
~~~

3. Train
~~~
make train
~~~

4. Make the samples
~~~
make sample
~~~

## Changing Training or Sampling Settings
You can either edit the Makefile or override variables on the command line:
~~~
make train STEPS=100000 CKPT_EVERY=10000
~~~

### Training variables

`FRAMES`・・・Number of frames per clip

`COND_FRAMES`・・・Conditioning frames

`BATCH`・・・Batch size

`BASE`・・・Base channel width for the UNet

`STEPS`・・・Training steps

`CKPT_EVERY`・・・Checkpoint save interval
