---
title: "How gpu-unlock Changes GeForce Tensor Throughput"
excerpt: "gpu-unlock lifts a limit on certain kinds of math on selected GeForce cards. It does this during Linux startup, before the NVIDIA driver takes over."
excerpt_separator: "<!--more-->"
tags:
  - GPUs
  - NVIDIA
  - Performance Engineering
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

Some GeForce cards can do certain kinds of AI math faster than they do with their default settings. **gpu-unlock** is a Linux project that changes one of those settings when the computer starts. The project reports that some AI workloads finish faster, particularly when the GPU spends much of its time multiplying matrices. [gpu-unlock repository](https://github.com/Zagitta/gpu-unlock)

<!--more-->

## How gpu-unlock lifts the limit

Think of a GPU as a factory with several workstations. One workstation, the tensor cores, specializes in the matrix math that AI models use. On the cards covered by this project, a setting limits how often the GPU can send certain kinds of work to that station. Lifting that limit lets the existing hardware take on those operations more frequently. [Project explanation](https://github.com/Zagitta/gpu-unlock#how-it-works)

The setting is protected. The project's Ampere write-up explains how a flaw in the GPU's startup firmware gives the module a way to change who can write to it. That lets the module lift the limit during Linux startup. By the time the regular NVIDIA driver takes over, the faster setting is already in place. [How Ampere was unlocked](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/docs/How%20Ampere%20was%20unlocked.md) · [Boot flow](https://github.com/Zagitta/gpu-unlock#boot-flow-and-opt-in)

The change lasts while the card has power. If the system cuts power to the GPU during sleep, the project can reapply the setting when the computer wakes. The project lists selected RTX 30 series cards as stable, with experimental support for selected RTX 20 and 40 series cards. [Supported GPUs and boot behavior](https://github.com/Zagitta/gpu-unlock#supported-gpus)

## What gets faster

Consider what happens when you ask a language model a question. It first processes your prompt, then begins generating an answer. During prompt processing, the GPU can work on many tokens together and reuse the same model weights across them. This gives its math units more work to do for each trip to memory, so a faster math unit can help. [NVIDIA’s matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/)

Once the answer starts streaming, a lightly loaded GPU may spend more of its time fetching model weights to produce each new token. In that case, the math units finish quickly and wait for the next data to arrive. You might see the first token sooner and then watch the answer stream at much the same pace. Batching several requests can shift that balance by letting the GPU reuse weights across requests. [NVIDIA’s GPU performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)

The project summarizes user reports of **25–55% higher prompt processing throughput** and **25–50% faster BF16 fine-tuning**, with little improvement for token generation when memory is the bottleneck. These figures come from specific setups and workloads. The practical comparison is how long your own job spends doing the kind of math the project accelerates. [Reported application benchmarks](https://github.com/Zagitta/gpu-unlock#what-you-get)

## Appendix

### Firmware access and driver handoff

The Linux kernel module runs from the initramfs and selects supported cards by PCI device ID after an explicit boot-time opt-in. On Ampere, the project documents a vulnerable bootloader path on the GPU's Security Engine 2. It uses the V67 firmware payload to change a privilege level mask governing write access to the speed selector registers. The host then sets the overrides in SS0 and SS1. The module performs a PCI function-level reset before the stock NVIDIA driver binds to the card. [Ampere firmware and register details](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/docs/How%20Ampere%20was%20unlocked.md) · [Module entry point](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/kernel/gpu_unlock/main.c) · [Driver handoff](https://github.com/Zagitta/gpu-unlock#how-it-works)

### Matrix multiplication benchmarks

The project targets the issue rate of selected tensor operations using FP16, BF16, or TF32 inputs with FP32 accumulation. Issue rate measures how frequently hardware can dispatch instructions of a given class. Clock frequency measures how quickly the chip cycles. The project reports the following BF16 matrix multiplication throughput with FP32 accumulation and stock power limits. [Published benchmark table](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/README.md#what-you-get)

| GPU | BF16 before | BF16 after |
| --- | ---: | ---: |
| RTX 3060 12 GB | 27 TFLOPS | 51 TFLOPS |
| RTX 3090 | 75 TFLOPS | 125 TFLOPS |
| RTX 4080 | 104 TFLOPS | 193 TFLOPS |

The README separately reports a near doubling of the RTX 4090's **peak** matrix instruction rate. It reports smaller gains in large matrix multiplies at stock power limits, where the card reaches its power ceiling. [RTX 4090 benchmark note](https://github.com/Zagitta/gpu-unlock#what-you-get)
