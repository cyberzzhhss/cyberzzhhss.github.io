---
title: "Computation Inside Language Models"
excerpt: "Five projects explore execution, internal training, extensible memory, and certified arithmetic—and clarify which parts of computation remain outside the network."
last_modified_at: 2026-11-10
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Language Models
  - Transformers
  - Formal Verification
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

A language model asked to multiply two matrices can write a program and hand it to an external runtime. Now imagine that its next forward pass executes an instruction from that program. The following pass executes another. Eventually, the answer is sitting in memory maintained by the model. The visible chat might look the same, but the division of labor has changed.

Five projects explore that boundary. Percepta builds a WebAssembly executor and later a Python environment with writable memory. qwenGPU adds a small parallel machine to Qwen. qwenLab gives Qwen an internal training lab for smaller transformers. Sketchpad adds a learned arithmetic engine whose local steps can be exhaustively checked. Together, they separate four decisions: **who chooses the computation, where it runs, where its state persists, and what establishes its correctness**.

<!--more-->

## A transformer that executes WebAssembly

In *Can LLMs Be Computers?*, Percepta compiles C programs to WebAssembly and runs them through an interpreter implemented in transformer weights. The model generates an execution trace whose tokens record operations such as changes to memory and the instruction pointer. Each step retrieves relevant information from previous steps. The authors demonstrate this with programs including a Sudoku solver and a minimum-cost matching algorithm. [Percepta, Can LLMs Be Computers?](https://www.percepta.ai/blog/can-llms-be-computers)

A growing trace creates a performance problem. Standard full attention examines an increasingly long prefix at each decoding step. Percepta's executor uses two-dimensional hard-max attention heads, where selecting the best key can be treated as a geometric query over a convex hull. In that structured setting, the authors describe logarithmic lookup time in the trace length in place of a linear scan. The speed claim concerns those lookups under the executor's constraints. [Percepta, Exponentially Fast Attention](https://www.percepta.ai/blog/can-llms-be-computers)

<figure>
  <iframe src="https://www.percepta.ai/blog/can-llms-be-computers#the-key-unlock-exponentially-fast-attention" title="Percepta’s original HullKVCache and standard KVCache comparison" width="100%" height="680" loading="lazy" style="border: 1px solid #d1d5db; border-radius: 8px;"></iframe>
  <figcaption>In the embedded article, find “The key unlock: Exponentially Fast Attention” for the interactive comparison. <a href="https://www.percepta.ai/blog/can-llms-be-computers#the-key-unlock-exponentially-fast-attention">Open the attention comparison on Percepta</a></figcaption>
</figure>

## qwenGPU puts a small parallel machine inside Qwen

qwenGPU inserts 45 constructed layers ahead of Qwen3.5-2B's original layers. These implement a four-lane machine in which lanes run instructions together and can take different branches. Registers and memory live in the recurrent state of Gated DeltaNet layers. Generated execution records show which blocks run and which values they produce. [qwenGPU repository](https://github.com/vixhal-baraiya/qwenGPU)

The host compiles CUDA source to PTX, lowers it to the machine's supported instructions, loads the program and inputs, and retrieves the output. The added model layers carry out the instructions. The repository reports that its reference interpreter matched a Tesla T4 on 49 test kernels and that checked model launches matched the interpreter. In a separate chat test, Qwen wrote a correct kernel on its first attempt for four of five sampled questions. That distinction matters because exact execution preserves the behavior of whatever program the model supplied, including a mistaken one. [qwenGPU results](https://github.com/vixhal-baraiya/qwenGPU#results)

<figure>
  <img src="https://raw.githubusercontent.com/vixhal-baraiya/qwenGPU/main/docs/qgpu-demo.gif" alt="Animated terminal demo showing the warp scheduler, registers, and memory during a CUDA launch" loading="lazy">
  <figcaption>A CUDA launch traced through qwenGPU’s simulated machine. <a href="https://github.com/vixhal-baraiya/qwenGPU/blob/main/docs/qgpu-demo.mp4">qwenGPU original video</a></figcaption>
</figure>

## qwenLab trains smaller transformers inside the model

qwenLab appends 2,382 constructed layers to Qwen3.5-9B. The lab holds six experiments, each with a small transformer learning addition modulo 97. Its layers maintain learner weights and optimizer state, run forward and backward passes, apply AdamW updates, and evaluate the learners. The base Qwen model chooses commands and settings under a supplied protocol. The system therefore has two levels of learning behavior: the small transformers update their parameters while Qwen selects experiments with its own weights fixed. [qwenLab repository](https://github.com/DeepakSingh98/qwenlab)

In the reported session, the best selected fork achieved 49.09% test accuracy versus 15.68% for the control. That result comes from one run with a fixed protocol and a narrow search over learning rate and weight decay. The authors report about 0.64 seconds per internal training update versus 5.5 milliseconds for the same update in PyTorch on the same H100. The demonstration shows how much of a training workflow can run inside a model, including branching an experiment and comparing its results. [qwenLab results and limits](https://github.com/DeepakSingh98/qwenlab#results)

<figure>
  <img src="https://raw.githubusercontent.com/DeepakSingh98/qwenlab/main/docs/demo.gif" alt="Animated qwenLab demo showing Qwen comparing experimental branches and their training curves" loading="lazy">
  <figcaption>Qwen directs a recorded training session. The loss curves come from PyTorch replays. <a href="https://github.com/DeepakSingh98/qwenlab/blob/main/docs/demo.mp4">qwenLab original video</a></figcaption>
</figure>

## Spotlight gives a fixed executor an extensible memory

Percepta's second article asks what an executor can gain by keeping code and data in writable memory. Its Spotlight architecture addresses a small number of memory cells per token, so the amount of memory accessed at each step can stay fixed as the stored state grows. The authors use a hand-built Python environment to demonstrate package loading and replacement. The same execution module can run a different procedure after its memory changes. [Percepta, Can LLMs Grow Their Own Capabilities?](https://www.percepta.ai/blog/can-llms-grow-their-own-capabilities)

One demonstration stores the weights of a digit classifier in memory. The execution module reads the weights in chunks and evaluates the classifier. This is a concrete example of capability growth through loading a procedure or model into memory. The article's longer-term proposal is that a system could also discover, validate, and retain its own methods over time. The package and classifier examples demonstrate the mechanism for using added capabilities. [Percepta, Can LLMs Grow Their Own Capabilities?](https://www.percepta.ai/blog/can-llms-grow-their-own-capabilities)

<figure>
  <iframe src="https://www.percepta.ai/blog/can-llms-grow-their-own-capabilities#constant-work-per-step-with-unlimited-context" title="Percepta’s original Spotlight Python execution demo" width="100%" height="680" loading="lazy" style="border: 1px solid #d1d5db; border-radius: 8px;"></iframe>
  <figcaption>In the embedded article, find “Constant work per step with unlimited context” for the Python demo. <a href="https://www.percepta.ai/blog/can-llms-grow-their-own-capabilities#constant-work-per-step-with-unlimited-context">Open the Spotlight Python demo on Percepta</a></figcaption>
</figure>

## Sketchpad makes local arithmetic exhaustively checkable

In *Sketchpad: certified exact arithmetic with a small vision-language model*, Mostafa “Neo” Mohsenvand adapts Qwen3.5-0.8B to work on a rendered sheet. Each step reads three cells and produces the arithmetic output and next operation. Conventional code writes those outputs and moves the view; the model computes the result digits. The low-rank adapter learns the local operations, while the runtime supplies the procedures and working memory. [Sketchpad repository](https://github.com/NeoVand/sketchpad)

The important restriction is that no conversation history or evolving hidden state carries information between steps. Everything needed for the next step is supplied through its bounded input. A longer multiplication requires more steps, but it does not introduce an unlimited variety of local cases.

That makes exhaustive checking possible. The paper enumerates **2,446 operation-specific views** and proves in Lean that correct local outputs imply correct runs of any length, provided the runtime presents only covered views and its reference procedures are correct. All views pass on the reported Apple M4 browser deployment. The runtime's view coverage and the correctness of its classical arithmetic procedures are **not themselves formalized in Lean**. [Paper, Sections 2–3](https://arxiv.org/html/2611.04434#S3)

<figure>
  <img src="https://arxiv.org/html/2611.04434v1/fig1.png" alt="Original Sketchpad overview showing a three-cell multiplication step, certified output margins, and arithmetic comparison results" loading="lazy">
  <figcaption>Sketchpad's original overview. Figure 1 from <a href="https://arxiv.org/html/2611.04434">Mohsenvand, Sketchpad</a>, reproduced under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>.</figcaption>
</figure>

The project reports 220 correct answers on 220 pre-registered long-addition and multiplication problems. That benchmark illustrates the system; the finite-view argument supplies the guarantee under its assumptions. The dialogue version still has to choose and transcribe a computation, and its reported performance on fresh GSM8K problems falls 5.8 points below the base model. Certified execution does not certify that the right problem was posed. [Results and limitations](https://github.com/NeoVand/sketchpad#results)

The [original interactive demo](https://neovand.github.io/sketchpad/demo) shows the sheet, local views, and model outputs. It starts with recorded runs; loading the model enables execution on the reader's GPU. Its default device check covers the weakest tenth of views; a full check is available separately. [Demo and device-check behavior](https://neovand.github.io/sketchpad/demo)

## Where a reusable capability lives

These projects connect to a question from my earlier post on computational words. Suppose a model identifies a repeated procedure and gives it a name. The name makes it easier to refer to the procedure. A separate design decision determines what happens when the name is used. [Creating New Words]({% post_url 2026-11-01-creating-new-words %})

A compiler can build the procedure's execution rules into layers. An interpreter can run code supplied as tokens. A memory architecture can keep that code available across invocations and replace it when a better version arrives. From this perspective, qwenLab illustrates how a model can choose among actions offered by a built-in subsystem, while Spotlight illustrates how the set of available procedures can change after the executor is built. Sketchpad adds a different question: can every local decision in a learned procedure be checked, so that confidence in execution no longer depends only on sampled problems?

The distinction matters when evaluating a claim that a model gained a capability. Running a supplied program tests execution. Keeping that program available tests storage and retrieval. Discovering a program worth retaining, verifying it, and applying it to a new problem would test something further. Each step can be evaluated on its own, which gives a more precise account of what improved and where that improvement resides.


## Appendix: what Sketchpad's certificate covers

The certificate is tied to the model files, glyphs, precision, and execution path. A different deployment needs a full check or a justified bound on numerical changes. The paper tests sampled batch compositions, rather than exhaustively bounding every possible batch. Its guarantee concerns the restricted sheet machine; it does not show that a finite benchmark certifies an unrestricted chat model. [Certification methods and discussion](https://arxiv.org/html/2611.04434#S8)
