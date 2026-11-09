---
title: "Redesigning Language Models During Training"
excerpt: "DoubleAI gives a coding agent a role in training. The agent diagnoses failures, proposes new computations, and tests where they fit inside a model."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Language Models
  - Model Architecture
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

A model learning to solve algorithm problems can do well on the input lengths it sees during training and stumble on the lengths in between. An engineer might inspect those failures and add a computation that handles the missing case. DoubleAI asks what happens when a coding agent can do that work while the model is still training. In one reported experiment, adding agent-written code raised accuracy on those intermediate lengths from 33% to 79%. [DoubleAI’s research article](https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#taking-on-jaggedness)

<!--more-->

## The agent diagnoses a failure and proposes a change

DoubleAI calls this approach **mechanistic intervention**. The agent inspects a model's internal representations and looks for recurring errors. One diagnostic, called a *phantom gradient*, measures how a small change to an internal representation would affect the loss. A recurring signal can suggest where a correction could help. The agent still has to work out what computation would produce it. [Nanochat experiment and diagnostics](https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#nanochat-where-does-a-language-model-need-help)

Imagine a model trying to keep track of the largest number in a list. A useful intervention might read the numbers from the model's internal representation, calculate the largest value seen so far, and return that result to a later layer. This splits the work into three pieces. The model supplies a representation, a program performs the operation, and the network learns how to use the result. In DoubleAI's experiments, a running maximum is one example of a computation a code layer can supply. [Code layers on CLRS-Text](https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#taking-on-jaggedness)

## How the code enters the model

In the training experiments, a **code layer** is an extra branch in the model's forward pass. It runs alongside the existing computation and adds a result to the information that flows to later layers. A learnable multiplier, called a *gate*, scales that addition. It starts at zero, so the new branch contributes nothing at first. As training resumes, the optimizer can change how much the branch contributes. [Code layer integration](https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#nanochat-where-does-a-language-model-need-help)

The gate gives two kinds of changes their own roles. The agent proposes a discrete edit by writing a program and choosing where to connect it. Training adjusts numerical weights, including the gate. DoubleAI compares candidates with runs from the same checkpoint and with controls that scramble the information a branch carries. Those comparisons help separate the benefit of the computation from the benefit of giving the model extra capacity. [Candidate comparisons](https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#nanochat-where-does-a-language-model-need-help)

<figure>
  <iframe src="{{ '/assets/html/doubleai-training-figures/code-layer-gate.html' | relative_url }}" title="DoubleAI Figure 5. A code layer enters behind a gate that starts at zero." width="100%" height="800" loading="lazy" sandbox="allow-scripts" style="border: 1px solid #d1d5db; border-radius: 8px;"></iframe>
  <figcaption>Figure 5 shows how the new branch begins to contribute as training resumes. <a href="https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#nanochat-where-does-a-language-model-need-help">View on DoubleAI</a></figcaption>
</figure>

## What the experiments measured

| Setup | Intervention | Result reported by DoubleAI |
| --- | --- | --- |
| Nanochat | Added three code layers during training | Lower validation loss through the end of the run |
| Sliding-window transformer | Added the same lookup halfway through training or at the start | 72% vs. 55% recall beyond the attention window |
| CLRS-Text | Added code layers during training | 84% vs. 55% overall accuracy and 79% vs. 33% at intermediate input lengths |
| DeepSeek-V4.1-Flash | Used a probe on completed solutions to decide when to retry | 71% vs. 64% contest math accuracy |
| Qwen3-8B | Wrote a policy that uses existing probes to stop, restart, or retry | 66% vs. 59% contest math accuracy, with more thinking tokens than a single attempt |

The rows compare different kinds of intervention. The first three concern changes to models during training. The DeepSeek and Qwen results concern decisions made while generating answers. For Qwen, the policy used about 2.8 times the thinking tokens of a single attempt. Its improvement over a length-based retry policy was 3.3 percentage points, compared with seven points over the single-attempt baseline. That makes the choice of comparison important when interpreting the gain. [DoubleAI’s experiments and baselines](https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train)

<figure>
  <iframe src="{{ '/assets/html/doubleai-training-figures/clrs-training-accuracy.html' | relative_url }}" title="DoubleAI Figure 8. CLRS-Text accuracy across training and code layer insertions." width="100%" height="800" loading="lazy" sandbox="allow-scripts" style="border: 1px solid #d1d5db; border-radius: 8px;"></iframe>
  <figcaption>Figure 8 shows CLRS-Text accuracy during training. Select an insertion to see what its code layer computes. <a href="https://www.doubleai.com/research/can-llms-redesign-themselves-as-they-train#taking-on-jaggedness">View on DoubleAI</a></figcaption>
</figure>

## Choosing what the model can compute

In my earlier post, [Computation Inside Language Models]({% post_url 2026-11-04-computation-inside-language-models %}), the examples mostly started with a procedure and asked where it could run. DoubleAI starts with an observed failure and asks what procedure should be added, where to connect it, and when to introduce it. The lookup experiment makes the timing question concrete. The same operation produced different recall depending on when the model acquired it.

The next question is whether a system can reuse what it learns about intervention. A tool that finds a bottleneck in one model might help diagnose another. To test that possibility, I would track which diagnostic methods transfer across tasks and architectures, and count the cost of the agent's search along with the cost of training the model it changes.
