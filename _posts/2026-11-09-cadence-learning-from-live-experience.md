---
title: "How Cadence Learns From Live Experience"
excerpt: "Cadence is a Python research library that uses a recurrent network to choose actions, remember outcomes, and adapt as an environment changes."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Continual Learning
  - Reinforcement Learning
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

Imagine a creature that sees one of two cues and must choose left or right. It learns which choice pays off. Halfway through its life, the rule flips. Its old behavior now earns the wrong outcome, and it has to recover while continuing to act.

Cadence is a Python research library built for experiments like this. Each interaction gives the system a new observation and, when available, the outcome of its previous action. The network settles on its next action, and the same stream can also teach it. The repository includes an example where the reward rule reverses partway through a run. [Cadence repository](https://github.com/muellerberndt/cadence) · [Continuous interaction example](https://github.com/muellerberndt/cadence/blob/main/docs/start.md#routine-and-repair-live)

In his article on embodied AGI, Cadence creator Bernhard Mueller puts the same problem in a kitchen. A robot has to learn what happens when it grips a particular cup or pulls on a drawer that sticks. Its experience has to carry forward into the next attempt, while the robot continues to operate within its compute budget. Cadence explores how recurrent activity, memory, and local updates could support that kind of learning. [Bernhard Mueller, The road to embodied AGI with Cadence](https://muellerberndt.medium.com/you-dont-need-attention-after-all-the-road-to-embodied-agi-with-cadence-8606a64e40df)

<!--more-->

<figure>
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*_S5bUYAovJa0Fq7Alh4u6g.png" width="700" height="477" loading="lazy" alt="Cadence readback loop connecting observations, candidate actions, stored records, and witnessed outcomes">
  <figcaption>A candidate action can consult memory. The result of a real action can update what the system remembers. <a href="https://muellerberndt.medium.com/you-dont-need-attention-after-all-the-road-to-embodied-agi-with-cadence-8606a64e40df#33cf">Diagram from Bernhard Mueller’s article</a>.</figcaption>
</figure>

## An answer forms as the network settles

Picture the cue activating one group of neurons while two other groups represent the possible actions. Connections carry influence in both directions. The cue shapes the network's interpretation, and the state of the action groups can influence earlier activity while the answer forms. Cadence repeatedly updates this shared state under the current observation. Once the state satisfies the network's equations within a tolerance, the system can use it to choose an action. This is what the project calls *settling*. [Cadence’s guide to settling](https://github.com/muellerberndt/cadence/blob/main/docs/settling.md#settling-and-equilibrium)

<figure>
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*53-rjQjHph5wQqX8-8Fj2Q.png" width="700" height="490" loading="lazy" alt="A Cadence neuron combining weighted activity from its neighbors and an external input">
  <figcaption>One neuron adjusts its state in response to its inputs. The example illustrates a local update. <a href="https://muellerberndt.medium.com/you-dont-need-attention-after-all-the-road-to-embodied-agi-with-cadence-8606a64e40df#77ba">Diagram from Bernhard Mueller’s article</a>.</figcaption>
</figure>

A settled answer can still be wrong. The settlement check tells us how closely the current state satisfies the network’s equations. The reward tells us whether the choice actually worked. Cadence records the solver's error and how many sweeps it needed, so a slow decision and a bad decision can be measured separately. [Settlement checks and refusal](https://github.com/muellerberndt/cadence/blob/main/docs/start.md#qualification-and-refusal)

## How the system learns from a correction

Suppose a teacher shows the creature that a cue calls for the left action. Cadence can let the network settle on its own, then run additional settles with small pushes toward and away from the taught answer. Each connection's update uses the difference in the activity of the neurons it joins across the nudged states. The rule comes from equilibrium propagation, a family of methods that uses differences between equilibria to estimate how weights should change. [Cadence’s learning rule](https://github.com/muellerberndt/cadence/blob/main/docs/settling.md#learning-the-freenudged-rule) · [Equilibrium Propagation](https://arxiv.org/abs/1602.05179)

When the environment supplies a reward instead of the correct answer, it gives the system a different kind of information. A connection keeps a fading record of its recent contribution to action choices. When the outcome arrives, a shared signal compares the earlier prediction with the reward received plus the estimated value of what follows. That signal scales the updates to those connections. This gives a later outcome a way to affect a recent choice. [Cadence’s reward learning guide](https://github.com/muellerberndt/cadence/blob/main/docs/reward.md#eligibility-traces-and-dopamine)

In the continuing interaction mode, Cadence uses a rule it calls *arousal* to decide when to spend effort learning. In routine conditions it can choose an action after one settle. An unexpected outcome or a drop in reward can switch on exploration, learning, and memory writes. In the reversal example, the interesting part is what happens immediately after the change. An action that used to pay now fails, and the system can revise its behavior while the task is still running. [Routine and repair in Cadence](https://github.com/muellerberndt/cadence/blob/main/docs/start.md#routine-and-repair-live)

## Three kinds of change

An action can change because the current observation is different, because the system remembers an earlier outcome, or because experience changed its connections. Cadence exposes those mechanisms separately. [State lifetimes in Cadence](https://github.com/muellerberndt/cadence/blob/main/docs/settling.md#three-state-lifetimes)

| What changes | What it does in the cue example |
| --- | --- |
| Current neural activity | Forms a choice for the cue in front of the creature |
| Carried state and memory | Brings information from earlier moments into the current choice |
| Learned connection strengths | Changes how the network responds to future cues |

<figure>
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*W5_OJgxH9Fthbo5QirOfVA.png" width="700" height="591" loading="lazy" alt="Three forms of state carried between decisions in Cadence: neuron potentials, fading traces, and stored records">
  <figcaption>The previous neural state gives the next solve a starting point. Fading traces and stored associations can also influence what the system sees in the next moment. <a href="https://muellerberndt.medium.com/you-dont-need-attention-after-all-the-road-to-embodied-agi-with-cadence-8606a64e40df#184a">Diagram from Bernhard Mueller’s article</a>.</figcaption>
</figure>

That separation makes the reward reversal a better test than a final accuracy score alone. If the creature recovers, we can ask whether it recalled an earlier association or changed its policy through new feedback. A comparison with memory disabled or learning frozen could help tell those explanations apart.

## From simulation to a physical robot

Mueller describes simulated bodies that learn from stored outcomes and a separate solver that reduces work when a change is local. Those results test parts of the design. The larger claim depends on putting them together in a physical robot that can learn a new task, adapt when conditions change, and retain older skills. The comparison would have to count the cost of settling, communication, memory, and learning as well as task success. [Mueller’s account of the evidence and open tests](https://muellerberndt.medium.com/you-dont-need-attention-after-all-the-road-to-embodied-agi-with-cadence-8606a64e40df#55f2)

## Appendix

### What the equilibrium comparison tests

Cadence's finite grammar benchmark compares its local update with an implicit equilibrium gradient on the same network. The task is to judge subject and verb agreement with a distractor noun between them, using supplied inputs that identify the words and their roles. The setup has 24 training sentences and eight test sentences. It tests whether the local update can learn a small nonlinear judgment from those inputs. The project separately describes full-sentence language as a research goal. [Local equilibrium benchmark](https://github.com/muellerberndt/cadence/tree/main/benchmarks/local_equilibrium) · [Cadence research goals](https://github.com/muellerberndt/cadence#why-cadence)

The gradient connection depends on conditions such as smooth, stable equilibria and symmetric effective weights. At a finite nudge and with imperfect solves, the local contrast is an estimate. Evaluating efficiency also means counting the work of all the solves used to answer and learn, along with memory maintenance. For the reversal task, I would compare recovery time and the cost of each interaction against a recurrent learner given the same observations and rewards. [Cadence’s learning assumptions and comparison](https://github.com/muellerberndt/cadence#where-the-learning-signal-comes-from)
