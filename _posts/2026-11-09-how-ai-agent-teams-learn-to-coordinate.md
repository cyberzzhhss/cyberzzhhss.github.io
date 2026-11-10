---
title: "How AI Agent Teams Learn to Coordinate"
excerpt: "General Reasoning trains agents with a shared reward and a deadline. The results show when parallel work can save time and why coordination still has to earn its cost."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Multi-Agent Systems
  - Reinforcement Learning
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

Imagine asking an AI agent to fix three bugs. One agent could tackle them in order, or three agents could work at once. The team can finish sooner if the fixes are independent. If two agents edit the same part of the code, they may spend much of that saved time sorting out their changes.

General Reasoning’s Avalon-M1 experiments ask whether AI agents can learn when it pays to work together. The researchers give the agents a shared score, tools for communicating, and a deadline. [General Reasoning’s How to Train a Swarm](https://www.gr.inc/blog/how-to-train-a-swarm)

<!--more-->

## What a deadline teaches

In Avalon-M1, the lead agent and its specialists are instances of the same model. The lead can recruit specialists, and the team can share work through messages, channels, and a common workspace. During training, the outcome of the whole run shapes how the model learns from each agent’s actions. General Reasoning calls this **reinforcement learning from collective rewards**, or RLCR. [Avalon-M1 technical report, methodology](https://www.gr.inc/avalonm1.pdf#page=9)

Say one team passes seven tests before time runs out and another passes four. The first team earns a higher score. If both pass seven, they tie even if one finished earlier. The deadline makes speed matter when it helps the team complete more work. [How to Train a Swarm, time limits](https://www.gr.inc/blog/how-to-train-a-swarm#making-parallel-work-pay)

To make that incentive work, the task needs a way to score partial progress. A test suite can do this because it can check the code whenever a run stops. General Reasoning increases the time limit over training so teams can practice on tasks long enough for the work of dividing and coordinating a job to pay off. [Avalon-M1 technical report, training horizons](https://www.gr.inc/avalonm1.pdf#page=13)

<figure>
  <iframe src="{{ '/assets/html/swarm-figures.html' | relative_url }}#deadline" title="General Reasoning’s interactive deadline schedule" width="100%" height="610" loading="lazy" sandbox="allow-scripts" style="border: 1px solid #d1d5db; border-radius: 8px;"></iframe>
  <figcaption>Move the slider to see how much time a team gets at different stages of training and how that affects the example runs. <a href="https://www.gr.inc/blog/how-to-train-a-swarm#making-parallel-work-pay">View the original figure in How to Train a Swarm</a>.</figcaption>
</figure>

## When parallel work pays

Suppose each of the three fixes takes 18 minutes and the review takes 10. One agent working through them in order finishes in **64 minutes**. Three agents working independently can finish in **28 minutes**, including the review. If one fix depends on another, the review has to wait for both, and the schedule grows to **46 minutes**.

These are illustrative times. Each schedule contains the same amount of work. What changes is how much can happen at once. A quick message can save time if it warns someone away from duplicate work. A handoff can also add a delay if the next agent has to wait for an answer.

## What Avalon-M1 measured

General Reasoning reports that, when the solo agent generated roughly as many output tokens as the team, its estimated run time was about four times as long. The researchers built that estimate from event logs, including the time to generate tokens, the time tools took to run, and which agents needed to wait for one another’s messages. Actual elapsed time also depends on how the serving system queues and batches requests. [General Reasoning’s results](https://www.gr.inc/blog/how-to-train-a-swarm#signs-of-life) · [Avalon-M1 technical report, latency methodology](https://www.gr.inc/avalonm1.pdf#page=59)

Giving a solo agent the team’s token budget closes much of the score gap on some tasks. On others, the team or the solo agent still comes out ahead. The EdgeBench figure shows a different comparison. It tracks how scores change over the course of a run, including the team’s slower start. [Avalon-M1 technical report, token comparison](https://www.gr.inc/avalonm1.pdf#page=20)

<figure>
  <iframe src="{{ '/assets/html/swarm-figures.html' | relative_url }}#edgebench" title="General Reasoning’s interactive EdgeBench reward chart" width="100%" height="470" loading="lazy" sandbox="allow-scripts" style="border: 1px solid #d1d5db; border-radius: 8px;"></iframe>
  <figcaption>On EdgeBench, the solo agent makes progress sooner. The team pulls ahead later. The time axis uses the researchers’ estimates. <a href="https://www.gr.inc/blog/how-to-train-a-swarm#signs-of-life">View the original figure in How to Train a Swarm</a>.</figcaption>
</figure>

## What a team score can miss

Suppose one agent finds the cause of a bug, another repeats the same search, and a third catches an error in the patch. All three share the score. It tells the training system how the team did overall, with little detail about which actions made the difference. That becomes harder to untangle as teams grow and runs get longer. [General Reasoning’s open problems](https://www.gr.inc/blog/how-to-train-a-swarm#outstanding-problems)

For a bug-fixing task, I’d compare a solo agent with more time to think, several independent attempts, and a team that can talk. I’d give them the same deadline and track tests passed, tokens used, and time spent waiting for handoffs. The most interesting cases are the ones where one agent learns something that lets another finish sooner.
