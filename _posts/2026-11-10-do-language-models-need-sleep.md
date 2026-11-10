---
title: "Do Language Models Need Sleep?"
excerpt: "Four papers connect the difficulty of staying adaptable with test-time training, offline memory processing, and learning across different timescales."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Continual Learning
  - Test-Time Training
  - Memory Consolidation
  - Large Language Models
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

Imagine a coding assistant that spends a week working in an unfamiliar repository. It encounters naming conventions, architectural decisions, and the reasons earlier fixes failed. How should that experience change what it can do next week?

Saving the conversations makes the experience retrievable. Learning from it raises further questions: what should become internal memory, when should that memory be updated, and can the model remain able to learn after many such updates?

In [*The Next Scaling Axis of LLM Is… Sleeping?*](https://www.youtube.com/watch?v=rAxTxqIHh9E), bycloud connects four papers around these questions. Their shared theme is the organization of learning over time. The two “sleep” papers propose different mechanisms, so understanding where each stores information is essential.

<!--more-->

<div style="position:relative;width:100%;aspect-ratio:16/9;margin:1.5em 0;">
  <iframe src="https://www.youtube-nocookie.com/embed/rAxTxqIHh9E" title="bycloud: The Next Scaling Axis of LLM Is… Sleeping?" loading="lazy" style="position:absolute;width:100%;height:100%;border:0;" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## The problem: retaining the ability to learn

**Can Scale Save Us From Plasticity Loss in Large Language Models?**, by Hernandez-Garcia, Figliolia, and Millidge, studies *plasticity*: how efficiently a trained network can learn something new. Catastrophic forgetting concerns damage to older capabilities; plasticity loss concerns deterioration in acquiring new ones.

The authors train GPT-style models on repeated cycles of eight languages. Periodically, they copy a checkpoint and train it on held-out Vietnamese with a fixed budget. These probe updates are discarded. Adaptation initially improves, then deteriorates with extended pretraining.

Across models with 5M–314M non-embedding parameters, larger models delay this deterioration. The estimated onset scales sublinearly with size, suggesting diminishing returns from scale alone. Smaller-model experiments also find the effect under a stationary language mixture.

The measured result concerns these architectures and training conditions. Predictions about billion-parameter models are extrapolations, rather than direct measurements in this study. [Paper, Sections IV–V](https://arxiv.org/html/2606.24752v1#S4)

## One response: make existing weights writable

**In-Place Test-Time Training**, by Feng and colleagues, repurposes the final projection in a transformer's feed-forward block as adaptable *fast weights*. Other projections remain stable during the inference-time updates, and attention remains part of the architecture.

Updates happen in chunks. Their objective is aligned with predicting the next token, so the memory is trained to store information useful for future prediction. The contribution is a way to introduce adaptive memory within an existing transformer architecture.

“Drop-in” describes architectural compatibility. The Qwen3-4B experiment still uses approximately 35 billion tokens of continued training for both the baseline and the enhanced model. At 128k context length, RULER accuracy rises from 74.8% to 77.0%.

This supports improved use of long context under the tested protocol. It does not establish indefinite retention across independent sessions or resolution of plasticity loss. [Paper, Sections 3–4 and Table 1](https://arxiv.org/html/2604.06169v1#S3)

## Sleep as computation before forgetting the context

**Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference**, by Lee, McLeish, Goldstein, and Fanti, asks how much computation is needed to form useful memory.

The model combines attention with state-space memory blocks. Before clearing an attention window, it performs repeated passes that refine its fast-weight state. That state survives cache eviction and supports later single-pass prediction.

Experiments cover cellular automata, graph traversal, and GSM-Infinite mathematical reasoning. Additional consolidation passes improve performance, especially on harder reasoning problems. The intervention changes the computation devoted to organizing memory, while preserving the prediction-phase constraint.

The extra computation still costs time: preserving wake-time prediction latency does not remove the sleep interval. And persistence beyond cache eviction is different from permanent acquisition in the pretrained parameters. Resetting the fast-weight state removes this memory. [Paper, Sections 5–7](https://arxiv.org/html/2605.26099v3#S5)

## Sleep as transfer into slower memory

**Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories**, by Behrouz, Hashemi, Javanmard, and Mirrokni, proposes a hierarchy of memories updated at different frequencies.

Its *knowledge seeding* stage activates additional capacity and distills knowledge from faster modules into slower ones. In its sparse implementation, new low-rank experts provide space for transferred knowledge. Its *dreaming* stage learns to generate synthetic training experiences, rewarding those that improve downstream performance after adaptation.

In the 200-passage SQuAD experiment, the four-level system reports 46.2% accuracy when answering without the passages in context, compared with 36.2% when dreaming is removed. This provides evidence for the role of synthetic rehearsal within that setup.

Here, sleep involves parameter expansion, knowledge transfer, and training. It is a different operation from Lee and colleagues' recurrent refinement of fast-weight state. [Paper, Sections 3–4 and Table 3](https://arxiv.org/html/2606.03979v2#S3)

## Four contributions, different questions

| Paper | Central question | Contribution |
| --- | --- | --- |
| Can Scale Save Us… | Does extended training preserve adaptability? | A multilingual probe and empirical scaling analysis |
| In-Place TTT | Where can inference-time adaptation live? | Writable feed-forward projections |
| Offline Recurrence | Can more memory-formation compute help? | Repeated consolidation before cache eviction |
| Self-Modification and Consolidation | How can recent learning move into slower storage? | Knowledge seeding and synthetic rehearsal |

These form a conceptual roadmap rather than a single experimentally validated pipeline. The first diagnoses a learning failure; the others investigate mechanisms for adaptation and consolidation. Combining them would require its own evidence.

For the coding assistant, the design questions become concrete. Which details need exact retrieval? Which recurring patterns should influence its internal behavior? How long should that influence persist? How much processing can happen between requests?

A useful synthesis is that memory needs both a location and an update schedule. A system can preserve a transcript yet fail to use it well. It can adapt successfully today yet become harder to teach later. Evaluating a continuing learner therefore requires tracking the acquisition of new skills, retention of earlier ones, and the computation spent achieving both.

## Appendix: technical distinctions behind the headlines

- **Plasticity measurement.** The Vietnamese probe uses the area under a validation-loss curve over adaptation steps. Lower area indicates faster adaptation. The fitted onset is an empirical relationship, rather than a universal lifetime for an LLM. [Sections IV–V](https://arxiv.org/html/2606.24752v1#S4)
- **In-Place TTT's update target.** The analysis uses a next-token embedding target; the implementation learns a local combination of future-token embeddings. Updates affect later chunks, and fast weights reset at document boundaries. [Sections 3.1–3.2](https://arxiv.org/html/2604.06169v1#S3.SS1)
- **Offline recurrence's learning rule.** During deployment, memory changes through local state updates. Training backpropagates through the recurrent consolidation process to learn how those updates support later predictions. [Section 5](https://arxiv.org/html/2605.26099v3#S5)
- **The ARC headline.** The self-modification paper's 80% result uses eight filtered held-out tasks and reports the fraction of independently applied dreams yielding a correct answer. It is not a full-benchmark ARC accuracy. [Appendix B.3](https://arxiv.org/html/2606.03979v2#A2.SS3)

## Sources

- bycloud, [The Next Scaling Axis of LLM Is… Sleeping?](https://www.youtube.com/watch?v=rAxTxqIHh9E), video and transcript.
- Hernandez-Garcia, Figliolia, and Millidge, [Can Scale Save Us From Plasticity Loss in Large Language Models?](https://arxiv.org/abs/2606.24752), v1.
- Feng et al., [In-Place Test-Time Training](https://arxiv.org/abs/2604.06169), v1.
- Lee et al., [Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference](https://arxiv.org/abs/2605.26099), v3.
- Behrouz et al., [Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories](https://arxiv.org/abs/2606.03979), v2.
