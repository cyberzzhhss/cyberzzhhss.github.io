---
title: "MiniMind and Mini-Omni: Learning How Language Models Are Built"
excerpt: "MiniMind makes small-model training inspectable. Mini-Omni shows how a pretrained language model can produce streaming speech."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Small Language Models
  - Multimodal Learning
  - Speech Models
  - Model Training
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

A conversational model hides several engineering problems behind a simple interface. It needs to learn language, respond to instructions, and turn its predictions into something a person can use. A voice assistant adds another challenge: it must begin speaking soon enough for the exchange to feel natural.

[MiniMind](https://github.com/jingyaogong/minimind) and [Mini-Omni](https://github.com/gpt-omni/mini-omni) expose different parts of this process. MiniMind provides a small language model and its training pipeline. Mini-Omni builds speech interaction around a pretrained text model, coordinating the generation of words and audio.

Their common value is making consequential design choices small enough to inspect. They are independent projects with different starting points, architectures, and goals.

<!--more-->

## Two projects, two questions

| | MiniMind | Mini-Omni |
| --- | --- | --- |
| Main question | How does a small model learn to become an assistant? | How can a text model generate spoken responses incrementally? |
| Starting point | A small model that can be trained from random initialization | A pretrained Qwen2-0.5B backbone |
| Main learning surface | Model code, datasets, training objectives, and evaluation | Audio features, speech tokens, parallel generation, and streaming inference |
| Mainline scope | Text, with separate related multimodal projects | Speech and text; Mini-Omni2 is a separate extension |

The size labels describe different things. MiniMind's current dense mainline is approximately 64 million parameters. Mini-Omni's 0.5-billion figure describes its language backbone, with audio components around it. Those numbers are not an equal-footing comparison of complete systems. [MiniMind model list](https://github.com/jingyaogong/minimind#readme) · [Mini-Omni technical report](https://arxiv.org/html/2408.16725)

## MiniMind: follow the path from tokens to an assistant

MiniMind is useful because it keeps the relationship between architecture, data, and training visible. Its mainline supports dense and mixture-of-experts models, alongside pretraining, supervised fine-tuning, LoRA, preference optimization, reinforcement-learning workflows, and distillation.

One headline deserves precise interpretation. The README's “two hours” refers to a measured **single SFT epoch on an NVIDIA RTX 3090**; the associated three-yuan figure refers to GPU rental for that period. It should not be read as the cost of every stage in the repository. [MiniMind README](https://github.com/jingyaogong/minimind#readme)

### The architecture is compact enough to trace

The model implementation contains token embeddings, causal attention, rotary position embeddings, normalization, and gated feed-forward layers. Its default configuration uses eight transformer layers, a hidden width of 768, and a vocabulary of 6,400 tokens.

Grouped-query attention lets multiple query heads share key/value heads. The feed-forward block combines two projections through a gate before projecting back to the model's hidden width. These mechanisms remain recognizable from larger model families, but their implementation is short enough to follow directly.

The MoE variant replaces the dense feed-forward block with a router and multiple experts. A token is sent to selected experts rather than every expert. That separates total stored capacity from the subset used for one token, while introducing routing and load-balancing decisions. [Model implementation](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

### Pretraining and instruction tuning change what is rewarded

The pretraining script exposes the ordinary training machinery: batches, model loss, backpropagation, gradient accumulation, clipping, optimizer updates, and checkpointing. For MoE models, the training loss also incorporates an auxiliary term. Following this loop makes the cost of a training step concrete. [Pretraining script](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py)

The dataset code reveals an especially important distinction. Pretraining examples contain text; padding positions are excluded from the prediction loss. SFT examples are formatted as conversations, and labels are constructed around assistant-response spans. The model sees the conversation as context, while the supervised target emphasizes the answer it should produce.

That is how a change in data formatting becomes a change in behavior. A text continuation objective and an assistant-response objective can use much of the same architecture and optimization code. [Dataset and label construction](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py) · [SFT training loop](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py)

### Preference learning adds a different signal

MiniMind's DPO implementation compares chosen and rejected responses. It computes their sequence log-probabilities under the trainable policy and a reference model, then optimizes the relative preference.

This provides a concrete example of training from comparisons: a pair communicates which answer is preferred, rather than supplying only one answer to imitate. The reference-model comparison is part of the objective, not merely an evaluation afterward. [DPO implementation](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py)

These stages answer different questions. Pretraining develops predictive structure; SFT demonstrates desired responses; preference optimization changes which responses receive relative advantage. A lower loss in one stage does not automatically establish better performance on every downstream task.

## Mini-Omni: coordinate language and speech

A voice assistant can be assembled from speech recognition, a text model, and speech synthesis. Mini-Omni explores a more integrated generation process: the model produces text and discrete audio tokens in parallel, with delays that allow text to guide the speech stream.

The technical report uses Qwen2-0.5B as its backbone and describes staged adaptation: align the new modality, adapt the language model, then perform multimodal fine-tuning. This transfers existing text capabilities into a speech system rather than relearning language from scratch. [Mini-Omni report, Sections 3–4](https://arxiv.org/html/2408.16725#S3)

### Audio still needs an encoder and a decoder

The runnable inference code loads a Whisper encoder and a SNAC audio codec. Incoming audio is converted into features and passed into the language-model pathway. Generated audio codes are reconstructed into playable sound.

This clarifies the project's claim that it needs no extra ASR or TTS models. It avoids requiring a separate transcription-to-answer-to-synthesis pipeline; it still relies on pretrained components that represent incoming sound and decode outgoing audio. The README also describes the release as trained for English output. [Inference code](https://github.com/gpt-omni/mini-omni/blob/main/inference.py) · [Repository FAQ](https://github.com/gpt-omni/mini-omni#faq)

### Parallel streams keep audio generation manageable

The implementation organizes generation around seven audio-token streams and one text stream. Flattening the audio streams into one long sequence would increase the number of sequential prediction steps. Parallel generation instead produces their outputs together, using a staggered schedule.

The streaming routine accumulates audio codes and decodes blocks as they become available. This allows playback to begin before the whole answer has been generated. Its `stream_stride` parameter controls how frequently code blocks are converted into audio. [Streaming generation routine](https://github.com/gpt-omni/mini-omni/blob/main/inference.py#L360)

There is also a batch-parallel decoding strategy. One branch generates text alone; another generates text and audio. The text-only branch's token is used to guide the speech-producing branch. The report motivates this with the observation that spoken answers could otherwise be simpler than text answers. [Decoding strategies](https://arxiv.org/html/2408.16725#S3.SS2)

“Talking while thinking” describes this coordinated generation. It does not require treating the generated text as a transparent account of internal reasoning.

### The release and the paper need to be distinguished

Mini-Omni's model code includes a conditional `post_adapter` path. The README identifies this as the TTS adapter and states that the open-source version does not support it. Reading the reported architecture alongside that release note prevents assuming that every component described in the paper is available in the shipped checkpoint. [Model code](https://github.com/gpt-omni/mini-omni/blob/main/litgpt/model.py) · [Release FAQ](https://github.com/gpt-omni/mini-omni#faq)

For someone exploring the repository, the most direct path is therefore its released inference and demo code, with the report explaining the broader training design.

## What these projects teach together

MiniMind makes changes to the learning objective easy to locate. You can connect a dataset's label mask to a loss calculation, then connect that loss to a parameter update. Mini-Omni makes the representation and scheduling of output equally concrete: text and audio have different token rates, and the way they are generated affects when a listener hears the response.

This suggests a useful way to read model code. Follow one example through the system and identify what is represented, what is predicted, and what must happen before the next useful output can appear.

For MiniMind, that path runs from text or conversation data through tokenization, labels, model outputs, loss, and optimization. For Mini-Omni, it runs from an audio file through features, coordinated token streams, codec reconstruction, and playback.

Both make the gap between a model and a working assistant visible. Architecture matters, but so do the supervision, interfaces, and execution schedule around it.

## Appendix: where to start reading

| Project | File | What to inspect |
| --- | --- | --- |
| MiniMind | [`model/model_minimind.py`](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py) | Attention, feed-forward layers, expert routing, and the model's loss |
| MiniMind | [`dataset/lm_dataset.py`](https://github.com/jingyaogong/minimind/blob/master/dataset/lm_dataset.py) | Text examples, chat formatting, assistant-label masks, preference pairs |
| MiniMind | [`trainer/train_pretrain.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py) | Gradient accumulation, optimizer steps, and checkpoints |
| MiniMind | [`trainer/train_dpo.py`](https://github.com/jingyaogong/minimind/blob/master/trainer/train_dpo.py) | Policy/reference comparisons and the preference objective |
| Mini-Omni | [`inference.py`](https://github.com/gpt-omni/mini-omni/blob/main/inference.py) | Audio preparation, batch inference, streaming stride, and decoding |
| Mini-Omni | [`litgpt/model.py`](https://github.com/gpt-omni/mini-omni/blob/main/litgpt/model.py) | Audio adaptation and transformer computation |

Mini-Omni's illustrated inference routine first loads an audio file, then streams the response. Streaming output alone does not establish simultaneous, interruption-aware input and output; inspect the frontend and turn handling separately if that is the behavior you need.

MiniMind's core architecture is implemented directly in PyTorch, while its code also imports infrastructure from libraries such as Transformers. “From scratch” refers to the ability to initialize and train the small model and inspect its core implementation; it does not mean eliminating dependencies or the work involved in preparing training data.

*Repository contents reviewed November 10, 2026. MiniMind's current mainline differs from older tutorials describing its earlier 26M-parameter models.*
