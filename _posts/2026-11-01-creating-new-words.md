---
title: "Creating New Words: From Learned Programs to Dynamic Vocabularies"
excerpt: "How six papers connect reusable program abstractions, neural representations, dynamic vocabularies, tool creation, and emergent language."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - Language Models
  - Program Synthesis
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

Our research began with a question about **computational concept formation**: can an AI identify a recurring computation, assign it a symbol, and compose it with subsequent discoveries? Repeated addition provides an elementary model:

$$
f(n,x)=\underbrace{x+\cdots+x}_{n\text{ terms}}
\quad\longrightarrow\quad
\sigma(n,x):=f(n,x).
$$

Here, $n$ is a positive integer, $x$ is the value being added, and $\sigma$ names the procedure $f$. If $\sigma$ is assigned a dedicated input token, let $\mathbf e_\sigma\in\mathbb R^{d_{\mathrm{model}}}$ denote that token's **input embedding vector**, where $d_{\mathrm{model}}$ is the model's embedding dimension. This vector lets the model receive the symbol as input; it does not itself guarantee that the model can execute $f$. An output vocabulary entry may require a separate output vector.

The association $\sigma\leftrightarrow f\leftrightarrow\mathbf e_\sigma$ is our motivating proposal, not a mechanism jointly demonstrated by the six papers. They address distinct parts of program induction, vocabulary expansion, tool creation, and emergent communication.

<!--more-->

## Research lineage and scope

The direct methodological lineage is **DreamCoder → Bayesian Program Learning by Decompiling Amortized Knowledge**. The remaining papers develop related mechanisms rather than a single sequence of extensions.

| Direction | Papers | Object being learned or extended |
| --- | --- | --- |
| Program abstraction | DreamCoder; Decompiling Amortized Knowledge | Executable library and neural search policy |
| Lexical representation | From Tokens to Words | Access to internal word representations |
| Dynamic generation | Generation with Dynamic Vocabulary | Input and output units available on demand |
| Tool construction | Large Language Models as Tool Makers | Reusable external procedures |
| Emergent communication | GlossoGen | Shared linguistic conventions |

The papers address different mechanisms. The following summaries distinguish their methods, principal contributions, and scope.

## 1. DreamCoder: learning a language of reusable programs

**Kevin Ellis and colleagues · 2020 preprint**  
[DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning](https://arxiv.org/abs/2006.08381)

DreamCoder searches for programs that explain task examples. It extracts reusable abstractions from discovered solutions and adds them to a library. A neural recognition model learns to guide subsequent searches using generated tasks and previously solved examples.

The library and search model improve together. Useful computations become available as components, and later abstractions can call earlier ones. This can make solutions shorter and more accessible within a limited search budget.

**Principal contribution:** a system that jointly develops a computational language and learns how to search within it. It provides the strongest conceptual precedent here for cumulative abstraction learning.

**Scope:** the learned concepts are executable program components within a designed synthesis framework. The paper does not establish the full combination of autonomous concept naming and language-model token allocation described in our opening idea.

## 2. Decompiling Amortized Knowledge: using neural knowledge to select abstractions

**Alessandro Palmarini and colleagues · 2023 preprint**  
[Bayesian Program Learning by Decompiling Amortized Knowledge](https://arxiv.org/abs/2306.07856)

This paper directly extends DreamCoder by using knowledge accumulated in its neural search policy to guide library learning. Its central distinction is between avoiding unpromising search branches and shortening the programs that must be constructed.

One proposed variant ranks candidate components by their average probability under the recognition model. The authors identify limitations of that approach and develop another variant that models whether adding a component is likely to benefit search overall.

**Principal contribution:** connect the knowledge used to navigate a search space with the knowledge used to restructure that space. The authors report faster domain proficiency and improved generalization in evaluated domains, particularly with fewer example solutions.

**Scope:** adding abstractions creates a tradeoff. Programs may become shorter, but the system also has more components to choose from. Library growth therefore needs a selection criterion; more abstractions do not automatically imply easier search.

## 3. From Tokens to Words: accessing an internal lexicon

**Guy Kaplan and colleagues · 2024 preprint, ICLR 2025**  
[From Tokens to Words: On the Inner Lexicon of LLMs](https://arxiv.org/abs/2410.05864)

The authors investigate how language models represent words split into multiple subword tokens. They find evidence that information about the complete word accumulates at its final token, primarily in early and middle layers. This process extends to alternative token splits and words outside the tokenizer's vocabulary.

Their vocabulary-expansion procedure identifies internal states from which a whole word can be recovered. Learned projections map those states into input and output vocabulary spaces, followed by a refinement step for the new entries.

**Principal contribution:** evidence that word-level representations extend beyond the tokenizer's explicit inventory, together with a method for exposing those representations as vocabulary entries.

**Scope:** the core model remains frozen, but auxiliary projections and refinement mappings are learned. The work concerns lexical representation and vocabulary access; it does not demonstrate the discovery of executable computational concepts.

## 4. Generation with Dynamic Vocabulary: generating text spans as units

**Yanting Liu and colleagues · 2024**  
[Generation with Dynamic Vocabulary](https://arxiv.org/abs/2410.08481)

This paper augments a language model's static vocabulary with text spans available on demand. A trained phrase encoder produces representations that let these spans function as input and output units. A phrase that ordinarily requires several tokens can then be generated in one step.

Supporting this flexibility requires more than appending vocabulary entries. The training procedure interleaves ordinary tokens and phrase units and uses informative negative examples to help the model learn useful phrase representations.

**Principal contribution:** a mechanism for flexible generation units, with reported improvements in evaluated generation settings and applications to domain adaptation and citation generation.

**Scope:** the method changes how text is represented and generated. It does not establish that the model discovers the conceptual or executable meaning of a new mathematical operation.

## 5. Large Language Models as Tool Makers: creating procedures for repeated use

**Tianle Cai and colleagues · 2023 preprint**  
[Large Language Models as Tool Makers](https://arxiv.org/abs/2305.17126)

The LATM framework separates tool construction from tool application. A tool-making LLM generates reusable Python functions from task demonstrations. A tool-using LLM converts subsequent problems into calls to those functions.

This division allows a stronger model to construct tools that a lighter model can use. The cost of building a tool is spread across related task instances. The paper also introduces a dispatcher that checks for suitable existing tools and helps trigger tool creation for new task types.

**Principal contribution:** a practical framework for reusable executable knowledge, demonstrating how tool creation can improve the capabilities and cost efficiency of tool-using models in evaluated settings.

**Scope:** the capability resides partly in external code and execution. Its benefits do not require adding a new token or changing the language model's internal representation of an operation.

## 6. GlossoGen: developing and transmitting shared conventions

**Elias Stengel-Eskin and colleagues · September 2026 preprint**  
[GlossoGen: Emergent Language in Complex Multi-Agent LLM Interactions](https://arxiv.org/abs/2609.01491)

GlossoGen studies language evolution in interacting LLM agents. Its SaveVeyru scenario gives agents partial information and requires communication under pressure. Agents both send and receive messages rather than occupying fixed speaker and listener roles.

The authors report conventions with compositional and morphologically productive structure. Efficiency pressure, model capability, and postmortem opportunities to discuss conventions influence emergence. They also study how newcomers acquire established conventions through usage.

**Principal contribution:** investigate both the emergence and transmission of structured communication in a controlled multi-agent setting. Productivity is tested through novel form–meaning combinations, going beyond the reuse of memorized codes.

**Scope:** the analysis uses induced linguistic paradigms and model-based semantic judgments, with separate robustness evaluations. The findings concern communication conventions in the studied environments, rather than a general mechanism for learning executable concepts or allocating embeddings.

## What connects these contributions

The direct lineage from DreamCoder to its decompilation extension concerns **learning and selecting computational abstractions**. The two vocabulary papers address **access to neural representations and flexible generation units**. LATM studies **external executable reuse**, while GlossoGen examines **shared conventions and their transmission**.

These are complementary forms of learning, with different evidence requirements. A shorter expression, a reusable procedure, an accessible word representation, and a convention understood by another agent each establish a different capability. Keeping those distinctions explicit makes the connection to computational concept formation precise without attributing a unified mechanism to the papers.
