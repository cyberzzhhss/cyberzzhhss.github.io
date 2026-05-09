---
title: "Solving International Mathematical Olympiad with GPT-OSS-120B (AIMO3) "
excerpt_separator: "Building a high-scoring AIMO3 solver with reasoning loops, majority voting, and aggressive prompt engineering"
tags:
  - Agent
  - GPT-OSS-120B
  -
categories:
  - Machine Learning
  - Computer Science
  - Mathematics
  - Artificial Intelligence
classes: wide
layout: single
---


<h3>My Work</h3>

<p>
This post documents my hosted implementation and results.
</p>

<p>
<a href="/assets/html/aimo3-gpt-oss-120b.html">
AIMO3 + GPT-OSS 120B Jupyter Notebook HTML
</a>
</p>

<p>
<strong>Result:</strong> On the IMO-style test set, my solver achieved a <strong>10/10</strong> solve rate.
</p>


<p>
AIMO3-style math problems reward more than a single model call: the strongest solvers combine 
<strong>reasoning</strong>, <strong>verification</strong>, <strong>self-consistency</strong>, and careful 
<strong>answer extraction</strong>.
</p>

<!--more-->

<h3>Background</h3>

<p>
This work is based on the 
<a href="https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3">
AI Mathematical Olympiad — Progress Prize 3 (AIMO3)
</a>, a Kaggle competition focused on solving challenging IMO-style math problems using large language models.
</p>

<p>
Unlike standard benchmarks, AIMO3 emphasizes <strong>multi-step reasoning</strong>, <strong>robust answer extraction</strong>, and <strong>system-level design</strong> rather than single-pass model performance.
</p>



<h3>Reference</h3>

<p>
This writeup was informed by a public Kaggle baseline:
</p>

<p>
<a href="https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook">
Kaggle baseline
</a>
</p>
---

<h3>Intro</h3>

When I first approached AIMO3, the obvious solution was to ask the model for a step-by-step answer and hope the final number was correct.

That is not enough.

These problems are long, fragile, and often require several failed paths before the right one appears. So the solver I built treats the model less like a deterministic oracle and more like a group of mathematical problem-solvers working in parallel.

The main idea is:

> Don’t trust one answer. Generate several lines of reasoning, verify where possible, and only stop when the evidence is strong enough.

---

<h3>System Overview</h3>

My actual implementation uses a local GPT-OSS-120B model served through vLLM’s OpenAI-compatible API. I use Harmony formatting for tool-aware conversations and keep a pool of persistent Jupyter kernels so the model can execute Python during reasoning.

At a high level, the pipeline looks like this:

```text
Problem
  → Build Harmony conversation
  → Run 9 parallel attempts per batch
  → Use different prompt modes and temperatures
  → Allow Python tool calls inside each attempt
  → Extract boxed integer answers
  → Check convergence
  → Run more batches if needed
  → Select final answer with vote + entropy score
```

The important part is that this is not just “sample 10 answers and vote.” Each attempt can reason differently, call Python, fail, recover, and produce metadata that later helps with selection.

---

<h3>High-Value Improvement 1 — Diverse Prompt Modes</h3>

The first major upgrade was making the attempts think differently.

Instead of sending the same prompt every time, each batch contains three styles:

- <strong>standard</strong>: careful olympiad-style reasoning
- <strong>verify</strong>: analytical solving plus mandatory computational verification
- <strong>alt</strong>: deliberately tries a less obvious approach

In code, the batch is assigned by position:

```python
attempt_temperatures = [
    0.2,  # standard
    0.2,  # standard
    0.2,  # standard
    0.2,  # verify
    0.6,  # verify
    0.6,  # verify
    0.6,  # alt
    0.6,  # alt
    0.6,  # alt
]

verify_prompt_attempts = {3, 4, 5}
alt_prompt_attempts    = {6, 7, 8}
```

And the solver chooses the prompt like this:

```python
def _pick_prompt(self, position: int) -> str:
    if position in self.cfg.verify_prompt_attempts:
        return self.cfg.system_prompt_verify
    elif position in self.cfg.alt_prompt_attempts:
        return self.cfg.system_prompt_alt
    else:
        return self.cfg.system_prompt
```

This matters because repeated reasoning is not the same as diverse reasoning. If every attempt follows the same mental path, the system can confidently converge on the same wrong idea. Different prompt modes create useful disagreement.

---

<h3>High-Value Improvement 2 — Conservative Convergence</h3>

The second major upgrade was changing when the solver is allowed to stop.

The original version stopped when the most common answer reached a simple vote threshold. That was risky because early agreement can be misleading.

The updated version requires three conditions before stopping:

```text
1. Enough valid answers exist
2. The top answer has enough votes
3. The top answer leads the runner-up by a large enough margin
```

In code, the convergence settings are:

```python
early_stop             = 5
min_lead_first_batch   = 3
min_lead_later_batches = 2
min_valid_answers      = 6
attempts               = 9
```

The actual convergence check is:

```python
def _has_converged(self, valid_answers: list, is_first_batch: bool) -> bool:
    top_count, second_count, lead = self._compute_lead(valid_answers)
    total_valid = len(valid_answers)

    if total_valid < self.cfg.min_valid_answers:
        return False

    if top_count < self.cfg.early_stop:
        return False

    min_lead = (
        self.cfg.min_lead_first_batch
        if is_first_batch
        else self.cfg.min_lead_later_batches
    )

    if lead < min_lead:
        return False

    return True
```

This is one of the most important design choices in the system. It lets easy problems finish quickly, but forces hard problems to keep generating evidence.

For example, one problem converged immediately with six matching answers. Another problem needed four full batches and 36 attempts before the final answer had enough support.

That behavior is exactly what I wanted: fast when confident, patient when uncertain.

---

<h3>High-Value Improvement 3 — Entropy-Aware Answer Selection</h3>

The third major upgrade was using more than raw votes.

Each generation returns top-logprob information. I use that to estimate mean token entropy, then give lower-entropy outputs more weight during final answer selection.

The scoring logic is:

```python
def _select_answer(self, detailed_results: list) -> int:
    answer_weights = defaultdict(float)
    answer_votes   = defaultdict(int)

    for result in detailed_results:
        answer  = result["Answer"]
        entropy = result["Entropy"]

        if answer is not None:
            weight = 1.0 / max(entropy, 1e-9)
            answer_weights[answer] += weight
            answer_votes[answer]   += 1

    scored_answers = sorted(
        [
            {"answer": ans, "votes": answer_votes[ans], "score": score}
            for ans, score in answer_weights.items()
        ],
        key=lambda x: (x["votes"], x["score"]),
        reverse=True
    )

    return scored_answers[0]["answer"] if scored_answers else 0
```

This still respects majority vote first, but it also distinguishes between answers that were produced confidently and answers that appeared through noisy reasoning.

In practice, this made the output table much more useful. I could inspect not just the winning answer, but also the vote count, confidence score, response length, Python calls, and Python errors for every attempt.

---

<h3>Tool Use: Persistent Python Sandboxes</h3>

A key part of the solver is that each reasoning attempt can call Python.

Instead of starting a fresh Python process every time, the notebook initializes a pool of persistent Jupyter kernels:

```python
print(f"Initializing {self.cfg.workers} persistent Jupyter kernels...")

self.sandbox_pool = queue.Queue()

with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
    futures = [
        executor.submit(_create_sandbox)
        for _ in range(self.cfg.workers)
    ]

    for future in as_completed(futures):
        self.sandbox_pool.put(future.result())
```

Each sandbox preloads tools like:

```python
import math
import numpy
import sympy
import itertools
import collections
import mpmath
```

This made verification practical. The model could run symbolic checks, brute-force small cases, or test numerical patterns without paying the full startup cost each time.

The tool wrapper also adds a small quality-of-life fix: if the model’s last line is an expression, it wraps it in `print(...)` so the output is visible.

```python
def _ensure_last_print(self, code: str) -> str:
    lines = code.strip().split("\n")
    last_line = lines[-1].strip()

    if "print" in last_line or "import" in last_line:
        return code

    if not last_line or last_line.startswith("#"):
        return code

    lines[-1] = "print(" + last_line + ")"
    return "\n".join(lines)
```

This is a small detail, but small details matter in tool-using systems.

---

<h3>Answer Extraction</h3>

The solver expects a final non-negative integer between 0 and 99999. The primary extraction target is a boxed answer, with a fallback for phrases like “final answer is ...”.

```python
def _scan_for_answer(self, text: str) -> int | None:
    pattern = r"\\boxed\s*\{\s*([0-9,]+)\s*\}"
    matches = re.findall(pattern, text)

    if matches:
        value = int(matches[-1].replace(",", ""))
        if 0 <= value <= 99999:
            return value

    pattern = r"final\s+answer\s+is\s*([0-9,]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        value = int(matches[-1].replace(",", ""))
        if 0 <= value <= 99999:
            return value

    return None
```

This is less flashy than prompting, but just as important. A correct solution is useless if the parser misses the final answer.

---

<h3>What Changed From the Original Version</h3>

The original version was already useful: it ran multiple attempts, extracted answers, and selected a result.

But it had three weaknesses:

1. Every attempt used the same prompt.
2. Early stopping depended mostly on raw vote count.
3. The final selection did not fully use confidence information.

The updated solver fixes those issues by adding:

- prompt diversity across standard, verify, and alternative modes
- batch-level convergence with vote count, lead margin, and minimum valid answers
- entropy-weighted answer scoring
- variable temperature by attempt position
- extended time budgets for harder problems
- richer result logging for debugging

The biggest conceptual shift was this:

> I stopped treating agreement as enough. I started asking whether the agreement was strong, diverse, and confident.

---

<h3>Observed Behavior</h3>

The solver handled easy and hard problems very differently.

For easier problems, it often stopped after one batch once six or more attempts agreed. For harder problems, it kept running additional batches because the lead was not strong enough.

One example produced a messy distribution after the first batch, continued through multiple batches, and eventually selected the correct answer after 36 attempts. That was a good sign: the solver did not panic when early generations disagreed.

This is the behavior I wanted from the system:

- move fast when consensus is obvious
- slow down when answers are unstable
- keep track of uncertainty instead of hiding it

---

<h3>Takeaways</h3>

The biggest lesson from this project is that benchmark performance depends heavily on orchestration.

The model matters, but the surrounding system matters just as much:

- Prompt diversity creates better search.
- Verification reduces hallucinated confidence.
- Conservative convergence prevents premature stopping.
- Entropy scoring gives a useful confidence signal.
- Good answer extraction prevents silent failures.

AIMO3 is not just a test of math ability. It is a test of whether you can build a system that turns probabilistic reasoning into reliable final answers.

---

<h3>Final Thoughts</h3>

I started this project thinking the goal was to make the model solve harder problems.

By the end, the real goal felt different:

> Build a solver that knows when not to trust itself yet.

That means generating multiple attempts, forcing different reasoning styles, verifying with tools, and only committing once the evidence is strong enough.

For AIMO3-style problems, that made the difference.
