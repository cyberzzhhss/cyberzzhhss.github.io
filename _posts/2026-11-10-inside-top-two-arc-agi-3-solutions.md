---
title: "Inside the Top Two ARC-AGI-3 Teams’ Solutions"
excerpt: "How threerabbits and Tufa Labs organize perception, experimentation, executable world models, memory, and inference in their published ARC-AGI-3 notebooks."
excerpt_separator: "<!--more-->"
tags:
  - Artificial Intelligence
  - ARC-AGI
  - AI Agents
  - Program Synthesis
  - World Models
categories:
  - Machine Learning
  - Computer Science
classes: wide
layout: single
---

An agent arrives in a game with no explanation of its rules. Colored objects fill the board. Some actions move things; others appear to do nothing. The goal itself may need to be discovered. Every experiment consumes part of the action budget.

Solving this requires more than recognizing a pattern. The agent has to turn observations into hypotheses, test those hypotheses, preserve what it learns, and plan under rules that are still uncertain.

As of **November 10, 2026**, Kaggle's **public leaderboard** for ARC Prize 2026—ARC-AGI-3 lists these teams first and second:

| Public rank | Team | Public score | Notebook examined |
| --- | --- | --- | --- |
| 1 | Yi-Chia Chen @ Fields Model (`threerabbits`) | 75.56 | [ARC-Code HS110 Click](https://www.kaggle.com/code/threerabbits/arc-code-hs110-click/notebook) |
| 2 | Tufa Labs | 70.24 | [Tufa Labs ARC3 Final Solution](https://www.kaggle.com/code/jeroencottaar/tufa-labs-arc3-final-solution) |

Kaggle states that the private leaderboard is awaiting verification and that final standings may differ. These are team rankings: a published notebook snapshot is not, by itself, proof that its exact code and configuration produced the team's highest score. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

<!--more-->

## What the agents are trying to learn

ARC-AGI-3 uses interactive environments. Agents must discover goals and dynamics through actions, then carry that understanding across increasingly difficult levels. The benchmark evaluates how efficiently they acquire and apply those skills, with action counts compared against human baselines. [ARC Prize's benchmark overview](https://arcprize.org/arc-agi/3)

Both solutions give a language model the means to write and execute Python. That creates a productive division of labor: the model proposes explanations and programs; code measures changes, simulates consequences, and searches possibilities.

The critical engineering question is how to connect those programs to reality. A perfectly executed search can still fail if its simulator encodes the wrong rule. A correct observation can still be lost when the conversation becomes too long.

## First-ranked team: ARC-Code as an experimental workspace

The `arc-code-hs110-click` notebook packages an entire agent deployment. Its embedded archive contains the launcher, game middleware, action client, recording system, prompts, reviewer, and supporting code. A pinned local model server supplies the language model.

The core idea is to give each game an independent coding workspace. The agent can inspect the board, write parsers and simulators, run bounded experiments, and keep a record of its conclusions. The notebook's prompts describe a scientific workflow: make a small prediction, spend a few actions testing it, compare the observations, and revise the explanation.

The code and configuration discussed below come from the [ARC-Code notebook](https://www.kaggle.com/code/threerabbits/arc-code-hs110-click/notebook), including its embedded `arcforge.arc_code` modules and `work/arc_code` files.

### A game action becomes recorded evidence

The agent interacts through a small command-line interface. It can initialize the game, inspect its status or board, and send individual actions or batches. The supplied interface describes a 64×64 board, with clicks addressed by column and row.

A middleware layer sits between the player and the upstream game gateway. It associates each player's credentials with one game, checks action availability and coordinates, and records the responses. The upstream gateway remains responsible for game behavior and the scorecard.

This placement matters. If the agent writes its own Python client, its actions still pass through the middleware and enter the record. The agent's own account of what happened is therefore separate from the observation history.

For each step, the workspace receives the final board as text and an image, the returned frame array, and intermediate images or an animation when the action produces multiple frames. A history entry connects the action to the level, outcome, changed-cell count, and the plan supplied with the batch.

The player is instructed to use images for overall structure and code for exact coordinates and differences. This avoids repeatedly transcribing a board by eye. Intermediate frames can also reveal a mechanism that the final frame hides: an object may move and return, or a click may trigger a sequence whose order matters.

### The “click” addition makes mistakes easier to interpret

The named notebook adds feedback about the cell an `ACTION6` click actually hit. Before sending the click, the client reads the current board and reports the target's color and the size and bounds of its same-colored connected region.

If the click lands on the inferred background, it looks nearby for another color. After execution, it can also report that the final board did not change.

This helps distinguish two very different explanations for an unproductive action: the agent clicked the intended object and misunderstood its behavior, or it missed the object entirely. Discovering the second case early can prevent a long investigation of a rule that was never actually tested.

The feedback is computed from visible board data. Its description of a region is geometric; it does not reveal the region's role in the game. Its background label is also a heuristic based on the most frequent color.

### Memory lives in files as well as conversation

ARC-Code separates several kinds of memory:

| Artifact | Purpose |
| --- | --- |
| `record/` | Server-written observations and action history |
| `notes.md` | Current beliefs, rejected hypotheses, and ongoing questions |
| `levels/` | Explanations and action sequences for cleared levels |
| `lib/` | Reusable parsing, simulation, and search code |
| `scripts/` | Individual experiments |

The record is read-only to the player. The remaining files are working material that the player can revise.

Before context compaction, a reminder asks the agent to update its notes. The compaction prompt separates confirmed mechanics, hypotheses, solved levels, current state, useful files, and the next step. Afterward, the player is instructed to reread its durable notes.

This is more than retaining a transcript. It tries to preserve the distinction between a rule supported by an experiment and an attractive guess. That distinction is especially valuable when an assumption would otherwise be copied into a summary and treated as established knowledge.

### Reviewers challenge beliefs without spending game actions

For stalled games, the launcher can start a separate reviewer. The reviewer receives a snapshot of the action record, notes, and code, with updates supplied while the player continues.

Its task is to identify a small number of consequential beliefs that are contradicted or insufficiently tested. Findings must cite recorded steps or reproducible checks. The reviewer cannot operate the game; it delivers a report, and the player decides how to use it.

This arrangement targets a common failure mode: an agent builds a simulator, searches it unsuccessfully, and concludes that the real game has no solution. The reviewer is prompted to ask whether the simulator's assumptions were ever justified.

The configuration makes this help conditional on available capacity. In the competition configuration, reviews begin only after two hours, target games stalled for at least 45 minutes, and require player-plus-reviewer occupancy below 64. This does not replace the main launcher’s 110-game concurrency setting; it restricts when additional reviewers can enter.

### The serving system supports many long-running players

The notebook serves the local checkpoint under the name `qwen38fn`, using a custom SGLang runtime and a patched coding-agent CLI. The CLI is the tool-using driver; requests are routed to the local checkpoint.

The deployment combines low-precision model and cache storage, speculative decoding, and session-cache machinery. These features aim to keep many long conversations affordable. Separately, the launcher adjusts compaction thresholds according to the shared cache budget and the number of running players.

Each game's processes are also bounded. The configuration sets memory and CPU limits, and a watchdog can stop runaway work. A search that exhausts the machine would otherwise take time and capacity away from every other game.

## Second-ranked team: Tufa's SealAgent and executable world models

Tufa's notebook uses the Tufa ARC-AGI Framework, or TAAF, to stage a source bundle, prepare inference, load the benchmark, and run its solver. The saved configuration identifies that solver as **`SealAgent`**.

The notebook is primarily the deployment entry point. Its linked source bundle contains the actual implementation in the `seal`, `platypus`, and supporting packages. This analysis uses both the notebook's saved configuration and that bundle, rather than assuming every feature in the repository is enabled. [Tufa notebook](https://www.kaggle.com/code/jeroencottaar/tufa-labs-arc3-final-solution) · [Linked source bundle](https://www.kaggle.com/datasets/jeroencottaar/platypus-demo-seal-xkb-keepalive-ds)

### A persistent Python session becomes the agent's workbench

SealAgent provides structured objects for actions, frames, and the history of the current level. The model writes Python in a persistent kernel. Its code can inspect the state, compute a result, submit an action, inspect the returned observation, and continue within the same call.

This reduces the need to convert every useful operation into a separate conversational exchange. A conditional procedure can remain a procedure: inspect a piece, attempt a move, and branch on what actually happened.

In the saved configuration, the kernel persists for a level. Files and journaled knowledge can carry across levels, while level-specific variables are cleared at the boundary. This favors reusing general mechanics without accidentally reusing coordinates or object identities from an earlier board.

### Perception includes structure and exact change descriptions

The agent receives the current grid as an image and can access integer-valued board data in Python. A segmentation helper exposes connected structure for further inspection.

The `framediff` machinery adds another representation: a short block of NumPy operations that reconstructs a newly observed frame from an available baseline. That baseline may be the previous frame, the level's initial frame, or an incorrect prediction from the agent's world model.

The proposed reconstruction is executed and checked before being shown. This gives the model a compact, exact account of a visual difference when one can be found within the search budget. It still has to infer what the difference means.

The saved configuration enables settled-frame differences and per-action change reports, with animation descriptions available on request. It does not enable every visual aid: the separate difference-image option is off.

### A world model is an executable hypothesis

SealAgent can maintain a `world_model.py` file describing the game's dynamics. Its interface separates an abstract initial state, a transition function, and a renderer that predicts the visible board. An optional distance-to-goal function can guide search.

This turns a verbal explanation into a program with testable consequences. Instead of merely saying that a colored piece slides until blocked, the agent must specify how its position changes and what the resulting board should look like.

`test_world_model()` replays the current level's recorded history through that program. It reports the first mismatch between a prediction and the observations. A passing replay also supplies the agent's predicted state at each recorded turn.

Crucially, the supplied configuration uses **optional world models**. An agent without a passing model can still act. The system offers verified simulation as a tool, rather than requiring a complete simulator before any progress is possible.

Passing the test validates the modeled history under the configured comparison rules. It does not establish that the program predicts every unseen situation. Partial rendering and tolerance controls make this distinction important.

### Search becomes useful once the simulator earns some trust

After a Python call explicitly tests the model successfully and makes no real move, the harness can launch a bounded search in a separate process. It rechecks the model there before planning.

The planner explores simulated transitions. It uses a supplied distance-to-goal heuristic when available and falls back to breadth-first ordering otherwise. It can seek a win or a boundary where the model cannot predict what happens next.

A result is returned as suggested actions and predicted intermediate history. The agent retains the decision to execute it. Search therefore operates inside a larger learning loop: build a hypothesis, check it, explore its consequences, and test those consequences against the game.

While a world model passes, real action sequences are checked step by step. A mismatched prediction stops the remaining sequence and returns control for investigation. Previously executed actions still count; the guard limits further damage rather than undoing the mistake.

Click search also uses a reduced action set. It selects representative cells from connected monochromatic regions, with a configured cap, instead of treating all 4,096 board coordinates as equally useful candidates. Failure to find a plan is consequently a result about the modeled transitions and searched candidates.

### Reflection, audits, and compaction serve different purposes

SealAgent's memory system includes a journal, saved files, a level-local Python session, and conversation summaries.

After clearing a level, a reflection turn lets it consolidate what it learned before proceeding. Separately, the saved configuration requests a strategy audit after 300,000 completion tokens on a level. During that turn, acting is disabled and the agent is prompted to reconsider its approach, including neglected actions or click targets.

This differs from ARC-Code's separate reviewer. Tufa schedules a reconsideration phase in the player's own control loop; ARC-Code can assign another agent to independently inspect the evidence.

The saved Tufa configuration uses a working context budget of 181,184 tokens and a target of 33,925 after compaction, within an engine ceiling of 262,144. Files, relevant Python state, and explicit knowledge reduce the amount that must survive in the prose summary alone.

### Compute allocation is part of the solution

The configuration permits up to 80 concurrent games and chooses the `least_game_tokens` scheduling policy. The source implements this as priority for games with the lowest accumulated token expenditure, helping distribute service across the game list.

Inference runs through a custom vLLM backend. The saved deployment uses low-precision weights and KV storage, CPU cache offloading, prefix-cache features, and a speculative draft model. These are serving optimizations; the notebook itself loads existing checkpoints and runs inference rather than training the model on the games.

The distinction matters when interpreting “learning.” Here, adaptation can occur through better code, a corrected simulator, a journal entry, or a revised plan. Those changes can improve behavior without updating neural-network weights during the run.

## What the supplied runs actually show

The leaderboard scores and the scores printed inside these notebooks describe different evaluations.

| Evidence | ARC-Code saved run | Tufa saved run |
| --- | --- | --- |
| Run type | Public stand-in gateway | Offline demonstration benchmark |
| Instances | 110, expanded from 25 public games | 110 benchmark entries |
| Duration | 90-minute play budget | Approximately 8 hours 48 minutes in diagnostics |
| Saved action-efficiency mean | 21.3384 | 68.18 under the “Official ARC” scorer |
| Public team score on November 10 | 75.56 | 70.24 |

ARC-Code's output reports 295 cleared levels, 11,892 actions, and six wins. Its local ledger agrees with the stand-in gateway's scorecard. That agreement checks accounting within the rehearsal; it does not convert the rehearsal into a held-out leaderboard evaluation.

Its reviewer log also records 25 launches, three delivered reports, and 22 failures. Some reviews were abandoned because the player moved to another level; others failed to produce the report in time. The record supports the existence of the reviewer mechanism, but it does not establish that reviewers improved the score.

Tufa's diagnostic report contains its own original learning-progress plot:

<figure>
  <img src="{{ '/assets/images/arc3-solutions/tufa-original-score-vs-wallclock.png' | relative_url }}" loading="lazy" alt="Original Tufa diagnostic plot of mean ARC action-efficiency score against elapsed job time, rising to approximately 68.18 over the demonstration run">
  <figcaption>Original figure extracted from the supplied Tufa notebook's diagnostics. The curve uses its official-formula ARC scorer on that offline benchmark. It is not the 70.24 public leaderboard result. Source: <a href="https://www.kaggle.com/code/jeroencottaar/tufa-labs-arc3-final-solution">Tufa Labs ARC3 Final Solution</a>.</figcaption>
</figure>

The two saved runs differ in game selection, checkpoints, time budgets, and deployment details. Their means cannot isolate which harness is better or explain the public ranking. The leaderboard provides the ranking; the notebooks explain concrete systems from those teams.

## The shared lesson: make beliefs accountable to observations

| Design question | ARC-Code | Tufa SealAgent |
| --- | --- | --- |
| Main work surface | Shell and a per-game directory | Persistent Python kernel and file tools |
| Observation evidence | Server-written boards, frames, animations, history | Structured turns, frames, segmentation, verified frame differences |
| Simulation | Agent-written code supported by prompts | Optional world-model interface with replay validation |
| Planning | Agent-written searches and experiments | Agent code plus harness-provided bounded search |
| Recovery from a wrong theory | Notes, experiments, optional independent reviewer | Prediction mismatch, replay test, strategy audit |
| Durable knowledge | Notes, level documents, reusable code | Journal, files, summaries, executable model |

Both designs make a language model responsible for forming and revising explanations. They also give ordinary software a substantial role in checking details that are difficult to maintain reliably in prose.

ARC-Code emphasizes an auditable experimental workspace. Tufa more explicitly connects a learned simulator to replay tests, planning, and guarded execution. In both, the useful unit of progress is a belief that has survived a relevant test and can be carried into the next decision.

That is a demanding standard. A simulator may fit the past and still fail on a new configuration. A summary may preserve an error. A reviewer may arrive too late. The systems are interesting because their architecture gives those failures places to become visible and correctable.

## Technical appendix

### A. Reading the first notebook's embedded source

The ARC-Code notebook's large configuration cell contains a Base64-encoded, gzip-compressed tar archive. The analysis unpacked it as data and checked the hash prefixes of all **539** carried files against its manifest. The important paths are:

- `work/arc_code/run_arm.py`: orchestration, per-game workspaces, sessions, context budgets, and reviewer integration.
- `src/arcforge/arc_code/middleware.py`: game-bound access, action validation, recording, and upstream communication.
- `src/arcforge/arc_code/act_client.py`: the generated action client and click feedback.
- `src/arcforge/arc_code/record.py`: text boards, images, frame arrays, and animations.
- `work/arc_code/prompts/v14/`: the active player brief and compaction prompts.
- `work/arc_code/review_scheduler.py` and `review_prompts/r7/`: review admission and the active review instructions.

The archive contains other versions and optional features. The two `*-review-click.json` configurations identify the paths relevant to this notebook.

### B. Context capacity is not one number

ARC-Code configures a 240,000-token server context, a 235,000-token driver context, and a 150,000-token automatic-compaction cap. Its dynamic threshold is the safety-adjusted share of the server cache pool per active player, clamped between 100,000 tokens and that cap. A separate reminder helps save notes before compaction.

These values serve different purposes. The server ceiling constrains a request; the driver budget reserves room around it; compaction governs how much history remains active under shared load.

Tufa similarly separates its 262,144-token engine ceiling, 181,184-token working context, and 33,925-token compaction target. Comparing only the largest context numbers would miss how both systems actually manage conversations.

### C. Scoring rewards completion and action efficiency

The ARC-Code scorer gives a cleared level an efficiency value of `min(100 * (baseline_actions / agent_actions) ** 2, 115)`. It weights levels by their one-based index and includes every level's weight in the denominator, including uncleared levels. A further cap limits the game's score by the weighted share of levels completed.

Consequently, solving early levels efficiently cannot fully compensate for never reaching later ones. Exploration can be worthwhile when it unlocks progress, but repeated actions that provide no new information reduce efficiency.

### D. Where Tufa's logic lives

The linked source archive identifies the submission repository snapshot as `8863e380`. Within `packages/seal/src/seal/`:

- `agent.py` defines the playing loop, active configuration fields, context handling, reflection, and audits.
- `harness/game.py` and `harness/test.py` connect Python actions and world-model checks to the recorded level history.
- `frame_delta.py` builds and verifies compact frame reconstructions.
- `search.py`, `search_process.py`, and `planner.py` run bounded planning and report its provenance.
- `move_lister.py` selects candidate actions, including representative clicks.

The saved configuration enables optional world models and search, with a 130-second search budget and 2 GiB of additional search-process memory. The planner's name is `BFSPlanner`, but it uses best-first ordering when a usable heuristic is provided; breadth-first ordering is its fallback.

The configuration also has `expose_undo=False` and `confine_filesystem=False`. The presence of undo or Landlock confinement code in the package therefore does not establish that those features operated in this saved run.

### E. Reproduction and interpretation

Both notebooks depend on mounted model checkpoints, packaged runtimes, and competition infrastructure. Their visible top-level cells are deployment programs rather than self-contained solver algorithms that run unchanged in an ordinary Python environment.

ARC-Code switches between a public stand-in and the competition gateway. Tufa loads its bundled benchmark in demonstration mode and obtains the game list from the competition gateway during a true rerun. Both write a placeholder submission in non-rerun operation; that file's dummy score is not evidence of game performance.

The attached ARC-Code export also contains configuration notes dated October 27 while its saved execution metadata dates the run to October 26. Its output should therefore not be treated as an ablation proving the effect of the added click feedback.

## Sources

- [Kaggle public leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard), checked November 10, 2026.
- threerabbits, [ARC-Code HS110 Click](https://www.kaggle.com/code/threerabbits/arc-code-hs110-click/notebook): supplied notebook, embedded source, configurations, and saved output.
- Jeroen Cottaar / Tufa Labs, [Tufa Labs ARC3 Final Solution](https://www.kaggle.com/code/jeroencottaar/tufa-labs-arc3-final-solution): supplied notebook, saved configuration, and diagnostics.
- Tufa Labs, [Platypus demonstration source bundle](https://www.kaggle.com/datasets/jeroencottaar/platypus-demo-seal-xkb-keepalive-ds): linked solver source.
- ARC Prize, [ARC-AGI-3 overview](https://arcprize.org/arc-agi/3).
