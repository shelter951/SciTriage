# SciTriage Research Next Steps

This project should not be sold as "another AutoResearch agent." The stronger story is:

> AutoResearch agents already produce many candidate discoveries. SciTriage asks whether those discoveries are valid enough to enter the research record.

The next phase should therefore make SciTriage look less like a hand-written filter and more like a reusable evidence layer that improves real research loops.

## What We Should Do Next

### 1. Run Real Same-Agent Loops

The most important experiment is a same-agent comparison:

- same base agent
- same task
- same budget
- with vs. without SciTriage

Measure whether SciTriage changes what the agent claims at the end, not just whether it flags a prepared candidate. This is the cleanest way to show that SciTriage helps AutoResearch in practice.

Primary metrics:

- invalid accept rate
- valid final score
- cost per valid claim
- number of extra probes triggered
- number of misleading high-score candidates blocked

### 2. Expand External Benchmarks

The current MLAgentBench evidence is promising, but too small for a top-tier paper. We should expand to more tasks where agents can produce tempting but invalid wins.

Priority task families:

- runtime optimization tasks with shortcut risks
- classification tasks with leakage risks
- checkpoint/model-loading tasks with artifact validity risks
- seed-sensitive tasks where single-run improvements are unreliable
- code-generation or data-science tasks where visible scores can be gamed

The key is not to make SciTriage win on our own toy tasks. The key is to show that public agent benchmarks have a real false-discovery problem, then show that SciTriage reduces it.

### 3. Build a Failure Corpus

Every blocked candidate should become a small, reproducible failure case:

- task name
- agent claim
- visible score
- why the claim is invalid
- which SciTriage gate caught it
- cheapest probe that would have exposed it

This can become one of the most valuable open-source assets in the project: a benchmark of AutoResearch false discoveries.

### 4. Make the Plugin Boundary Real

SciTriage should remain framework-agnostic. It should be easy to call from:

- Karpathy-style autoresearch loops
- ARIS-like automatic research systems
- Claude Code / Codex workflows
- custom MLAgentBench runners

The practical API should stay simple:

```text
claim + patch + logs + scores + artifacts -> allow / block / probe
```

The project should provide adapters, not force people to adopt a new harness.

### 5. Strengthen Baselines and Ablations

For a serious paper, compare against:

- official-score-only selection
- static artifact rules
- LLM judge-only
- simple rerun policy
- SciTriage without each gate

The strongest result would show that LLM judging alone sounds reasonable but misses benchmark-specific validity failures, while SciTriage catches them because it executes targeted probes.

### 6. Improve the Paper Story

The paper should center on one claim:

> AutoResearch needs evidence triage because visible benchmark scores are not the same as valid research progress.

The contribution should be framed as:

- a taxonomy of AutoResearch false discoveries
- a plugin-style evidence gate
- targeted probes for semantic validity, leakage, artifact validity, and seed noise
- external benchmark evidence that the gate reduces invalid claims without destroying valid progress

## Current Honest Status

SciTriage is promising but not yet a top-tier paper by itself. The current results are a strong prototype:

- it catches real invalid winners in external MLAgentBench-style audits
- it improves same-agent policy selection in controlled evaluations
- it has useful ablations and baselines

To become top-tier solid, it needs larger external coverage and real closed-loop agent runs where SciTriage changes the final research behavior of the agent.

## Near-Term Milestone

The next milestone should be:

> 10 to 20 public benchmark tasks, same-agent with/without SciTriage, with a released failure corpus and reproducible scripts.

That result would be understandable, useful to other AutoResearch projects, and much harder to dismiss as self-defined evaluation.
