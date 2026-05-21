# Paper Readiness Assessment

Last updated: 2026-05-21.

## Short Answer

SciTriage is not yet a complete top-conference submission, but it has a credible top-conference core.

The current result is strong enough for a serious paper direction because it has:

- a clear problem: AutoResearch agents can convert misleading visible scores into false research claims;
- a simple reusable method: gate claims and candidate selection with task-validity evidence;
- external benchmark evidence: four score-bearing MLAgentBench audits;
- a practical open-source artifact: command-line tools, plugin APIs, reports, and public artifacts.

It is not yet enough for AAAI/ICLR/NeurIPS main-conference strength because it still needs:

- broader task coverage;
- live agent-loop comparisons, not only curated candidate audits;
- stronger baselines against simpler static checks;
- a systematic false-positive/false-negative analysis;
- automatic invariant generation or at least a reusable invariant template library.

## Current Main Result

Across curated MLAgentBench external audits:

| Metric | Value |
|---|---:|
| score-bearing external tasks | 4 |
| audited candidates | 20 |
| candidates blocked by SciTriage | 8 |
| invalid visible-score winners blocked | 3 / 4 visible winners were invalid; all invalid visible winners blocked |
| visible-winner block rate | 0.750 |

Tasks:

- `MLAgentBench/vectorization`: blocks invalid runtime shortcuts while preserving a 235x valid speedup.
- `MLAgentBench/cifar10`: blocks a perfect-score CIFAR-10 test-label oracle.
- `MLAgentBench/imdb`: blocks a perfect-score IMDB test-label oracle.
- `MLAgentBench/CLRS`: keeps a valid checkpointed visible-score winner and blocks a missing-checkpoint candidate.

## Current Paper Claim

The defensible claim today is:

```text
Visible benchmark scores are insufficient evidence for AutoResearch claims.
SciTriage is a lightweight validity gate that blocks high-scoring invalid candidates
and selects the best candidate that still satisfies task-valid evidence.
```

The claim we cannot make yet is:

```text
SciTriage broadly improves autonomous research agents across many natural tasks.
```

That broader claim needs live agent-loop experiments.

## What A Top-Conference Version Needs

Minimum strong version:

1. 5-8 score-bearing external tasks.
2. At least 2 task families: semantic equivalence, leakage/evaluator gaming, seed-noise claim gating.
3. Same-agent comparison: with SciTriage vs without SciTriage.
4. Metrics:
   - invalid accept rate,
   - valid best-score retention,
   - extra cost,
   - false block rate,
   - claim rewrite quality,
   - final valid task score.
5. Ablations:
   - no semantic gate,
   - no leakage gate,
   - no seed-noise gate,
   - static-only baseline,
   - LLM-judge-only baseline.

Ideal version:

- automatic task-contract extraction from benchmark files;
- generated semantic invariants for function-level tasks;
- integration hooks for at least two AutoResearch harnesses;
- qualitative examples of prevented false discoveries;
- public replay package for all audits.

## Current Verdict

The project is paper-worthy but not finished-paper-ready.

Best near-term strategy:

```text
Do not oversell "agentic RL" or "complete AutoResearch".
Position SciTriage as a reusable evidence gate for AutoResearch agents.
Win by showing that current agent benchmarks have a score-validity gap,
and that a small plugin can close it in practical cases.
```

This is a good story because it is humble, concrete, and experimentally checkable.

## Updated Experimental Status

The project now has:

- 4 complete score-bearing candidate audits;
- 1 checkpoint-style MLAgentBench task (`CLRS`);
- an aggregate audit summary over the four complete audits;
- a public README that presents the problem and results for non-specialist readers.

The highest-leverage next experiment is now a live same-agent comparison: run the same AutoResearch agent with and without SciTriage on the candidate-audit tasks and measure invalid accept rate, final valid score, and extra cost.
