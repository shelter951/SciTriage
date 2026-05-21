# External Benchmark Plan

Last updated: 2026-05-21.

## Why This Plan Exists

The current Karpathy autoresearch evidence board is useful, but it is not enough as the main claim of the project.

It was built from our own candidate pool and our own steering policy. That makes it a good feasibility signal, not a strong external result. A skeptical reader could fairly ask:

> Did SciTriage solve a real benchmark problem, or did it only look good on a self-defined metric?

From this point forward, the main evidence for SciTriage should come from external benchmarks or externally defined tasks.

## Main Hypothesis

SciTriage should improve existing research agents under fixed budgets.

The target claim is:

> Given the same base agent, task set, model budget, and GPU budget, adding SciTriage reduces wasted runs and unsupported claims while preserving or improving external benchmark scores.

## Primary Metrics

Use external benchmark metrics as the primary outcome:

- task success rate,
- final task score,
- pass/fail against benchmark evaluator,
- cost to reach a target score,
- number of failed runs before a valid improvement.

SciTriage-specific metrics are secondary:

- blocked unsupported claims,
- resource-fit diagnoses,
- noisy one-shot results filtered,
- recommended probes followed,
- seed budget saved.

These secondary metrics are only convincing when paired with an external task outcome.

## Benchmark Priority

### 1. MLAgentBench

Reason:

- It is close to the exact loop SciTriage targets: an agent edits ML code, runs experiments, and tries to improve a metric.
- It produces logs and failures that can be triaged.
- It is more feasible on 4x RTX 4090 than full paper-replication benchmarks.

Initial experiment:

```text
base agent on 3-6 small MLAgentBench tasks
base agent + SciTriage on the same tasks
fixed max iterations and wall-clock budget
compare benchmark score, failed run count, and cost
```

### 2. MLE-bench Lite / Low-complexity Subset

Reason:

- It is externally credible and widely legible.
- It evaluates ML engineering agents on Kaggle-like tasks.

Risk:

- Heavier setup and data requirements.
- Full benchmark is likely too expensive for the first pass.

### 3. ScienceAgentBench Verified Tasks

Reason:

- Scientific workflow framing is close to the paper story.

Risk:

- May be less directly compatible with the current code-experiment-log loop.

## Seven-step Execution Plan

1. Install and smoke-test the external benchmark.
2. Select a small task subset that fits 4x RTX 4090 and has clear evaluation.
3. Run the base agent under a fixed iteration/time budget.
4. Run the same agent with SciTriage in the loop.
5. Compare external scores, success rate, failed runs, and cost.
6. If SciTriage does not help, inspect failure modes and improve the policy.
7. Re-run the same subset with the improved policy and report both before/after results.

## What Would Count As A Strong Result

Strong:

- higher task success rate at the same budget,
- same success rate with lower GPU/API cost,
- fewer failed or invalid runs without hurting final score,
- fewer unsupported final claims while benchmark score is unchanged or better.

Weak:

- only better internal risk labels,
- only prettier reports,
- only seed savings on our own generated candidate pool,
- cherry-picked single-task success.

## Current Status

SciTriage has internal feasibility evidence:

- Karpathy autoresearch resource-fit diagnosis,
- one-shot false discovery filtering,
- evidence board over 24 observed candidates,
- family-landmark policy that saves seed runs while preserving supported research directions.

The next milestone is external validation, starting with MLAgentBench.
