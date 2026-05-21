# Project Story

SciTriage is built around a simple observation: AutoResearch agents can produce plausible discoveries faster than they can produce reliable evidence.

The project does not try to replace AutoResearch harnesses. It provides an evidence layer that can sit beside them:

1. Read the candidate patch, metric logs, and research claim.
2. Diagnose risk labels such as noisy result, evaluator drift, resource mismatch, and unsupported claim.
3. Recommend the cheapest useful probe.
4. Gate the claim after the probe.

## Why This Is Not Just Multi-Seed

Multi-seed reruns are necessary but not enough. Some failures require frozen evaluators, ablations, resource-fit checks, or claim rewriting. SciTriage treats "what evidence is missing?" as the central object.

## Current Karpathy Autoresearch Evidence

The current public anchor is a 4090 quick-probe adaptation of Karpathy `autoresearch`.

Main result:

- one-shot selection accepts 3 of 4 initial candidates,
- seed-group evidence accepts only 1,
- two one-shot discoveries are blocked as inconclusive,
- robust discoveries such as `depth7` and `batch_small` are allowed,
- low-value follow-up probes such as `depth7_batch_small` are deferred.

This makes the contribution concrete: SciTriage reduces false discoveries without freezing research progress.

## Current External Benchmark Evidence

The current public external anchor is MLAgentBench. SciTriage has two score-bearing candidate audits:

- `vectorization`: the official runtime-only winner is invalid because it skips the convolution. SciTriage blocks it and selects `im2col_einsum`, which preserves the computation and is still about 235x faster than baseline.
- `cifar10`: the official accuracy winner is a test-label oracle with 1.0000 accuracy. SciTriage blocks it as benchmark leakage before the AutoResearch loop can claim a perfect classifier.

These two tasks matter because they show two distinct failure modes:

1. A candidate can optimize the metric while breaking the computation.
2. A candidate can exploit the benchmark surface while producing no scientific progress.

The project story is therefore not "we get higher scores." It is:

```text
AutoResearch needs a validity layer between score improvement and research claims.
```

SciTriage is that layer.

