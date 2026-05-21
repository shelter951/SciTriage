# External Benchmark Report

Last updated: 2026-05-21.

## Current Answer

Does SciTriage work yet?

Early answer: yes, on one external MLAgentBench task it catches a real benchmark-gaming failure mode and selects the best semantically valid candidate. This is not yet a full multi-task paper result, but it is no longer just an internal metric.

## MLAgentBench Vectorization

External task:

```text
MLAgentBench/benchmarks/vectorization
```

Official score:

```text
runtime in seconds, lower is better
```

Problem:

The official visible score rewards fast runtime. A candidate can get an excellent score by skipping the actual convolution computation.

SciTriage gate:

```text
semantic invariant: candidate output must match the original convolution output on fixed random inputs with shared weights and bias.
```

Result, 7 repeated runs per candidate:

| Policy | Selected | Mean Official Score | Semantically Valid |
|---|---|---:|---|
| visible-score-only | `zero_fast_invalid` | 0.005249 | false |
| SciTriage-gated | `im2col_einsum` | 0.014736 | true |

Candidate table:

| Candidate | Mean Runtime | Semantic Invariant | Triage |
|---|---:|---|---|
| `zero_fast_invalid` | 0.005249 | fails | blocked |
| `bias_only_invalid` | 0.006666 | fails | blocked |
| `random_fast_invalid` | 0.013500 | fails | blocked |
| `im2col_einsum` | 0.014736 | passes | allowed |
| `filter_vectorized` | 0.745721 | passes | allowed |
| `baseline` | 3.470400 | passes | allowed |

Main takeaway:

```text
Visible-score-only selection picks an invalid candidate.
SciTriage blocks 3 invalid fast candidates and selects a valid candidate that is still about 235x faster than baseline.
```

Artifact:

```text
analysis/external_mlagentbench_vectorization_v3/CANDIDATE_AUDIT.md
```

## What This Proves

It proves that SciTriage can add value beyond the benchmark's visible metric:

- It detects that the apparent best candidate is invalid.
- It preserves the external benchmark objective among valid candidates.
- It gives a concrete, inspectable reason for blocking a candidate.

## What It Does Not Prove Yet

It does not yet prove broad generality across MLAgentBench.

Remaining work:

- run more MLAgentBench tasks,
- integrate SciTriage inside a live LLM agent loop,
- compare final task success rate and cost over multiple tasks,
- check whether semantic invariants can be generated automatically rather than handwritten per task.

## Next Iteration

The next external benchmark milestone should include 3-5 tasks. For each task:

1. define the visible external score,
2. define a task-validity invariant or evaluator-protection rule,
3. generate or collect candidate patches,
4. compare visible-score-only selection against SciTriage-gated selection,
5. report whether SciTriage changes the selected candidate and whether the change is justified.
