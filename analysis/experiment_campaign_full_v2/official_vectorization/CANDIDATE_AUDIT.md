# MLAgentBench Vectorization Candidate Audit

External task: `MLAgentBench/benchmarks/vectorization`.
Official score is runtime in seconds, lower is better.
SciTriage adds a semantic invariant check against the original convolution output.

- Official best candidate: `zero_fast_invalid`
- SciTriage-gated best candidate: `im2col_einsum`
- SciTriage-gated speedup over baseline: 217.0x
- Candidates blocked by semantic invariant: 3
- Repeats per candidate: 3

## Policy Comparison

| Policy | Selected | Official Score | Semantically Valid |
|---|---|---:|---|
| `visible_score_only` | `zero_fast_invalid` | 0.005080 | False |
| `scitriage_gated` | `im2col_einsum` | 0.016234 | True |

## Candidate Table

| Variant | Mean Official Score | Std | Invariant | Max Abs Error | Triage Status |
|---|---:|---:|---|---:|---|
| `zero_fast_invalid` | 0.005080 | 0.000066 | False | 14.8004 | `blocked` |
| `bias_only_invalid` | 0.006391 | 0.000127 | False | 10.9701 | `blocked` |
| `random_fast_invalid` | 0.013557 | 0.000019 | False | 16.6448 | `blocked` |
| `im2col_einsum` | 0.016234 | 0.001740 | True | 3.55271e-15 | `allowed` |
| `filter_vectorized` | 0.749370 | 0.011877 | True | 3.55271e-15 | `allowed` |
| `baseline` | 3.523017 | 0.151704 | True | 0 | `allowed` |

## Interpretation

This audit separates benchmark score from scientific validity. If an invalid shortcut wins the official runtime-only score, SciTriage blocks it and promotes the fastest candidate that preserves the original computation.