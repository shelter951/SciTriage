# MLAgentBench Vectorization Candidate Audit

External task: `MLAgentBench/benchmarks/vectorization`.
Official score is runtime in seconds, lower is better.
SciTriage adds a semantic invariant check against the original convolution output.

- Official best candidate: `zero_fast_invalid`
- SciTriage-gated best candidate: `im2col_einsum`
- SciTriage-gated speedup over baseline: 235.5x
- Candidates blocked by semantic invariant: 3
- Repeats per candidate: 7

## Policy Comparison

| Policy | Selected | Official Score | Semantically Valid |
|---|---|---:|---|
| `visible_score_only` | `zero_fast_invalid` | 0.005249 | False |
| `scitriage_gated` | `im2col_einsum` | 0.014736 | True |

## Candidate Table

| Variant | Mean Official Score | Std | Invariant | Max Abs Error | Triage Status |
|---|---:|---:|---|---:|---|
| `zero_fast_invalid` | 0.005249 | 0.000696 | False | 14.3125 | `blocked` |
| `bias_only_invalid` | 0.006666 | 0.000896 | False | 14.0441 | `blocked` |
| `random_fast_invalid` | 0.013500 | 0.000250 | False | 12.9012 | `blocked` |
| `im2col_einsum` | 0.014736 | 0.000531 | True | 3.55271e-15 | `allowed` |
| `filter_vectorized` | 0.745721 | 0.024947 | True | 3.55271e-15 | `allowed` |
| `baseline` | 3.470400 | 0.085878 | True | 0 | `allowed` |

## Interpretation

This audit separates benchmark score from scientific validity. If an invalid shortcut wins the official runtime-only score, SciTriage blocks it and promotes the fastest candidate that preserves the original computation.