# MLAgentBench Vectorization Candidate Audit

External task: `MLAgentBench/benchmarks/vectorization`.
Official score is runtime in seconds, lower is better.
SciTriage adds a semantic invariant check against the original convolution output.

- Official best candidate: `zero_fast_invalid`
- SciTriage-gated best candidate: `im2col_einsum`
- Candidates blocked by semantic invariant: 1

## Candidate Table

| Variant | Official Score | Invariant | Max Abs Error | Triage Status |
|---|---:|---|---:|---|
| `zero_fast_invalid` | 0.005201 | False | 13.2365 | `blocked` |
| `im2col_einsum` | 0.014278 | True | 3.55271e-15 | `allowed` |
| `filter_vectorized` | 0.743157 | True | 3.55271e-15 | `allowed` |
| `baseline` | 3.551149 | True | 0 | `allowed` |

## Interpretation

This audit separates benchmark score from scientific validity. If an invalid shortcut wins the official runtime-only score, SciTriage blocks it and promotes the fastest candidate that preserves the original computation.