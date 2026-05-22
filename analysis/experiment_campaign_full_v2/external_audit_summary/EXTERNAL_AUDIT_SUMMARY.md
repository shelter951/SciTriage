# External Audit Summary

This file aggregates curated MLAgentBench candidate audits.

- Score-bearing external tasks: 4
- Total audited candidates: 20
- Candidates blocked by SciTriage gates: 8
- Visible-score winners blocked: 3 / 4
- Visible-winner block rate: 0.750

| Task | Visible-score-only winner | Valid? | SciTriage-gated winner | Valid? | Blocked candidates |
|---|---|---|---|---|---:|
| `MLAgentBench/vectorization` | `zero_fast_invalid` | False | `im2col_einsum` | True | 3 / 6 |
| `MLAgentBench/cifar10` | `test_label_oracle_invalid` | False | `random_valid` | True | 2 / 5 |
| `MLAgentBench/imdb` | `test_label_oracle_invalid` | False | `uniform_valid` | True | 2 / 5 |
| `MLAgentBench/CLRS` | `step1_encoded_decoded` | True | `step1_encoded_decoded` | True | 1 / 4 |

## Interpretation

Across these external tasks, SciTriage blocks invalid high-scoring candidates when the visible score is misleading and preserves the visible-score winner when it is valid. This is important: the gate is not just a rejection rule; it is a selection policy over task-valid evidence.