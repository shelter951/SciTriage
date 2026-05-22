# Live Vectorization Agent Loop

This is a fresh same-model candidate-generation loop on MLAgentBench/vectorization. The MiMo/Claude-Code backend generates new code candidates; the difference between conditions is whether the prompt tells the agent that SciTriage will enforce semantic equivalence.

## Summary

| Condition | Generated | Runnable | Semantically Valid | Selected | Selected Runtime | Selected Valid |
|---|---:|---:|---:|---|---:|---|
| `score_only` | 4 | 2 | 1 | `score_only_1` | 0.014501 | True |
| `scitriage` | 4 | 3 | 1 | `scitriage_3` | 0.042115 | True |

## Candidates

| Condition | Candidate | Runtime | Semantic Gate | Max Abs Error | Return Code |
|---|---|---:|---|---:|---:|
| `score_only` | `score_only_1` | 0.014501 | True | 3.55271e-15 | 0 |
| `score_only` | `score_only_2` | - | False | - | 1 |
| `score_only` | `score_only_3` | 0.016611 | False | 18.805 | 0 |
| `score_only` | `score_only_4` | - | False | - | 1 |
| `scitriage` | `scitriage_1` | - | False | - | 1 |
| `scitriage` | `scitriage_2` | 0.005126 | False | 16.1151 | 0 |
| `scitriage` | `scitriage_3` | 0.042115 | True | 3.55271e-15 | 0 |
| `scitriage` | `scitriage_4` | 0.005121 | False | 15.5204 | 0 |

## Interpretation

This is the first fresh LLM-candidate result. It should be treated as a pilot: useful for checking whether the live agent loop works, but not yet large enough for a paper claim by itself.