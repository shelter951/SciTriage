# MLAgentBench CLRS Candidate Audit

External task: `MLAgentBench/benchmarks/CLRS`.
Official score is CLRS test score, higher is better.
SciTriage checks checkpoint loadability via official eval.

- Official best candidate: `step1_encoded_decoded`
- SciTriage-gated best candidate: `step1_encoded_decoded`
- Candidates blocked by validity gate: 1

## Policy Comparison

| Policy | Selected | Official Score | Validity Gate |
|---|---|---:|---|
| `visible_score_only` | `step1_encoded_decoded` | 0.020592 | True |
| `scitriage_gated` | `step1_encoded_decoded` | 0.020592 | True |

## Candidate Table

| Variant | Official Score | Checkpoint | Official Eval | Triage Status |
|---|---:|---|---|---|
| `step1_encoded_decoded` | 0.020592 | True | True | `allowed` |
| `step1_decoded_only` | 0.017929 | True | True | `allowed` |
| `step1_no_hints` | 0.016693 | True | True | `allowed` |
| `invalid_no_checkpoint` | - | False | False | `blocked` |

## Interpretation

This audit expands SciTriage beyond CSV submissions. On CLRS, a candidate is valid only if it produces checkpoint artifacts that the official evaluator can load and score.