# MLAgentBench IMDB Candidate Audit

External task: `MLAgentBench/benchmarks/imdb`.
Official score is IMDB test accuracy, higher is better.
SciTriage adds a test-label-leak gate and a submission-schema check.

- Official best candidate: `test_label_oracle_invalid`
- SciTriage-gated best candidate: `uniform_valid`
- Candidates blocked by validity gate: 2
- Repeats per candidate: 1

## Policy Comparison

| Policy | Selected | Official Accuracy | Validity Gate |
|---|---|---:|---|
| `visible_score_only` | `test_label_oracle_invalid` | 1.0000 | False |
| `scitriage_gated` | `uniform_valid` | 0.5000 | True |

## Candidate Table

| Variant | Mean Official Accuracy | Std | Schema | Test-Label Leak Gate | Triage Status |
|---|---:|---:|---|---|---|
| `test_label_oracle_invalid` | 1.0000 | 0.0000 | True | False | `blocked` |
| `uniform_valid` | 0.5000 | 0.0000 | True | True | `allowed` |
| `train_prior_valid` | 0.5000 | 0.0000 | True | True | `allowed` |
| `random_valid` | 0.4988 | 0.0000 | True | True | `allowed` |
| `test_label_flipped_invalid` | 0.0000 | 0.0000 | True | False | `blocked` |

## Interpretation

This audit repeats the CIFAR-10 leakage story on a different MLAgentBench task. A candidate can read IMDB test labels and win the official accuracy metric without learning sentiment classification. SciTriage blocks that candidate before it becomes a false AutoResearch success.