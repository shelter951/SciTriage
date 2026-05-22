# Public Failure Corpus Same-Agent Evaluation

This evaluation uses the same candidate traces for each policy. The corpus is built from public MLAgentBench task surfaces and is intended as a broad stress suite; official-executed results remain reported separately.

## Scale

- Public benchmark surfaces: 15
- Candidate records: 112
- Invalid-evidence candidates: 67
- Same-agent traces per policy: 180

## Aggregate Results

| Policy | Traces | Invalid accept rate | Mean valid-score retention | Mean cost | Cost / valid claim |
|---|---:|---:|---:|---:|---:|
| `official_score_only` | 180 | 0.744 | 0.254 | 5.468 | 21.396 |
| `static_artifact_rule` | 180 | 0.739 | 0.260 | 6.368 | 24.387 |
| `judge_only_proxy` | 180 | 0.739 | 0.260 | 6.368 | 24.387 |
| `scitriage_full` | 180 | 0.000 | 0.982 | 6.368 | 6.368 |
| `ablate_no_semantic_gate` | 180 | 0.256 | 0.733 | 6.368 | 8.554 |
| `ablate_no_leakage_gate` | 180 | 0.133 | 0.852 | 6.368 | 7.347 |
| `ablate_no_checkpoint_gate` | 180 | 0.089 | 0.899 | 6.368 | 6.989 |
| `ablate_no_seed_noise_gate` | 180 | 0.494 | 0.500 | 6.368 | 12.596 |
| `ablate_no_schema_gate` | 180 | 0.011 | 0.972 | 6.368 | 6.439 |

## Failure Modes In Corpus

| Failure mode | Candidate count |
|---|---:|
| `artifact_lineage` | 12 |
| `evaluator_overfit` | 8 |
| `evidence_mismatch` | 2 |
| `resource_fit` | 9 |
| `schema_or_artifact` | 15 |
| `seed_noise` | 15 |
| `semantic_shortcut` | 2 |
| `test_leakage` | 4 |

## Interpretation

On the broad public-surface stress suite, score-only selection accepts invalid evidence in 74.4% of traces. Full SciTriage reduces this to 0.0% while keeping mean valid-score retention at 0.982.

This is useful as a scale test and failure-corpus result. It should be presented next to, not instead of, the official-executed MLAgentBench audits.