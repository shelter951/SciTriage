# Seed-Noise Policy Evaluation

This file summarizes the Karpathy-style AutoResearch evidence-board policies.

- Candidates: 24
- One-shot accepts: 15
- Group-tested candidates: 8
- Baseline seed std: 0.004211

| Policy | Extra Seed Runs | True Accepts | False Accepts | Precision | Recall | Family Precision | Family Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `one_shot_claiming` | 0 | 5 | 2 | 0.714 | 1.000 | 0.500 | 1.000 |
| `verify_all_one_shot_positives` | 14 | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| `scitriage_high_priority_only` | 12 | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| `scitriage_family_landmarks` | 4 | 2 | 0 | 1.000 | 0.400 | 1.000 | 1.000 |
| `exhaustive_verify_all_candidates` | 16 | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

## Interpretation

The no-seed-noise baseline (`one_shot_claiming`) accepts false discoveries. SciTriage policies remove those false accepts by spending targeted seed budget. The family-landmark policy preserves research-direction recall while using 71.4% fewer extra seed runs than verifying every one-shot positive.