# Public Failure Corpus Multi-Seed Evaluation

This repeats the public-surface same-agent evaluation across multiple deterministic trace seeds. It measures whether the SciTriage advantage is stable under candidate-order and observed-score noise.

## Scale

- Public benchmark surfaces: 15
- Candidate records: 112
- Invalid-evidence candidates: 67
- Trace seeds: 40
- Traces per seed per policy: 180

## Aggregate Across Seeds

| Policy | Invalid accept rate | Valid-score retention | Mean cost | Cost / valid claim |
|---|---:|---:|---:|---:|
| `official_score_only` | 0.770 +/- 0.008 | 0.228 +/- 0.008 | 5.389 +/- 0.015 | 23.749 +/- 0.828 |
| `static_artifact_rule` | 0.753 +/- 0.007 | 0.245 +/- 0.007 | 6.289 +/- 0.015 | 25.666 +/- 0.762 |
| `judge_only_proxy` | 0.753 +/- 0.007 | 0.245 +/- 0.007 | 6.289 +/- 0.015 | 25.666 +/- 0.762 |
| `scitriage_full` | 0.000 +/- 0.000 | 0.973 +/- 0.002 | 6.289 +/- 0.015 | 6.348 +/- 0.017 |
| `ablate_no_semantic_gate` | 0.254 +/- 0.002 | 0.725 +/- 0.003 | 6.289 +/- 0.015 | 8.540 +/- 0.036 |
| `ablate_no_leakage_gate` | 0.133 +/- 0.000 | 0.845 +/- 0.002 | 6.289 +/- 0.015 | 7.306 +/- 0.021 |
| `ablate_no_checkpoint_gate` | 0.104 +/- 0.007 | 0.883 +/- 0.007 | 6.289 +/- 0.015 | 7.025 +/- 0.055 |
| `ablate_no_seed_noise_gate` | 0.507 +/- 0.010 | 0.487 +/- 0.010 | 6.289 +/- 0.015 | 12.818 +/- 0.273 |
| `ablate_no_schema_gate` | 0.024 +/- 0.003 | 0.959 +/- 0.004 | 6.289 +/- 0.015 | 6.447 +/- 0.022 |

## Interpretation

Across trace seeds, score-only selection has invalid accept rate 0.770 +/- 0.008; full SciTriage has invalid accept rate 0.000 +/- 0.000.