# SciTriage Evidence Board

## Summary

- Candidates observed: 24
- One-shot accepts: 15
- High-priority candidates: 8
- Group-tested candidates: 8
- Group-supported discoveries: 5
- False discoveries among tested one-shot accepts: 2 (FDR 0.286)

## Budget Policy Evaluation

| Policy | Extra seed runs | True accepts | False accepts | Missed true accepts | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| `one_shot_claiming` | 0 | 5 | 2 | 0 | 0.714 | 1.000 |
| `verify_all_one_shot_positives` | 14 | 5 | 0 | 0 | 1.000 | 1.000 |
| `scitriage_high_priority_only` | 12 | 5 | 0 | 0 | 1.000 | 1.000 |
| `scitriage_family_landmarks` | 4 | 2 | 0 | 3 | 1.000 | 0.400 |
| `exhaustive_verify_all_candidates` | 16 | 5 | 0 | 0 | 1.000 | 1.000 |

## Research-Direction Evaluation

| Policy | Extra seed runs | True families | False families | Missed true families | Family precision | Family recall |
|---|---:|---:|---:|---:|---:|---:|
| `one_shot_claiming` | 0 | 2 | 2 | 0 | 0.500 | 1.000 |
| `verify_all_one_shot_positives` | 14 | 2 | 0 | 0 | 1.000 | 1.000 |
| `scitriage_high_priority_only` | 12 | 2 | 0 | 0 | 1.000 | 1.000 |
| `scitriage_family_landmarks` | 4 | 2 | 0 | 0 | 1.000 | 1.000 |
| `exhaustive_verify_all_candidates` | 16 | 2 | 0 | 0 | 1.000 | 1.000 |

SciTriage high-priority gating saves 14.3% of extra seed runs versus verifying every one-shot positive, while preserving the supported discoveries in this real candidate pool.
 It saves 25.0% versus exhaustive verification of every observed candidate.

For early-stage research steering, the family-landmark policy saves 71.4% versus verifying every one-shot positive while preserving the supported research directions in this candidate pool.

## Candidate Board

| Variant | Family | Seed1 Delta | Delta/Std | Priority | Group Delta | Margin | Verdict | Claim |
|---|---|---:|---:|---|---:|---:|---|---|
| `batch_small` | batch | 0.008483 | 2.014262 | `high` | 0.009168 | 0.006091 | `supports_improvement` | `allowed` |
| `depth5` | depth | 0.047804 | 11.350914 | `high` | 0.044186 | 0.005546 | `supports_improvement` | `allowed` |
| `depth6` | depth | 0.056189 | 13.341907 | `high` | 0.052982 | 0.009053 | `supports_improvement` | `allowed` |
| `depth7` | depth | 0.066303 | 15.743445 | `high` | 0.067128 | 0.005395 | `supports_improvement` | `allowed` |
| `depth8` | depth | 0.066069 | 15.687883 | `high` | 0.061321 | 0.006267 | `supports_improvement` | `allowed` |
| `matrix_lr_high` | learning_rate | 0.006960 | 1.652631 | `high` | 0.005740 | 0.006479 | `inconclusive_within_noise` | `blocked` |
| `depth5_matrix_lr_high` | depth | 0.049990 | 11.869973 | `high` | - | - | `not_tested` | `unknown` |
| `embedding_lr_high` | learning_rate | 0.006895 | 1.637197 | `high` | - | - | `not_tested` | `unknown` |
| `weight_decay_low` | regularization | 0.003722 | 0.883778 | `medium` | 0.004464 | 0.005706 | `inconclusive_within_noise` | `blocked` |
| `window_sssl` | attention_window | 0.005242 | 1.244697 | `medium` | - | - | `not_tested` | `unknown` |
| `embedding_lr_low` | learning_rate | 0.004534 | 1.076584 | `medium` | - | - | `not_tested` | `unknown` |
| `final_lr_frac_0_1` | learning_rate | 0.006291 | 1.493779 | `medium` | - | - | `not_tested` | `unknown` |
| `unembedding_lr_high` | learning_rate | 0.005784 | 1.373393 | `medium` | - | - | `not_tested` | `unknown` |
| `weight_decay_high` | regularization | 0.005016 | 1.191034 | `medium` | - | - | `not_tested` | `unknown` |
| `warmdown_short` | schedule | 0.003809 | 0.904435 | `medium` | - | - | `not_tested` | `unknown` |
| `matrix_lr_low` | learning_rate | -0.004994 | -1.185810 | `low` | -0.003535 | 0.005564 | `does_not_support_improvement` | `blocked` |
| `batch_large` | batch | -0.013404 | -3.182739 | `low` | - | - | `not_tested` | `unknown` |
| `unembedding_lr_low` | learning_rate | -0.000773 | -0.183546 | `low` | - | - | `not_tested` | `unknown` |
| `warmdown_long` | schedule | -0.001438 | -0.341449 | `low` | - | - | `not_tested` | `unknown` |
| `warmup_10` | schedule | -0.000300 | -0.071234 | `low` | - | - | `not_tested` | `unknown` |
| `depth7_matrix_lr_high` | depth | - | - | `blocked` | - | - | `not_tested` | `unknown` |
| `depth7_matrix_lr_low` | depth | - | - | `blocked` | - | - | `not_tested` | `unknown` |
| `depth7_warmdown_short` | depth | - | - | `blocked` | - | - | `not_tested` | `unknown` |
| `depth7_weight_decay_low` | depth | - | - | `blocked` | - | - | `not_tested` | `unknown` |