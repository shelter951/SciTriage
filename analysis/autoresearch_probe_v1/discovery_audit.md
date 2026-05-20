# Discovery Audit

- Variants audited: 4
- One-shot accepts: 3
- Group-gate accepts: 1
- One-shot false discoveries vs group gate: 2
- False discovery rate among one-shot accepts: 0.667

## Variant Table

| Variant | Seed1 Delta | Group Delta | 95% Margin | Group Verdict |
|---|---:|---:|---:|---|
| depth5 | 0.047804 | 0.044186 | 0.005546 | `supports_improvement` |
| matrix_lr_high | 0.006960 | 0.005740 | 0.006479 | `inconclusive_within_noise` |
| matrix_lr_low | -0.004994 | -0.003535 | 0.005564 | `does_not_support_improvement` |
| weight_decay_low | 0.003722 | 0.004464 | 0.005706 | `inconclusive_within_noise` |

## Interpretation

One-shot selection accepts every candidate whose first run improves the metric. The group gate requires the candidate-vs-baseline improvement to exceed the combined uncertainty margin. This turns noisy AutoResearch wins into explicit evidence debt instead of publishable claims.