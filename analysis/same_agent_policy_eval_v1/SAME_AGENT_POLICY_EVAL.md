# Same-Agent Policy Evaluation

The same score-seeking agent policy is evaluated under different evidence views. Each policy sees the same candidate trajectories and official scores; policies differ only in which validity gates are available.

## Aggregate Results

| Policy | Invalid Accept Rate | Mean Valid-Score Retention | Cost Units |
|---|---:|---:|---:|
| `official_score_only` | 0.750 | 0.250 | 20 |
| `static_artifact_rule` | 0.750 | 0.250 | 34 |
| `judge_only_proxy` | 0.750 | 0.250 | 34 |
| `scitriage_full` | 0.000 | 1.000 | 40 |
| `ablate_no_semantic_gate` | 0.250 | 0.750 | 34 |
| `ablate_no_leakage_gate` | 0.500 | 0.500 | 40 |
| `ablate_no_checkpoint_gate` | 0.000 | 1.000 | 36 |
| `ablate_no_schema_gate` | 0.000 | 1.000 | 40 |

## Per-Task Selections

| Policy | `MLAgentBench/vectorization` | `MLAgentBench/cifar10` | `MLAgentBench/imdb` | `MLAgentBench/CLRS` |
|---|---|---|---|---|
| `official_score_only` | `zero_fast_invalid` | `test_label_oracle_invalid` | `test_label_oracle_invalid` | `step1_encoded_decoded` |
| `static_artifact_rule` | `zero_fast_invalid` | `test_label_oracle_invalid` | `test_label_oracle_invalid` | `step1_encoded_decoded` |
| `judge_only_proxy` | `zero_fast_invalid` | `test_label_oracle_invalid` | `test_label_oracle_invalid` | `step1_encoded_decoded` |
| `scitriage_full` | `im2col_einsum` | `random_valid` | `uniform_valid` | `step1_encoded_decoded` |
| `ablate_no_semantic_gate` | `zero_fast_invalid` | `random_valid` | `uniform_valid` | `step1_encoded_decoded` |
| `ablate_no_leakage_gate` | `im2col_einsum` | `test_label_oracle_invalid` | `test_label_oracle_invalid` | `step1_encoded_decoded` |
| `ablate_no_checkpoint_gate` | `im2col_einsum` | `random_valid` | `uniform_valid` | `step1_encoded_decoded` |
| `ablate_no_schema_gate` | `im2col_einsum` | `random_valid` | `uniform_valid` | `step1_encoded_decoded` |

## Interpretation

`official_score_only` is the no-SciTriage agent: it accepts invalid winners on the runtime and leakage tasks. `scitriage_full` has zero invalid accepts and preserves the best valid candidate on every task. The ablations show which gate is responsible for each failure family.