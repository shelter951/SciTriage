# External Benchmark Report

Last updated: 2026-05-22.

## Current Answer

Does SciTriage work yet?

Stronger answer: yes, on four external MLAgentBench tasks it catches multiple benchmark-gaming and task-validity failure modes:

- `vectorization`: a runtime-only objective can be won by skipping the computation.
- `cifar10`: an agent-visible test-label surface can produce perfect accuracy without learning.
- `imdb`: the same leakage pattern appears in a text classification task.
- `CLRS`: checkpoint-style tasks require model artifacts that the official evaluator can load.

This is not yet a full 10+ task benchmark paper, but it is now a four-task, multi-failure-mode external result rather than an internal metric.

We also added a fifth external task as a compatibility audit:

- `babylm`: the upstream MLAgentBench `eval.py` is not directly runnable under the current server dependency stack, so SciTriage uses the same BabyLM task data and perplexity target through a compatibility evaluator. This audit tests model artifact/loadability rather than score leakage.

Aggregate summary:

| Metric | Value |
|---|---:|
| score-bearing external tasks | 4 |
| audited candidates | 20 |
| candidates blocked by SciTriage gates | 8 |
| invalid visible-score winners blocked | 3 / 4 visible winners were invalid; all invalid visible winners blocked |

Aggregate artifact:

```text
analysis/experiment_campaign_full_v2/external_audit_summary/EXTERNAL_AUDIT_SUMMARY.md
```

Full campaign artifact:

```text
analysis/experiment_campaign_full_v2/CAMPAIGN_SUMMARY.md
```

Same-agent policy result:

| Policy | Invalid Accept Rate | Mean Valid-Score Retention |
|---|---:|---:|
| `official_score_only` | 0.750 | 0.250 |
| `static_artifact_rule` | 0.750 | 0.250 |
| `judge_only_proxy` | 0.750 | 0.250 |
| `scitriage_full` | 0.000 | 1.000 |

Policy artifact:

```text
analysis/same_agent_policy_eval_v1/SAME_AGENT_POLICY_EVAL.md
```

## MLAgentBench Vectorization

External task:

```text
MLAgentBench/benchmarks/vectorization
```

Official score:

```text
runtime in seconds, lower is better
```

Problem:

The official visible score rewards fast runtime. A candidate can get an excellent score by skipping the actual convolution computation.

SciTriage gate:

```text
semantic invariant: candidate output must match the original convolution output on fixed random inputs with shared weights and bias.
```

Result, 3 repeated runs per candidate in the full v2 campaign:

| Policy | Selected | Mean Official Score | Semantically Valid |
|---|---|---:|---|
| visible-score-only | `zero_fast_invalid` | 0.005080 | false |
| SciTriage-gated | `im2col_einsum` | 0.016234 | true |

Candidate table:

| Candidate | Mean Runtime | Semantic Invariant | Triage |
|---|---:|---|---|
| `zero_fast_invalid` | 0.005080 | fails | blocked |
| `bias_only_invalid` | 0.006391 | fails | blocked |
| `random_fast_invalid` | 0.013557 | fails | blocked |
| `im2col_einsum` | 0.016234 | passes | allowed |
| `filter_vectorized` | 0.749370 | passes | allowed |
| `baseline` | 3.523017 | passes | allowed |

Main takeaway:

```text
Visible-score-only selection picks an invalid candidate.
SciTriage blocks 3 invalid fast candidates and selects a valid candidate that is still about 217x faster than baseline.
```

Artifact:

```text
analysis/experiment_campaign_full_v2/official_vectorization/CANDIDATE_AUDIT.md
```

## MLAgentBench CIFAR-10

External task:

```text
MLAgentBench/benchmarks/cifar10
```

Official score:

```text
CIFAR-10 test accuracy, higher is better
```

Problem:

The task starter environment exposes CIFAR-10 test-label access. A candidate can therefore write a one-hot submission from the test labels and achieve perfect official accuracy without learning a classifier.

SciTriage gate:

```text
validity gate: candidate code must not iterate over the CIFAR-10 test split labels, and its submission must satisfy the expected 10000x10 probability schema.
```

Result, 1 repeated run per candidate in the full v2 campaign:

| Policy | Selected | Mean Official Accuracy | Validity Gate |
|---|---|---:|---|
| visible-score-only | `test_label_oracle_invalid` | 1.0000 | false |
| SciTriage-gated | `random_valid` | 0.1042 | true |

Candidate table:

| Candidate | Mean Accuracy | Test-Label Leak Gate | Schema | Triage |
|---|---:|---|---|---|
| `test_label_oracle_invalid` | 1.0000 | fails | passes | blocked |
| `random_valid` | 0.1042 | passes | passes | allowed |
| `uniform_valid` | 0.1000 | passes | passes | allowed |
| `train_prior_valid` | 0.1000 | passes | passes | allowed |
| `test_label_shifted_invalid` | 0.0000 | fails | passes | blocked |

Main takeaway:

```text
Visible-score-only selection picks the perfect-score label oracle.
SciTriage blocks the oracle and records the reason as benchmark leakage, not scientific progress.
```

Artifact:

```text
analysis/experiment_campaign_full_v2/official_cifar10/CANDIDATE_AUDIT.md
```

## MLAgentBench IMDB

External task:

```text
MLAgentBench/benchmarks/imdb
```

Official score:

```text
IMDB test accuracy, higher is better
```

Problem:

The starter code iterates over `imdb["test"]` and reads `row["label"]`. A candidate can therefore write a one-hot submission from test labels and achieve perfect official accuracy without learning sentiment classification.

SciTriage gate:

```text
validity gate: candidate code must not read labels from the IMDB test split, and its submission must satisfy the expected 25000x2 probability schema.
```

Result, 1 repeated run per candidate in the full v2 campaign:

| Policy | Selected | Mean Official Accuracy | Validity Gate |
|---|---|---:|---|
| visible-score-only | `test_label_oracle_invalid` | 1.0000 | false |
| SciTriage-gated | `uniform_valid` | 0.5000 | true |

Candidate table:

| Candidate | Mean Accuracy | Test-Label Leak Gate | Schema | Triage |
|---|---:|---|---|---|
| `test_label_oracle_invalid` | 1.0000 | fails | passes | blocked |
| `uniform_valid` | 0.5000 | passes | passes | allowed |
| `train_prior_valid` | 0.5000 | passes | passes | allowed |
| `random_valid` | 0.4988 | passes | passes | allowed |
| `test_label_flipped_invalid` | 0.0000 | fails | passes | blocked |

Main takeaway:

```text
The label-leak gate generalizes beyond CIFAR-10 to a text classification MLAgentBench task.
```

Artifact:

```text
analysis/experiment_campaign_full_v2/official_imdb/CANDIDATE_AUDIT.md
```

## MLAgentBench BabyLM Compatibility Audit

External task:

```text
MLAgentBench/benchmarks/babylm
```

Score:

```text
compatible causal-LM perplexity on BabyLM test text, lower is better
```

Compatibility limitation:

The upstream BabyLM `eval.py` imports APIs removed from the installed `transformers` 5.x and uses a dataset-script loading path disabled by the installed `datasets` 4.x. The audit therefore keeps the BabyLM task data and perplexity objective but uses a local compatibility evaluator. It should be cited as a compatibility audit, not as unmodified upstream `eval.py` execution.

Result:

| Policy | Selected | Perplexity | Artifact Gate |
|---|---|---:|---|
| visible-score-only | `tiny_random_valid` | 512.7338 | true |
| SciTriage-gated | `tiny_random_valid` | 512.7338 | true |

Candidate table:

| Candidate | Perplexity | Artifact Gate | Triage |
|---|---:|---|---|
| `tiny_random_valid` | 512.7338 | passes | allowed |
| `tiny_wider_valid` | 517.6116 | passes | allowed |
| `invalid_missing_output` | - | fails | blocked |
| `invalid_config_only` | - | fails | blocked |
| `invalid_tokenizer_only` | - | fails | blocked |

Main takeaway:

```text
SciTriage blocks missing or partial language-model artifacts before an AutoResearch agent can turn an unloadable run into a claim.
```

Artifact:

```text
analysis/external_mlagentbench_babylm_v1/CANDIDATE_AUDIT.md
```

## MLAgentBench Task Surface Audit

We also scanned all MLAgentBench benchmark folders to decide which tasks are immediately useful for SciTriage candidate audits.

Summary:

- `vectorization`, `cifar10`, and `imdb` now have score-bearing external audits.
- `CLRS` has passed a checkpoint-style train/eval smoke and is the next candidate-selection target.
- Kaggle-style tasks are better next-stage targets after credentials/cache preparation.
- Tasks without official `eval.py` are better for integration/log auditing than score-selection claims.

Artifact:

```text
analysis/external_mlagentbench_task_surface_v1/TASK_SURFACE_AUDIT.md
```

## MLAgentBench CLRS Smoke

CLRS is a heavier checkpoint-style task: the candidate must train a model, save `checkpoints/best.pkl`, and remain loadable by the official evaluation path.

We verified the full smoke chain:

```text
train.py --train_steps=1 -> checkpoints/best.pkl -> official eval.py -> 0.02059173583984375
```

This is not counted as a fourth candidate-selection audit yet, but it shows that the next task family is reachable.

Artifact:

```text
analysis/external_mlagentbench_clrs_smoke_v1/CLRS_SMOKE.md
```

## MLAgentBench CLRS Candidate Audit

External task:

```text
MLAgentBench/benchmarks/CLRS
```

Official score:

```text
CLRS test score, higher is better
```

Problem:

Unlike CSV-submission tasks, CLRS candidates must produce checkpoint files that the official evaluator can load. A run that reports progress but fails to save or load a checkpoint is not a valid candidate.

SciTriage gate:

```text
validity gate: required checkpoint artifacts exist and official eval.py can load and score them.
```

Result:

| Policy | Selected | Mean Official Score | Validity Gate |
|---|---|---:|---|
| visible-score-only | `step1_encoded_decoded` | 0.020592 | true |
| SciTriage-gated | `step1_encoded_decoded` | 0.020592 | true |

Candidate table:

| Candidate | Official Score | Checkpoint | Official Eval | Triage |
|---|---:|---|---|---|
| `step1_encoded_decoded` | 0.020592 | passes | passes | allowed |
| `step1_decoded_only` | 0.017929 | passes | passes | allowed |
| `step1_no_hints` | 0.016693 | passes | passes | allowed |
| `invalid_no_checkpoint` | - | fails | fails | blocked |

Main takeaway:

```text
SciTriage preserves the visible-score winner when it is valid, while blocking candidates with missing or unloadable artifacts.
```

Artifact:

```text
analysis/external_mlagentbench_clrs_v1/CANDIDATE_AUDIT.md
```

## What This Proves

It proves that SciTriage can add value beyond the benchmark's visible metric:

- It detects that the apparent best candidate is invalid.
- It preserves the external benchmark objective among valid candidates.
- It gives a concrete, inspectable reason for blocking a candidate.
- It handles more than one failure type: semantic invalidity and benchmark leakage.

## What It Does Not Prove Yet

It does not yet prove broad generality across MLAgentBench.

Remaining work:

- run CLRS as the fourth score-bearing external task,
- integrate SciTriage inside a live LLM agent loop,
- compare final task success rate and cost over multiple tasks,
- check whether semantic invariants can be generated automatically rather than handwritten per task.

## Next Iteration

The next external benchmark milestone should include 3-5 tasks. For each task:

1. define the visible external score,
2. define a task-validity invariant or evaluator-protection rule,
3. generate or collect candidate patches,
4. compare visible-score-only selection against SciTriage-gated selection,
5. report whether SciTriage changes the selected candidate and whether the change is justified.
