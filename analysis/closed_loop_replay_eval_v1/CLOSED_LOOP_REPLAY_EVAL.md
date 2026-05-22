# Closed-Loop Replay Evaluation

Closed-loop replay over executed/curated candidate audits. Candidate order is randomized with shared seeds across policies. The base agent policy is identical; policies differ only in what evidence is allowed before a final claim.

This is stronger than one-shot winner selection because the agent observes candidates sequentially and may stop early with a final claim. It is still a replay over executed audits, not a fully autonomous LLM loop.

## Scale

- Tasks: 5
- Traces per task: 300
- Loops per policy: 1500
- Max observed candidates per loop: 5
- Claim tolerance: 0.02

## Aggregate Results

| Policy | Invalid Final Claim Rate | Valid Final Claim Rate | Valid-Score Retention | Mean Cost | Mean Steps |
|---|---:|---:|---:|---:|---:|
| `official_score_only` | 0.600 | 0.400 | 0.400 | 2.825 | 2.825 |
| `static_artifact_rule` | 0.600 | 0.400 | 0.399 | 3.400 | 2.987 |
| `judge_only_proxy` | 0.600 | 0.400 | 0.399 | 3.400 | 2.987 |
| `scitriage_full` | 0.000 | 1.000 | 0.960 | 5.112 | 4.027 |
| `ablate_no_semantic_gate` | 0.200 | 0.800 | 0.799 | 4.626 | 3.721 |
| `ablate_no_leakage_gate` | 0.400 | 0.600 | 0.560 | 3.886 | 3.293 |
| `ablate_no_checkpoint_gate` | 0.000 | 1.000 | 0.960 | 5.012 | 4.027 |
| `ablate_no_schema_gate` | 0.000 | 1.000 | 0.960 | 4.752 | 4.027 |

## Tasks

- `MLAgentBench/vectorization`
- `MLAgentBench/cifar10`
- `MLAgentBench/imdb`
- `MLAgentBench/CLRS`
- `MLAgentBench/babylm`

## Interpretation

Score-only final claims are invalid in 0.600 of closed-loop replays. Full SciTriage reduces this to 0.000, with 0.960 mean valid-score retention.
The remaining limitation is candidate generation: these candidates come from executed audit suites, not from a fresh LLM agent writing new patches during this evaluation.