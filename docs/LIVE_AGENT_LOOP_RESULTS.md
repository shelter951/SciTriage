# Live Agent Loop Results

Last updated: 2026-05-22.

## What This Experiment Tests

We evaluate the same score-seeking agent policy under different evidence views.

The candidate trajectories and official scores are fixed. The difference is what the agent is allowed to see before it writes a final claim:

- `official_score_only`: sees only benchmark scores.
- `static_artifact_rule`: sees score plus simple submission/checkpoint artifact checks.
- `judge_only_proxy`: simulates a text-only judge that sees result summaries and artifact presence, but not SciTriage semantic/leakage instrumentation.
- `scitriage_full`: sees score plus semantic, leakage, schema, checkpoint, and seed-noise gates.
- ablations: remove one SciTriage gate at a time.

This is not yet a fully autonomous LLM that writes and edits every candidate from scratch. It is a same-agent policy evaluation over real MLAgentBench candidate trajectories and Karpathy-style AutoResearch trajectories. That is the right intermediate step before running expensive live LLM loops.

## External MLAgentBench Policy Results

Artifact:

```text
analysis/same_agent_policy_eval_v1/SAME_AGENT_POLICY_EVAL.md
```

Aggregate result over 4 score-bearing MLAgentBench tasks:

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

Interpretation:

- Without SciTriage, the score-seeking agent accepts invalid winners on 3 of 4 tasks.
- Full SciTriage reduces invalid accepts to 0.
- Removing the semantic gate reintroduces the vectorization failure.
- Removing the leakage gate reintroduces CIFAR-10 and IMDB label-oracle failures.
- Removing checkpoint/schema gates has no effect on current winners, but CLRS shows those gates block missing-checkpoint candidates.

## Public-Surface Stress Results

The official-executed set above is intentionally conservative and still small. To test broader failure coverage, the repository now also includes a public false-discovery corpus built from all 15 MLAgentBench task surfaces.

Artifacts:

```text
benchmarks/false_discovery_corpus/INDEX.md
analysis/public_failure_corpus_eval_v1/PUBLIC_FAILURE_CORPUS_EVAL.md
```

This is a stress suite, not official benchmark execution. It uses deterministic candidate traces derived from public task metadata and failure modes, then evaluates the same candidate trajectories with and without SciTriage.

| Policy | Traces | Invalid accept rate | Mean valid-score retention | Mean cost |
|---|---:|---:|---:|---:|
| `official_score_only` | 180 | 0.744 | 0.254 | 5.468 |
| `static_artifact_rule` | 180 | 0.739 | 0.260 | 6.368 |
| `judge_only_proxy` | 180 | 0.739 | 0.260 | 6.368 |
| `scitriage_full` | 180 | 0.000 | 0.982 | 6.368 |

This makes the current evidence less toy-like while keeping the boundary honest: official-executed audits show real benchmark behavior; the public-surface corpus shows scale and failure-mode coverage.

We also repeat the public-surface evaluation across 40 deterministic trace seeds:

```text
analysis/public_failure_corpus_multiseed_eval_v1/PUBLIC_FAILURE_CORPUS_MULTISEED_EVAL.md
```

| Policy | Invalid accept rate | Mean valid-score retention | Cost / valid claim |
|---|---:|---:|---:|
| `official_score_only` | 0.770 +/- 0.008 | 0.228 +/- 0.008 | 23.749 +/- 0.828 |
| `static_artifact_rule` | 0.753 +/- 0.007 | 0.245 +/- 0.007 | 25.666 +/- 0.762 |
| `judge_only_proxy` | 0.753 +/- 0.007 | 0.245 +/- 0.007 | 25.666 +/- 0.762 |
| `scitriage_full` | 0.000 +/- 0.000 | 0.973 +/- 0.002 | 6.348 +/- 0.017 |

## Closed-Loop Replay Results

Artifact:

```text
analysis/closed_loop_replay_eval_v1/CLOSED_LOOP_REPLAY_EVAL.md
```

This evaluation uses executed/curated audit candidates from `vectorization`, `cifar10`, `imdb`, `CLRS`, and the `babylm` compatibility audit. The same agent observes candidates sequentially in randomized order and may stop early with a final claim. The only difference across policies is which evidence gates are available before claiming.

| Policy | Invalid Final Claim Rate | Valid Final Claim Rate | Valid-Score Retention | Mean Cost |
|---|---:|---:|---:|---:|
| `official_score_only` | 0.600 | 0.400 | 0.400 | 2.825 |
| `static_artifact_rule` | 0.600 | 0.400 | 0.399 | 3.400 |
| `judge_only_proxy` | 0.600 | 0.400 | 0.399 | 3.400 |
| `scitriage_full` | 0.000 | 1.000 | 0.960 | 5.112 |

Interpretation: this is stronger than one-shot winner selection because the agent makes a sequential final-claim decision under equal candidate order. It is still not a fresh LLM-generation experiment; it is a replay over executed audit suites.

## Seed-Noise Policy Results

Artifact:

```text
analysis/seed_noise_policy_eval_v1/SEED_NOISE_POLICY_EVAL.md
```

Karpathy-style AutoResearch result:

| Policy | Extra Seed Runs | True Accepts | False Accepts | Precision | Recall | Family Precision | Family Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `one_shot_claiming` | 0 | 5 | 2 | 0.714 | 1.000 | 0.500 | 1.000 |
| `verify_all_one_shot_positives` | 14 | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| `scitriage_high_priority_only` | 12 | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| `scitriage_family_landmarks` | 4 | 2 | 0 | 1.000 | 0.400 | 1.000 | 1.000 |
| `exhaustive_verify_all_candidates` | 16 | 5 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |

Interpretation:

- Without a seed-noise gate, one-shot claiming accepts false discoveries.
- Targeted SciTriage reruns remove false accepts.
- The family-landmark policy preserves research-direction recall while using 71.4% fewer extra seed runs than verifying every one-shot positive.

## Current Paper-Level Claim

The stronger claim is now:

```text
Across external MLAgentBench tasks and a Karpathy-style AutoResearch trace,
SciTriage reduces invalid or unsupported accepts while preserving valid progress.
```

This is a credible experimental core. The remaining top-conference gap is a fully autonomous LLM loop that creates candidates from scratch under both conditions.

## Optional LLM Judge Baseline

The repository includes a dry-run-safe OpenAI-compatible baseline harness:

```bash
python scripts/run_llm_judge_baseline.py --repo-root . --out analysis/llm_judge_baseline_v1 --dry-run
```

To run an actual judge, set:

```bash
SCITRIAGE_LLM_API_BASE=https://your-openai-compatible-endpoint/v1
SCITRIAGE_LLM_API_KEY=...
SCITRIAGE_LLM_MODEL=your-model
```

This keeps keys out of the repository and lets us compare `judge_only_proxy` with real LLM judge behavior when a suitable API is available.
