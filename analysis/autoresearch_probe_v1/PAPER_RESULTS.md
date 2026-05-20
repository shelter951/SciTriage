# SciTriage Karpathy Autoresearch Probe V1

## Plain-English Finding
SciTriage turns noisy AutoResearch candidate claims into evidence-gated decisions. In the 4090 quick-probe setting, one-shot selection would accept several candidates that do not survive seed-group uncertainty checks, while robust candidates such as depth7 and batch_small are allowed.

## Result 1: False Discovery Reduction

| Metric | Value |
|---|---:|
| num_variants | 4 |
| one_shot_accepts | 3 |
| group_accepts | 1 |
| one_shot_false_discoveries_vs_group_gate | 2 |
| false_discovery_rate_among_one_shot_accepts | 0.6666666666666666 |

## Result 2: Depth Scaling Under Fixed 4090 Probe Budget

| Variant | Mean val_bpb | Std | Delta vs depth4 | 95% Margin | Verdict |
|---|---:|---:|---:|---:|---|
| depth4_baseline | 1.245304 | 0.004211 | 0.000000 | 0.000000 | `baseline` |
| depth5 | 1.201118 | 0.002507 | 0.044186 | 0.005546 | `supports_improvement` |
| depth6 | 1.192322 | 0.006802 | 0.052982 | 0.009053 | `supports_improvement` |
| depth7 | 1.178177 | 0.002234 | 0.067128 | 0.005395 | `supports_improvement` |
| depth8 | 1.183984 | 0.003596 | 0.061321 | 0.006267 | `supports_improvement` |

Best current candidate: `depth7`.

## Result 3: Non-Depth Batch/Schedule Family

| Variant | Seed1 val_bpb | Seed1 delta vs baseline | Probe Priority |
|---|---:|---:|---|
| batch_large | 1.260044 | -0.013404 | `low` |
| batch_small | 1.238157 | 0.008483 | `high` |
| warmdown_long | 1.248078 | -0.001438 | `low` |
| warmdown_short | 1.242831 | 0.003809 | `medium` |

Multi-seed accepted non-depth candidate: `batch_small`.

| Candidate | Mean val_bpb | Std | Delta vs baseline | 95% Margin | Verdict |
|---|---:|---:|---:|---:|---|
| batch_small | 1.236136 | 0.003353 | 0.009168 | 0.006091 | `supports_improvement` |

## Result 4: Interaction Check

Combining the two accepted directions did not justify more seed budget:

| Candidate | Candidate val_bpb | Baseline mean | Delta | Delta/std | Priority |
|---|---:|---:|---:|---:|---|
| depth7_batch_small | 1.178042 | 1.178177 | 0.000135 | 0.060271 | `low` |

## Paper Claim

In this real Karpathy autoresearch probe, SciTriage (1) diagnoses resource-contract mismatch, (2) estimates seed noise, (3) blocks unsupported one-shot discoveries, (4) accepts robust discoveries, and (5) avoids wasting compute on low-value follow-up probes.

## Caveat

The 4090 quick-probe contract is a resource-normalized diagnostic setting, not the original H100-oriented Karpathy default. Claims should be scoped to this contract unless rerun under the original contract.