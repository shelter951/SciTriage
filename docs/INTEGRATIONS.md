# Integrations

SciTriage is meant to sit beside an AutoResearch harness, not replace it.

The harness keeps doing the creative work: generate ideas, edit code, run experiments, and write logs. SciTriage reads those artifacts and answers three practical questions:

- Did the idea fail, or did the run fail for a fixable reason?
- Is this result big enough to deserve more compute?
- Is the current claim allowed, or does it need more evidence?

## Integration Shapes

### 1. Command-line Sidecar

Use this when the harness already writes run logs and seed folders.

```bash
scitriage resource-fit --run-log runs/candidate_seed1.log --out runs/scitriage_fit

scitriage prioritize-probe \
  --candidate-log runs/candidate_seed1.log \
  --baseline-summary runs/baseline_seed_summary.json \
  --metric val_bpb \
  --out runs/scitriage_priority

scitriage compare-seed-groups \
  --baseline-logs baseline/runs/seed_*.log \
  --candidate-logs candidate/runs/seed_*.log \
  --metric val_bpb \
  --out runs/candidate_vs_baseline.json
```

### 2. Python Plugin API

Use this when an AutoResearch project has its own orchestrator.

```python
from scitriage.plugin import assess_autoresearch_run, seed_group_gate, write_bundle

bundle = assess_autoresearch_run(
    repo=".",
    run_log="runs/candidate_seed1.log",
    baseline_val_bpb=1.245304,
    baseline_std=0.004211,
    claims=["The candidate improves val_bpb."],
)
write_bundle(bundle, "runs/scitriage_candidate_seed1")

gate = seed_group_gate(
    baseline_logs=["baseline/seed1.log", "baseline/seed2.log", "baseline/seed3.log"],
    candidate_logs=["candidate/seed1.log", "candidate/seed2.log", "candidate/seed3.log"],
    metric="val_bpb",
    claim="The candidate improves val_bpb under the 4090 probe contract.",
)
```

### 3. Generated Post-run Hook

Use this when you want a small script that can be called after every candidate.

```bash
scitriage write-autoresearch-hook --out tools/scitriage_after_run.py

python tools/scitriage_after_run.py \
  --repo . \
  --run-log runs/candidate_seed1.log \
  --baseline-val-bpb 1.245304 \
  --baseline-std 0.004211 \
  --claim "The candidate improves val_bpb." \
  --out runs/scitriage_candidate_seed1
```

## What SciTriage Emits

Each integration produces JSON artifacts that another agent can read:

- `triage_report.json`: risk labels, blocked claims, allowed claims, and the next probe.
- `probe_plan.json`: concrete follow-up actions.
- `scitriage_bundle.json`: one file containing the available diagnosis results.

This makes SciTriage usable as a guardrail in Karpathy-style loops, ARIS-like skill systems, or any harness that can pass a log path and a few claims.
