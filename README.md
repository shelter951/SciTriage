# SciTriage

SciTriage is a harness-agnostic diagnostic and evidence-gating plugin for AutoResearch agents.

AutoResearch systems can generate candidate discoveries faster than they can generate reliable evidence. SciTriage sits beside an existing harness, reads traces/logs/claims, and decides what evidence is still missing before a claim should be written into the research record.

It does not reuse AutoResearchGuard. It uses a clean interface:

- ingest a structured AutoResearch trace,
- assign failure/risk labels,
- identify evidence debt,
- recommend the cheapest next diagnostic probe,
- compare seed groups against measured noise,
- allow or block claims,
- emit machine-readable and human-readable reports.

## Karpathy Autoresearch Result

Current primary integration target:

```text
Karpathy autoresearch
```

On our 4x RTX 4090 server, the original default run reaches training but OOMs on a single 4090. SciTriage treats this as a resource-contract mismatch and materializes a 4090 quick-probe contract:

```text
MAX_SEQ_LEN=1024
TIME_BUDGET=60
EVAL_TOKENS=2 * 65536
DEPTH=4
DEVICE_BATCH_SIZE=32
TOTAL_BATCH_SIZE=2**16
WINDOW_PATTERN="L"
```

Under this real Karpathy quick-probe setting:

- baseline depth4 noise: mean `1.245304`, std `0.004211`,
- one-shot candidate selection accepts 3/4 initial candidates,
- seed-group evidence accepts only 1/4,
- one-shot false discovery rate among accepted candidates: `0.667`,
- depth scaling finds a robust compute-aware optimum at `depth7`,
- a non-depth batch-size candidate, `batch_small`, is also accepted,
- combining `depth7 + batch_small` is deprioritized because its delta is far below the depth7 noise scale.

Paper-style results are in:

```text
analysis/autoresearch_probe_v1/PAPER_RESULTS.md
```

## Useful Commands

```bash
cd ~/projects/scitriage
```

Assess a generic trace:

```bash
python3 -m scitriage.cli assess examples/confounded_noisy_trace.json --out runs/confounded_noisy
```

Diagnose a failed Karpathy autoresearch run:

```bash
python3 -m scitriage.cli resource-fit \
  --run-log /path/to/run.log \
  --out runs/resource_fit
```

Create a 4090-friendly autoresearch probe copy:

```bash
python3 -m scitriage.cli materialize-autoresearch-probe \
  --source-repo /path/to/autoresearch \
  --target-dir /path/to/autoresearch_probe_4090
```

Compare candidate and baseline seed groups:

```bash
python3 -m scitriage.cli compare-seed-groups \
  --baseline-logs baseline/runs/seed_*.log \
  --candidate-logs candidate/runs/seed_*.log \
  --metric val_bpb \
  --out runs/candidate_vs_baseline.json
```

Gate a claim:

```bash
python3 -m scitriage.cli claim-gate \
  --group-compare runs/candidate_vs_baseline.json \
  --claim "This candidate improves val_bpb under the probe contract." \
  --out runs/claim_gate
```

Prioritize a follow-up probe:

```bash
python3 -m scitriage.cli prioritize-probe \
  --candidate-log candidate/runs/seed_1.log \
  --baseline-summary runs/baseline_seed_summary.json \
  --metric val_bpb \
  --out runs/probe_priority
```

## Regression Checks

```bash
python3 -m scitriage.cli eval-benchmark benchmarks/failure_injected/cases --out runs/benchmark_latest
python3 -m scitriage.cli toy-policy-sweep --n-per-type 20 --out runs/toy_policy_sweep_latest
```
