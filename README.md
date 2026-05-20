# SciTriage

**Evidence gates for AutoResearch agents.**

AutoResearch agents are getting good at producing ideas, patches, and experiment logs. The hard part is knowing what happened next:

- Did the idea fail, or did the run fail?
- Is the result bigger than seed noise?
- Is the current claim safe to write down?
- Is the next experiment worth the GPU time?

SciTriage is a small diagnostic layer for that moment. It does not replace Karpathy autoresearch, ARIS, Claude Code, Codex, or your own research loop. It sits beside them, reads their artifacts, and returns a clear next action.

```text
AutoResearch harness  ->  logs, claims, patches  ->  SciTriage  ->  pass / block / probe
```

## What It Does

SciTriage answers three practical questions.

| Question | SciTriage output |
|---|---|
| Why did this idea fail? | implementation bug, evaluator drift, noisy result, resource mismatch, unsupported claim, or likely invalid idea |
| Should we keep spending compute? | probe priority based on one-shot delta versus measured seed noise |
| Can the agent claim this result? | claim gate from seed-group evidence and uncertainty margin |

The goal is simple: let AutoResearch systems move fast without turning noisy one-run wins into research claims.

## Current Result

We use [Karpathy autoresearch](https://github.com/karpathy/autoresearch) as the first real harness because it is compact, widely known, and has a clean metric: `val_bpb`.

On a 4x RTX 4090 server, the default run is too large for a single 4090. SciTriage classifies that as a **resource-contract mismatch**, not as an idea failure, and creates a 4090-friendly quick-probe contract:

```text
MAX_SEQ_LEN=1024
TIME_BUDGET=60
EVAL_TOKENS=2 * 65536
DEPTH=4
DEVICE_BATCH_SIZE=32
TOTAL_BATCH_SIZE=2**16
WINDOW_PATTERN="L"
```

Under this real quick-probe setting:

| Finding | Result |
|---|---:|
| baseline seed std | `0.004211` val_bpb |
| one-shot candidates accepted | `3 / 4` |
| candidates accepted after seed-group gate | `1 / 4` |
| false discovery rate among one-shot accepts | `0.667` |
| best depth candidate | `depth7` |
| accepted non-depth candidate | `batch_small` |

The important point is not just that `depth7` wins. The important point is that two plausible one-shot “discoveries” disappear once we compare against measured seed noise.

Full results: [`analysis/autoresearch_probe_v1/PAPER_RESULTS.md`](analysis/autoresearch_probe_v1/PAPER_RESULTS.md)

## Install

```bash
git clone https://github.com/shelter951/SciTriage.git
cd SciTriage
python -m pip install -e .
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## Quick Start

Assess a generic trace:

```bash
scitriage assess examples/confounded_noisy_trace.json --out runs/confounded_noisy
```

Check whether a failed run is really a resource problem:

```bash
scitriage resource-fit \
  --run-log /path/to/run.log \
  --out runs/resource_fit
```

Compare a candidate against baseline seed noise:

```bash
scitriage compare-seed-groups \
  --baseline-logs baseline/runs/seed_*.log \
  --candidate-logs candidate/runs/seed_*.log \
  --metric val_bpb \
  --out runs/candidate_vs_baseline.json
```

Gate a claim:

```bash
scitriage claim-gate \
  --group-compare runs/candidate_vs_baseline.json \
  --claim "The candidate improves val_bpb under the 4090 probe contract." \
  --out runs/claim_gate
```

Decide whether a one-shot candidate deserves more seeds:

```bash
scitriage prioritize-probe \
  --candidate-log candidate/runs/seed_1.log \
  --baseline-summary runs/baseline_seed_summary.json \
  --metric val_bpb \
  --out runs/probe_priority
```

## Use It As A Plugin

SciTriage can be used as a command-line sidecar, a Python API, or a generated post-run hook.

```python
from scitriage.plugin import seed_group_gate

result = seed_group_gate(
    baseline_logs=["baseline/seed1.log", "baseline/seed2.log", "baseline/seed3.log"],
    candidate_logs=["candidate/seed1.log", "candidate/seed2.log", "candidate/seed3.log"],
    metric="val_bpb",
    claim="The candidate improves val_bpb.",
)

print(result["claim_gate"]["status"])
```

Generate a hook script for Karpathy-style AutoResearch workspaces:

```bash
scitriage write-autoresearch-hook --out tools/scitriage_after_run.py
```

More integration examples: [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md)

## Outputs

SciTriage writes artifacts that another agent can read:

- `triage_report.json`: what went wrong and which claims are blocked.
- `probe_plan.json`: the cheapest useful next check.
- `claim_gate.json`: whether a claim is allowed by current evidence.
- `probe_priority.json`: whether a candidate deserves more seeds.
- `scitriage_bundle.json`: a compact bundle for external harnesses.

## Why This Exists

AutoResearch loops often optimize for momentum: propose, patch, run, summarize, repeat. That is powerful, but it creates a new failure mode: the agent can create more “discoveries” than the experiment budget can verify.

SciTriage adds a research hygiene layer:

- separate implementation failure from idea failure,
- separate resource mismatch from scientific weakness,
- separate one-shot improvement from robust improvement,
- separate what the logs support from what the agent wants to claim.

That is the story we want this project to make concrete.
