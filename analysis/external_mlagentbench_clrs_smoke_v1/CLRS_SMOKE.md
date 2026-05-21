# MLAgentBench CLRS Smoke

External task: `MLAgentBench/benchmarks/CLRS`.

This is a checkpoint-style MLAgentBench task rather than a CSV-submission task. The candidate must train a model, save `checkpoints/best.pkl`, and remain loadable by the official `BaselineModel` evaluation path.

## What We Verified

On the server, with the MLAgentBench CLRS environment copied into a SciTriage scratch workspace:

```text
train.py --train_steps=1 --eval_every=1 --test_every=1 --log_every=1
```

completed successfully and saved a checkpoint.

Then the official `MLAgentBench/benchmarks/CLRS/scripts/eval.py` loaded the checkpoint and returned:

```text
0.02059173583984375
```

## Why This Matters

The first three MLAgentBench audits are candidate-selection audits:

- `vectorization`
- `cifar10`
- `imdb`

CLRS is not yet a full candidate-selection audit. It is a heavier task-family smoke showing that SciTriage can reach a checkpoint-based MLAgentBench workflow. This is the right next target for a fourth score-bearing audit because it tests a different surface:

```text
CSV prediction tasks -> checkpoint and model-loadability tasks
```

## Reproduction Notes

The PyPI package named `clrs` is not the usable package for this benchmark. The working package is:

```text
dm-clrs
```

The smoke used:

```text
dm-clrs==2.0.3
jax==0.6.2
jaxlib==0.6.2
tensorflow==2.21.0
```

JAX fell back to CPU because the installed `jaxlib` is not CUDA-enabled. That is acceptable for this smoke, but a larger CLRS audit should either use CPU intentionally or install a CUDA-enabled JAX stack.

## Current Status

```text
status: smoke_passed
official_eval_score: 0.02059173583984375
next_step: build candidate audit around checkpoint validity, loadability, and score retention
```
