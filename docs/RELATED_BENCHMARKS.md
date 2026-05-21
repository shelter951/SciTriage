# Related Benchmarks And Positioning

Last updated: 2026-05-21.

SciTriage is not meant to compete with AutoResearch harnesses. It is a validity and evidence layer that can be plugged into them.

## Benchmark Landscape

| Project | What it measures | Why it matters for SciTriage |
|---|---|---|
| [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) | End-to-end ML experimentation tasks for language agents | Best immediate external harness for task artifacts, official eval scripts, logs, and candidate selection audits |
| [MLE-bench](https://github.com/openai/mle-bench) | Kaggle-style ML engineering benchmark with competition-style scoring | Stronger hidden-score realism; heavier data/setup target after MLAgentBench |
| [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench) | Scientific discovery agent benchmark | Useful for the long-term claim that agents need evidence discipline, not only score optimization |
| [AI Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | Automated scientific discovery pipeline | Natural integration target because SciTriage can gate generated claims and experiment conclusions |
| [Karpathy autoresearch](https://github.com/karpathy/autoresearch) | Compact AutoResearch loop around a real training metric | Good first harness for seed noise, resource-fit, and claim-gating experiments |

## Current Position

The strongest current claim is:

```text
SciTriage prevents AutoResearch loops from promoting high-scoring but invalid candidates.
```

We support this with:

- Karpathy-style seed-noise evidence on 24 observed candidates.
- MLAgentBench `vectorization`: blocks invalid runtime shortcuts and selects a 235x faster valid implementation.
- MLAgentBench `cifar10`: blocks a perfect-score test-label oracle.
- MLAgentBench `imdb`: blocks the same perfect-score oracle pattern on text classification.
- MLAgentBench task-surface audit over all benchmark folders to choose the next external tasks.

## Why Not Only Chase Higher Scores?

For AutoResearch, a higher benchmark score is not enough if the route to that score breaks the task contract. The paper story should focus on selection quality:

```text
visible-score-only policy -> selects the apparent best result
SciTriage-gated policy   -> selects the best result that is still valid evidence
```

That keeps the project aligned with current agent benchmarks while making a distinct contribution: evidence discipline as a reusable layer.

## Next Benchmark Targets

Near-term:

- MLAgentBench `CLRS`: candidate audit if dependencies are manageable.
- More MLAgentBench Kaggle-style tasks once credentials and data caches are ready.
- More vectorization-style tasks where semantic invariants can be generated automatically.

Mid-term:

- MLE-bench subset with cached data and strict no-leakage policies.
- Live AutoResearch loop comparison: same agent, with and without SciTriage, measuring invalid accept rate, cost, and final valid score.
