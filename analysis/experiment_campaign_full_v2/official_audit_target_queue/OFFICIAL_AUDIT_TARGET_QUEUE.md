# Official Audit Target Queue

This queue ranks public MLAgentBench tasks for conversion from stress-suite coverage into official-executed SciTriage audits.

## Current Official Anchors

`cifar10`, `clrs`, `imdb`, `vectorization`

## Recommended Next 5-8 Targets

| Rank | Task | Priority | Main failure modes | Blockers | Why this target |
|---:|---|---:|---|---|---|
| 1 | `babylm` | 9.000 | `checkpoint_or_model_artifact`, `seed_noise`, `resource_fit` | `external_download` | Language-model task with model artifact validity and perplexity; heavier but on-story for AutoResearch. |
| 2 | `ogbn-arxiv` | 8.000 | `artifact_lineage`, `seed_noise`, `resource_fit` | `external_download` | Non-Kaggle public graph task with official evaluator; heavier dependencies but strong externality. |
| 3 | `feedback` | 6.200 | `answer_leakage`, `schema_or_artifact`, `seed_noise` | `kaggle_data` | Text-regression CSV task; good for claim gating and evaluator leakage after data prep. |
| 4 | `house-price` | 6.200 | `answer_leakage`, `schema_or_artifact`, `seed_noise` | `kaggle_data` | CSV regression task with cheap official MAE eval once Kaggle data is prepared. |
| 5 | `spaceship-titanic` | 6.200 | `answer_leakage`, `schema_or_artifact`, `seed_noise` | `kaggle_data` | CSV classification task with cheap official accuracy eval once Kaggle data is prepared. |

## Full Queue

| Task | Status | Priority | Eval | Kaggle | External download | Data presence |
|---|---|---:|---|---|---|---|
| `babylm` | `next` | 9.000 | True | False | True | 1 / 2 |
| `ogbn-arxiv` | `next` | 8.000 | True | False | True | 0 / 1 |
| `feedback` | `next` | 6.200 | True | True | True | 0 / 2 |
| `house-price` | `next` | 6.200 | True | True | True | 0 / 2 |
| `spaceship-titanic` | `next` | 6.200 | True | True | True | 0 / 2 |
| `amp-parkinsons-disease-progression-prediction` | `later` | 4.200 | True | True | True | 0 / 5 |
| `fathomnet` | `later` | 4.200 | True | True | True | 0 / 4 |
| `identify-contrails` | `later` | 4.200 | True | True | True | 0 / 6 |
| `bibtex-generation` | `integration` | -4.000 | False | False | False | 0 / 0 |
| `literature-review-tool` | `integration` | -4.000 | False | False | False | 0 / 0 |
| `llama-inference` | `integration` | -4.000 | False | False | False | 0 / 0 |
| `CLRS` | `done` | -100.000 | True | False | False | 0 / 0 |
| `cifar10` | `done` | -100.000 | True | False | True | 1 / 1 |
| `imdb` | `done` | -100.000 | True | False | False | 0 / 0 |
| `vectorization` | `done` | -100.000 | True | False | False | 0 / 0 |

## Why This Is Our Advantage

SciTriage does not compete with stronger base agents. It improves the research loop at the evidence boundary: after an agent proposes a patch and reports a score, SciTriage asks whether the result is valid, reproducible, within resource contract, and safe to claim. This makes it plugin-friendly for Karpathy-style loops, ARIS-like systems, Codex/Claude Code workflows, and MLAgentBench runners.