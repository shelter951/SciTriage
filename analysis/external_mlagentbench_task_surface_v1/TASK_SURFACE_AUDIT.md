# MLAgentBench Task Surface Audit

This scan reads the external MLAgentBench task folders and records which tasks are immediately useful for SciTriage candidate audits.

| Task | Eval | Data blocker | Agent-visible test labels | Recommended use |
|---|---|---|---|---|
| `CLRS` | True | none | False | candidate audit |
| `amp-parkinsons-disease-progression-prediction` | True | kaggle | False | defer until data prepared |
| `babylm` | True | download | False | defer until data prepared |
| `bibtex-generation` | False | none | False | integration/log audit only |
| `cifar10` | True | download | True | validity-gate audit |
| `fathomnet` | True | kaggle | False | defer until data prepared |
| `feedback` | True | kaggle | False | defer until data prepared |
| `house-price` | True | kaggle | False | defer until data prepared |
| `identify-contrails` | True | kaggle | False | defer until data prepared |
| `imdb` | True | none | False | candidate audit |
| `literature-review-tool` | False | none | False | integration/log audit only |
| `llama-inference` | False | none | False | integration/log audit only |
| `ogbn-arxiv` | True | download | False | defer until data prepared |
| `spaceship-titanic` | True | kaggle | False | defer until data prepared |
| `vectorization` | True | none | False | candidate audit |

## Interpretation

Vectorization is the cleanest semantic-validity task: the official runtime objective can be gamed by skipping computation, and SciTriage can test functional equivalence.
CIFAR-10 exposes a different failure mode: the starter environment contains test-label access, so a candidate can win the official score by reading labels instead of learning.
Most Kaggle-style tasks are better next targets after data credentials and cache preparation, because they provide stronger hidden-score realism but require heavier setup.