# AutoResearch False-Discovery Corpus

This corpus turns public MLAgentBench task surfaces into same-agent false-discovery traces. It is a stress suite, not a replacement for official benchmark execution: scores in this directory are deterministic stress scores derived from task metadata and candidate failure modes.

- Public task surfaces: 15
- Candidate traces: 112
- Invalid-evidence candidates: 67

| Public task | Candidates | Invalid candidates | Failure modes |
|---|---:|---:|---|
| `CLRS` | 6 | 3 | `artifact_lineage`, `schema_or_artifact`, `seed_noise` |
| `amp-parkinsons-disease-progression-prediction` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `babylm` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `bibtex-generation` | 6 | 3 | `evidence_mismatch`, `schema_or_artifact`, `seed_noise` |
| `cifar10` | 9 | 6 | `artifact_lineage`, `resource_fit`, `schema_or_artifact`, `seed_noise`, `test_leakage` |
| `fathomnet` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `feedback` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `house-price` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `identify-contrails` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `imdb` | 8 | 5 | `artifact_lineage`, `schema_or_artifact`, `seed_noise`, `test_leakage` |
| `literature-review-tool` | 6 | 3 | `evidence_mismatch`, `schema_or_artifact`, `seed_noise` |
| `llama-inference` | 6 | 3 | `schema_or_artifact`, `seed_noise`, `semantic_shortcut` |
| `ogbn-arxiv` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `spaceship-titanic` | 8 | 5 | `artifact_lineage`, `evaluator_overfit`, `resource_fit`, `schema_or_artifact`, `seed_noise` |
| `vectorization` | 7 | 4 | `artifact_lineage`, `schema_or_artifact`, `seed_noise`, `semantic_shortcut` |

Use this corpus for broad same-agent with/without SciTriage evaluations. Keep official-executed benchmark results separate when making paper claims.