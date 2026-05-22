# MLAgentBench BabyLM Candidate Audit

External task: `MLAgentBench/benchmarks/babylm`.
Score is compatible causal-LM perplexity on BabyLM test text, lower is better.
The upstream MLAgentBench BabyLM `eval.py` is incompatible with the current server dependency stack (`transformers` 5.x and `datasets` 4.x remove APIs it imports). This audit therefore uses the same task data and perplexity objective with a compatibility evaluator, and marks that limitation explicitly.

- Visible-score best candidate: `tiny_random_valid`
- SciTriage-gated best candidate: `tiny_random_valid`
- Candidates blocked by artifact gate: 3
- Repeats per candidate: 1
- Eval texts: 200

## Policy Comparison

| Policy | Selected | Perplexity | Artifact Gate |
|---|---|---:|---|
| `visible_score_only` | `tiny_random_valid` | 512.7338 | True |
| `scitriage_gated` | `tiny_random_valid` | 512.7338 | True |

## Candidate Table

| Variant | Mean Perplexity | Std | Artifact Gate | Triage Status |
|---|---:|---:|---|---|
| `tiny_random_valid` | 512.7338 | 0.0000 | True | `allowed` |
| `tiny_wider_valid` | 517.6116 | 0.0000 | True | `allowed` |
| `invalid_missing_output` | - | 0.0000 | False | `blocked` |
| `invalid_config_only` | - | 0.0000 | False | `blocked` |
| `invalid_tokenizer_only` | - | 0.0000 | False | `blocked` |

## Interpretation

This task adds a model-artifact validity surface: a language-model benchmark can only support a research claim if the candidate writes a loadable model and tokenizer. SciTriage blocks missing or partial artifacts before the agent can present an unloadable run as a result.