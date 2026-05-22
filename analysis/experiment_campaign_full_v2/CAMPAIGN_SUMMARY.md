# SciTriage Experiment Campaign

This campaign runs a batch of reproducible experiments without requiring interactive prompting.

| Step | Return code | Seconds | Timed out |
|---|---:|---:|---|
| `target_queue` | 0 | 0.064 | False |
| `build_public_failure_corpus` | 0 | 0.083 | False |
| `public_failure_corpus_single_seed` | 0 | 0.169 | False |
| `public_failure_corpus_multiseed` | 0 | 2.109 | False |
| `same_agent_policy_eval` | 0 | 0.053 | False |
| `external_audit_summary` | 0 | 0.061 | False |
| `unit_tests` | 0 | 0.124 | False |
| `official_vectorization` | 0 | 17.165 | False |
| `official_cifar10` | 0 | 52.209 | False |
| `official_imdb` | 0 | 28.104 | False |

All campaign steps completed successfully.