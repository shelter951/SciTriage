# Paper Readiness Assessment

Last updated: 2026-05-22.

## Short Answer

SciTriage is not yet a complete top-conference submission, but it has a credible top-conference core.

The current result is strong enough for a serious paper direction because it has:

- a clear problem: AutoResearch agents can convert misleading visible scores into false research claims;
- a simple reusable method: gate claims and candidate selection with task-validity evidence;
- external benchmark evidence: four score-bearing MLAgentBench audits;
- one additional BabyLM compatibility audit for language-model artifact validity;
- a practical open-source artifact: command-line tools, plugin APIs, reports, and public artifacts.

It is not yet enough for AAAI/ICLR/NeurIPS main-conference strength because it still needs:

- broader task coverage;
- fully autonomous LLM loops that generate candidates from scratch, not only same-agent policy evaluation over candidate trajectories;
- stronger baselines against simpler static checks;
- a systematic false-positive/false-negative analysis;
- automatic invariant generation or at least a reusable invariant template library.

The latest improvement is operational: `analysis/experiment_campaign_full_v2` now reruns the main evidence package end to end, including unit tests, the 40-seed public-surface stress suite, same-agent policy evaluation, and three official MLAgentBench audits (`vectorization`, `cifar10`, `imdb`). This makes the current evidence easier to reproduce, but it does not remove the need for larger live-loop experiments.

After that, we added `analysis/external_mlagentbench_babylm_v1` as a compatibility audit. It expands the task family to language-model artifacts, but because the upstream BabyLM `eval.py` is dependency-incompatible on the current server, it should be presented separately from the four direct score-bearing audits.

## Current Main Result

Across curated MLAgentBench external audits:

| Metric | Value |
|---|---:|
| score-bearing external tasks | 4 |
| audited candidates | 20 |
| candidates blocked by SciTriage | 8 |
| invalid visible-score winners blocked | 3 / 4 visible winners were invalid; all invalid visible winners blocked |
| visible-winner block rate | 0.750 |

Tasks:

- `MLAgentBench/vectorization`: blocks invalid runtime shortcuts while preserving a 235x valid speedup.
- `MLAgentBench/cifar10`: blocks a perfect-score CIFAR-10 test-label oracle.
- `MLAgentBench/imdb`: blocks a perfect-score IMDB test-label oracle.
- `MLAgentBench/CLRS`: keeps a valid checkpointed visible-score winner and blocks a missing-checkpoint candidate.

## Current Paper Claim

The defensible claim today is:

```text
Visible benchmark scores are insufficient evidence for AutoResearch claims.
SciTriage is a lightweight validity gate that blocks high-scoring invalid candidates
and selects the best candidate that still satisfies task-valid evidence.
```

The claim we cannot make yet is:

```text
SciTriage broadly improves autonomous research agents across many natural tasks.
```

That broader claim needs live agent-loop experiments.

We now have an intermediate same-agent policy evaluation: the same score-seeking policy sees the same candidate trajectories with and without SciTriage evidence. This closes part of the gap, but does not replace fully autonomous LLM loops.

We also now have a stronger closed-loop replay: the same agent observes executed audit candidates sequentially and decides when to write a final claim. This reduces the gap between static selection and live agent behavior, but still does not replace a fresh LLM-generation loop.

We now have a first fresh LLM-generation pilot on `MLAgentBench/vectorization`. It is not large enough to claim broad live-agent improvement, but it is scientifically useful: prompt-level awareness of SciTriage did not prevent invalid high-speed shortcuts, while the executable semantic gate did.

## What A Top-Conference Version Needs

Minimum strong version:

1. 5-8 score-bearing external tasks.
2. At least 2 task families: semantic equivalence, leakage/evaluator gaming, seed-noise claim gating.
3. Same-agent comparison: with SciTriage vs without SciTriage.
4. Metrics:
   - invalid accept rate,
   - valid best-score retention,
   - extra cost,
   - false block rate,
   - claim rewrite quality,
   - final valid task score.
5. Ablations:
   - no semantic gate,
   - no leakage gate,
   - no seed-noise gate,
   - static-only baseline,
   - LLM-judge-only baseline.

Ideal version:

- automatic task-contract extraction from benchmark files;
- generated semantic invariants for function-level tasks;
- integration hooks for at least two AutoResearch harnesses;
- qualitative examples of prevented false discoveries;
- public replay package for all audits.

## Current Verdict

The project is paper-worthy but not finished-paper-ready.

After the full v2 campaign, the honest assessment is:

```text
Strong workshop-free research direction: yes.
Credible short-paper / systems-demo style story: yes.
AAAI main-conference ready today: not yet.
```

The current evidence is strong enough to show that the score-validity gap is real and that SciTriage can close it on selected public benchmark surfaces. A top-conference version still needs the stronger claim that the same autonomous agent, under the same budget, makes better final research decisions when SciTriage is attached.

Best near-term strategy:

```text
Do not oversell "agentic RL" or "complete AutoResearch".
Position SciTriage as a reusable evidence gate for AutoResearch agents.
Win by showing that current agent benchmarks have a score-validity gap,
and that a small plugin can close it in practical cases.
```

This is a good story because it is humble, concrete, and experimentally checkable.

## Updated Experimental Status

The project now has:

- 4 complete score-bearing candidate audits;
- 1 BabyLM compatibility audit for loadable model/tokenizer artifacts;
- 1 checkpoint-style MLAgentBench task (`CLRS`);
- an aggregate audit summary over the four complete audits;
- a public README that presents the problem and results for non-specialist readers.
- a full v2 campaign report that reruns the lightweight official audits and stress evaluations without interactive prompting.
- a closed-loop replay result over 5 executed/compatibility audits, with 1,500 loops per policy.
- a first MiMo live-agent vectorization pilot with fresh generated candidate code.

The highest-leverage next experiment is now a fully autonomous same-LLM comparison: run the same LLM agent with and without SciTriage while it creates candidates from scratch, then measure invalid accept rate, final valid score, and extra cost.

Current same-agent policy result:

| Policy | Invalid Accept Rate | Mean Valid-Score Retention |
|---|---:|---:|
| `official_score_only` | 0.750 | 0.250 |
| `static_artifact_rule` | 0.750 | 0.250 |
| `judge_only_proxy` | 0.750 | 0.250 |
| `scitriage_full` | 0.000 | 1.000 |

Closed-loop replay result:

| Policy | Invalid Final Claim Rate | Valid Final Claim Rate | Valid-Score Retention |
|---|---:|---:|---:|
| `official_score_only` | 0.600 | 0.400 | 0.400 |
| `static_artifact_rule` | 0.600 | 0.400 | 0.399 |
| `judge_only_proxy` | 0.600 | 0.400 | 0.399 |
| `scitriage_full` | 0.000 | 1.000 | 0.960 |

Additional scale result:

We added a public false-discovery corpus over all 15 MLAgentBench task surfaces. This is not official benchmark execution; it is a deterministic public-surface stress suite designed to test broad failure-mode coverage and same-agent policy behavior.

| Policy | Traces | Invalid accept rate | Mean valid-score retention |
|---|---:|---:|---:|
| `official_score_only` | 180 | 0.744 | 0.254 |
| `static_artifact_rule` | 180 | 0.739 | 0.260 |
| `judge_only_proxy` | 180 | 0.739 | 0.260 |
| `scitriage_full` | 180 | 0.000 | 0.982 |

Paper implication: this helps with scale and failure taxonomy, but the strongest next result is still fully autonomous live LLM loops over a larger official-executed subset.

See:

```text
docs/LIVE_AGENT_LOOP_RESULTS.md
analysis/experiment_campaign_full_v2/CAMPAIGN_SUMMARY.md
```
