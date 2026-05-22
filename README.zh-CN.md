# SciTriage

<p align="center">
  <img src="assets/scitriage_icon_cartoon.png" alt="SciTriage cartoon icon" width="132">
</p>

[English](README.md) | [中文](README.zh-CN.md)

![SciTriage banner](assets/readme_banner.svg)

**给 AutoResearch 智能体用的研究证据闸门。**

现在的 AutoResearch 智能体已经能提出想法、改代码、跑实验、写总结。新的问题是：

> 一个结果可以在可见分数上看起来很强，但作为科研证据完全不可靠。

SciTriage 是一个插件式证据层。它不替代 Karpathy autoresearch、ARIS、Claude Code、Codex 或你自己的科研循环，而是站在它们旁边，读取 agent 已经产生的日志、patch、分数和 claim，然后给出三个动作之一：

| 决策 | 含义 |
|---|---|
| `allow` | 当前证据足够支持这个 claim |
| `block` | 分数/结论有误导性、无效或证据不足 |
| `probe` | 先做一个最便宜的补充验证，再决定能不能 claim |

```text
AutoResearch agent -> idea, patch, logs, scores -> SciTriage -> allow / block / probe
```

## 为什么值得看

AutoResearch 不只需要更会想 idea 的模型。它还需要一个机制，阻止“假发现”进入研究记录。

SciTriage 关注的是每次实验结束后的关键时刻：

- 到底是 idea 不行，还是实现坏了？
- 这个提升真的超过 seed noise 了吗？
- 候选方案是在解决任务，还是在钻 benchmark 的空子？
- 现在这个结论能写进报告吗？
- 下一步最值得花算力验证什么？

## 核心结果

SciTriage 已经在外部 MLAgentBench 任务上抓到了高分但无效的候选。

| 外部任务 | 只看可见分数会选什么 | SciTriage 做什么 |
|---|---|---|
| `MLAgentBench/vectorization` | `0.005080s` 的 shortcut，跳过真实卷积计算 | 阻断它，选择有效且 `217x` 加速的实现 |
| `MLAgentBench/cifar10` | `1.0000` accuracy 的 test-label oracle | 标记为 benchmark leakage 并阻断 |
| `MLAgentBench/imdb` | `1.0000` accuracy 的 test-label oracle | 标记为 benchmark leakage 并阻断 |
| `MLAgentBench/CLRS` | 一个有效 checkpoint 模型 | 保留可见分数 winner，同时阻断缺 checkpoint 的候选 |

在 Karpathy-style AutoResearch 轨迹上，SciTriage 也能减少 false discoveries：24 个候选里，family-landmark policy 在保持 research-direction recall 为 `1.000` 的同时，比验证所有 one-shot positives 少用 `71.4%` 的额外 seed runs。

聚合外部审计结果：4 个 score-bearing MLAgentBench 任务，20 个候选，8 个无效候选被阻断。4 个 visible-score winners 里有 3 个是无效的，SciTriage 全部挡住；CLRS 的 visible winner 是有效的，所以 SciTriage 保留它。

Same-agent policy evaluation：没有 SciTriage 的 score-seeking agent invalid accept rate 是 `0.750`，mean valid-score retention 是 `0.250`；完整 SciTriage 在同一批候选轨迹上 invalid accept rate 是 `0.000`，valid-score retention 是 `1.000`。

现在我们也把规模从 4 个官方执行审计扩展到了一个 public false-discovery corpus：它覆盖全部 15 个 MLAgentBench 公开任务表面。基于 112 个候选记录和 180 条 same-agent stress traces，只看分数的策略 invalid accept rate 是 `0.744`；完整 SciTriage 降到 `0.000`，同时保持 `0.982` mean valid-score retention。这个结果会明确标注为 public-surface stress test，不伪装成官方 benchmark 执行。

跨 40 个 deterministic trace seeds 后结果仍然稳定：score-only invalid accept rate 是 `0.770 +/- 0.008`，完整 SciTriage 是 `0.000 +/- 0.000`，并保持 `0.973 +/- 0.002` valid-score retention。

## 可复现 Campaign

最新服务器 campaign 会把核心证据链端到端跑一遍：

```bash
python scripts/run_experiment_campaign.py \
  --repo-root . \
  --out analysis/experiment_campaign_full_v2 \
  --mlagentbench-root /path/to/MLAgentBench \
  --seeds 40 \
  --traces-per-task 12 \
  --max-candidates 5 \
  --include-official
```

已完成步骤：

| 步骤类型 | 状态 |
|---|---|
| public failure corpus 构建 | passed |
| 40-seed public-surface stress evaluation | passed |
| same-agent 有/无 SciTriage policy evaluation | passed |
| external audit summary | passed |
| unit tests | passed |
| official MLAgentBench vectorization audit | passed |
| official MLAgentBench CIFAR-10 audit | passed |
| official MLAgentBench IMDB audit | passed |

Campaign 报告：[analysis/experiment_campaign_full_v2/CAMPAIGN_SUMMARY.md](analysis/experiment_campaign_full_v2/CAMPAIGN_SUMMARY.md)

## 它具体做什么

SciTriage 回答三个实用问题。

| 问题 | SciTriage 输出 |
|---|---|
| 为什么失败？ | implementation bug、evaluator drift、noisy result、resource mismatch、unsupported claim 或 likely invalid idea |
| 值不值得继续花算力？ | 基于 one-shot delta 和 measured seed noise 的 probe priority |
| 能不能写 claim？ | 基于 seed-group evidence 和 uncertainty margin 的 claim gate |

目标很简单：让 AutoResearch 系统跑得快，但不要把 noisy one-run wins 写成科研结论。

## 当前真实实验

我们用 [Karpathy autoresearch](https://github.com/karpathy/autoresearch) 作为第一个真实 harness，因为它足够小、足够知名，而且有清晰指标 `val_bpb`。

在 4 张 RTX 4090 的服务器上，默认配置对单卡太大。SciTriage 把这类失败归为 **resource-contract mismatch**，而不是 idea failure，并给出 4090-friendly quick-probe contract：

```text
MAX_SEQ_LEN=1024
TIME_BUDGET=60
EVAL_TOKENS=2 * 65536
DEPTH=4
DEVICE_BATCH_SIZE=32
TOTAL_BATCH_SIZE=2**16
WINDOW_PATTERN="L"
```

在这个真实 quick-probe 设置下：

| 发现 | 结果 |
|---|---:|
| observed candidate variants | `24` |
| baseline seed std | `0.004211` val_bpb |
| one-shot candidates accepted | `3 / 4` |
| candidates accepted after seed-group gate | `1 / 4` |
| false discovery rate among one-shot accepts | `0.667` |
| best depth candidate | `depth7` |
| accepted non-depth candidate | `batch_small` |
| family-landmark seed saving | `71.4%` fewer extra seed runs |

重要的不是 `depth7` 赢了，而是两个看起来像“发现”的 one-shot 结果，在 seed-noise 检验后消失了。

完整结果：[analysis/autoresearch_probe_v1/PAPER_RESULTS.md](analysis/autoresearch_probe_v1/PAPER_RESULTS.md)

Evidence board：[analysis/autoresearch_probe_v1/evidence_board_v1/EVIDENCE_BOARD.md](analysis/autoresearch_probe_v1/evidence_board_v1/EVIDENCE_BOARD.md)

## 外部 Benchmark 结果

我们用 [MLAgentBench](https://github.com/snap-stanford/MLAgentBench) 做外部验证，并使用其原始任务目录和官方 eval scripts。

当前外部结果覆盖 4 个 score-bearing MLAgentBench audits 和 3 类 failure modes：

| MLAgentBench 任务 | visible-score-only winner | SciTriage-gated winner | SciTriage 阻断什么 |
|---|---|---|---|
| `vectorization` | `zero_fast_invalid` at `0.005080s` | `im2col_einsum` at `0.016234s` | invalid runtime shortcut |
| `cifar10` | `test_label_oracle_invalid` at `1.0000` acc | `random_valid` at `0.1042` acc | test-label leakage |
| `imdb` | `test_label_oracle_invalid` at `1.0000` acc | `uniform_valid` at `0.5000` acc | test-label leakage |
| `CLRS` | `step1_encoded_decoded` at `0.020592` | `step1_encoded_decoded` at `0.020592` | missing checkpoint / unloadable result |

补充的 compatibility audit：

| MLAgentBench 任务 | 结果 | 为什么单独标注 |
|---|---|---|
| `babylm` | 阻断 3 个缺失/不完整模型 artifact，保留 2 个可加载 tiny language model | 上游 `eval.py` 和当前服务器的 `transformers`/`datasets` 版本不兼容，所以这里使用同一任务数据和 perplexity 目标，但通过兼容 evaluator 执行 |

### Vectorization

官方分数是 runtime，所以一个无效 shortcut 可以通过跳过真实计算拿到很好的分数。SciTriage 加入 semantic invariant：候选输出必须匹配原始卷积输出。

| Candidate | Official runtime | Semantic invariant | Triage status |
|---|---:|---|---|
| `zero_fast_invalid` | `0.005080s` | fails | blocked |
| `im2col_einsum` | `0.016234s` | passes | allowed |
| `filter_vectorized` | `0.749370s` | passes | allowed |
| `baseline` | `3.523017s` | passes | allowed |

官方 runtime-only winner 是无效的。SciTriage 阻断它，并选择最快的 semantically valid candidate，它仍然比 baseline 快约 `217x`。

审计文件：[analysis/experiment_campaign_full_v2/official_vectorization/CANDIDATE_AUDIT.md](analysis/experiment_campaign_full_v2/official_vectorization/CANDIDATE_AUDIT.md)

### CIFAR-10

这个任务暴露了另一类问题：starter environment 可以访问 CIFAR-10 test labels。候选可以不训练模型，直接写出完美 one-hot submission。

| Candidate | Official accuracy | Test-label leak gate | Triage status |
|---|---:|---|---|
| `test_label_oracle_invalid` | `1.0000` | fails | blocked |
| `random_valid` | `0.1042` | passes | allowed |
| `uniform_valid` | `0.1000` | passes | allowed |
| `train_prior_valid` | `0.1000` | passes | allowed |

审计文件：[analysis/experiment_campaign_full_v2/official_cifar10/CANDIDATE_AUDIT.md](analysis/experiment_campaign_full_v2/official_cifar10/CANDIDATE_AUDIT.md)

### IMDB

IMDB 在文本分类任务上重复了同样的 leakage pattern。候选可以读取 `imdb["test"]` labels，并写出完美 submission。

| Candidate | Official accuracy | Test-label leak gate | Triage status |
|---|---:|---|---|
| `test_label_oracle_invalid` | `1.0000` | fails | blocked |
| `uniform_valid` | `0.5000` | passes | allowed |
| `train_prior_valid` | `0.5000` | passes | allowed |
| `random_valid` | `0.4988` | passes | allowed |

审计文件：[analysis/experiment_campaign_full_v2/official_imdb/CANDIDATE_AUDIT.md](analysis/experiment_campaign_full_v2/official_imdb/CANDIDATE_AUDIT.md)

### BabyLM Compatibility Audit

BabyLM 补上了 language-model artifact 这一类失败面：候选必须写出可加载的 model 和 tokenizer，perplexity claim 才有意义。

| Candidate | Compatible perplexity | Artifact gate | Triage status |
|---|---:|---|---|
| `tiny_random_valid` | `512.7338` | passes | allowed |
| `tiny_wider_valid` | `517.6116` | passes | allowed |
| `invalid_missing_output` | `-` | fails | blocked |
| `invalid_config_only` | `-` | fails | blocked |
| `invalid_tokenizer_only` | `-` | fails | blocked |

这个结果会明确标成 compatibility audit：上游 MLAgentBench BabyLM `eval.py` 依赖当前环境里已经移除的 `transformers`/`datasets` API。我们保留同一任务数据和 perplexity 目标，但不声称这是未修改上游 `eval.py` 的直接执行。

审计文件：[analysis/external_mlagentbench_babylm_v1/CANDIDATE_AUDIT.md](analysis/external_mlagentbench_babylm_v1/CANDIDATE_AUDIT.md)

### CLRS

CLRS 是 checkpoint-style 任务。候选必须训练模型、保存 `checkpoints/best.pkl`，并且能被官方 evaluator 加载。

| Candidate | Official score | Checkpoint | Triage status |
|---|---:|---|---|
| `step1_encoded_decoded` | `0.020592` | passes | allowed |
| `step1_decoded_only` | `0.017929` | passes | allowed |
| `step1_no_hints` | `0.016693` | passes | allowed |
| `invalid_no_checkpoint` | `-` | fails | blocked |

这里 visible-score winner 本来就是有效的，所以 SciTriage 保留它。这说明 gate 不是“乱挡”，而是在有效证据上选择最优候选。

审计文件：[analysis/external_mlagentbench_clrs_v1/CANDIDATE_AUDIT.md](analysis/external_mlagentbench_clrs_v1/CANDIDATE_AUDIT.md)

## 关键文档

- Task surface audit：[analysis/external_mlagentbench_task_surface_v1/TASK_SURFACE_AUDIT.md](analysis/external_mlagentbench_task_surface_v1/TASK_SURFACE_AUDIT.md)
- Benchmark positioning：[docs/RELATED_BENCHMARKS.md](docs/RELATED_BENCHMARKS.md)
- Paper readiness：[docs/PAPER_READINESS.md](docs/PAPER_READINESS.md)
- Live-loop policy results：[docs/LIVE_AGENT_LOOP_RESULTS.md](docs/LIVE_AGENT_LOOP_RESULTS.md)
- Research next steps：[docs/RESEARCH_NEXT_STEPS.md](docs/RESEARCH_NEXT_STEPS.md)
- Public failure corpus：[benchmarks/false_discovery_corpus/INDEX.md](benchmarks/false_discovery_corpus/INDEX.md)
- Public-surface same-agent evaluation：[analysis/public_failure_corpus_eval_v1/PUBLIC_FAILURE_CORPUS_EVAL.md](analysis/public_failure_corpus_eval_v1/PUBLIC_FAILURE_CORPUS_EVAL.md)
- Public-surface multi-seed evaluation：[analysis/public_failure_corpus_multiseed_eval_v1/PUBLIC_FAILURE_CORPUS_MULTISEED_EVAL.md](analysis/public_failure_corpus_multiseed_eval_v1/PUBLIC_FAILURE_CORPUS_MULTISEED_EVAL.md)
- Full campaign v2：[analysis/experiment_campaign_full_v2/CAMPAIGN_SUMMARY.md](analysis/experiment_campaign_full_v2/CAMPAIGN_SUMMARY.md)
- BabyLM compatibility audit：[analysis/external_mlagentbench_babylm_v1/CANDIDATE_AUDIT.md](analysis/external_mlagentbench_babylm_v1/CANDIDATE_AUDIT.md)
- Official audit target queue：[analysis/official_audit_target_queue_v1/OFFICIAL_AUDIT_TARGET_QUEUE.md](analysis/official_audit_target_queue_v1/OFFICIAL_AUDIT_TARGET_QUEUE.md)

可选 LLM judge baseline：

```bash
SCITRIAGE_LLM_API_BASE=https://your-openai-compatible-endpoint/v1 \
SCITRIAGE_LLM_API_KEY=... \
SCITRIAGE_LLM_MODEL=your-model \
python scripts/run_llm_judge_baseline.py --repo-root . --out analysis/llm_judge_baseline_v1
```

## 安装

```bash
git clone https://github.com/shelter951/SciTriage.git
cd SciTriage
python -m pip install -e .
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

## 快速开始

评估一个通用 trace：

```bash
scitriage assess examples/confounded_noisy_trace.json --out runs/confounded_noisy
```

检查一个失败是不是资源不匹配：

```bash
scitriage resource-fit \
  --run-log /path/to/run.log \
  --out runs/resource_fit
```

比较 candidate 和 baseline 的 seed noise：

```bash
scitriage compare-seed-groups \
  --baseline-logs baseline/runs/seed_*.log \
  --candidate-logs candidate/runs/seed_*.log \
  --metric val_bpb \
  --out runs/candidate_vs_baseline.json
```

对 claim 做 gate：

```bash
scitriage claim-gate \
  --group-compare runs/candidate_vs_baseline.json \
  --claim "The candidate improves val_bpb under the 4090 probe contract." \
  --out runs/claim_gate
```

判断一个 one-shot candidate 是否值得继续跑更多 seeds：

```bash
scitriage prioritize-probe \
  --candidate-log candidate/runs/seed_1.log \
  --baseline-summary runs/baseline_seed_summary.json \
  --metric val_bpb \
  --out runs/probe_priority
```

## 作为插件使用

SciTriage 可以作为 command-line sidecar、Python API，或者 post-run hook 使用。

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

生成 Karpathy-style AutoResearch hook：

```bash
scitriage write-autoresearch-hook --out tools/scitriage_after_run.py
```

如果从源码目录使用：

```bash
python tools/scitriage_after_run.py --scitriage-root /path/to/SciTriage ...
```

更多集成方式：[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)

## 输出

SciTriage 会写出其他 agent 能读的结构化 artifact：

- `triage_report.json`：哪里出了问题，哪些 claim 被阻断。
- `probe_plan.json`：最便宜且有用的下一步验证。
- `claim_gate.json`：当前证据是否允许 claim。
- `probe_priority.json`：candidate 是否值得继续花 seeds。
- `scitriage_bundle.json`：给外部 harness 使用的紧凑 bundle。

## 为什么存在

AutoResearch 循环通常追求速度：propose、patch、run、summarize、repeat。这很强，但会制造新的失败模式：agent 产生“发现”的速度，可能远超实验预算能验证的速度。

SciTriage 加入一个 research hygiene layer：

- 区分 implementation failure 和 idea failure；
- 区分 resource mismatch 和 scientific weakness；
- 区分 one-shot improvement 和 robust improvement；
- 区分日志真正支持的内容和 agent 想写下来的 claim。

这就是 SciTriage 想讲清楚的故事。
