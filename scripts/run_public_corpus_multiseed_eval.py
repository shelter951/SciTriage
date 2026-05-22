from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

import run_false_discovery_corpus_eval as corpus_eval


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    std = stdev(values) if len(values) >= 2 else 0.0
    return {
        "mean": round(mean(values), 6),
        "std": round(std, 6),
        "ci95": round(1.96 * std / math.sqrt(len(values)), 6) if len(values) >= 2 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def run_multiseed(
    corpus_path: Path,
    seeds: List[int],
    traces_per_task: int,
    max_candidates: int,
    keep_traces: bool = False,
) -> Dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    seed_results = []
    for seed in seeds:
        seed_results.append(
            corpus_eval.evaluate(corpus["tasks"], traces_per_task, max_candidates, seed)
        )
    policies = [row["policy"] for row in seed_results[0]["aggregate"]]
    aggregate = []
    for policy in policies:
        rows = [
            next(row for row in result["aggregate"] if row["policy"] == policy)
            for result in seed_results
        ]
        aggregate.append({
            "policy": policy,
            "invalid_accept_rate": _summary([row["invalid_accept_rate"] for row in rows]),
            "mean_valid_score_retention": _summary([row["mean_valid_score_retention"] for row in rows]),
            "mean_cost_units": _summary([row["mean_cost_units"] for row in rows]),
            "cost_per_valid_claim": _summary([
                row["cost_per_valid_claim"]
                for row in rows
                if row["cost_per_valid_claim"] is not None
            ]),
        })
    return {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "traces_per_task": traces_per_task,
        "max_candidates_per_trace": max_candidates,
        "task_count": seed_results[0]["task_count"],
        "candidate_count": seed_results[0]["candidate_count"],
        "invalid_candidate_count": seed_results[0]["invalid_candidate_count"],
        "aggregate": aggregate,
        "seed_aggregates": [
            {
                "seed": seed,
                "aggregate": result["aggregate"],
            }
            for seed, result in zip(seeds, seed_results)
        ],
        "seed_results": seed_results if keep_traces else None,
    }


def render(result: Dict[str, Any]) -> str:
    lines = ["# Public Failure Corpus Multi-Seed Evaluation\n"]
    lines.append(
        "This repeats the public-surface same-agent evaluation across multiple deterministic trace seeds. "
        "It measures whether the SciTriage advantage is stable under candidate-order and observed-score noise."
    )
    lines.append("")
    lines.append("## Scale\n")
    lines.append(f"- Public benchmark surfaces: {result['task_count']}")
    lines.append(f"- Candidate records: {result['candidate_count']}")
    lines.append(f"- Invalid-evidence candidates: {result['invalid_candidate_count']}")
    lines.append(f"- Trace seeds: {result['num_seeds']}")
    lines.append(f"- Traces per seed per policy: {result['task_count'] * result['traces_per_task']}")
    lines.append("")
    lines.append("## Aggregate Across Seeds\n")
    lines.append("| Policy | Invalid accept rate | Valid-score retention | Mean cost | Cost / valid claim |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in result["aggregate"]:
        invalid = row["invalid_accept_rate"]
        retention = row["mean_valid_score_retention"]
        cost = row["mean_cost_units"]
        cpvc = row["cost_per_valid_claim"]
        lines.append(
            f"| `{row['policy']}` | "
            f"{invalid['mean']:.3f} +/- {invalid['ci95']:.3f} | "
            f"{retention['mean']:.3f} +/- {retention['ci95']:.3f} | "
            f"{cost['mean']:.3f} +/- {cost['ci95']:.3f} | "
            f"{cpvc['mean']:.3f} +/- {cpvc['ci95']:.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    full = next(row for row in result["aggregate"] if row["policy"] == "scitriage_full")
    score_only = next(row for row in result["aggregate"] if row["policy"] == "official_score_only")
    lines.append(
        "Across trace seeds, score-only selection has invalid accept rate "
        f"{score_only['invalid_accept_rate']['mean']:.3f} +/- {score_only['invalid_accept_rate']['ci95']:.3f}; "
        "full SciTriage has invalid accept rate "
        f"{full['invalid_accept_rate']['mean']:.3f} +/- {full['invalid_accept_rate']['ci95']:.3f}."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--corpus", default="benchmarks/false_discovery_corpus/corpus.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=20260522)
    parser.add_argument("--traces-per-task", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--keep-traces", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    seeds = [args.seed_start + idx for idx in range(args.seeds)]
    result = run_multiseed(
        root / args.corpus,
        seeds,
        args.traces_per_task,
        args.max_candidates,
        args.keep_traces,
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "public_failure_corpus_multiseed_eval.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (out / "PUBLIC_FAILURE_CORPUS_MULTISEED_EVAL.md").write_text(
        render(result),
        encoding="utf-8",
    )
    print(out / "public_failure_corpus_multiseed_eval.json")
    print(out / "PUBLIC_FAILURE_CORPUS_MULTISEED_EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
