from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


POLICIES = [
    "official_score_only",
    "static_artifact_rule",
    "judge_only_proxy",
    "scitriage_full",
    "ablate_no_semantic_gate",
    "ablate_no_leakage_gate",
    "ablate_no_checkpoint_gate",
    "ablate_no_seed_noise_gate",
    "ablate_no_schema_gate",
]


def _read(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(row: Dict[str, Any]) -> float | None:
    value = row.get("official_score")
    return float(value) if value is not None else None


def _variant(row: Dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return str(row.get("variant") or row.get("candidate_id") or row.get("name"))


def _higher_is_better(task: Dict[str, Any]) -> bool:
    if "higher_is_better" in task:
        return bool(task["higher_is_better"])
    return "vectorization" not in str(task.get("task", "")).lower()


def _score_key(row: Dict[str, Any], higher_is_better: bool) -> float:
    score = _score(row)
    if score is None:
        return float("inf")
    return -score if higher_is_better else score


def _component_passes(row: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "score_present": row.get("official_score") is not None,
        "schema": bool(row.get("schema_passed", True)),
        "semantic": bool(row.get("semantic_invariant_passed", True)),
        "leakage": bool(row.get("test_label_gate_passed", True)),
        "checkpoint": bool(row.get("checkpoint_passed", True)) and bool(row.get("official_eval_passed", True)),
        "seed_noise": bool(row.get("seed_noise_gate_passed", True)),
    }


def _full_valid(row: Dict[str, Any] | None) -> bool:
    if row is None:
        return False
    c = _component_passes(row)
    if "is_valid_evidence" in row:
        return bool(row["is_valid_evidence"]) and c["score_present"]
    if "validity_gate_passed" in row:
        return bool(row["validity_gate_passed"]) and c["score_present"]
    return all(c.values())


def _policy_valid(row: Dict[str, Any], policy: str) -> bool:
    c = _component_passes(row)
    if policy == "official_score_only":
        return c["score_present"]
    if policy == "static_artifact_rule":
        return c["score_present"] and c["schema"] and c["checkpoint"]
    if policy == "judge_only_proxy":
        return c["score_present"] and c["schema"] and c["checkpoint"]
    if policy == "scitriage_full":
        return _full_valid(row)
    if policy == "ablate_no_semantic_gate":
        return c["score_present"] and c["schema"] and c["leakage"] and c["checkpoint"] and c["seed_noise"]
    if policy == "ablate_no_leakage_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["checkpoint"] and c["seed_noise"]
    if policy == "ablate_no_checkpoint_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["leakage"] and c["seed_noise"]
    if policy == "ablate_no_seed_noise_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["leakage"] and c["checkpoint"]
    if policy == "ablate_no_schema_gate":
        return c["score_present"] and c["semantic"] and c["leakage"] and c["checkpoint"] and c["seed_noise"]
    raise ValueError(f"unknown policy {policy}")


def _select(rows: List[Dict[str, Any]], policy: str, higher_is_better: bool) -> Dict[str, Any] | None:
    candidates = [row for row in rows if _score(row) is not None and _policy_valid(row, policy)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _score_key(row, higher_is_better))


def _best_valid(rows: Iterable[Dict[str, Any]], higher_is_better: bool) -> Dict[str, Any] | None:
    valid = [row for row in rows if _score(row) is not None and _full_valid(row)]
    if not valid:
        return None
    return min(valid, key=lambda row: _score_key(row, higher_is_better))


def _retention(selected: Dict[str, Any] | None, best: Dict[str, Any] | None, higher_is_better: bool) -> float:
    if selected is None or best is None or not _full_valid(selected):
        return 0.0
    selected_score = _score(selected)
    best_score = _score(best)
    if selected_score is None or best_score is None or best_score == 0:
        return 0.0
    if higher_is_better:
        return min(1.0, selected_score / best_score)
    return min(1.0, best_score / selected_score) if selected_score else 0.0


def _stable_seed(task: str, trace_id: int, global_seed: int) -> int:
    digest = hashlib.sha256(f"{global_seed}:{trace_id}:{task}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _make_trace(task: Dict[str, Any], trace_id: int, global_seed: int, max_candidates: int) -> List[Dict[str, Any]]:
    rows = list(task["rows"])
    rng = random.Random(_stable_seed(str(task["task"]), trace_id, global_seed))
    scored = []
    higher = _higher_is_better(task)
    for row in rows:
        score = _score(row)
        score_std = float(row.get("official_score_std", 0.0) or 0.0)
        if score is not None:
            noisy_score = score + rng.gauss(0, score_std)
            row = dict(row)
            row["observed_trace_score"] = round(noisy_score, 6)
            row["official_score"] = round(max(0.000001, noisy_score), 6)
        attractiveness = -_score_key(row, higher) + rng.random() * 0.08
        if not _full_valid(row):
            attractiveness += rng.random() * 0.035
        scored.append((attractiveness, rng.random(), row))
    scored.sort(reverse=True)
    chosen = [row for _, _, row in scored[:max_candidates]]
    rng.shuffle(chosen)
    return chosen


def _gate_checks(policy: str, rows: List[Dict[str, Any]]) -> int:
    if policy == "official_score_only":
        return 0
    keys_by_policy = {
        "static_artifact_rule": ["schema_passed", "checkpoint_passed"],
        "judge_only_proxy": ["schema_passed", "checkpoint_passed"],
        "ablate_no_semantic_gate": ["test_label_gate_passed", "checkpoint_passed", "seed_noise_gate_passed", "schema_passed"],
        "ablate_no_leakage_gate": ["semantic_invariant_passed", "checkpoint_passed", "seed_noise_gate_passed", "schema_passed"],
        "ablate_no_checkpoint_gate": ["semantic_invariant_passed", "test_label_gate_passed", "seed_noise_gate_passed", "schema_passed"],
        "ablate_no_seed_noise_gate": ["semantic_invariant_passed", "test_label_gate_passed", "checkpoint_passed", "schema_passed"],
        "ablate_no_schema_gate": ["semantic_invariant_passed", "test_label_gate_passed", "checkpoint_passed", "seed_noise_gate_passed"],
    }
    keys = keys_by_policy.get(policy, [
        "semantic_invariant_passed",
        "test_label_gate_passed",
        "checkpoint_passed",
        "seed_noise_gate_passed",
        "schema_passed",
    ])
    return sum(1 for row in rows if any(key in row for key in keys))


def _cost(policy: str, rows: List[Dict[str, Any]]) -> float:
    candidate_cost = sum(float(row.get("cost_units", 1.0)) for row in rows)
    gate_cost = 0.18 * _gate_checks(policy, rows)
    return round(candidate_cost + gate_cost, 3)


def evaluate(tasks: List[Dict[str, Any]], traces_per_task: int, max_candidates: int, global_seed: int) -> Dict[str, Any]:
    trace_rows = []
    for task in tasks:
        higher = _higher_is_better(task)
        best = _best_valid(task["rows"], higher)
        for trace_id in range(traces_per_task):
            trace = _make_trace(task, trace_id, global_seed, max_candidates)
            for policy in POLICIES:
                selected = _select(trace, policy, higher)
                trace_rows.append({
                    "task": task["task"],
                    "benchmark_task": task.get("benchmark_task", task["task"]),
                    "evaluation_tier": task.get("evaluation_tier", "unknown"),
                    "trace_id": trace_id,
                    "policy": policy,
                    "observed_candidates": [_variant(row) for row in trace],
                    "selected": _variant(selected),
                    "selected_score": _score(selected) if selected else None,
                    "selected_failure_mode": selected.get("failure_mode") if selected else None,
                    "invalid_accept": selected is not None and not _full_valid(selected),
                    "valid_score_retention": _retention(selected, best, higher),
                    "cost_units": _cost(policy, trace),
                })
    aggregate = []
    for policy in POLICIES:
        rows = [row for row in trace_rows if row["policy"] == policy]
        valid_rows = [row for row in rows if not row["invalid_accept"] and row["selected"] is not None]
        aggregate.append({
            "policy": policy,
            "traces": len(rows),
            "tasks": len({row["task"] for row in rows}),
            "invalid_accepts": sum(1 for row in rows if row["invalid_accept"]),
            "invalid_accept_rate": sum(1 for row in rows if row["invalid_accept"]) / len(rows) if rows else 0.0,
            "mean_valid_score_retention": sum(row["valid_score_retention"] for row in rows) / len(rows) if rows else 0.0,
            "mean_cost_units": sum(row["cost_units"] for row in rows) / len(rows) if rows else 0.0,
            "cost_per_valid_claim": (
                sum(row["cost_units"] for row in rows) / len(valid_rows)
                if valid_rows else None
            ),
        })
    failure_modes: Dict[str, int] = {}
    for task in tasks:
        for row in task["rows"]:
            mode = row.get("failure_mode")
            if mode:
                failure_modes[str(mode)] = failure_modes.get(str(mode), 0) + 1
    return {
        "traces_per_task": traces_per_task,
        "max_candidates_per_trace": max_candidates,
        "global_seed": global_seed,
        "task_count": len(tasks),
        "candidate_count": sum(len(task["rows"]) for task in tasks),
        "invalid_candidate_count": sum(1 for task in tasks for row in task["rows"] if not _full_valid(row)),
        "failure_modes": dict(sorted(failure_modes.items())),
        "aggregate": aggregate,
        "traces": trace_rows,
    }


def render(result: Dict[str, Any]) -> str:
    lines = ["# Public Failure Corpus Same-Agent Evaluation\n"]
    lines.append(
        "This evaluation uses the same candidate traces for each policy. "
        "The corpus is built from public MLAgentBench task surfaces and is intended as a broad stress suite; official-executed results remain reported separately."
    )
    lines.append("")
    lines.append("## Scale\n")
    lines.append(f"- Public benchmark surfaces: {result['task_count']}")
    lines.append(f"- Candidate records: {result['candidate_count']}")
    lines.append(f"- Invalid-evidence candidates: {result['invalid_candidate_count']}")
    lines.append(f"- Same-agent traces per policy: {result['task_count'] * result['traces_per_task']}")
    lines.append("")
    lines.append("## Aggregate Results\n")
    lines.append("| Policy | Traces | Invalid accept rate | Mean valid-score retention | Mean cost | Cost / valid claim |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in result["aggregate"]:
        cpvc = row["cost_per_valid_claim"]
        cpvc_text = "n/a" if cpvc is None else f"{cpvc:.3f}"
        lines.append(
            f"| `{row['policy']}` | {row['traces']} | {row['invalid_accept_rate']:.3f} | "
            f"{row['mean_valid_score_retention']:.3f} | {row['mean_cost_units']:.3f} | {cpvc_text} |"
        )
    lines.append("")
    lines.append("## Failure Modes In Corpus\n")
    lines.append("| Failure mode | Candidate count |")
    lines.append("|---|---:|")
    for mode, count in result["failure_modes"].items():
        lines.append(f"| `{mode}` | {count} |")
    lines.append("")
    lines.append("## Interpretation\n")
    full = next(row for row in result["aggregate"] if row["policy"] == "scitriage_full")
    official = next(row for row in result["aggregate"] if row["policy"] == "official_score_only")
    lines.append(
        f"On the broad public-surface stress suite, score-only selection accepts invalid evidence in "
        f"{official['invalid_accept_rate']:.1%} of traces. Full SciTriage reduces this to "
        f"{full['invalid_accept_rate']:.1%} while keeping mean valid-score retention at "
        f"{full['mean_valid_score_retention']:.3f}."
    )
    lines.append("")
    lines.append(
        "This is useful as a scale test and failure-corpus result. It should be presented next to, not instead of, the official-executed MLAgentBench audits."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--corpus", default="benchmarks/false_discovery_corpus/corpus.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--traces-per-task", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260522)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    corpus = _read(root / args.corpus)
    result = evaluate(corpus["tasks"], args.traces_per_task, args.max_candidates, args.seed)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "public_failure_corpus_eval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "PUBLIC_FAILURE_CORPUS_EVAL.md").write_text(render(result), encoding="utf-8")
    print(out / "public_failure_corpus_eval.json")
    print(out / "PUBLIC_FAILURE_CORPUS_EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
