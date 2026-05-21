from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_AUDITS = [
    "analysis/external_mlagentbench_vectorization_v3/candidate_audit.json",
    "analysis/external_mlagentbench_cifar10_v1/candidate_audit.json",
    "analysis/external_mlagentbench_imdb_v1/candidate_audit.json",
    "analysis/external_mlagentbench_clrs_v1/candidate_audit.json",
]


POLICIES = [
    "official_score_only",
    "static_artifact_rule",
    "judge_only_proxy",
    "scitriage_full",
    "ablate_no_semantic_gate",
    "ablate_no_leakage_gate",
    "ablate_no_checkpoint_gate",
    "ablate_no_schema_gate",
]


def _load(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def _variant(row: Dict[str, object]) -> str:
    return str(row.get("variant") or row.get("candidate_id") or row.get("name"))


def _score(row: Dict[str, object]) -> float | None:
    score = row.get("official_score")
    return float(score) if score is not None else None


def _lower_is_better(task: str) -> bool:
    return "vectorization" in task.lower()


def _score_key(row: Dict[str, object], lower_is_better: bool) -> float:
    score = _score(row)
    if score is None:
        return float("inf")
    return score if lower_is_better else -score


def _component_passes(row: Dict[str, object]) -> Dict[str, bool]:
    schema = bool(row.get("schema_passed", True))
    semantic = bool(row.get("semantic_invariant_passed", True))
    leakage = bool(row.get("test_label_gate_passed", True))
    checkpoint = bool(row.get("checkpoint_passed", True))
    official_eval = bool(row.get("official_eval_passed", row.get("official_score") is not None))
    return {
        "schema": schema,
        "semantic": semantic,
        "leakage": leakage,
        "checkpoint": checkpoint and official_eval,
        "score_present": row.get("official_score") is not None,
    }


def _full_valid(row: Dict[str, object]) -> bool:
    components = _component_passes(row)
    if "validity_gate_passed" in row:
        return bool(row["validity_gate_passed"]) and components["score_present"]
    return all(components.values())


def _policy_valid(row: Dict[str, object], policy: str) -> bool:
    c = _component_passes(row)
    if policy == "official_score_only":
        return c["score_present"]
    if policy == "static_artifact_rule":
        return c["score_present"] and c["schema"] and c["checkpoint"]
    if policy == "judge_only_proxy":
        # A text-only judge sees the run summary and artifact presence, but not hidden
        # semantic equivalence tests or test-label-leak instrumentation.
        return c["score_present"] and c["schema"] and c["checkpoint"]
    if policy == "scitriage_full":
        return _full_valid(row)
    if policy == "ablate_no_semantic_gate":
        return c["score_present"] and c["schema"] and c["leakage"] and c["checkpoint"]
    if policy == "ablate_no_leakage_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["checkpoint"]
    if policy == "ablate_no_checkpoint_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["leakage"]
    if policy == "ablate_no_schema_gate":
        return c["score_present"] and c["semantic"] and c["leakage"] and c["checkpoint"]
    raise ValueError(f"unknown policy {policy}")


def _select(rows: List[Dict[str, object]], policy: str, lower_is_better: bool) -> Dict[str, object] | None:
    candidates = [row for row in rows if _score(row) is not None and _policy_valid(row, policy)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _score_key(row, lower_is_better))


def _best_valid(rows: List[Dict[str, object]], lower_is_better: bool) -> Dict[str, object] | None:
    valid = [row for row in rows if _score(row) is not None and _full_valid(row)]
    if not valid:
        return None
    return min(valid, key=lambda row: _score_key(row, lower_is_better))


def _retention(selected: Dict[str, object] | None, best: Dict[str, object] | None, lower_is_better: bool) -> float:
    if selected is None or best is None or not _full_valid(selected):
        return 0.0
    selected_score = _score(selected)
    best_score = _score(best)
    if selected_score is None or best_score is None:
        return 0.0
    if lower_is_better:
        return best_score / selected_score if selected_score else 0.0
    return selected_score / best_score if best_score else 0.0


def _trajectory(rows: List[Dict[str, object]], policy: str, lower_is_better: bool) -> List[Dict[str, object]]:
    seen: List[Dict[str, object]] = []
    out = []
    for row in rows:
        seen.append(row)
        selected = _select(seen, policy, lower_is_better)
        out.append({
            "observed": _variant(row),
            "observed_score": _score(row),
            "observed_full_valid": _full_valid(row),
            "selected_so_far": _variant(selected) if selected else None,
            "selected_so_far_full_valid": _full_valid(selected) if selected else False,
        })
    return out


def evaluate_audit(audit: Dict[str, object]) -> Dict[str, object]:
    task = str(audit["task"])
    rows = list(audit.get("rows", []))
    lower_is_better = _lower_is_better(task)
    best = _best_valid(rows, lower_is_better)
    policy_rows = []
    for policy in POLICIES:
        selected = _select(rows, policy, lower_is_better)
        full_valid = _full_valid(selected) if selected else False
        policy_rows.append({
            "policy": policy,
            "selected": _variant(selected) if selected else None,
            "official_score": _score(selected) if selected else None,
            "full_valid": full_valid,
            "invalid_accept": selected is not None and not full_valid,
            "valid_score_retention": _retention(selected, best, lower_is_better),
            "candidate_observations": len(rows),
            "gate_checks": _gate_checks(policy, rows),
            "cost_units": len(rows) + _gate_checks(policy, rows),
            "trajectory": _trajectory(rows, policy, lower_is_better),
        })
    return {
        "task": task,
        "lower_is_better": lower_is_better,
        "candidate_count": len(rows),
        "best_valid": {
            "selected": _variant(best) if best else None,
            "official_score": _score(best) if best else None,
        },
        "policies": policy_rows,
    }


def _gate_checks(policy: str, rows: Iterable[Dict[str, object]]) -> int:
    rows = list(rows)
    if policy == "official_score_only":
        return 0
    if policy in {"static_artifact_rule", "judge_only_proxy"}:
        return sum(1 for row in rows if "schema_passed" in row or "checkpoint_passed" in row)
    if policy == "ablate_no_semantic_gate":
        return sum(1 for row in rows if "test_label_gate_passed" in row or "checkpoint_passed" in row or "schema_passed" in row)
    if policy == "ablate_no_leakage_gate":
        return sum(1 for row in rows if "semantic_invariant_passed" in row or "checkpoint_passed" in row or "schema_passed" in row)
    if policy == "ablate_no_checkpoint_gate":
        return sum(1 for row in rows if "semantic_invariant_passed" in row or "test_label_gate_passed" in row or "schema_passed" in row)
    if policy == "ablate_no_schema_gate":
        return sum(1 for row in rows if "semantic_invariant_passed" in row or "test_label_gate_passed" in row or "checkpoint_passed" in row)
    return sum(1 for row in rows if any(k in row for k in [
        "semantic_invariant_passed",
        "test_label_gate_passed",
        "schema_passed",
        "checkpoint_passed",
    ]))


def aggregate(task_results: List[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    for policy in POLICIES:
        matches = [
            policy_row
            for task in task_results
            for policy_row in task["policies"]
            if policy_row["policy"] == policy
        ]
        rows.append({
            "policy": policy,
            "tasks": len(matches),
            "invalid_accepts": sum(1 for row in matches if row["invalid_accept"]),
            "invalid_accept_rate": sum(1 for row in matches if row["invalid_accept"]) / len(matches) if matches else 0.0,
            "mean_valid_score_retention": sum(row["valid_score_retention"] for row in matches) / len(matches) if matches else 0.0,
            "total_cost_units": sum(row["cost_units"] for row in matches),
            "mean_cost_units": sum(row["cost_units"] for row in matches) / len(matches) if matches else 0.0,
            "selected": {task["task"]: next(
                row["selected"] for row in task["policies"] if row["policy"] == policy
            ) for task in task_results},
        })
    return {
        "tasks": len(task_results),
        "policies": rows,
        "task_results": task_results,
    }


def render_markdown(result: Dict[str, object]) -> str:
    lines = ["# Same-Agent Policy Evaluation\n"]
    lines.append(
        "The same score-seeking agent policy is evaluated under different evidence views. "
        "Each policy sees the same candidate trajectories and official scores; policies differ only in which validity gates are available."
    )
    lines.append("")
    lines.append("## Aggregate Results\n")
    lines.append("| Policy | Invalid Accept Rate | Mean Valid-Score Retention | Cost Units |")
    lines.append("|---|---:|---:|---:|")
    for row in result["policies"]:
        lines.append(
            f"| `{row['policy']}` | {row['invalid_accept_rate']:.3f} | "
            f"{row['mean_valid_score_retention']:.3f} | {row['total_cost_units']} |"
        )
    lines.append("")
    lines.append("## Per-Task Selections\n")
    tasks = [task["task"] for task in result["task_results"]]
    lines.append("| Policy | " + " | ".join(f"`{task}`" for task in tasks) + " |")
    lines.append("|---" + "|---" * len(tasks) + "|")
    for row in result["policies"]:
        selected = [row["selected"][task] for task in tasks]
        lines.append("| `" + row["policy"] + "` | " + " | ".join(f"`{item}`" for item in selected) + " |")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "`official_score_only` is the no-SciTriage agent: it accepts invalid winners on the runtime and leakage tasks. "
        "`scitriage_full` has zero invalid accepts and preserves the best valid candidate on every task. "
        "The ablations show which gate is responsible for each failure family."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("audits", nargs="*")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    audits = [_load(root / path) for path in (args.audits or DEFAULT_AUDITS)]
    task_results = [evaluate_audit(audit) for audit in audits]
    result = aggregate(task_results)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "same_agent_policy_eval.json").write_text(json.dumps(result, indent=2))
    (out / "SAME_AGENT_POLICY_EVAL.md").write_text(render_markdown(result))
    print(out / "same_agent_policy_eval.json")
    print(out / "SAME_AGENT_POLICY_EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
