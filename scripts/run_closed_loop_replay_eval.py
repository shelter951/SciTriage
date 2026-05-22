from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


DEFAULT_AUDITS = [
    "analysis/experiment_campaign_full_v2/official_vectorization/candidate_audit.json",
    "analysis/experiment_campaign_full_v2/official_cifar10/candidate_audit.json",
    "analysis/experiment_campaign_full_v2/official_imdb/candidate_audit.json",
    "analysis/external_mlagentbench_clrs_v1/candidate_audit.json",
    "analysis/external_mlagentbench_babylm_v1/candidate_audit.json",
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


def _load(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    data["_path"] = str(path)
    return data


def _variant(row: Dict[str, Any] | None) -> str | None:
    if row is None:
        return None
    return str(row.get("variant") or row.get("candidate_id") or row.get("name"))


def _score(row: Dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    score = row.get("official_score")
    return float(score) if score is not None else None


def _lower_is_better(audit: Dict[str, Any]) -> bool:
    if "lower_is_better" in audit:
        return bool(audit["lower_is_better"])
    task = str(audit.get("task", "")).lower()
    return "vectorization" in task or "babylm" in task


def _score_key(row: Dict[str, Any], lower_is_better: bool) -> float:
    score = _score(row)
    if score is None:
        return float("inf")
    return score if lower_is_better else -score


def _components(row: Dict[str, Any]) -> Dict[str, bool]:
    schema = bool(row.get("schema_passed", True))
    semantic = bool(row.get("semantic_invariant_passed", True))
    leakage = bool(row.get("test_label_gate_passed", True))
    checkpoint = bool(row.get("checkpoint_passed", True))
    artifact = bool(row.get("artifact_gate_passed", True))
    official_eval = bool(row.get("official_eval_passed", row.get("official_score") is not None))
    score_present = row.get("official_score") is not None
    return {
        "schema": schema,
        "semantic": semantic,
        "leakage": leakage,
        "checkpoint": checkpoint and official_eval,
        "artifact": artifact and official_eval,
        "score_present": score_present,
    }


def _full_valid(row: Dict[str, Any] | None) -> bool:
    if row is None:
        return False
    c = _components(row)
    if "validity_gate_passed" in row:
        return bool(row["validity_gate_passed"]) and c["score_present"]
    return all(c.values())


def _policy_valid(row: Dict[str, Any], policy: str) -> bool:
    c = _components(row)
    if policy == "official_score_only":
        return c["score_present"]
    if policy in {"static_artifact_rule", "judge_only_proxy"}:
        return c["score_present"] and c["schema"] and c["checkpoint"] and c["artifact"]
    if policy == "scitriage_full":
        return _full_valid(row)
    if policy == "ablate_no_semantic_gate":
        return c["score_present"] and c["schema"] and c["leakage"] and c["checkpoint"] and c["artifact"]
    if policy == "ablate_no_leakage_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["checkpoint"] and c["artifact"]
    if policy == "ablate_no_checkpoint_gate":
        return c["score_present"] and c["schema"] and c["semantic"] and c["leakage"] and c["artifact"]
    if policy == "ablate_no_schema_gate":
        return c["score_present"] and c["semantic"] and c["leakage"] and c["checkpoint"] and c["artifact"]
    raise ValueError(f"unknown policy {policy}")


def _select(seen: Iterable[Dict[str, Any]], policy: str, lower_is_better: bool) -> Dict[str, Any] | None:
    candidates = [row for row in seen if _policy_valid(row, policy)]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _score_key(row, lower_is_better))


def _best(rows: Iterable[Dict[str, Any]], lower_is_better: bool, valid_only: bool = False) -> Dict[str, Any] | None:
    candidates = [row for row in rows if _score(row) is not None and (not valid_only or _full_valid(row))]
    if not candidates:
        return None
    return min(candidates, key=lambda row: _score_key(row, lower_is_better))


def _within_claim_band(selected: Dict[str, Any], visible_best: Dict[str, Any], lower_is_better: bool, tolerance: float) -> bool:
    selected_score = _score(selected)
    best_score = _score(visible_best)
    if selected_score is None or best_score is None:
        return False
    if lower_is_better:
        return selected_score <= best_score * (1.0 + tolerance)
    return selected_score >= best_score * (1.0 - tolerance)


def _retention(selected: Dict[str, Any] | None, best_valid: Dict[str, Any] | None, lower_is_better: bool) -> float:
    if selected is None or best_valid is None or not _full_valid(selected):
        return 0.0
    selected_score = _score(selected)
    best_score = _score(best_valid)
    if selected_score is None or best_score is None:
        return 0.0
    if lower_is_better:
        return best_score / selected_score if selected_score else 0.0
    return selected_score / best_score if best_score else 0.0


def _gate_cost(policy: str, row: Dict[str, Any]) -> float:
    if policy == "official_score_only":
        return 0.0
    if policy in {"static_artifact_rule", "judge_only_proxy"}:
        keys = ["schema_passed", "checkpoint_passed", "artifact_gate_passed"]
    elif policy == "ablate_no_semantic_gate":
        keys = ["schema_passed", "test_label_gate_passed", "checkpoint_passed", "artifact_gate_passed"]
    elif policy == "ablate_no_leakage_gate":
        keys = ["schema_passed", "semantic_invariant_passed", "checkpoint_passed", "artifact_gate_passed"]
    elif policy == "ablate_no_checkpoint_gate":
        keys = ["schema_passed", "semantic_invariant_passed", "test_label_gate_passed", "artifact_gate_passed"]
    elif policy == "ablate_no_schema_gate":
        keys = ["semantic_invariant_passed", "test_label_gate_passed", "checkpoint_passed", "artifact_gate_passed"]
    else:
        keys = [
            "schema_passed",
            "semantic_invariant_passed",
            "test_label_gate_passed",
            "checkpoint_passed",
            "artifact_gate_passed",
        ]
    return 0.18 * sum(1 for key in keys if key in row)


def run_loop(
    audit: Dict[str, Any],
    policy: str,
    rng: random.Random,
    max_steps: int,
    min_steps_before_claim: int,
    claim_tolerance: float,
) -> Dict[str, Any]:
    rows = [row for row in audit.get("rows", []) if _score(row) is not None or policy != "official_score_only"]
    lower = _lower_is_better(audit)
    visible_best = _best(rows, lower, valid_only=False)
    best_valid = _best(rows, lower, valid_only=True)
    order = list(rows)
    rng.shuffle(order)
    order = order[:max_steps]

    seen: List[Dict[str, Any]] = []
    trace = []
    total_cost = 0.0
    selected: Dict[str, Any] | None = None
    claimed = False
    stop_reason = "budget_exhausted"

    for step, row in enumerate(order, start=1):
        seen.append(row)
        total_cost += float(row.get("cost_units", 1.0)) + _gate_cost(policy, row)
        selected = _select(seen, policy, lower)
        can_claim = (
            selected is not None
            and visible_best is not None
            and step >= min_steps_before_claim
            and _within_claim_band(selected, visible_best, lower, claim_tolerance)
        )
        trace.append({
            "step": step,
            "observed": _variant(row),
            "observed_score": _score(row),
            "observed_full_valid": _full_valid(row),
            "selected": _variant(selected),
            "selected_score": _score(selected),
            "selected_full_valid": _full_valid(selected),
            "can_claim": can_claim,
        })
        if can_claim:
            claimed = True
            stop_reason = "claim_band_reached"
            break

    if selected is None:
        stop_reason = "no_selectable_candidate"
    return {
        "task": audit.get("task"),
        "policy": policy,
        "observed_candidates": [_variant(row) for row in seen],
        "steps": len(seen),
        "claimed": claimed or selected is not None,
        "stop_reason": stop_reason,
        "selected": _variant(selected),
        "official_score": _score(selected),
        "full_valid": _full_valid(selected),
        "invalid_final_claim": selected is not None and not _full_valid(selected),
        "valid_score_retention": _retention(selected, best_valid, lower),
        "cost_units": round(total_cost, 3),
        "best_visible": _variant(visible_best),
        "best_valid": _variant(best_valid),
        "trace": trace,
    }


def evaluate(
    audits: List[Dict[str, Any]],
    traces_per_task: int,
    seed: int,
    max_steps: int,
    min_steps_before_claim: int,
    claim_tolerance: float,
    include_all_loops: bool,
) -> Dict[str, Any]:
    loop_rows = []
    for task_idx, audit in enumerate(audits):
        for trace_idx in range(traces_per_task):
            base_seed = seed + task_idx * 100_003 + trace_idx
            for policy in POLICIES:
                rng = random.Random(base_seed)
                row = run_loop(audit, policy, rng, max_steps, min_steps_before_claim, claim_tolerance)
                row["trace_id"] = trace_idx
                loop_rows.append(row)

    aggregate = []
    for policy in POLICIES:
        rows = [row for row in loop_rows if row["policy"] == policy]
        invalids = sum(1 for row in rows if row["invalid_final_claim"])
        valid_claims = sum(1 for row in rows if row["full_valid"])
        aggregate.append({
            "policy": policy,
            "loops": len(rows),
            "invalid_final_claims": invalids,
            "invalid_final_claim_rate": invalids / len(rows) if rows else 0.0,
            "valid_final_claim_rate": valid_claims / len(rows) if rows else 0.0,
            "mean_valid_score_retention": mean(row["valid_score_retention"] for row in rows) if rows else 0.0,
            "mean_cost_units": mean(row["cost_units"] for row in rows) if rows else 0.0,
            "mean_steps": mean(row["steps"] for row in rows) if rows else 0.0,
        })
    result = {
        "task_count": len(audits),
        "tasks": [audit.get("task") for audit in audits],
        "traces_per_task": traces_per_task,
        "loops_per_policy": traces_per_task * len(audits),
        "max_steps": max_steps,
        "min_steps_before_claim": min_steps_before_claim,
        "claim_tolerance": claim_tolerance,
        "aggregate": aggregate,
        "sample_loops": loop_rows[:24],
        "interpretation": (
            "Closed-loop replay over executed/curated candidate audits. Candidate order is randomized "
            "with shared seeds across policies. The base agent policy is identical; policies differ only "
            "in what evidence is allowed before a final claim."
        ),
    }
    if include_all_loops:
        result["loops"] = loop_rows
    return result


def render_markdown(result: Dict[str, Any]) -> str:
    lines = ["# Closed-Loop Replay Evaluation\n"]
    lines.append(result["interpretation"])
    lines.append("")
    lines.append("This is stronger than one-shot winner selection because the agent observes candidates sequentially and may stop early with a final claim. It is still a replay over executed audits, not a fully autonomous LLM loop.")
    lines.append("")
    lines.append("## Scale\n")
    lines.append(f"- Tasks: {result['task_count']}")
    lines.append(f"- Traces per task: {result['traces_per_task']}")
    lines.append(f"- Loops per policy: {result['loops_per_policy']}")
    lines.append(f"- Max observed candidates per loop: {result['max_steps']}")
    lines.append(f"- Claim tolerance: {result['claim_tolerance']}")
    lines.append("")
    lines.append("## Aggregate Results\n")
    lines.append("| Policy | Invalid Final Claim Rate | Valid Final Claim Rate | Valid-Score Retention | Mean Cost | Mean Steps |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in result["aggregate"]:
        lines.append(
            f"| `{row['policy']}` | {row['invalid_final_claim_rate']:.3f} | "
            f"{row['valid_final_claim_rate']:.3f} | {row['mean_valid_score_retention']:.3f} | "
            f"{row['mean_cost_units']:.3f} | {row['mean_steps']:.3f} |"
        )
    lines.append("")
    lines.append("## Tasks\n")
    for task in result["tasks"]:
        lines.append(f"- `{task}`")
    lines.append("")
    score_only = next(row for row in result["aggregate"] if row["policy"] == "official_score_only")
    full = next(row for row in result["aggregate"] if row["policy"] == "scitriage_full")
    lines.append("## Interpretation\n")
    lines.append(
        f"Score-only final claims are invalid in {score_only['invalid_final_claim_rate']:.3f} of closed-loop replays. "
        f"Full SciTriage reduces this to {full['invalid_final_claim_rate']:.3f}, with "
        f"{full['mean_valid_score_retention']:.3f} mean valid-score retention."
    )
    lines.append(
        "The remaining limitation is candidate generation: these candidates come from executed audit suites, "
        "not from a fresh LLM agent writing new patches during this evaluation."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("--traces-per-task", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--min-steps-before-claim", type=int, default=2)
    parser.add_argument("--claim-tolerance", type=float, default=0.02)
    parser.add_argument("--include-all-loops", action="store_true")
    parser.add_argument("audits", nargs="*")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    audits = [_load(root / path) for path in (args.audits or DEFAULT_AUDITS)]
    result = evaluate(
        audits,
        traces_per_task=args.traces_per_task,
        seed=args.seed,
        max_steps=args.max_steps,
        min_steps_before_claim=args.min_steps_before_claim,
        claim_tolerance=args.claim_tolerance,
        include_all_loops=args.include_all_loops,
    )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_replay_eval.json").write_text(json.dumps(result, indent=2))
    (out / "CLOSED_LOOP_REPLAY_EVAL.md").write_text(render_markdown(result))
    print(out / "closed_loop_replay_eval.json")
    print(out / "CLOSED_LOOP_REPLAY_EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
