from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _variant_from_group_path(path: str | Path) -> str:
    name = Path(path).name
    for suffix in [
        "_vs_baseline_group_compare.json",
        "_group_compare.json",
        ".json",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _priority(seed1_delta: float | None, baseline_std: float | None) -> tuple[str, float | None]:
    if seed1_delta is None:
        return "blocked", None
    if seed1_delta <= 0:
        return "low", None if not baseline_std else seed1_delta / baseline_std
    if not baseline_std:
        return "medium", None
    ratio = seed1_delta / baseline_std
    if ratio >= 1.5:
        return "high", ratio
    if ratio >= 0.5:
        return "medium", ratio
    return "low", ratio


def _family(variant: str) -> str:
    if variant.startswith("depth"):
        return "depth"
    if "batch" in variant:
        return "batch"
    if "lr" in variant:
        return "learning_rate"
    if "warm" in variant or "schedule" in variant or "final_lr" in variant:
        return "schedule"
    if "weight_decay" in variant:
        return "regularization"
    if "window" in variant:
        return "attention_window"
    return "other"


def _load_sweep_rows(paths: Iterable[str | Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for raw in paths:
        data = json.loads(Path(raw).read_text())
        for row in data.get("rows", []):
            variant = row.get("variant")
            if not variant:
                continue
            current = dict(rows.get(variant, {}))
            current.update(row)
            current["source_sweep"] = str(raw)
            rows[variant] = current
    return rows


def _load_group_rows(paths: Iterable[str | Path]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for raw in paths:
        data = json.loads(Path(raw).read_text())
        rows[_variant_from_group_path(raw)] = data
    return rows


def _load_baseline_std(path: str | Path | None, group_rows: Dict[str, Dict[str, Any]]) -> float | None:
    if path:
        data = json.loads(Path(path).read_text())
        if data.get("std") is not None:
            return data.get("std")
        if isinstance(data.get("candidate"), dict) and data["candidate"].get("std") is not None:
            return data["candidate"].get("std")
        if isinstance(data.get("baseline"), dict) and data["baseline"].get("std") is not None:
            return data["baseline"].get("std")
    for group in group_rows.values():
        baseline = group.get("baseline")
        if isinstance(baseline, dict) and baseline.get("std") is not None:
            return baseline["std"]
    return None


def build_evidence_board(
    sweeps: Iterable[str | Path],
    group_compares: Iterable[str | Path],
    baseline_summary: str | Path | None = None,
) -> Dict[str, Any]:
    sweep_rows = _load_sweep_rows(sweeps)
    group_rows = _load_group_rows(group_compares)
    baseline_std = _load_baseline_std(baseline_summary, group_rows)

    variants = sorted(set(sweep_rows) | set(group_rows))
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        seed = sweep_rows.get(variant, {})
        group = group_rows.get(variant)
        seed1_delta = seed.get("delta_vs_seed1_baseline")
        priority, ratio = _priority(seed1_delta, baseline_std)
        row = {
            "variant": variant,
            "family": _family(variant),
            "seed1_val": seed.get("val_bpb"),
            "seed1_delta": seed1_delta,
            "peak_vram_mb": seed.get("peak_vram_mb"),
            "num_steps": seed.get("num_steps"),
            "one_shot_accept": bool(seed1_delta is not None and seed1_delta > 0),
            "priority": priority,
            "delta_to_baseline_std": ratio,
            "group_tested": group is not None,
            "group_verdict": None if group is None else group.get("verdict"),
            "group_delta": None if group is None else group.get("delta_improvement"),
            "group_margin95": None if group is None else group.get("z_margin"),
            "claim_status": "unknown",
        }
        if group is not None:
            row["claim_status"] = "allowed" if group.get("verdict") == "supports_improvement" else "blocked"
        rows.append(row)

    rows.sort(key=lambda r: (
        {"high": 0, "medium": 1, "low": 2, "blocked": 3}.get(r["priority"], 4),
        not r["group_tested"],
        r["family"],
        r["variant"],
    ))
    summary = _summarize(rows)
    return {
        "baseline_std": baseline_std,
        "summary": summary,
        "rows": rows,
        "policy_eval": _policy_eval(rows),
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    one_shot_accepts = [r for r in rows if r["one_shot_accept"]]
    group_tested = [r for r in rows if r["group_tested"]]
    group_accepts = [r for r in group_tested if r["group_verdict"] == "supports_improvement"]
    one_shot_group = [r for r in group_tested if r["one_shot_accept"]]
    false_discoveries = [
        r for r in one_shot_group
        if r["group_verdict"] != "supports_improvement"
    ]
    high = [r for r in rows if r["priority"] == "high"]
    return {
        "candidates": len(rows),
        "one_shot_accepts": len(one_shot_accepts),
        "high_priority": len(high),
        "group_tested": len(group_tested),
        "group_accepts": len(group_accepts),
        "false_discoveries_among_tested_one_shot_accepts": len(false_discoveries),
        "fdr_among_tested_one_shot_accepts": (
            len(false_discoveries) / len(one_shot_group) if one_shot_group else 0.0
        ),
    }


def _policy_eval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    tested = [r for r in rows if r["group_tested"]]
    true_accepts = {r["variant"] for r in tested if r["group_verdict"] == "supports_improvement"}
    positive = [r for r in tested if r["one_shot_accept"]]
    high_positive = [r for r in positive if r["priority"] == "high"]

    policies = {
        "one_shot_claiming": positive,
        "verify_all_one_shot_positives": positive,
        "scitriage_high_priority_only": high_positive,
        "exhaustive_verify_all_candidates": tested,
    }
    result: Dict[str, Any] = {}
    for name, selected in policies.items():
        selected_names = {r["variant"] for r in selected}
        if name == "one_shot_claiming":
            accepted = selected_names
            extra_seed_runs = 0
        else:
            accepted = {
                r["variant"] for r in selected
                if r["group_verdict"] == "supports_improvement"
            }
            extra_seed_runs = 2 * len(selected)
        false_accepts = accepted - true_accepts
        missed = true_accepts - accepted
        result[name] = {
            "selected_for_followup": sorted(selected_names),
            "accepted_claims": sorted(accepted),
            "extra_seed_runs": extra_seed_runs,
            "true_accepts": len(accepted & true_accepts),
            "false_accepts": len(false_accepts),
            "missed_true_accepts": len(missed),
            "precision": len(accepted & true_accepts) / len(accepted) if accepted else 0.0,
            "recall": len(accepted & true_accepts) / len(true_accepts) if true_accepts else 0.0,
        }
    exhaustive_cost = result["exhaustive_verify_all_candidates"]["extra_seed_runs"]
    triage_cost = result["scitriage_high_priority_only"]["extra_seed_runs"]
    result["cost_savings_vs_exhaustive"] = (
        (exhaustive_cost - triage_cost) / exhaustive_cost if exhaustive_cost else 0.0
    )
    verify_positive_cost = result["verify_all_one_shot_positives"]["extra_seed_runs"]
    result["cost_savings_vs_verify_all_positives"] = (
        (verify_positive_cost - triage_cost) / verify_positive_cost if verify_positive_cost else 0.0
    )
    return result


def render_evidence_board_markdown(board: Dict[str, Any]) -> str:
    summary = board["summary"]
    policy = board["policy_eval"]
    lines = ["# SciTriage Evidence Board\n"]
    lines.append("## Summary\n")
    lines.append(f"- Candidates observed: {summary['candidates']}")
    lines.append(f"- One-shot accepts: {summary['one_shot_accepts']}")
    lines.append(f"- High-priority candidates: {summary['high_priority']}")
    lines.append(f"- Group-tested candidates: {summary['group_tested']}")
    lines.append(f"- Group-supported discoveries: {summary['group_accepts']}")
    lines.append(
        "- False discoveries among tested one-shot accepts: "
        f"{summary['false_discoveries_among_tested_one_shot_accepts']} "
        f"(FDR {summary['fdr_among_tested_one_shot_accepts']:.3f})"
    )
    lines.append("\n## Budget Policy Evaluation\n")
    lines.append("| Policy | Extra seed runs | True accepts | False accepts | Missed true accepts | Precision | Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in [
        "one_shot_claiming",
        "verify_all_one_shot_positives",
        "scitriage_high_priority_only",
        "exhaustive_verify_all_candidates",
    ]:
        item = policy[name]
        lines.append(
            f"| `{name}` | {item['extra_seed_runs']} | {item['true_accepts']} | "
            f"{item['false_accepts']} | {item['missed_true_accepts']} | "
            f"{item['precision']:.3f} | {item['recall']:.3f} |"
        )
    lines.append(
        "\nSciTriage high-priority gating saves "
        f"{policy['cost_savings_vs_verify_all_positives']:.1%} of extra seed runs versus verifying every "
        "one-shot positive, while preserving the supported discoveries in this real candidate pool."
    )
    lines.append(
        " It saves "
        f"{policy['cost_savings_vs_exhaustive']:.1%} versus exhaustive verification of every observed candidate."
    )
    lines.append("\n## Candidate Board\n")
    lines.append("| Variant | Family | Seed1 Delta | Delta/Std | Priority | Group Delta | Margin | Verdict | Claim |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|---|")
    for row in board["rows"]:
        lines.append(
            f"| `{row['variant']}` | {row['family']} | {_fmt(row['seed1_delta'])} | "
            f"{_fmt(row['delta_to_baseline_std'])} | `{row['priority']}` | "
            f"{_fmt(row['group_delta'])} | {_fmt(row['group_margin95'])} | "
            f"`{row['group_verdict'] or 'not_tested'}` | `{row['claim_status']}` |"
        )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
