from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def audit_discoveries(seed1_sweep_json: str | Path, group_compare_jsons: Iterable[str | Path]) -> Dict[str, object]:
    seed1_data = json.loads(Path(seed1_sweep_json).read_text())
    seed1_rows = seed1_data.get("rows", [])
    seed1_by_variant = {row["variant"]: row for row in seed1_rows}

    rows: List[Dict[str, object]] = []
    for raw_path in group_compare_jsons:
        path = Path(raw_path)
        group = json.loads(path.read_text())
        variant = path.name.replace("_vs_baseline_group_compare.json", "")
        seed1 = seed1_by_variant.get(variant, {})
        seed1_delta = seed1.get("delta_vs_seed1_baseline")
        one_shot_accept = bool(seed1_delta is not None and seed1_delta > 0)
        group_accept = group.get("verdict") == "supports_improvement"
        rows.append({
            "variant": variant,
            "seed1_delta": seed1_delta,
            "baseline_mean": group.get("baseline", {}).get("mean"),
            "baseline_std": group.get("baseline", {}).get("std"),
            "candidate_mean": group.get("candidate", {}).get("mean"),
            "candidate_std": group.get("candidate", {}).get("std"),
            "delta": group.get("delta_improvement"),
            "margin95": group.get("z_margin"),
            "group_verdict": group.get("verdict"),
            "one_shot_accept": one_shot_accept,
            "group_accept": group_accept,
        })

    rows.sort(key=lambda row: (row["group_verdict"] != "supports_improvement", row["variant"]))
    one_shot_accepts = sum(1 for row in rows if row["one_shot_accept"])
    group_accepts = sum(1 for row in rows if row["group_accept"])
    false_discoveries = sum(1 for row in rows if row["one_shot_accept"] and not row["group_accept"])
    return {
        "num_variants": len(rows),
        "one_shot_accepts": one_shot_accepts,
        "group_accepts": group_accepts,
        "one_shot_false_discoveries_vs_group_gate": false_discoveries,
        "false_discovery_rate_among_one_shot_accepts": (
            false_discoveries / one_shot_accepts if one_shot_accepts else 0.0
        ),
        "rows": rows,
    }


def render_discovery_audit_markdown(audit: Dict[str, object]) -> str:
    lines = ["# Discovery Audit\n"]
    lines.append(f"- Variants audited: {audit['num_variants']}")
    lines.append(f"- One-shot accepts: {audit['one_shot_accepts']}")
    lines.append(f"- Group-gate accepts: {audit['group_accepts']}")
    lines.append(
        "- One-shot false discoveries vs group gate: "
        f"{audit['one_shot_false_discoveries_vs_group_gate']}"
    )
    lines.append(
        "- False discovery rate among one-shot accepts: "
        f"{audit['false_discovery_rate_among_one_shot_accepts']:.3f}"
    )
    lines.append("\n## Variant Table\n")
    lines.append("| Variant | Seed1 Delta | Group Delta | 95% Margin | Group Verdict |")
    lines.append("|---|---:|---:|---:|---|")
    for row in audit["rows"]:
        seed1_delta = row["seed1_delta"]
        group_delta = row["delta"]
        margin = row["margin95"]
        lines.append(
            f"| {row['variant']} | "
            f"{seed1_delta:.6f} | "
            f"{group_delta:.6f} | "
            f"{margin:.6f} | "
            f"`{row['group_verdict']}` |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "One-shot selection accepts every candidate whose first run improves the metric. "
        "The group gate requires the candidate-vs-baseline improvement to exceed the "
        "combined uncertainty margin. This turns noisy AutoResearch wins into explicit "
        "evidence debt instead of publishable claims."
    )
    return "\n".join(lines)
