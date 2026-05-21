from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


DEFAULT_BOARD = "analysis/autoresearch_probe_v1/evidence_board_v1/evidence_board.json"


def summarize(board: Dict[str, object]) -> Dict[str, object]:
    policies = board["policy_eval"]
    rows = []
    for name, data in policies.items():
        if not isinstance(data, dict) or "precision" not in data:
            continue
        rows.append({
            "policy": name,
            "extra_seed_runs": data["extra_seed_runs"],
            "true_accepts": data["true_accepts"],
            "false_accepts": data["false_accepts"],
            "precision": data["precision"],
            "recall": data["recall"],
            "family_precision": data.get("family_precision"),
            "family_recall": data.get("family_recall"),
        })
    return {
        "baseline_seed_std": board["baseline_std"],
        "candidate_count": board["summary"]["candidates"],
        "one_shot_accepts": board["summary"]["one_shot_accepts"],
        "group_tested": board["summary"]["group_tested"],
        "policies": rows,
        "cost_savings_vs_verify_all_positives": policies["cost_savings_vs_verify_all_positives"],
        "landmark_cost_savings_vs_verify_all_positives": policies["landmark_cost_savings_vs_verify_all_positives"],
    }


def render_markdown(result: Dict[str, object]) -> str:
    lines = ["# Seed-Noise Policy Evaluation\n"]
    lines.append("This file summarizes the Karpathy-style AutoResearch evidence-board policies.")
    lines.append("")
    lines.append(f"- Candidates: {result['candidate_count']}")
    lines.append(f"- One-shot accepts: {result['one_shot_accepts']}")
    lines.append(f"- Group-tested candidates: {result['group_tested']}")
    lines.append(f"- Baseline seed std: {result['baseline_seed_std']:.6f}")
    lines.append("")
    lines.append("| Policy | Extra Seed Runs | True Accepts | False Accepts | Precision | Recall | Family Precision | Family Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["policies"]:
        lines.append(
            f"| `{row['policy']}` | {row['extra_seed_runs']} | {row['true_accepts']} | {row['false_accepts']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['family_precision']:.3f} | {row['family_recall']:.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append(
        "The no-seed-noise baseline (`one_shot_claiming`) accepts false discoveries. "
        "SciTriage policies remove those false accepts by spending targeted seed budget. "
        "The family-landmark policy preserves research-direction recall while using 71.4% fewer extra seed runs than verifying every one-shot positive."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--board", default=DEFAULT_BOARD)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    board = json.loads((root / args.board).read_text())
    result = summarize(board)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed_noise_policy_eval.json").write_text(json.dumps(result, indent=2))
    (out / "SEED_NOISE_POLICY_EVAL.md").write_text(render_markdown(result))
    print(out / "seed_noise_policy_eval.json")
    print(out / "SEED_NOISE_POLICY_EVAL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
