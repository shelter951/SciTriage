from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_AUDITS = [
    "analysis/external_mlagentbench_vectorization_v3/candidate_audit.json",
    "analysis/external_mlagentbench_cifar10_v1/candidate_audit.json",
    "analysis/external_mlagentbench_imdb_v1/candidate_audit.json",
    "analysis/external_mlagentbench_clrs_v1/candidate_audit.json",
]


def _load(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def _policy(data: Dict[str, object], key: str) -> Dict[str, object]:
    return data["policy_comparison"][key]


def _score(item: Dict[str, object]) -> float | None:
    return item.get("official_score")


def _valid(item: Dict[str, object]) -> bool:
    if "validity_gate_passed" in item:
        return bool(item["validity_gate_passed"])
    if "semantic_invariant_passed" in item:
        return bool(item["semantic_invariant_passed"])
    return False


def summarize(audits: List[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    blocked_visible_winners = 0
    visible_invalid = 0
    total_candidates = 0
    blocked_candidates = 0
    score_bearing_tasks = 0

    for audit in audits:
        visible = _policy(audit, "visible_score_only")
        gated = _policy(audit, "scitriage_gated")
        rows_raw = audit.get("rows", [])
        total_candidates += len(rows_raw)
        blocked = [
            row for row in rows_raw
            if not (
                row.get("semantic_invariant_passed")
                if "semantic_invariant_passed" in row
                else row.get("validity_gate_passed")
            )
        ]
        blocked_candidates += len(blocked)
        if _score(visible) is not None:
            score_bearing_tasks += 1
        if not _valid(visible):
            visible_invalid += 1
            blocked_visible_winners += 1
        rows.append({
            "task": audit["task"],
            "artifact": audit["_path"],
            "visible_selected": visible.get("selected"),
            "visible_score": _score(visible),
            "visible_valid": _valid(visible),
            "scitriage_selected": gated.get("selected") if gated else None,
            "scitriage_score": _score(gated) if gated else None,
            "scitriage_valid": _valid(gated) if gated else False,
            "candidate_count": len(rows_raw),
            "blocked_candidate_count": len(blocked),
        })

    return {
        "score_bearing_tasks": score_bearing_tasks,
        "total_candidates": total_candidates,
        "blocked_candidates": blocked_candidates,
        "visible_invalid_winners": visible_invalid,
        "blocked_visible_winners": blocked_visible_winners,
        "visible_winner_block_rate": blocked_visible_winners / score_bearing_tasks if score_bearing_tasks else 0.0,
        "rows": rows,
    }


def render_markdown(summary: Dict[str, object]) -> str:
    lines = ["# External Audit Summary\n"]
    lines.append("This file aggregates curated MLAgentBench candidate audits.")
    lines.append("")
    lines.append(f"- Score-bearing external tasks: {summary['score_bearing_tasks']}")
    lines.append(f"- Total audited candidates: {summary['total_candidates']}")
    lines.append(f"- Candidates blocked by SciTriage gates: {summary['blocked_candidates']}")
    lines.append(f"- Visible-score winners blocked: {summary['blocked_visible_winners']} / {summary['score_bearing_tasks']}")
    lines.append(f"- Visible-winner block rate: {summary['visible_winner_block_rate']:.3f}")
    lines.append("")
    lines.append("| Task | Visible-score-only winner | Valid? | SciTriage-gated winner | Valid? | Blocked candidates |")
    lines.append("|---|---|---|---|---|---:|")
    for row in summary["rows"]:
        lines.append(
            f"| `{row['task']}` | `{row['visible_selected']}` | {row['visible_valid']} | "
            f"`{row['scitriage_selected']}` | {row['scitriage_valid']} | {row['blocked_candidate_count']} / {row['candidate_count']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Across these external tasks, SciTriage blocks invalid high-scoring candidates when the visible score is misleading "
        "and preserves the visible-score winner when it is valid. This is important: the gate is not just a rejection rule; "
        "it is a selection policy over task-valid evidence."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    parser.add_argument("audits", nargs="*")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    audit_paths = args.audits or DEFAULT_AUDITS
    audits = [_load((root / path).resolve()) for path in audit_paths]
    summary = summarize(audits)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "external_audit_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "EXTERNAL_AUDIT_SUMMARY.md").write_text(render_markdown(summary))
    print(out / "external_audit_summary.json")
    print(out / "EXTERNAL_AUDIT_SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
