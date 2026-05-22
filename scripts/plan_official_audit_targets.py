from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


DONE_TASKS = {
    "vectorization",
    "cifar10",
    "imdb",
    "clrs",
}

TASK_NOTES = {
    "house-price": {
        "tier": "next",
        "failure_modes": ["answer_leakage", "schema_or_artifact", "seed_noise"],
        "why": "CSV regression task with cheap official MAE eval once Kaggle data is prepared.",
    },
    "spaceship-titanic": {
        "tier": "next",
        "failure_modes": ["answer_leakage", "schema_or_artifact", "seed_noise"],
        "why": "CSV classification task with cheap official accuracy eval once Kaggle data is prepared.",
    },
    "feedback": {
        "tier": "next",
        "failure_modes": ["answer_leakage", "schema_or_artifact", "seed_noise"],
        "why": "Text-regression CSV task; good for claim gating and evaluator leakage after data prep.",
    },
    "ogbn-arxiv": {
        "tier": "next",
        "failure_modes": ["artifact_lineage", "seed_noise", "resource_fit"],
        "why": "Non-Kaggle public graph task with official evaluator; heavier dependencies but strong externality.",
    },
    "babylm": {
        "tier": "next",
        "failure_modes": ["checkpoint_or_model_artifact", "seed_noise", "resource_fit"],
        "why": "Language-model task with model artifact validity and perplexity; heavier but on-story for AutoResearch.",
    },
    "amp-parkinsons-disease-progression-prediction": {
        "tier": "later",
        "failure_modes": ["answer_leakage", "resource_fit", "schema_or_artifact"],
        "why": "Interesting tabular/time-series task, but Kaggle data and custom format make it slower to stabilize.",
    },
    "fathomnet": {
        "tier": "later",
        "failure_modes": ["resource_fit", "schema_or_artifact"],
        "why": "Vision/data-heavy task; useful later but not first priority on 4x4090 time budget.",
    },
    "identify-contrails": {
        "tier": "later",
        "failure_modes": ["answer_leakage", "resource_fit", "schema_or_artifact"],
        "why": "Image-heavy task with large Kaggle data; good eventual stress test, expensive first target.",
    },
    "bibtex-generation": {
        "tier": "integration",
        "failure_modes": ["evidence_mismatch", "citation_hallucination"],
        "why": "No official eval in MLAgentBench; useful for integration/log audits, not score-bearing official audit.",
    },
    "literature-review-tool": {
        "tier": "integration",
        "failure_modes": ["evidence_mismatch", "citation_hallucination"],
        "why": "No official eval; relevant to AutoResearch narratives but needs a separate evidence benchmark.",
    },
    "llama-inference": {
        "tier": "integration",
        "failure_modes": ["semantic_shortcut", "resource_fit"],
        "why": "No official eval; useful for harness integration and runtime-contract checks.",
    },
}


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _data_presence(benchmarks_root: Path | None, task: str, read_only_files: List[str]) -> Dict[str, Any]:
    if benchmarks_root is None:
        return {"checked": False, "present_count": None, "expected_count": len(read_only_files), "missing": []}
    task_dir = benchmarks_root / task
    missing = []
    present = 0
    for pattern in read_only_files:
        clean = pattern.strip().lstrip("./")
        matches = list((task_dir / "env").glob(clean)) if clean else []
        if matches:
            present += 1
        else:
            missing.append(pattern)
    return {
        "checked": True,
        "present_count": present,
        "expected_count": len(read_only_files),
        "missing": missing,
    }


def _score(row: Dict[str, Any], note: Dict[str, Any], data: Dict[str, Any]) -> float:
    if row["task"].lower() in DONE_TASKS:
        return -100.0
    score = 0.0
    if row.get("has_eval"):
        score += 3.0
    if row.get("has_env_train"):
        score += 1.0
    if row.get("agent_visible_test_labels"):
        score += 2.0
    if not row.get("needs_kaggle"):
        score += 1.0
    if not row.get("needs_external_download"):
        score += 1.0
    if note.get("tier") == "next":
        score += 3.0
    elif note.get("tier") == "later":
        score += 1.0
    elif note.get("tier") == "integration":
        score -= 2.0
    if data.get("checked") and data.get("expected_count"):
        score += 2.0 * (data.get("present_count", 0) / data["expected_count"])
    if row.get("needs_kaggle"):
        score -= 0.8
    if "no_official_eval" in row.get("blockers", []):
        score -= 4.0
    return round(score, 3)


def plan(surface: Dict[str, Any], benchmarks_root: Path | None = None) -> Dict[str, Any]:
    rows = []
    for row in surface["rows"]:
        task = str(row["task"])
        note = TASK_NOTES.get(task, {
            "tier": "unknown",
            "failure_modes": [],
            "why": "Not yet manually prioritized.",
        })
        data = _data_presence(benchmarks_root, task, list(row.get("read_only_files", [])))
        item = {
            "task": task,
            "status": "done" if task.lower() in DONE_TASKS else note["tier"],
            "priority_score": _score(row, note, data),
            "has_eval": row.get("has_eval"),
            "needs_kaggle": row.get("needs_kaggle"),
            "needs_external_download": row.get("needs_external_download"),
            "agent_visible_test_labels": row.get("agent_visible_test_labels"),
            "data_presence": data,
            "target_failure_modes": note["failure_modes"],
            "why_this_target": note["why"],
            "blockers": row.get("blockers", []),
        }
        rows.append(item)
    rows.sort(key=lambda item: item["priority_score"], reverse=True)
    return {
        "done_official_audits": sorted(DONE_TASKS),
        "recommended_next": [row for row in rows if row["status"] == "next"][:8],
        "all_rows": rows,
    }


def render(result: Dict[str, Any]) -> str:
    lines = ["# Official Audit Target Queue\n"]
    lines.append(
        "This queue ranks public MLAgentBench tasks for conversion from stress-suite coverage into official-executed SciTriage audits."
    )
    lines.append("")
    lines.append("## Current Official Anchors\n")
    lines.append(", ".join(f"`{task}`" for task in result["done_official_audits"]))
    lines.append("")
    lines.append("## Recommended Next 5-8 Targets\n")
    lines.append("| Rank | Task | Priority | Main failure modes | Blockers | Why this target |")
    lines.append("|---:|---|---:|---|---|---|")
    for idx, row in enumerate(result["recommended_next"], 1):
        modes = ", ".join(f"`{mode}`" for mode in row["target_failure_modes"]) or "-"
        blockers = ", ".join(f"`{blocker}`" for blocker in row["blockers"]) or "none"
        lines.append(
            f"| {idx} | `{row['task']}` | {row['priority_score']:.3f} | {modes} | {blockers} | {row['why_this_target']} |"
        )
    lines.append("")
    lines.append("## Full Queue\n")
    lines.append("| Task | Status | Priority | Eval | Kaggle | External download | Data presence |")
    lines.append("|---|---|---:|---|---|---|---|")
    for row in result["all_rows"]:
        data = row["data_presence"]
        if data["checked"]:
            data_text = f"{data['present_count']} / {data['expected_count']}"
        else:
            data_text = "not checked"
        lines.append(
            f"| `{row['task']}` | `{row['status']}` | {row['priority_score']:.3f} | "
            f"{row['has_eval']} | {row['needs_kaggle']} | {row['needs_external_download']} | {data_text} |"
        )
    lines.append("")
    lines.append("## Why This Is Our Advantage\n")
    lines.append(
        "SciTriage does not compete with stronger base agents. It improves the research loop at the evidence boundary: "
        "after an agent proposes a patch and reports a score, SciTriage asks whether the result is valid, reproducible, "
        "within resource contract, and safe to claim. This makes it plugin-friendly for Karpathy-style loops, ARIS-like "
        "systems, Codex/Claude Code workflows, and MLAgentBench runners."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--surface", default="analysis/external_mlagentbench_task_surface_v1/task_surface_audit.json")
    parser.add_argument("--mlagentbench-root", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    benchmarks_root = None
    if args.mlagentbench_root:
        mlagentbench_root = Path(args.mlagentbench_root).expanduser().resolve()
        benchmarks_root = mlagentbench_root / "MLAgentBench" / "benchmarks"
    result = plan(_load(root / args.surface), benchmarks_root)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "official_audit_target_queue.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "OFFICIAL_AUDIT_TARGET_QUEUE.md").write_text(render(result), encoding="utf-8")
    print(out / "official_audit_target_queue.json")
    print(out / "OFFICIAL_AUDIT_TARGET_QUEUE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
