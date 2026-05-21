from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def audit_task(task_dir: Path) -> Dict[str, object]:
    env_train = task_dir / "env" / "train.py"
    eval_py = task_dir / "scripts" / "eval.py"
    prepare_py = task_dir / "scripts" / "prepare.py"
    read_only = task_dir / "scripts" / "read_only_files.txt"
    env_text = _read(env_train)
    eval_text = _read(eval_py)
    prepare_text = _read(prepare_py)
    ro_text = _read(read_only)
    combined_prepare = "\n".join([prepare_text, _read(task_dir / "env" / "data_description.txt")])

    agent_visible_test_labels = (
        "train=False" in env_text
        and ("test_dataset" in env_text or "test_dataloader" in env_text)
        and ("labels" in env_text or "label" in env_text or " y" in env_text)
    )
    eval_uses_labels = any(token in eval_text for token in [
        "train=False",
        "answer",
        "ground_truth",
        "label",
        "y_true",
    ])
    needs_kaggle = "kaggle" in combined_prepare.lower()
    needs_external_download = any(token in combined_prepare.lower() for token in [
        "download",
        "wget",
        "requests.",
        "load_dataset",
        "torchvision",
        "ogb",
    ])
    read_only_files = [line.strip() for line in ro_text.splitlines() if line.strip()]

    blockers = []
    if not eval_py.exists():
        blockers.append("no_official_eval")
    if needs_kaggle:
        blockers.append("kaggle_data")
    elif needs_external_download:
        blockers.append("external_download")
    if agent_visible_test_labels:
        blockers.append("agent_visible_test_label_surface")

    return {
        "task": task_dir.name,
        "has_eval": eval_py.exists(),
        "has_prepare": prepare_py.exists(),
        "has_env_train": env_train.exists(),
        "read_only_files": read_only_files,
        "read_only_count": len(read_only_files),
        "eval_uses_labels": eval_uses_labels,
        "agent_visible_test_labels": agent_visible_test_labels,
        "needs_kaggle": needs_kaggle,
        "needs_external_download": needs_external_download,
        "blockers": blockers,
        "recommended_use": _recommended_use(blockers, eval_py.exists()),
    }


def _recommended_use(blockers: List[str], has_eval: bool) -> str:
    if "agent_visible_test_label_surface" in blockers:
        return "validity-gate audit"
    if has_eval and not blockers:
        return "candidate audit"
    if has_eval:
        return "defer until data prepared"
    return "integration/log audit only"


def render_markdown(rows: List[Dict[str, object]]) -> str:
    lines = ["# MLAgentBench Task Surface Audit\n"]
    lines.append(
        "This scan reads the external MLAgentBench task folders and records which tasks "
        "are immediately useful for SciTriage candidate audits."
    )
    lines.append("\n| Task | Eval | Data blocker | Agent-visible test labels | Recommended use |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        data_blocker = "kaggle" if row["needs_kaggle"] else ("download" if row["needs_external_download"] else "none")
        lines.append(
            f"| `{row['task']}` | {row['has_eval']} | {data_blocker} | "
            f"{row['agent_visible_test_labels']} | {row['recommended_use']} |"
        )
    lines.append("\n## Interpretation\n")
    lines.append(
        "Vectorization is the cleanest semantic-validity task: the official runtime objective can be "
        "gamed by skipping computation, and SciTriage can test functional equivalence."
    )
    lines.append(
        "CIFAR-10 exposes a different failure mode: the starter environment contains test-label access, "
        "so a candidate can win the official score by reading labels instead of learning."
    )
    lines.append(
        "Most Kaggle-style tasks are better next targets after data credentials and cache preparation, "
        "because they provide stronger hidden-score realism but require heavier setup."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlagentbench-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.mlagentbench_root).expanduser().resolve()
    benchmarks = root / "MLAgentBench" / "benchmarks"
    rows = [
        audit_task(path)
        for path in sorted(benchmarks.iterdir())
        if path.is_dir()
    ]
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "task_surface_audit.json").write_text(json.dumps({"rows": rows}, indent=2))
    (out / "TASK_SURFACE_AUDIT.md").write_text(render_markdown(rows))
    print(out / "task_surface_audit.json")
    print(out / "TASK_SURFACE_AUDIT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
