from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional

from ..schema import MetricObservation, ResearchTrace


VAL_RE = re.compile(r"^val_bpb:\s*([0-9.]+)", re.MULTILINE)
VRAM_RE = re.compile(r"^peak_vram_mb:\s*([0-9.]+)", re.MULTILINE)


def _git_changed_files(repo: Path, diff_ref: Optional[str]) -> List[str]:
    if diff_ref is None:
        cmd = ["git", "-C", str(repo), "diff", "--name-only", "HEAD"]
    else:
        cmd = ["git", "-C", str(repo), "diff", "--name-only", diff_ref]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _git_diff_summary(repo: Path, diff_ref: Optional[str]) -> str:
    if diff_ref is None:
        cmd = ["git", "-C", str(repo), "diff", "--stat", "HEAD"]
    else:
        cmd = ["git", "-C", str(repo), "diff", "--stat", diff_ref]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def trace_from_autoresearch(
    repo: str,
    run_log: str,
    baseline_val_bpb: float,
    claims: List[str],
    proposal: str = "",
    diff_ref: Optional[str] = None,
    baseline_std: Optional[float] = None,
    seeds: int = 1,
) -> ResearchTrace:
    repo_path = Path(repo).expanduser().resolve()
    log_path = Path(run_log).expanduser().resolve()
    logs = log_path.read_text(errors="replace")
    m = VAL_RE.search(logs)
    candidate = float(m.group(1)) if m else None
    changed_files = _git_changed_files(repo_path, diff_ref)
    diff_summary = _git_diff_summary(repo_path, diff_ref)
    if not proposal:
        proposal = "Autoresearch candidate patch in train.py."
    metric = MetricObservation(
        name="val_bpb",
        candidate=candidate,
        baseline=baseline_val_bpb,
        higher_is_better=False,
        baseline_std=baseline_std,
        seeds=seeds,
    )
    trace_id = f"autoresearch_{log_path.stem}"
    return ResearchTrace(
        trace_id=trace_id,
        question="Improve val_bpb in Karpathy autoresearch without modifying prepare.py or the evaluation protocol.",
        proposal=proposal,
        claims=claims,
        changed_files=changed_files,
        protected_files=["prepare.py", "evaluate_bpb"],
        diff_summary=diff_summary,
        logs=logs,
        metrics=[metric] if candidate is not None else [],
        experiment={"budget_minutes": 5, "harness": "karpathy_autoresearch"},
        history=[],
    )
