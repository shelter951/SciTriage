from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .aggregate import parse_metric


def _extract_baseline_stats(data: Dict[str, object]) -> tuple[float | None, float | None]:
    if data.get("mean") is not None:
        return data.get("mean"), data.get("std")
    candidate = data.get("candidate")
    if isinstance(candidate, dict) and candidate.get("mean") is not None:
        return candidate.get("mean"), candidate.get("std")
    baseline = data.get("baseline")
    if isinstance(baseline, dict) and baseline.get("mean") is not None:
        return baseline.get("mean"), baseline.get("std")
    return None, None


def prioritize_probe(
    candidate_log: str | Path,
    baseline_summary_json: str | Path,
    metric: str,
    higher_is_better: bool = False,
) -> Dict[str, object]:
    baseline = json.loads(Path(baseline_summary_json).read_text())
    baseline_mean, baseline_std = _extract_baseline_stats(baseline)
    candidate_text = Path(candidate_log).read_text(errors="replace")
    candidate_value = parse_metric(candidate_text, metric)

    delta = None
    ratio = None
    if candidate_value is not None and baseline_mean is not None:
        raw = candidate_value - baseline_mean
        delta = raw if higher_is_better else -raw
    if delta is not None and baseline_std:
        ratio = delta / baseline_std

    if candidate_value is None:
        priority = "blocked"
        action = "Run failed or metric missing; perform implementation/resource diagnosis first."
    elif delta is None:
        priority = "blocked"
        action = "Baseline summary is missing mean; cannot prioritize probe."
    elif delta <= 0:
        priority = "low"
        action = "Do not spend multi-seed budget unless the candidate has another scientific reason."
    elif ratio is not None and ratio < 0.5:
        priority = "low"
        action = "Positive delta is far below observed seed noise; defer multi-seed rerun."
    elif ratio is not None and ratio < 1.5:
        priority = "medium"
        action = "Possible small improvement; run more seeds only if this claim is important."
    else:
        priority = "high"
        action = "Delta is large enough relative to observed noise to justify a seed-group probe."

    return {
        "metric": metric,
        "candidate_log": str(Path(candidate_log)),
        "candidate_value": candidate_value,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "higher_is_better": higher_is_better,
        "delta_improvement": delta,
        "delta_to_baseline_std": ratio,
        "priority": priority,
        "recommended_action": action,
    }


def render_probe_priority_markdown(result: Dict[str, object]) -> str:
    lines = [f"# Probe Priority: {result['priority']}\n"]
    lines.append(f"- Metric: `{result['metric']}`")
    lines.append(f"- Candidate value: {result['candidate_value']}")
    lines.append(f"- Baseline mean: {result['baseline_mean']}")
    lines.append(f"- Baseline std: {result['baseline_std']}")
    lines.append(f"- Delta improvement: {result['delta_improvement']}")
    lines.append(f"- Delta / baseline std: {result['delta_to_baseline_std']}")
    lines.append(f"- Recommended action: {result['recommended_action']}")
    return "\n".join(lines)
