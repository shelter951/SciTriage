from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .adapters.autoresearch import trace_from_autoresearch
from .adapters.filesystem import trace_from_filesystem
from .aggregate import aggregate_logs, compare_seed_groups
from .claim_gate import gate_claim
from .probe_plan import build_probe_plan
from .probe_priority import prioritize_probe
from .resource_fit import diagnose_resource_fit
from .rules import diagnose
from .schema import ResearchTrace


def assess_trace(data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess a structured AutoResearch trace and return a JSON-safe bundle."""
    trace = ResearchTrace.from_dict(data)
    report = diagnose(trace)
    plan = build_probe_plan(trace, report)
    return {
        "trace": data,
        "report": report.to_dict(),
        "probe_plan": plan,
    }


def assess_filesystem(root: str | Path, trace_id: str | None = None) -> Dict[str, Any]:
    """Assess an artifact directory produced by an external research harness."""
    trace = trace_from_filesystem(root, trace_id=trace_id)
    report = diagnose(trace)
    return {
        "trace": _trace_to_dict(trace),
        "report": report.to_dict(),
        "probe_plan": build_probe_plan(trace, report),
    }


def assess_autoresearch_run(
    repo: str | Path,
    run_log: str | Path,
    baseline_val_bpb: float,
    claims: Iterable[str],
    proposal: str = "",
    diff_ref: str | None = None,
    baseline_std: float | None = None,
    seeds: int = 1,
) -> Dict[str, Any]:
    """Assess a Karpathy autoresearch run log without taking over the harness."""
    trace = trace_from_autoresearch(
        repo=str(repo),
        run_log=str(run_log),
        baseline_val_bpb=baseline_val_bpb,
        baseline_std=baseline_std,
        seeds=seeds,
        claims=list(claims),
        proposal=proposal,
        diff_ref=diff_ref,
    )
    report = diagnose(trace)
    return {
        "trace": _trace_to_dict(trace),
        "report": report.to_dict(),
        "probe_plan": build_probe_plan(trace, report),
    }


def resource_fit(run_log: str | Path) -> Dict[str, Any]:
    """Classify resource-contract failures such as CUDA OOM."""
    return diagnose_resource_fit(run_log)


def seed_group_gate(
    baseline_logs: Iterable[str | Path],
    candidate_logs: Iterable[str | Path],
    metric: str,
    claim: str,
    higher_is_better: bool = False,
    z: float = 1.96,
    min_margin_ratio: float = 1.0,
) -> Dict[str, Any]:
    """Compare seed groups and gate a claim from the comparison."""
    comparison = compare_seed_groups(
        baseline_logs=baseline_logs,
        candidate_logs=candidate_logs,
        metric=metric,
        higher_is_better=higher_is_better,
        z=z,
    )
    gate = gate_claim_from_dict(comparison, claim, min_margin_ratio=min_margin_ratio)
    return {
        "comparison": comparison,
        "claim_gate": gate,
    }


def prioritize_candidate(
    candidate_log: str | Path,
    baseline_summary_json: str | Path,
    metric: str,
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Decide whether a one-shot candidate deserves multi-seed budget."""
    return prioritize_probe(candidate_log, baseline_summary_json, metric, higher_is_better=higher_is_better)


def aggregate_seed_logs(
    logs: Iterable[str | Path],
    metric: str,
    baseline: float | None = None,
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Compute seed mean/std for a harness that emits plain text logs."""
    return aggregate_logs(logs, metric=metric, baseline=baseline, higher_is_better=higher_is_better)


def write_bundle(bundle: Dict[str, Any], out_dir: str | Path) -> Dict[str, str]:
    """Write a plugin bundle using stable filenames external harnesses can read."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "bundle": out / "scitriage_bundle.json",
        "report": out / "triage_report.json",
        "probe_plan": out / "probe_plan.json",
    }
    paths["bundle"].write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    if "report" in bundle:
        paths["report"].write_text(json.dumps(bundle["report"], indent=2, ensure_ascii=False))
    if "probe_plan" in bundle:
        paths["probe_plan"].write_text(json.dumps(bundle["probe_plan"], indent=2, ensure_ascii=False))
    return {key: str(path) for key, path in paths.items() if path.exists()}


def gate_claim_from_dict(
    group_compare: Dict[str, Any],
    claim: str,
    min_margin_ratio: float = 1.0,
) -> Dict[str, Any]:
    """Gate a claim from an in-memory seed-group comparison."""
    verdict = group_compare.get("verdict")
    delta = group_compare.get("delta_improvement")
    margin = group_compare.get("z_margin")
    allowed = verdict == "supports_improvement"
    margin_ratio = None
    if delta is not None and margin:
        margin_ratio = delta / margin
        allowed = allowed and margin_ratio >= min_margin_ratio
    return {
        "claim": claim,
        "status": "allowed" if allowed else "blocked",
        "reason": (
            "Seed-group improvement clears the uncertainty margin."
            if allowed
            else "Claim is not supported by the current seed-group evidence."
        ),
        "group_verdict": verdict,
        "delta_improvement": delta,
        "z_margin": margin,
        "margin_ratio": margin_ratio,
        "min_margin_ratio": min_margin_ratio,
    }


def _trace_to_dict(trace: ResearchTrace) -> Dict[str, Any]:
    return {
        "trace_id": trace.trace_id,
        "question": trace.question,
        "proposal": trace.proposal,
        "claims": trace.claims,
        "changed_files": trace.changed_files,
        "protected_files": trace.protected_files,
        "diff_summary": trace.diff_summary,
        "logs": trace.logs,
        "metrics": [m.__dict__ for m in trace.metrics],
        "experiment": trace.experiment,
        "history": trace.history,
    }
