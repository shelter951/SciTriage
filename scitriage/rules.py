from __future__ import annotations

import re
from typing import List, Set, Tuple

from .schema import MetricObservation, ProbeRecommendation, ResearchTrace, TriageReport


RISK_ORDER = [
    "IMPLEMENTATION_BUG",
    "EVALUATOR_DRIFT",
    "OBJECTIVE_DRIFT",
    "NOISY_RESULT",
    "UNDERPOWERED_EXPERIMENT",
    "CONFOUNDED_PATCH",
    "UNSUPPORTED_CLAIM",
    "IDEA_INVALID",
]


def _subsystem_for_path(path: str) -> str:
    p = path.lower()
    if any(k in p for k in ["eval", "metric", "score", "benchmark"]):
        return "evaluation"
    if any(k in p for k in ["data", "dataset", "loader", "dataloader"]):
        return "data"
    if any(k in p for k in ["model", "net", "module", "layer", "attention", "mlp"]):
        return "model"
    if any(k in p for k in ["optim", "train", "schedule", "loss", "lr"]):
        return "training"
    if any(k in p for k in ["config", "yaml", "json", "toml"]):
        return "config"
    if any(k in p for k in ["test", "unit"]):
        return "test"
    return "other"


def _claim_requirements(claim: str) -> List[str]:
    c = claim.lower()
    req = []
    if any(k in c for k in ["caused", "because", "due to", "attributed to", "responsible for"]):
        req.append("mechanism ablation")
    if any(k in c for k in ["robust", "stable", "consistently", "across seeds", "multi-seed"]):
        req.append("multi-seed evidence")
    if any(k in c for k in ["generalizes", "generalization", "across datasets", "across tasks", "held-out"]):
        req.append("held-out task or dataset")
    if any(k in c for k in ["efficient", "faster", "speed", "cost", "cheaper"]):
        req.append("fair cost/runtime comparison")
    if any(k in c for k in ["novel", "first", "new state of the art", "sota"]):
        req.append("novelty search")
    if any(k in c for k in ["improves", "improved", "better", "outperforms", "beats", "lower", "higher"]):
        req.append("baseline comparison")
    if any(k in c for k in ["significant", "significantly", "statistically"]):
        req.append("statistical evidence")
    return req


def _metric_is_noisy(metric: MetricObservation) -> Tuple[bool, str]:
    delta = metric.delta()
    if delta is None:
        return False, ""
    stds = [v for v in [metric.baseline_std, metric.candidate_std] if v is not None]
    if metric.seeds <= 1:
        return True, f"{metric.name}: one-shot delta={delta:.4g}; no multi-seed evidence."
    if stds and abs(delta) < max(stds):
        return True, f"{metric.name}: delta={delta:.4g} is smaller than observed std={max(stds):.4g}."
    return False, ""



def _looks_like_objective_drift(trace: ResearchTrace, subsystems: Set[str]) -> bool:
    """Conservative objective-drift detector.

    We avoid generic lexical-overlap scoring because research proposals often use
    different words from the original question. Drift should be flagged only when
    the action is structurally aimed away from experiment progress.
    """
    text = (trace.proposal + " " + " ".join(trace.claims) + " " + " ".join(trace.changed_files)).lower()
    drift_terms = ["readme", "dashboard", "visualization", "paper presentation", "blog", "slides", "ui", "report styling"]
    has_experiment_signal = bool(trace.metrics) or any(s in subsystems for s in ["training", "model", "data", "evaluation"])
    if any(term in text for term in drift_terms) and not has_experiment_signal:
        return True
    return False


def diagnose(trace: ResearchTrace) -> TriageReport:
    risks: Set[str] = set()
    rationales: List[str] = []
    evidence_debt = []

    logs_lower = trace.logs.lower()
    if re.search(r"traceback|runtimeerror|assertionerror|\bnan\b|\binf\b|cuda out of memory|\boom\b|shape mismatch", logs_lower):
        risks.add("IMPLEMENTATION_BUG")
        rationales.append("Logs contain runtime, numerical, OOM, or shape-failure signals.")

    subsystems = {_subsystem_for_path(path) for path in trace.changed_files}
    if "evaluation" in subsystems or any(path in trace.protected_files for path in trace.changed_files):
        risks.add("EVALUATOR_DRIFT")
        rationales.append("Patch touches evaluation, metric, benchmark, or protected files.")

    major_subsystems = {s for s in subsystems if s not in {"other", "test", "config"}}
    if len(major_subsystems) >= 2:
        risks.add("CONFOUNDED_PATCH")
        rationales.append("Patch spans multiple semantic subsystems: " + ", ".join(sorted(major_subsystems)) + ".")

    max_steps = trace.experiment.get("max_steps") or trace.experiment.get("steps")
    budget_minutes = trace.experiment.get("budget_minutes")
    if (isinstance(max_steps, int) and max_steps < 1000) or (isinstance(budget_minutes, (int, float)) and budget_minutes <= 10):
        risks.add("UNDERPOWERED_EXPERIMENT")
        rationales.append("Experiment appears to be a short pilot run.")

    for metric in trace.metrics:
        noisy, reason = _metric_is_noisy(metric)
        if noisy:
            risks.add("NOISY_RESULT")
            rationales.append(reason)

    if _looks_like_objective_drift(trace, subsystems):
        risks.add("OBJECTIVE_DRIFT")
        rationales.append("Action appears structurally aimed away from the original experimental objective.")

    allowed_claims = []
    blocked_claims = []
    for claim in trace.claims:
        requirements = _claim_requirements(claim)
        missing = []
        if "multi-seed evidence" in requirements and all(m.seeds <= 1 for m in trace.metrics):
            missing.append("multi-seed evidence")
        if "mechanism ablation" in requirements and "CONFOUNDED_PATCH" in risks:
            missing.append("mechanism ablation")
        if "held-out task or dataset" in requirements:
            missing.append("held-out task or dataset")
        if "fair cost/runtime comparison" in requirements:
            missing.append("fair cost/runtime comparison")
        if "novelty search" in requirements:
            missing.append("novelty search")
        if "baseline comparison" in requirements and not trace.metrics:
            missing.append("baseline comparison")
        if "baseline comparison" in requirements and "EVALUATOR_DRIFT" in risks:
            missing.append("frozen-evaluator baseline comparison")
        if "baseline comparison" in requirements and "NOISY_RESULT" in risks and all(m.seeds <= 1 for m in trace.metrics):
            missing.append("multi-seed evidence")
        if "statistical evidence" in requirements and all(m.seeds <= 1 for m in trace.metrics):
            missing.append("statistical or multi-seed evidence")
        if missing:
            risks.add("UNSUPPORTED_CLAIM")
            blocked_claims.append(claim)
            evidence_debt.append({"claim": claim, "missing_evidence": sorted(set(missing)), "severity": "high" if len(missing) > 1 else "medium"})
        else:
            allowed_claims.append(claim)

    deltas = [m.delta() for m in trace.metrics if m.delta() is not None and m.seeds >= 2]
    if deltas and all(delta <= 0 for delta in deltas) and not ({"IMPLEMENTATION_BUG", "EVALUATOR_DRIFT"} & risks):
        risks.add("IDEA_INVALID")
        rationales.append("Repeated-run metric deltas are non-positive after basic checks.")

    ordered = [risk for risk in RISK_ORDER if risk in risks]
    status = "pass" if not ordered else "needs_probe"
    if "IMPLEMENTATION_BUG" in risks or "EVALUATOR_DRIFT" in risks:
        status = "blocked"

    return TriageReport(
        trace_id=trace.trace_id,
        status=status,
        risk_labels=ordered,
        evidence_debt=evidence_debt,
        recommended_probe=choose_probe(ordered, trace),
        allowed_claims=allowed_claims,
        blocked_claims=blocked_claims,
        rationales=rationales,
    )


def choose_probe(risks: List[str], trace: ResearchTrace) -> ProbeRecommendation | None:
    if not risks:
        return None
    if "IMPLEMENTATION_BUG" in risks:
        return ProbeRecommendation("implementation_invariant", "Rule out implementation or numerical failure before judging the idea.", 0.05, {"checks": ["smoke test", "finite loss", "shape invariants"]})
    if "EVALUATOR_DRIFT" in risks:
        return ProbeRecommendation("frozen_evaluator", "Evaluation/protected files changed; rerun with a frozen evaluator before trusting scores.", 0.2, {"protected_files": trace.protected_files})
    mechanism_claim = any(
        any(k in claim.lower() for k in ["caused", "because", "due to", "attributed to", "responsible for"])
        for claim in trace.claims
    )
    if "CONFOUNDED_PATCH" in risks and mechanism_claim:
        return ProbeRecommendation("minimal_ablation", "Patch spans multiple subsystems and the claim is causal; isolate the claimed mechanism before accepting it.", 1.5, {"changed_files": trace.changed_files})
    if "NOISY_RESULT" in risks:
        return ProbeRecommendation("multi_seed_rerun", "Observed result is one-shot or smaller than variance; verify across seeds.", 1.0, {"seeds": [1, 2, 3]})
    if "CONFOUNDED_PATCH" in risks:
        return ProbeRecommendation("minimal_ablation", "Patch spans multiple subsystems; isolate the claimed mechanism.", 1.5, {"changed_files": trace.changed_files})
    if "OBJECTIVE_DRIFT" in risks:
        return ProbeRecommendation("objective_anchor", "Compare current proposal against the original research contract.", 0.05, {})
    if "UNSUPPORTED_CLAIM" in risks:
        return ProbeRecommendation("claim_reconstruction", "Current claim exceeds evidence; reconstruct claims from raw logs and metrics.", 0.1, {})
    if "UNDERPOWERED_EXPERIMENT" in risks:
        return ProbeRecommendation("budget_extension", "Pilot run is too short to support the current conclusion.", 1.0, {"suggested_multiplier": 3})
    return ProbeRecommendation("human_review", "No automatic probe matched this risk profile.", 0.0, {})
