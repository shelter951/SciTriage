from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .rules import diagnose
from .schema import MetricObservation, ResearchTrace


@dataclass(frozen=True)
class ToyScenario:
    name: str
    description: str
    kind: str
    changed_files: List[str]
    claim: str
    baseline_values: Dict[int, float]
    candidate_values: Dict[int, float]
    official_candidate_values: Dict[int, float] | None = None
    logs: str = "completed"
    expected_truth: str = "reject"  # accept | reject | blocked


SCENARIOS = [
    ToyScenario(
        name="noisy_false_positive",
        description="One seed looks better, but multi-seed mean is not better than baseline.",
        kind="noisy",
        changed_files=["train.py"],
        claim="The patch significantly improves validation loss.",
        baseline_values={1: 1.000, 2: 1.000, 3: 1.000},
        candidate_values={1: 0.990, 2: 1.012, 3: 1.008},
        expected_truth="reject",
    ),
    ToyScenario(
        name="true_positive",
        description="Candidate improves across seeds.",
        kind="true_positive",
        changed_files=["train.py"],
        claim="The patch improves validation loss under three seeds.",
        baseline_values={1: 1.000, 2: 1.002, 3: 0.998},
        candidate_values={1: 0.960, 2: 0.965, 3: 0.958},
        expected_truth="accept",
    ),
    ToyScenario(
        name="evaluator_drift_false_positive",
        description="Changed evaluator creates a fake improvement; official evaluator removes it.",
        kind="evaluator_drift",
        changed_files=["train.py", "eval.py"],
        claim="The patch significantly improves validation loss.",
        baseline_values={1: 1.000, 2: 1.000, 3: 1.000},
        candidate_values={1: 0.700, 2: 0.705, 3: 0.710},
        official_candidate_values={1: 1.010, 2: 1.000, 3: 1.005},
        expected_truth="reject",
    ),
    ToyScenario(
        name="implementation_bug",
        description="Candidate crashes before producing trustworthy evidence.",
        kind="implementation_bug",
        changed_files=["train.py", "loss.py"],
        claim="The normalized loss improves stability.",
        baseline_values={1: 1.000, 2: 1.000, 3: 1.000},
        candidate_values={1: 999.0, 2: 999.0, 3: 999.0},
        logs="Traceback RuntimeError: shape mismatch in loss normalization",
        expected_truth="blocked",
    ),
    ToyScenario(
        name="confounded_mechanism",
        description="Full patch improves, but causal mechanism claim is unsupported without ablation.",
        kind="confounded",
        changed_files=["model.py", "train.py", "data_loader.py"],
        claim="The improvement is caused by dropout.",
        baseline_values={1: 1.000, 2: 1.002, 3: 0.998},
        candidate_values={1: 0.940, 2: 0.945, 3: 0.942},
        expected_truth="reject",
    ),
]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def make_trace(scenario: ToyScenario, seed: int = 1, official_evaluator: bool = False, seeds_observed: int = 1) -> ResearchTrace:
    baseline = scenario.baseline_values[seed]
    candidate_source = scenario.candidate_values
    if official_evaluator and scenario.official_candidate_values is not None:
        candidate_source = scenario.official_candidate_values
    candidate = candidate_source[seed]
    baseline_std = _std(list(scenario.baseline_values.values())) or None
    return ResearchTrace(
        trace_id=f"toy_{scenario.name}_seed{seed}",
        question="Improve validation loss for a tiny research task while preserving the official evaluator.",
        proposal=scenario.description,
        claims=[scenario.claim],
        changed_files=scenario.changed_files,
        protected_files=["eval.py", "metrics.py", "official_evaluator.py"],
        diff_summary="; ".join(scenario.changed_files),
        logs=scenario.logs + f"\nval_loss={candidate:.6f} baseline={baseline:.6f}",
        metrics=[MetricObservation(
            name="val_loss",
            candidate=candidate,
            baseline=baseline,
            higher_is_better=False,
            baseline_std=baseline_std,
            seeds=seeds_observed,
        )] if "Traceback" not in scenario.logs else [],
        experiment={"budget_minutes": 30, "max_steps": 3000, "toy_scenario": scenario.name},
        history=[],
    )


def multiseed_verdict(scenario: ToyScenario, official_evaluator: bool = False) -> Tuple[str, Dict[str, float]]:
    candidate_source = scenario.candidate_values
    if official_evaluator and scenario.official_candidate_values is not None:
        candidate_source = scenario.official_candidate_values
    baseline_mean = _mean(list(scenario.baseline_values.values()))
    candidate_mean = _mean(list(candidate_source.values()))
    delta = baseline_mean - candidate_mean  # positive is improvement for loss
    verdict = "accept" if delta > 0.01 else "reject"
    return verdict, {"baseline_mean": baseline_mean, "candidate_mean": candidate_mean, "delta_improvement": delta}


def make_post_probe_trace(scenario: ToyScenario, official_evaluator: bool = False) -> ResearchTrace:
    candidate_source = scenario.candidate_values
    if official_evaluator and scenario.official_candidate_values is not None:
        candidate_source = scenario.official_candidate_values
    baseline_values = list(scenario.baseline_values.values())
    candidate_values = list(candidate_source.values())
    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    baseline_std = _std(baseline_values) or None
    candidate_std = _std(candidate_values) or None
    return ResearchTrace(
        trace_id=f"toy_{scenario.name}_post_probe",
        question="Improve validation loss for a tiny research task while preserving the official evaluator.",
        proposal=scenario.description,
        claims=[scenario.claim],
        changed_files=scenario.changed_files,
        protected_files=["eval.py", "metrics.py", "official_evaluator.py"],
        diff_summary="; ".join(scenario.changed_files),
        logs=scenario.logs + f"\nval_loss={candidate_mean:.6f} baseline={baseline_mean:.6f}",
        metrics=[MetricObservation(
            name="val_loss",
            candidate=candidate_mean,
            baseline=baseline_mean,
            higher_is_better=False,
            baseline_std=baseline_std,
            candidate_std=candidate_std,
            seeds=3,
        )] if "Traceback" not in scenario.logs else [],
        experiment={"budget_minutes": 30, "max_steps": 3000, "toy_scenario": scenario.name, "post_probe": True},
        history=[],
    )


def evaluate_no_triage(scenario: ToyScenario) -> Dict[str, object]:
    if scenario.kind == "implementation_bug":
        return {"decision": "blocked", "cost": 1.0, "final_truth": scenario.expected_truth, "false_discovery": False, "unsupported_claim_allowed": False}
    trace = make_trace(scenario, seed=1)
    metric = trace.metrics[0]
    decision = "accept" if metric.delta() is not None and metric.delta() > 0 else "reject"
    false_discovery = decision == "accept" and scenario.expected_truth != "accept"
    unsupported = decision == "accept" and scenario.kind in {"noisy", "evaluator_drift", "confounded"}
    return {"decision": decision, "cost": 1.0, "final_truth": scenario.expected_truth, "false_discovery": false_discovery, "unsupported_claim_allowed": unsupported}


def evaluate_fixed_multiseed(scenario: ToyScenario) -> Dict[str, object]:
    if scenario.kind == "implementation_bug":
        return {"decision": "blocked", "cost": 1.0, "final_truth": scenario.expected_truth, "false_discovery": False, "unsupported_claim_allowed": False}
    # Fixed policy always spends three seed runs but does not know to use frozen evaluator or ablation.
    verdict, stats = multiseed_verdict(scenario, official_evaluator=False)
    false_discovery = verdict == "accept" and scenario.expected_truth != "accept"
    unsupported = verdict == "accept" and scenario.kind in {"evaluator_drift", "confounded"}
    return {"decision": verdict, "cost": 3.0, "final_truth": scenario.expected_truth, "false_discovery": false_discovery, "unsupported_claim_allowed": unsupported, "stats": stats}


def evaluate_scitriage(scenario: ToyScenario) -> Dict[str, object]:
    trace = make_trace(scenario, seed=1)
    report = diagnose(trace)
    probe = report.recommended_probe.probe_type if report.recommended_probe else None
    cost = 1.0
    decision = "reject"
    stats = {}
    final_report = report

    if report.status == "blocked" and probe == "implementation_invariant":
        decision = "blocked"
        cost += 0.05
    elif report.status == "blocked" and probe == "frozen_evaluator":
        verdict, stats = multiseed_verdict(scenario, official_evaluator=True)
        final_trace = make_post_probe_trace(scenario, official_evaluator=True)
        final_report = diagnose(final_trace)
        decision = "accept" if verdict == "accept" and not final_report.blocked_claims else "reject"
        cost += 1.5
    elif probe == "multi_seed_rerun":
        verdict, stats = multiseed_verdict(scenario, official_evaluator=False)
        final_trace = make_post_probe_trace(scenario, official_evaluator=False)
        final_report = diagnose(final_trace)
        decision = "accept" if verdict == "accept" and not final_report.blocked_claims else "reject"
        cost += 2.0
    elif probe == "minimal_ablation":
        # A mechanism claim is rejected until ablation supports it.
        decision = "reject"
        cost += 1.5
        stats = {"ablation_needed": True}
    elif probe in {"claim_reconstruction", "objective_anchor", "budget_extension"}:
        decision = "reject"
        cost += 0.2
    elif report.status == "pass":
        decision = "accept" if not report.blocked_claims else "reject"

    false_discovery = decision == "accept" and scenario.expected_truth != "accept"
    unsupported = decision == "accept" and bool(final_report.blocked_claims)
    return {
        "decision": decision,
        "cost": cost,
        "probe": probe,
        "risk_labels": report.risk_labels,
        "post_probe_risk_labels": final_report.risk_labels,
        "final_truth": scenario.expected_truth,
        "false_discovery": false_discovery,
        "unsupported_claim_allowed": unsupported,
        "stats": stats,
    }


def run_toy_policy_eval() -> Dict[str, object]:
    rows = []
    for scenario in SCENARIOS:
        rows.append({
            "scenario": scenario.name,
            "truth": scenario.expected_truth,
            "no_triage": evaluate_no_triage(scenario),
            "fixed_multiseed": evaluate_fixed_multiseed(scenario),
            "scitriage": evaluate_scitriage(scenario),
        })

    def summarize(policy: str) -> Dict[str, object]:
        items = [row[policy] for row in rows]
        false_discoveries = sum(1 for item in items if item["false_discovery"])
        unsupported = sum(1 for item in items if item["unsupported_claim_allowed"])
        cost = sum(float(item["cost"]) for item in items)
        correct = 0
        for row in rows:
            item = row[policy]
            truth = row["truth"]
            correct += int(item["decision"] == truth or (truth == "reject" and item["decision"] == "reject"))
        return {
            "false_discoveries": false_discoveries,
            "unsupported_claims_allowed": unsupported,
            "total_cost_units": cost,
            "decision_accuracy": correct / len(rows),
        }

    return {
        "num_scenarios": len(rows),
        "summary": {
            "no_triage": summarize("no_triage"),
            "fixed_multiseed": summarize("fixed_multiseed"),
            "scitriage": summarize("scitriage"),
        },
        "rows": rows,
    }


def render_toy_policy_markdown(result: Dict[str, object]) -> str:
    lines = ["# Toy Executable Policy Evaluation\n"]
    lines.append(f"Scenarios: {result['num_scenarios']}\n")
    lines.append("## Summary\n")
    lines.append("| Policy | Decision Acc. | False Discoveries | Unsupported Claims Allowed | Cost Units |")
    lines.append("|---|---:|---:|---:|---:|")
    for policy, summary in result["summary"].items():
        lines.append(
            f"| {policy} | {summary['decision_accuracy']:.2f} | {summary['false_discoveries']} | "
            f"{summary['unsupported_claims_allowed']} | {summary['total_cost_units']:.2f} |"
        )
    lines.append("\n## Scenario Details\n")
    for row in result["rows"]:
        lines.append(f"### {row['scenario']}")
        lines.append(f"- Truth: `{row['truth']}`")
        for policy in ["no_triage", "fixed_multiseed", "scitriage"]:
            item = row[policy]
            probe = f", probe={item.get('probe')}" if policy == "scitriage" else ""
            lines.append(
                f"- {policy}: decision=`{item['decision']}`, cost={item['cost']}, "
                f"false_discovery={item['false_discovery']}, unsupported={item['unsupported_claim_allowed']}{probe}"
            )
        lines.append("")
    return "\n".join(lines)


def generate_sweep_scenarios(n_per_type: int = 20) -> List[ToyScenario]:
    scenarios: List[ToyScenario] = []
    idx = 0
    for i in range(n_per_type):
        idx += 1
        # One seed is lucky, mean is neutral or worse.
        scenarios.append(ToyScenario(
            name=f"sweep_noisy_{i:03d}",
            description="Lucky one-shot improvement but no verified mean gain.",
            kind="noisy",
            changed_files=["train.py"],
            claim="The patch significantly improves validation loss.",
            baseline_values={1: 1.000, 2: 1.000, 3: 1.000},
            candidate_values={1: 0.985 - 0.001 * (i % 3), 2: 1.006 + 0.001 * (i % 5), 3: 1.010 + 0.001 * (i % 4)},
            expected_truth="reject",
        ))
        # Stable improvement.
        scenarios.append(ToyScenario(
            name=f"sweep_true_{i:03d}",
            description="Stable candidate improvement across seeds.",
            kind="true_positive",
            changed_files=["train.py"],
            claim="The patch improves validation loss under three seeds.",
            baseline_values={1: 1.000, 2: 1.002, 3: 0.998},
            candidate_values={1: 0.955 - 0.001 * (i % 4), 2: 0.962 - 0.001 * (i % 3), 3: 0.959 - 0.001 * (i % 5)},
            expected_truth="accept",
        ))
        # Evaluator drift: changed metric creates fake gain.
        scenarios.append(ToyScenario(
            name=f"sweep_eval_drift_{i:03d}",
            description="Fake improvement created by evaluator changes.",
            kind="evaluator_drift",
            changed_files=["train.py", "eval.py"],
            claim="The patch significantly improves validation loss.",
            baseline_values={1: 1.000, 2: 1.000, 3: 1.000},
            candidate_values={1: 0.700, 2: 0.705, 3: 0.710},
            official_candidate_values={1: 1.002 + 0.001 * (i % 3), 2: 1.000, 3: 1.004 + 0.001 * (i % 2)},
            expected_truth="reject",
        ))
        # Implementation bug.
        scenarios.append(ToyScenario(
            name=f"sweep_bug_{i:03d}",
            description="Candidate fails with an implementation error.",
            kind="implementation_bug",
            changed_files=["train.py", "loss.py"],
            claim="The normalized loss improves stability.",
            baseline_values={1: 1.000, 2: 1.000, 3: 1.000},
            candidate_values={1: 999.0, 2: 999.0, 3: 999.0},
            logs="Traceback RuntimeError: shape mismatch in loss normalization",
            expected_truth="blocked",
        ))
        # Confounded mechanism: performance may be real, mechanism claim is unsupported.
        scenarios.append(ToyScenario(
            name=f"sweep_confounded_{i:03d}",
            description="Full patch improves but mechanism claim is confounded.",
            kind="confounded",
            changed_files=["model.py", "train.py", "data_loader.py"],
            claim="The improvement is caused by dropout.",
            baseline_values={1: 1.000, 2: 1.002, 3: 0.998},
            candidate_values={1: 0.930, 2: 0.940, 3: 0.935},
            expected_truth="reject",
        ))
    return scenarios


def run_toy_policy_sweep(n_per_type: int = 20) -> Dict[str, object]:
    global SCENARIOS
    old = SCENARIOS
    try:
        SCENARIOS = generate_sweep_scenarios(n_per_type)
        return run_toy_policy_eval()
    finally:
        SCENARIOS = old
