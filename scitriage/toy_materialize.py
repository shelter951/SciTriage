from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .probe_plan import build_probe_plan, render_probe_plan_markdown
from .render import render_markdown
from .rules import diagnose
from .schema import ResearchTrace
from .toy_eval import SCENARIOS, evaluate_scitriage, make_trace


def trace_to_dict(trace: ResearchTrace) -> Dict[str, object]:
    return {
        'trace_id': trace.trace_id,
        'question': trace.question,
        'proposal': trace.proposal,
        'claims': trace.claims,
        'changed_files': trace.changed_files,
        'protected_files': trace.protected_files,
        'diff_summary': trace.diff_summary,
        'logs': trace.logs,
        'metrics': [m.__dict__ for m in trace.metrics],
        'experiment': trace.experiment,
        'history': trace.history,
    }


def materialize_toy_cases(out_dir: str | Path) -> Dict[str, object]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for scenario in SCENARIOS:
        case_dir = out / scenario.name
        case_dir.mkdir(parents=True, exist_ok=True)
        trace = make_trace(scenario, seed=1)
        report = diagnose(trace)
        plan = build_probe_plan(trace, report)
        verdict = evaluate_scitriage(scenario)
        (case_dir / 'trace.json').write_text(json.dumps(trace_to_dict(trace), indent=2, ensure_ascii=False))
        (case_dir / 'triage_report.json').write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        (case_dir / 'triage_report.md').write_text(render_markdown(report))
        (case_dir / 'probe_plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        (case_dir / 'probe_plan.md').write_text(render_probe_plan_markdown(plan))
        (case_dir / 'scitriage_verdict.json').write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
        index.append({
            'scenario': scenario.name,
            'truth': scenario.expected_truth,
            'risk_labels': report.risk_labels,
            'probe': report.recommended_probe.probe_type if report.recommended_probe else None,
            'decision': verdict['decision'],
            'cost': verdict['cost'],
            'false_discovery': verdict['false_discovery'],
            'unsupported_claim_allowed': verdict['unsupported_claim_allowed'],
        })
    summary = {'num_cases': len(index), 'cases': index}
    (out / 'index.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    lines = ['# Materialized Toy Cases\n']
    lines.append('| Scenario | Truth | Probe | Decision | False Discovery | Unsupported Allowed |')
    lines.append('|---|---|---|---|---:|---:|')
    for item in index:
        lines.append(f"| {item['scenario']} | {item['truth']} | {item['probe']} | {item['decision']} | {item['false_discovery']} | {item['unsupported_claim_allowed']} |")
    (out / 'index.md').write_text('\n'.join(lines) + '\n')
    return summary
