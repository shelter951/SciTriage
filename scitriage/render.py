from __future__ import annotations

from .schema import TriageReport


def render_markdown(report: TriageReport) -> str:
    lines = [f"# SciTriage Report: {report.trace_id}\n", f"**Status**: `{report.status}`\n"]
    lines.append("## Risk Labels\n")
    lines.extend([f"- `{risk}`" for risk in report.risk_labels] or ["- none"])
    lines.append("\n## Recommended Probe\n")
    if report.recommended_probe:
        p = report.recommended_probe
        lines.append(f"- **Type**: `{p.probe_type}`")
        lines.append(f"- **Estimated cost**: {p.estimated_cost_gpu_hours} GPU-hours")
        lines.append(f"- **Reason**: {p.reason}")
        if p.details:
            lines.append(f"- **Details**: `{p.details}`")
    else:
        lines.append("- no probe needed")
    lines.append("\n## Evidence Debt\n")
    if report.evidence_debt:
        for item in report.evidence_debt:
            lines.append(f"- **Claim**: {item['claim']}")
            lines.append(f"  - Missing: {', '.join(item['missing_evidence'])}")
            lines.append(f"  - Severity: {item['severity']}")
    else:
        lines.append("- none")
    lines.append("\n## Blocked Claims\n")
    lines.extend([f"- {claim}" for claim in report.blocked_claims] or ["- none"])
    lines.append("\n## Allowed Claims\n")
    lines.extend([f"- {claim}" for claim in report.allowed_claims] or ["- none"])
    lines.append("\n## Rationales\n")
    lines.extend([f"- {rationale}" for rationale in report.rationales] or ["- none"])
    return "\n".join(lines) + "\n"
