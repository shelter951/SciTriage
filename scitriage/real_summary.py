from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def summarize_real_runs(run_dirs: List[str | Path]) -> Dict[str, object]:
    rows = []
    for raw in run_dirs:
        d = Path(raw)
        report_path = d / 'triage_report.json'
        trace_path = d / 'trace.json'
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
        rows.append({
            'run_dir': str(d),
            'trace_id': report.get('trace_id'),
            'status': report.get('status'),
            'risk_labels': report.get('risk_labels', []),
            'num_claims': len(trace.get('claims', [])),
            'blocked_claims': len(report.get('blocked_claims', [])),
            'allowed_claims': len(report.get('allowed_claims', [])),
            'probe': (report.get('recommended_probe') or {}).get('type'),
            'num_files_scanned': (trace.get('experiment') or {}).get('num_files_scanned'),
            'num_metrics': len(trace.get('metrics', [])),
        })
    return {'num_runs': len(rows), 'runs': rows}


def render_real_summary_markdown(summary: Dict[str, object]) -> str:
    lines = ['# Real Artifact SciTriage Summary\n']
    lines.append('| Trace | Status | Risks | Claims Blocked/Total | Probe | Files | Metrics |')
    lines.append('|---|---|---|---:|---|---:|---:|')
    for row in summary['runs']:
        risks = ', '.join(row['risk_labels']) if row['risk_labels'] else 'none'
        lines.append(
            f"| {row['trace_id']} | {row['status']} | {risks} | "
            f"{row['blocked_claims']}/{row['num_claims']} | {row['probe']} | "
            f"{row.get('num_files_scanned')} | {row['num_metrics']} |"
        )
    return '\n'.join(lines) + '\n'
