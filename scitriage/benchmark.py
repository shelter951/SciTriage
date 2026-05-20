from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

from .rules import diagnose
from .schema import ResearchTrace


@dataclass
class CaseResult:
    case_id: str
    expected: Set[str]
    predicted: Set[str]
    expected_status: str | None
    predicted_status: str
    expected_probe: str | None
    predicted_probe: str | None

    @property
    def tp(self) -> int:
        return len(self.expected & self.predicted)

    @property
    def fp(self) -> int:
        return len(self.predicted - self.expected)

    @property
    def fn(self) -> int:
        return len(self.expected - self.predicted)

    @property
    def exact(self) -> bool:
        return self.expected == self.predicted

    @property
    def status_ok(self) -> bool:
        return self.expected_status is None or self.expected_status == self.predicted_status

    @property
    def probe_ok(self) -> bool:
        return self.expected_probe is None or self.expected_probe == self.predicted_probe


def iter_cases(path: str | Path) -> Iterable[Path]:
    root = Path(path)
    if root.is_file():
        yield root
    else:
        yield from sorted(root.glob('*.json'))


def evaluate_benchmark(path: str | Path) -> Dict[str, object]:
    results: List[CaseResult] = []
    for case_path in iter_cases(path):
        data = json.loads(case_path.read_text())
        expected_info = data.get('expected', {})
        trace = ResearchTrace.from_dict(data)
        report = diagnose(trace)
        expected = set(expected_info.get('risk_labels', []))
        predicted = set(report.risk_labels)
        probe = report.recommended_probe.probe_type if report.recommended_probe else None
        results.append(CaseResult(
            case_id=data.get('trace_id', case_path.stem),
            expected=expected,
            predicted=predicted,
            expected_status=expected_info.get('status'),
            predicted_status=report.status,
            expected_probe=expected_info.get('probe_type'),
            predicted_probe=probe,
        ))

    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    exact = sum(r.exact for r in results) / len(results) if results else 0.0
    status_acc = sum(r.status_ok for r in results) / len(results) if results else 0.0
    probe_acc = sum(r.probe_ok for r in results) / len(results) if results else 0.0

    return {
        'num_cases': len(results),
        'label_precision': precision,
        'label_recall': recall,
        'label_f1': f1,
        'exact_match': exact,
        'status_accuracy': status_acc,
        'probe_accuracy': probe_acc,
        'cases': [
            {
                'case_id': r.case_id,
                'expected': sorted(r.expected),
                'predicted': sorted(r.predicted),
                'missing': sorted(r.expected - r.predicted),
                'extra': sorted(r.predicted - r.expected),
                'expected_status': r.expected_status,
                'predicted_status': r.predicted_status,
                'expected_probe': r.expected_probe,
                'predicted_probe': r.predicted_probe,
                'exact': r.exact,
                'status_ok': r.status_ok,
                'probe_ok': r.probe_ok,
            }
            for r in results
        ],
    }


def render_benchmark_markdown(summary: Dict[str, object]) -> str:
    lines = ['# SciTriage Benchmark Report\n']
    lines.append(f"- Cases: {summary['num_cases']}")
    lines.append(f"- Label precision: {summary['label_precision']:.3f}")
    lines.append(f"- Label recall: {summary['label_recall']:.3f}")
    lines.append(f"- Label F1: {summary['label_f1']:.3f}")
    lines.append(f"- Exact match: {summary['exact_match']:.3f}")
    lines.append(f"- Status accuracy: {summary['status_accuracy']:.3f}")
    lines.append(f"- Probe accuracy: {summary['probe_accuracy']:.3f}")
    lines.append('\n## Case Details\n')
    for case in summary['cases']:
        mark = 'PASS' if case['exact'] and case['status_ok'] and case['probe_ok'] else 'CHECK'
        lines.append(f"### {case['case_id']} ? {mark}")
        lines.append(f"- Expected labels: `{case['expected']}`")
        lines.append(f"- Predicted labels: `{case['predicted']}`")
        if case['missing']:
            lines.append(f"- Missing: `{case['missing']}`")
        if case['extra']:
            lines.append(f"- Extra: `{case['extra']}`")
        lines.append(f"- Status: expected `{case['expected_status']}`, predicted `{case['predicted_status']}`")
        lines.append(f"- Probe: expected `{case['expected_probe']}`, predicted `{case['predicted_probe']}`")
        lines.append('')
    return '\n'.join(lines)
