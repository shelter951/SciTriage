from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from ..schema import MetricObservation, ResearchTrace

DEFAULT_PATTERNS = [
    'IDEA_REPORT.md',
    'IDEA_CANDIDATES.md',
    'NARRATIVE_REPORT.md',
    'OVERNIGHT_FINAL_REPORT.md',
    'PRE_EXPERIMENT_GUARD_REPORT.md',
    'EXPERIMENT_LOG.md',
    'RESULTS.md',
    'AUTO_REVIEW.md',
    'COST_REALISM_AUDIT.md',
    'CLAUDE.md',
]

CLAIM_CUE = re.compile(r'(^|\b)(we claim|we propose|we show|we demonstrate|this work|our method|conclusion|finding|claim|hypothesis)\b', re.I)
METRIC_RE = re.compile(r'(?P<name>val_loss|val_bpb|accuracy|success_rate|reward|score|f1)\s*[:=]\s*(?P<value>[0-9]+(?:\.[0-9]+)?)', re.I)


def _read_if_exists(path: Path, limit_chars: int = 30000) -> str:
    if not path.exists() or not path.is_file():
        return ''
    return path.read_text(errors='replace')[:limit_chars]


def _candidate_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in DEFAULT_PATTERNS:
        files.extend(root.rglob(pattern))
    # Include root-level research notes for lightweight project folders.
    files.extend(sorted(root.glob('*.md'))[:30])

    # Include high-signal markdown files in common ARIS directories.
    for sub in ['idea-stage', 'outputs', 'review-stage', 'experiments']:
        d = root / sub
        if d.exists():
            files.extend(sorted(d.rglob('*.md')))
            files.extend(sorted(d.rglob('*.log')))
            files.extend(sorted(d.rglob('*.json')))
    # Deduplicate preserving order.
    seen = set()
    out = []
    for f in files:
        if f not in seen and f.is_file():
            seen.add(f)
            out.append(f)
    return out


def _extract_claims(text: str, max_claims: int = 12) -> List[str]:
    claims = []
    seen = set()
    strong_cues = [
        'core hypothesis', 'hypothesis', 'core claim', 'we propose', 'we show',
        'we demonstrate', 'we find', 'novelty', 'novel intersection',
        'outperforms', 'significant', 'significantly', 'improves', 'robust',
        'efficient', 'cost-efficient', 'generalizes', 'caused by', 'because'
    ]
    weak_prefixes = ('pipeline', 'selection criteria', 'why promising', 'why interesting', 'gap')
    for raw in re.split(r'\n+|(?<=[.!?])\s+', text):
        line = raw.strip(' -*#\t')
        low = line.lower().strip()
        if len(line) < 35 or len(line) > 320:
            continue
        if low.startswith('|') or line.count('|') >= 2:
            continue
        if any(low.startswith(p) for p in weak_prefixes):
            continue
        if low.startswith('===== file:') or low.startswith('http'):
            continue
        is_claim = CLAIM_CUE.search(line) is not None or any(k in low for k in strong_cues)
        if not is_claim:
            continue
        key = re.sub(r'\s+', ' ', low)
        if key in seen:
            continue
        seen.add(key)
        claims.append(line)
        if len(claims) >= max_claims:
            break
    return claims


def _extract_metrics(text: str) -> List[MetricObservation]:
    metrics = []
    for match in METRIC_RE.finditer(text):
        name = match.group('name').lower()
        value = float(match.group('value'))
        higher = name in {'accuracy', 'success_rate', 'reward', 'score', 'f1'}
        metrics.append(MetricObservation(name=name, candidate=value, baseline=None, higher_is_better=higher, seeds=1))
    return metrics[:8]


def _changed_files_from_git(root: Path) -> List[str]:
    import subprocess
    try:
        inside = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--is-inside-work-tree'], text=True, stderr=subprocess.DEVNULL).strip()
        if inside != 'true':
            return []
        out = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', 'HEAD'], text=True, stderr=subprocess.DEVNULL)
        return [line.strip() for line in out.splitlines() if line.strip()][:80]
    except Exception:
        return []


def trace_from_filesystem(root: str, trace_id: str | None = None) -> ResearchTrace:
    root_path = Path(root).expanduser().resolve()
    files = _candidate_files(root_path)
    doc_sections = []
    log_sections = []
    for f in files[:60]:
        rel = f.relative_to(root_path)
        content = _read_if_exists(f)
        if not content:
            continue
        block = f'\n\n===== FILE: {rel} =====\n{content}'
        if f.suffix.lower() in {'.log'} or any(part in str(rel).lower() for part in ['results', 'outputs', 'run_', 'metrics']):
            log_sections.append(block)
        if f.suffix.lower() in {'.md', '.json', '.jsonl'}:
            doc_sections.append(block)
    combined = ''.join(doc_sections)
    logs_combined = ''.join(log_sections)
    claims = _extract_claims(combined)
    metrics = _extract_metrics(logs_combined) or _extract_metrics(combined)
    changed_files = _changed_files_from_git(root_path)
    protected_files = ['evaluation.py', 'eval.py', 'metrics.py', 'benchmark.py', 'official_evaluator.py']
    proposal = ''
    for name in ['IDEA_REPORT.md', 'IDEA_CANDIDATES.md', 'NARRATIVE_REPORT.md', 'OVERNIGHT_FINAL_REPORT.md']:
        p = next((f for f in files if f.name == name), None)
        if p:
            proposal = _read_if_exists(p, 5000)
            break
    if not proposal:
        proposal = combined[:5000]
    return ResearchTrace(
        trace_id=trace_id or f'filesystem_{root_path.name}',
        question='Assess an AutoResearch project artifact directory for unsupported claims, drift, and missing evidence.',
        proposal=proposal,
        claims=claims,
        changed_files=changed_files,
        protected_files=protected_files,
        diff_summary='filesystem artifact scan: ' + ', '.join(str(f.relative_to(root_path)) for f in files[:20]),
        logs=logs_combined[:50000],
        metrics=metrics,
        experiment={'harness': 'filesystem', 'root': str(root_path), 'num_files_scanned': len(files)},
        history=[],
    )
