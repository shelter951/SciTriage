from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List


METRIC_PATTERNS = {
    'val_bpb': re.compile(r'^val_bpb:\s*([0-9.]+)', re.MULTILINE),
    'val_loss': re.compile(r'(?:^|\b)val_loss[:=]\s*([0-9.]+)', re.MULTILINE),
    'accuracy': re.compile(r'(?:^|\b)accuracy[:=]\s*([0-9.]+)', re.MULTILINE),
}


def parse_metric(log_text: str, metric: str) -> float | None:
    pattern = METRIC_PATTERNS.get(metric)
    if pattern is None:
        pattern = re.compile(rf'(?:^|\b){re.escape(metric)}[:=]\s*([0-9.]+)', re.MULTILINE)
    match = pattern.search(log_text)
    return float(match.group(1)) if match else None


def aggregate_logs(paths: Iterable[str | Path], metric: str, baseline: float | None = None, higher_is_better: bool = False) -> Dict[str, object]:
    observations: List[Dict[str, object]] = []
    values: List[float] = []
    for raw in paths:
        path = Path(raw)
        value = parse_metric(path.read_text(errors='replace'), metric)
        observations.append({'path': str(path), 'value': value, 'ok': value is not None})
        if value is not None:
            values.append(value)

    n = len(values)
    mean = sum(values) / n if n else None
    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
        stderr = std / math.sqrt(n)
    else:
        std = None
        stderr = None

    delta = None
    verdict = 'insufficient_data'
    if mean is not None and baseline is not None:
        raw_delta = mean - baseline
        delta = raw_delta if higher_is_better else -raw_delta
        if n < 2:
            verdict = 'needs_more_seeds'
        elif delta > 0 and (stderr is None or abs(delta) > stderr):
            verdict = 'supports_improvement'
        elif delta <= 0:
            verdict = 'does_not_support_improvement'
        else:
            verdict = 'inconclusive_within_noise'

    return {
        'metric': metric,
        'n': n,
        'mean': mean,
        'std': std,
        'stderr': stderr,
        'baseline': baseline,
        'higher_is_better': higher_is_better,
        'delta_improvement': delta,
        'verdict': verdict,
        'observations': observations,
    }


def _sample_summary(paths: Iterable[str | Path], metric: str) -> Dict[str, object]:
    observations: List[Dict[str, object]] = []
    values: List[float] = []
    for raw in paths:
        path = Path(raw)
        value = parse_metric(path.read_text(errors='replace'), metric)
        observations.append({'path': str(path), 'value': value, 'ok': value is not None})
        if value is not None:
            values.append(value)

    n = len(values)
    mean = sum(values) / n if n else None
    if n >= 2:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = None
    return {'n': n, 'mean': mean, 'std': std, 'values': values, 'observations': observations}


def compare_seed_groups(
    baseline_logs: Iterable[str | Path],
    candidate_logs: Iterable[str | Path],
    metric: str,
    higher_is_better: bool = False,
    z: float = 1.96,
) -> Dict[str, object]:
    baseline = _sample_summary(baseline_logs, metric)
    candidate = _sample_summary(candidate_logs, metric)
    verdict = 'insufficient_data'
    delta = None
    combined_se = None
    margin = None
    if baseline['mean'] is not None and candidate['mean'] is not None:
        raw = candidate['mean'] - baseline['mean']
        delta = raw if higher_is_better else -raw
        if baseline['n'] >= 2 and candidate['n'] >= 2:
            combined_se = math.sqrt((baseline['std'] ** 2 / baseline['n']) + (candidate['std'] ** 2 / candidate['n']))
            margin = z * combined_se
            if delta > margin:
                verdict = 'supports_improvement'
            elif delta <= 0:
                verdict = 'does_not_support_improvement'
            else:
                verdict = 'inconclusive_within_noise'
        else:
            verdict = 'needs_more_seeds'

    return {
        'metric': metric,
        'higher_is_better': higher_is_better,
        'baseline': baseline,
        'candidate': candidate,
        'delta_improvement': delta,
        'combined_standard_error': combined_se,
        'z_margin': margin,
        'z': z,
        'verdict': verdict,
    }
