from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _safe_rewrite(claim: str, missing: List[str]) -> str:
    missing_text = ', '.join(missing) if missing else 'additional evidence'
    low = claim.lower().strip()
    if low.startswith('how can'):
        return f"Open research question, not a validated claim: {claim} Evidence needed to answer it: {missing_text}."
    if 'novelty search' in missing and len(missing) == 1:
        return f"Provisional novelty statement: {claim} This should not be treated as confirmed until targeted prior-work search is attached."
    if 'fair cost/runtime comparison' in missing and len(missing) == 1:
        return f"Provisional efficiency/cost statement: {claim} This requires a fair cost/runtime comparison before being claimed."
    if 'statistical or multi-seed evidence' in missing or 'multi-seed evidence' in missing:
        return f"Provisional empirical observation: {claim} This requires statistical or multi-seed verification before being reported as a result."
    if 'baseline comparison' in missing:
        return f"Provisional empirical observation: {claim} This requires a reproduced baseline comparison before being claimed."
    if 'mechanism ablation' in missing:
        return f"Provisional mechanism hypothesis: {claim} This requires ablation before attribution is justified."
    if 'held-out task or dataset' in missing:
        return f"Narrow-scope observation only: {claim} This does not yet support generalization beyond the tested setting."
    return f"Provisional claim: {claim} Missing evidence: {missing_text}."


def reconstruct_claims(trace_path: str | Path, report_path: str | Path) -> Dict[str, object]:
    trace = json.loads(Path(trace_path).read_text())
    report = json.loads(Path(report_path).read_text())
    debt_by_claim = {item['claim']: item.get('missing_evidence', []) for item in report.get('evidence_debt', [])}
    supported = []
    revised = []
    for claim in report.get('allowed_claims', []):
        supported.append({'claim': claim, 'status': 'allowed'})
    for claim in report.get('blocked_claims', []):
        missing = debt_by_claim.get(claim, [])
        revised.append({
            'original': claim,
            'status': 'revised',
            'missing_evidence': missing,
            'safe_rewrite': _safe_rewrite(claim, missing),
        })
    return {
        'trace_id': report.get('trace_id', trace.get('trace_id')),
        'supported_claims': supported,
        'revised_claims': revised,
        'num_allowed': len(supported),
        'num_revised': len(revised),
    }


def render_reconstructed_claims(result: Dict[str, object]) -> str:
    lines = [f"# Supported Claims: {result['trace_id']}\n"]
    lines.append(f"- Allowed claims: {result['num_allowed']}")
    lines.append(f"- Revised blocked claims: {result['num_revised']}\n")
    lines.append('## Allowed Claims\n')
    if not result['supported_claims']:
        lines.append('- none')
    for item in result['supported_claims']:
        lines.append(f"- {item['claim']}")
    lines.append('\n## Revised Claims\n')
    if not result['revised_claims']:
        lines.append('- none')
    for idx, item in enumerate(result['revised_claims'], 1):
        lines.append(f"### R{idx}")
        lines.append(f"- Original: {item['original']}")
        lines.append(f"- Missing evidence: {', '.join(item['missing_evidence']) if item['missing_evidence'] else 'unspecified'}")
        lines.append(f"- Safe rewrite: {item['safe_rewrite']}\n")
    return '\n'.join(lines)
