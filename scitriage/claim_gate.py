from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def gate_claim(group_compare_json: str | Path, claim: str, min_margin_ratio: float = 1.0) -> Dict[str, object]:
    data = json.loads(Path(group_compare_json).read_text())
    verdict = data.get("verdict")
    delta = data.get("delta_improvement")
    margin = data.get("z_margin")
    ratio = None
    if isinstance(delta, (int, float)) and isinstance(margin, (int, float)) and margin > 0:
        ratio = delta / margin

    if verdict == "supports_improvement" and ratio is not None and ratio >= min_margin_ratio:
        status = "allowed"
        safe_claim = claim
        reason = "The candidate improvement exceeds the group-comparison uncertainty margin."
    elif verdict == "inconclusive_within_noise":
        status = "blocked"
        safe_claim = (
            "The candidate showed a positive mean trend, but the improvement is within "
            "the measured seed-group uncertainty and should be reported as inconclusive."
        )
        reason = "The improvement does not exceed the configured uncertainty margin."
    elif verdict == "does_not_support_improvement":
        status = "blocked"
        safe_claim = "The available seed-group evidence does not support an improvement claim."
        reason = "The candidate mean does not improve over the baseline mean."
    else:
        status = "needs_more_evidence"
        safe_claim = "The claim needs more evidence before it can be reported."
        reason = f"Unhandled or insufficient verdict: {verdict}."

    return {
        "claim": claim,
        "status": status,
        "safe_claim": safe_claim,
        "reason": reason,
        "group_verdict": verdict,
        "delta_improvement": delta,
        "z_margin": margin,
        "delta_to_margin_ratio": ratio,
        "min_margin_ratio": min_margin_ratio,
    }


def render_claim_gate_markdown(result: Dict[str, object]) -> str:
    lines = [f"# Claim Gate: {result['status']}\n"]
    lines.append(f"- Original claim: {result['claim']}")
    lines.append(f"- Safe claim: {result['safe_claim']}")
    lines.append(f"- Reason: {result['reason']}")
    lines.append(f"- Group verdict: `{result['group_verdict']}`")
    lines.append(f"- Delta improvement: {result['delta_improvement']}")
    lines.append(f"- 95% margin: {result['z_margin']}")
    lines.append(f"- Delta / margin: {result['delta_to_margin_ratio']}")
    return "\n".join(lines)
