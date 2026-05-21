from __future__ import annotations

from typing import Any, Dict, Iterable, List


def select_with_semantic_gate(
    candidates: Iterable[Dict[str, Any]],
    score_key: str = "official_score",
    invariant_key: str = "semantic_invariant_passed",
    lower_is_better: bool = True,
) -> Dict[str, Any]:
    """Compare visible-score selection against semantic-invariant-gated selection."""
    rows: List[Dict[str, Any]] = [
        row for row in candidates
        if row.get(score_key) is not None
    ]
    if not rows:
        return {
            "status": "no_scored_candidates",
            "visible_score_only": None,
            "scitriage_gated": None,
            "blocked_candidates": [],
        }

    key = (lambda row: row[score_key]) if lower_is_better else (lambda row: -row[score_key])
    visible = min(rows, key=key)
    valid = [row for row in rows if bool(row.get(invariant_key))]
    gated = min(valid, key=key) if valid else None
    blocked = [row for row in rows if not bool(row.get(invariant_key))]

    return {
        "status": "ok" if gated is not None else "no_semantically_valid_candidate",
        "visible_score_only": _policy_item(visible, score_key, invariant_key),
        "scitriage_gated": None if gated is None else _policy_item(gated, score_key, invariant_key),
        "blocked_candidates": [_policy_item(row, score_key, invariant_key) for row in blocked],
        "visible_winner_blocked": not bool(visible.get(invariant_key)),
    }


def _policy_item(row: Dict[str, Any], score_key: str, invariant_key: str) -> Dict[str, Any]:
    return {
        "selected": row.get("variant") or row.get("candidate_id") or row.get("name"),
        "score": row.get(score_key),
        "semantic_invariant_passed": bool(row.get(invariant_key)),
    }
