from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricObservation:
    name: str
    candidate: Optional[float] = None
    baseline: Optional[float] = None
    higher_is_better: bool = False
    baseline_std: Optional[float] = None
    candidate_std: Optional[float] = None
    seeds: int = 1

    def delta(self) -> Optional[float]:
        if self.candidate is None or self.baseline is None:
            return None
        raw = self.candidate - self.baseline
        return raw if self.higher_is_better else -raw


@dataclass
class ResearchTrace:
    trace_id: str
    question: str
    proposal: str
    claims: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    protected_files: List[str] = field(default_factory=list)
    diff_summary: str = ""
    logs: str = ""
    metrics: List[MetricObservation] = field(default_factory=list)
    experiment: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchTrace":
        metrics = [MetricObservation(**item) for item in data.get("metrics", [])]
        return cls(
            trace_id=data.get("trace_id", "unknown"),
            question=data.get("question", ""),
            proposal=data.get("proposal", ""),
            claims=list(data.get("claims", [])),
            changed_files=list(data.get("changed_files", [])),
            protected_files=list(data.get("protected_files", [])),
            diff_summary=data.get("diff_summary", ""),
            logs=data.get("logs", ""),
            metrics=metrics,
            experiment=dict(data.get("experiment", {})),
            history=list(data.get("history", [])),
        )


@dataclass
class ProbeRecommendation:
    probe_type: str
    reason: str
    estimated_cost_gpu_hours: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.probe_type,
            "reason": self.reason,
            "estimated_cost_gpu_hours": self.estimated_cost_gpu_hours,
            "details": self.details,
        }


@dataclass
class TriageReport:
    trace_id: str
    status: str
    risk_labels: List[str]
    evidence_debt: List[Dict[str, Any]]
    recommended_probe: Optional[ProbeRecommendation]
    allowed_claims: List[str]
    blocked_claims: List[str]
    rationales: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "status": self.status,
            "risk_labels": self.risk_labels,
            "evidence_debt": self.evidence_debt,
            "recommended_probe": None if self.recommended_probe is None else self.recommended_probe.to_dict(),
            "allowed_claims": self.allowed_claims,
            "blocked_claims": self.blocked_claims,
            "rationales": self.rationales,
        }
