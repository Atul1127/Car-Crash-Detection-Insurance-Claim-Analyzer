"""Typed data contracts shared across the ClaimVision pipeline."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ImageQualityResult:
    valid: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DamageDetection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass
class DamageAssessment:
    detections: list[DamageDetection] = field(default_factory=list)
    severity: str | None = None
    severity_score: float | None = None


@dataclass
class ClaimInformation:
    claim_id: str | None = None
    policy_number: str | None = None
    claimant_name: str | None = None
    vehicle_registration: str | None = None
    incident_date: str | None = None
    incident_description: str | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyEvidence:
    text: str
    source: str | None = None
    page: int | None = None
    score: float | None = None


@dataclass
class ClaimDecision:
    decision: str
    coverage_status: str
    risk_score: float
    rationale: str
    evidence: list[PolicyEvidence] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ClaimReport:
    image_quality: ImageQualityResult
    damage: DamageAssessment
    claim_info: ClaimInformation = field(default_factory=ClaimInformation)
    decision: ClaimDecision | None = None
    explainability: dict[str, Any] = field(default_factory=dict)
