from pydantic import BaseModel, ConfigDict, Field


class ComplianceScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float | None = Field(default=None, ge=0, le=100)
    total_findings: int
    assessed_findings: int
    potential_compliance_count: int
    potential_non_compliance_count: int
    insufficient_evidence_count: int
    requires_review_count: int
    not_applicable_count: int
    assessment_coverage_percent: float | None = Field(
        default=None, ge=0, le=100
    )
    minimum_coverage_percent: float = Field(default=50.0, ge=0, le=100)
    score_withheld_reason: str | None = None
    score_description: str
