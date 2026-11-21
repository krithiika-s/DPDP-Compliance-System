from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.legal_rag.schemas import (
    FailedLLMCall,
    FailedRetrieval,
    LegalAnalysisResult,
    LegalFinding,
)
from agents.scoring.schemas import ComplianceScore


class ReportRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    priority: Literal["low", "medium", "high"]
    action: str
    reason: str


class ComplianceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    score: ComplianceScore
    findings: list[LegalFinding]
    recommendations: list[ReportRecommendation]
    sources_consulted: list[str]
    executive_summary: str
    disclaimer: str
    failed_calls: list[FailedLLMCall] = Field(default_factory=list)
    retrieval_failures: list[FailedRetrieval] = Field(default_factory=list)


class ReportGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    legal_analysis: LegalAnalysisResult
    score: ComplianceScore
