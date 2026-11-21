from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from agents.alignment_agent.schemas import AlignmentResult
from agents.implementation_agent.schemas import ImplementationReport



class LegalAssessmentStatus(str, Enum):
    POTENTIAL_NON_COMPLIANCE = "potential_non_compliance"
    POTENTIAL_COMPLIANCE = "potential_compliance"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NOT_APPLICABLE = "not_applicable"
    REQUIRES_REVIEW = "requires_review"


class RetrievedLegalChunk(BaseModel):
    """A legal text chunk retrieved from the authoritative knowledge base."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    text: str
    source_file: str
    document_type: str
    authority: str
    pdf_page: int = Field(ge=1)
    provision: str = "not_identified"
    distance: float | None = None


class LegalCitation(BaseModel):
    """A traceable citation to a retrieved legal source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    provision: str
    document_type: str
    source_file: str
    pdf_page: int = Field(ge=1)
    excerpt: str = Field(min_length=1)

    @classmethod
    def model_validate(cls, obj, **kwargs):
        instance = super().model_validate(obj, **kwargs)
        if not instance.excerpt.strip():
            raise ValueError("Citation excerpt cannot be whitespace-only.")
        if not all((
            instance.source_id.strip(),
            instance.provision.strip(),
            instance.document_type.strip(),
            instance.source_file.strip(),
        )):
            raise ValueError("Citation metadata fields cannot be empty.")
        return instance


class LegalFinding(BaseModel):
    """Legal assessment corresponding to one Agent 3 alignment finding."""

    model_config = ConfigDict(extra="forbid")

    alignment_finding_id: str
    status: LegalAssessmentStatus
    applicable_provisions: list[str] = Field(default_factory=list)
    legal_reasoning: str
    evidence_fact_ids: list[str] = Field(default_factory=list)
    citations: list[LegalCitation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    limitations: list[str] = Field(default_factory=list)


class FailedLLMCall(BaseModel):
    """A failed Agent 4 LLM call or response-validation attempt."""

    model_config = ConfigDict(extra="forbid")

    alignment_finding_id: str
    attempt: int = Field(ge=1)
    error_type: str


class FailedRetrieval(BaseModel):
    """A failed attempt to retrieve legal source passages."""

    model_config = ConfigDict(extra="forbid")

    alignment_finding_id: str
    error_type: str


class LegalAnalysisResult(BaseModel):
    """Complete structured output from Agent 4 for Agent 5."""

    model_config = ConfigDict(extra="forbid")

    findings: list[LegalFinding]
    summary: dict[str, int] = Field(default_factory=dict)
    sources_consulted: list[str] = Field(default_factory=list)
    failed_calls: list[FailedLLMCall] = Field(default_factory=list)
    retrieval_failures: list[FailedRetrieval] = Field(default_factory=list)


class LegalAnalysisRequest(BaseModel):
    """Input contract for Agent 4."""

    model_config = ConfigDict(extra="forbid")

    alignment_result: AlignmentResult
    implementation_report: ImplementationReport
