
"""Data models for the central orchestrator.

These are validated containers, not agents. They describe:
  1. what the user submits (AssessmentRequest),
  2. everything known about one assessment run (AssessmentState),
  3. an audit log of what the orchestrator did (StepRecord).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.policy_agent.schemas import PolicyClaim


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PolicyDocument(BaseModel):
    """One policy or governance document supplied by the user."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AssessmentRequest(BaseModel):
    """Everything the user submits to start an assessment."""

    model_config = ConfigDict(extra="forbid")

    assessment_id: str = Field(min_length=1)
    policy_documents: list[PolicyDocument] = Field(min_length=1)
    project_path: str | None = None

    @field_validator("policy_documents")
    @classmethod
    def _names_must_be_unique(
        cls, documents: list[PolicyDocument]
    ) -> list[PolicyDocument]:
        names = [doc.name for doc in documents]
        if len(names) != len(set(names)):
            raise ValueError("Policy document names must be unique.")
        return documents


class StepStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class StepRecord(BaseModel):
    """One entry in the orchestrator's audit log."""

    model_config = ConfigDict(extra="forbid")

    step: str
    target: str = ""
    status: StepStatus
    attempt: int = Field(default=1, ge=1)
    message: str = ""
    timestamp: datetime = Field(default_factory=_now)


class AssessmentState(BaseModel):
    """The orchestrator's single record of an assessment in progress."""

    model_config = ConfigDict(extra="forbid")

    request: AssessmentRequest

    policy_claims: list[PolicyClaim] = Field(default_factory=list)
    rejected_claims: list[PolicyClaim] = Field(default_factory=list)

    # Agent 2's report as a plain dict (model_dump(mode="json")).
    implementation_report: dict[str, Any] | None = None

    # Agent 3's result as a plain dict (model_dump(mode="json")).
    alignment_result: dict[str, Any] | None = None

    # Evidence gate decisions (GateResult.to_dict()).
    evidence_gate: dict[str, Any] | None = None

    # Agent 4's original legal result, before the legal gate.
    legal_analysis_original: dict[str, Any] | None = None

    # Agent 4's legal result after evidence-gate safeguards.
    legal_analysis_gated: dict[str, Any] | None = None

    # Status changes made by the legal gate.
    legal_gate_application: dict[str, Any] | None = None

    # Agent 4's gated result scored by the deterministic scoring engine.
    compliance_score: dict[str, Any] | None = None

    # Agent 5's structured final report.
    compliance_report: dict[str, Any] | None = None

    step_log: list[StepRecord] = Field(default_factory=list)

    def log(
        self,
        step: str,
        status: StepStatus,
        *,
        target: str = "",
        attempt: int = 1,
        message: str = "",
    ) -> StepRecord:
        """Add an audit entry and return it."""
        record = StepRecord(
            step=step,
            target=target,
            status=status,
            attempt=attempt,
            message=message,
        )
        self.step_log.append(record)
        return record
