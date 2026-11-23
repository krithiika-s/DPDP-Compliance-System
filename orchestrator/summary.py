
"""Assessment summary for the complete DPDP pipeline.

Reads a finished AssessmentState and reports what was decided, what was
not assessed, and which stages failed. All calculations are deterministic.
No LLM is involved.

A run that does not start the final stages retains the original Phase A
summary behavior. Once final-stage execution starts, its outputs and
failures are included in the completeness assessment.
"""

from pydantic import BaseModel, ConfigDict, Field

from orchestrator.alignment_stage import STEP_NAME as ALIGNMENT_STEP
from orchestrator.analysis_stage import WARNING_STEP
from orchestrator.implementation_stage import STEP_NAME as IMPLEMENTATION_STEP
from orchestrator.models import AssessmentState, StepStatus
from orchestrator.policy_stage import STEP_NAME as POLICY_STEP
from orchestrator.sufficiency import Outcome, count_outcomes, outcome_for_status

MODEL_PROBLEM_CODES = ("model_failure", "invalid_model_output")
MESSAGE_LIMIT = 200

FINAL_STEPS = {
    "legal_analysis",
    "legal_gate",
    "compliance_scoring",
    "report_generation",
}


class StepIssue(BaseModel):
    """A step whose last log entry is not COMPLETED."""

    model_config = ConfigDict(extra="forbid")

    step: str
    target: str = ""
    status: StepStatus
    message: str = ""


class AssessmentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str

    # Agent 3 assessment.
    outcome_counts: dict[str, int] | None = None
    insufficient_by_reason: dict[str, int] = Field(default_factory=dict)
    decided_percent: float | None = None
    policy_text_chars: int | None = None
    policy_text_truncated: bool | None = None
    implementation_llm_failures: int | None = None
    alignment_llm_failures: int | None = None
    alignment_failed_calls: list[str] = Field(default_factory=list)
    alignment_unknown_verdict_ids: list[str] = Field(default_factory=list)
    documents_without_claims: list[str] = Field(default_factory=list)

    # Evidence gate for Agent 3 findings.
    gate_counts: dict[str, int] | None = None
    gate_warnings: list[str] = Field(default_factory=list)

    # Final legal assessment.
    legal_status_counts: dict[str, int] | None = None
    legal_failed_llm_calls: int | None = None
    legal_retrieval_failures: int | None = None
    legal_gate_changes: int | None = None

    # Scoring and report.
    compliance_score_available: bool | None = None
    compliance_score: float | None = None
    score_withheld_reason: str | None = None
    report_generated: bool | None = None

    failed_steps: list[StepIssue] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    incomplete_reasons: list[str] = Field(default_factory=list)
    is_complete: bool = False


def _short(text: str, limit: int = MESSAGE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _failed_steps(state: AssessmentState) -> list[StepIssue]:
    """Return steps whose last log entry is not COMPLETED."""

    last = {}

    for record in state.step_log:
        if record.step == WARNING_STEP:
            continue

        target = record.target if record.step == POLICY_STEP else ""
        last[(record.step, target)] = record

    return [
        StepIssue(
            step=record.step,
            target=record.target,
            status=record.status,
            message=_short(record.message),
        )
        for record in last.values()
        if record.status != StepStatus.COMPLETED
    ]


def _documents_without_claims(state: AssessmentState) -> list[str]:
    completed = {
        record.target
        for record in state.step_log
        if record.step == POLICY_STEP
        and record.status == StepStatus.COMPLETED
    }

    return [
        document.name
        for document in state.request.policy_documents
        if document.name not in completed
    ]


def _final_stage_started(state: AssessmentState) -> bool:
    """Determine whether the final stages were attempted."""

    if any(record.step in FINAL_STEPS for record in state.step_log):
        return True

    return any(
        value is not None
        for value in (
            state.legal_analysis_original,
            state.legal_analysis_gated,
            state.legal_gate_application,
            state.compliance_score,
            state.compliance_report,
        )
    )


def summarize_assessment(state: AssessmentState) -> AssessmentSummary:
    """Summarise the complete assessment without hiding missing results."""

    failed_steps = _failed_steps(state)
    missing_documents = _documents_without_claims(state)
    warnings = [
        record.message
        for record in state.step_log
        if record.step == WARNING_STEP
    ]

    reasons: list[str] = []

    if failed_steps:
        names = ", ".join(
            issue.step + (f" ({issue.target})" if issue.target else "")
            for issue in failed_steps
        )
        reasons.append(
            f"{len(failed_steps)} step(s) did not complete: {names}."
        )

    if missing_documents:
        reasons.append(
            "No claims were extracted from: "
            + ", ".join(missing_documents)
            + ". Their claims were not checked."
        )

    if warnings:
        reasons.append(
            f"{len(warnings)} analysis warning(s) were logged; see warnings."
        )

    # Agent 2.
    implementation = state.implementation_report
    implementation_failures = None

    if implementation is None:
        reasons.append(
            "Agent 2 produced no report, so the code was not analysed."
        )
    else:
        implementation_failures = int(
            implementation.get("llm_failures", 0)
        )

        if implementation_failures > 0:
            reasons.append(
                f"Agent 2 gave up on {implementation_failures} model call(s). "
                "Categories without facts are not evidence that the code does "
                "nothing there."
            )

    # Agent 3.
    alignment = state.alignment_result
    outcome_counts = None
    insufficient_by_reason: dict[str, int] = {}
    decided_percent = None
    text_chars = None
    truncated = None
    alignment_failures = None
    failed_calls: list[str] = []
    unknown_verdict_ids: list[str] = []

    if alignment is None:
        reasons.append(
            "Agent 3 produced no result, so nothing was assessed."
        )
    else:
        findings = alignment.get("findings", [])
        counts = count_outcomes(
            finding.get("status", "") for finding in findings
        )

        outcome_counts = {
            outcome.value: count
            for outcome, count in counts.items()
        }

        for finding in findings:
            if outcome_for_status(
                finding.get("status", "")
            ) == Outcome.INSUFFICIENT_EVIDENCE:
                code = finding.get("reason_code", "unknown")
                insufficient_by_reason[code] = (
                    insufficient_by_reason.get(code, 0) + 1
                )

        decided = counts[Outcome.COMPLIANT] + counts[Outcome.GAP]
        in_scope = decided + counts[Outcome.INSUFFICIENT_EVIDENCE]

        if in_scope:
            decided_percent = round(100 * decided / in_scope, 1)

        if not findings:
            reasons.append(
                "Agent 3 returned no findings, so nothing was assessed."
            )

        alignment_failures = int(alignment.get("llm_failures", 0))
        failed_calls = [
            str(call) for call in alignment.get("failed_calls", [])
        ]
        unknown_verdict_ids = [
            str(item) for item in alignment.get("unknown_verdict_ids", [])
        ]
        text_chars = alignment.get("policy_text_chars")
        truncated = bool(
            alignment.get("policy_text_truncated", False)
        )

        if alignment_failures > 0 or failed_calls:
            detail = (
                f": {', '.join(failed_calls)}"
                if failed_calls else ""
            )
            reasons.append(
                f"Agent 3 gave up on {alignment_failures} model call(s)"
                f"{detail}."
            )

        if "undeclared" in failed_calls:
            reasons.append(
                "The undeclared-behaviour check did not run, so "
                "'no undeclared findings' is not a clean result."
            )

        model_problems = sum(
            insufficient_by_reason.get(code, 0)
            for code in MODEL_PROBLEM_CODES
        )

        if model_problems:
            reasons.append(
                f"{model_problems} finding(s) could not be assessed "
                "because of a model problem; a retry may fix them."
            )

        if truncated:
            reasons.append(
                f"The policy text was cut ({text_chars} characters before "
                "the cut), so Agent 3 did not see the whole policy and "
                "'undeclared' findings may be wrong."
            )

    # Agent 3 evidence gate.
    gate = state.evidence_gate
    gate_counts = None
    gate_warnings: list[str] = []

    if gate is not None:
        gate_counts = {
            str(key): int(value)
            for key, value in (gate.get("counts") or {}).items()
        }
        gate_warnings = [
            str(warning) for warning in (gate.get("warnings") or [])
        ]

        blocked = gate_counts.get("blocked", 0)

        if blocked:
            reasons.append(
                f"{blocked} finding(s) were blocked by the evidence gate: "
                "the scan behind them was incomplete or a cited fact was "
                "missing, so they are not reported as gaps."
            )

    # Agent 4, legal gate, scoring and Agent 5.
    legal_status_counts = None
    legal_failed_llm_calls = None
    legal_retrieval_failures = None
    legal_gate_changes = None
    score_available = None
    score_value = None
    score_withheld_reason = None
    report_generated = None

    if _final_stage_started(state):
        original = state.legal_analysis_original
        gated = state.legal_analysis_gated

        if original is None:
            reasons.append(
                "Agent 4 produced no legal analysis result."
            )
        else:
            legal_findings = original.get("findings", [])

            legal_status_counts = {}
            for finding in legal_findings:
                status = str(finding.get("status", "unknown"))
                legal_status_counts[status] = (
                    legal_status_counts.get(status, 0) + 1
                )

            legal_failed_llm_calls = len(
                original.get("failed_calls", [])
            )
            legal_retrieval_failures = len(
                original.get("retrieval_failures", [])
            )

            if not legal_findings:
                reasons.append(
                    "Agent 4 returned no legal findings, so no legal "
                    "outcomes were assessed."
                )

            if legal_failed_llm_calls:
                reasons.append(
                    f"Agent 4 recorded {legal_failed_llm_calls} "
                    "failed LLM call(s)."
                )

            if legal_retrieval_failures:
                reasons.append(
                    f"Agent 4 recorded {legal_retrieval_failures} "
                    "retrieval failure(s)."
                )

        if gated is None:
            reasons.append(
                "No gated legal result is available."
            )

        if state.legal_gate_application is not None:
            changes = state.legal_gate_application.get("changes", [])
            legal_gate_changes = len(changes)

        score = state.compliance_score

        if score is None:
            reasons.append(
                "No compliance scoring result is available."
            )
        else:
            score_value = score.get("score")
            score_available = score_value is not None
            score_withheld_reason = score.get("score_withheld_reason")

        report_generated = state.compliance_report is not None

        if not report_generated:
            reasons.append(
                "Agent 5 did not produce a compliance report."
            )

    return AssessmentSummary(
        assessment_id=state.request.assessment_id,
        outcome_counts=outcome_counts,
        insufficient_by_reason=insufficient_by_reason,
        decided_percent=decided_percent,
        policy_text_chars=text_chars,
        policy_text_truncated=truncated,
        implementation_llm_failures=implementation_failures,
        alignment_llm_failures=alignment_failures,
        alignment_failed_calls=failed_calls,
        alignment_unknown_verdict_ids=unknown_verdict_ids,
        documents_without_claims=missing_documents,
        gate_counts=gate_counts,
        gate_warnings=gate_warnings,
        legal_status_counts=legal_status_counts,
        legal_failed_llm_calls=legal_failed_llm_calls,
        legal_retrieval_failures=legal_retrieval_failures,
        legal_gate_changes=legal_gate_changes,
        compliance_score_available=score_available,
        compliance_score=score_value,
        score_withheld_reason=score_withheld_reason,
        report_generated=report_generated,
        failed_steps=failed_steps,
        warnings=warnings,
        incomplete_reasons=reasons,
        is_complete=not reasons,
    )
