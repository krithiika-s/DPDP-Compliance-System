
"""Final assessment stages: legal reasoning, legal gate, scoring and report.

All agents and the scoring engine are injected. This keeps the orchestrator
independent of concrete agent constructors and allows tests to use fakes.
"""

from typing import Any

from agents.alignment_agent.schemas import AlignmentResult
from agents.implementation_agent.schemas import ImplementationReport
from agents.legal_rag.schemas import (
    LegalAnalysisRequest,
    LegalAnalysisResult,
)
from agents.report_agent.schemas import ReportGenerationRequest
from agents.scoring.schemas import ComplianceScore
from orchestrator.legal_gate import apply_gate
from orchestrator.models import AssessmentState, StepStatus

LEGAL_ANALYSIS_STEP = "legal_analysis"
LEGAL_GATE_STEP = "legal_gate"
SCORING_STEP = "compliance_scoring"
REPORT_STEP = "report_generation"


def _log_failure(state: AssessmentState, step: str, exc: Exception) -> None:
    """Record a stage failure without exposing sensitive data."""
    state.log(
        step,
        StepStatus.FAILED,
        message=f"{type(exc).__name__}: {str(exc)[:300]}",
    )


def run_final_assessment_stage(
    state: AssessmentState,
    legal_agent: Any,
    scoring_engine: Any,
    report_agent: Any,
) -> None:
    """Run Agent 4, legal gate, deterministic scoring and Agent 5.

    Results are stored as JSON-friendly dictionaries in AssessmentState.
    The original and gated legal results are both preserved for audit.

    If a stage fails, later stages are not run. A missing evidence-gate
    result is passed to apply_gate as None, which safely downgrades
    potential non-compliance findings to requires_review.
    """

    # The final stages require both upstream agent results.
    if state.implementation_report is None:
        state.log(
            LEGAL_ANALYSIS_STEP,
            StepStatus.FAILED,
            message="Agent 2 produced no implementation report; legal analysis was skipped.",
        )
        return

    if state.alignment_result is None:
        state.log(
            LEGAL_ANALYSIS_STEP,
            StepStatus.FAILED,
            message="Agent 3 produced no alignment result; legal analysis was skipped.",
        )
        return

    # Convert the dictionaries stored by the orchestrator into the exact
    # Pydantic contracts required by Agent 4.
    try:
        implementation = ImplementationReport.model_validate(
            state.implementation_report
        )
        alignment = AlignmentResult.model_validate(state.alignment_result)
        request = LegalAnalysisRequest(
            alignment_result=alignment,
            implementation_report=implementation,
        )
    except Exception as exc:
        _log_failure(state, LEGAL_ANALYSIS_STEP, exc)
        return

    # Stage 1: Agent 4 legal reasoning.
    try:
        legal_result = legal_agent.run(request)
        if not isinstance(legal_result, LegalAnalysisResult):
            legal_result = LegalAnalysisResult.model_validate(legal_result)

        original_data = legal_result.model_dump(mode="json")
        state.legal_analysis_original = original_data
        state.log(LEGAL_ANALYSIS_STEP, StepStatus.COMPLETED)
    except Exception as exc:
        _log_failure(state, LEGAL_ANALYSIS_STEP, exc)
        return

    # Stage 2: Apply the evidence gate to Agent 4's result.
    try:
        gate_application = apply_gate(
            legal_result=state.legal_analysis_original,
            gate=state.evidence_gate,
        )

        gated_result = LegalAnalysisResult.model_validate(
            gate_application.result
        )

        state.legal_analysis_gated = gated_result.model_dump(mode="json")
        state.legal_gate_application = gate_application.to_dict()

        message = ""
        if gate_application.changes:
            message = (
                f"{len(gate_application.changes)} legal finding(s) "
                "were downgraded by the evidence gate."
            )
        elif state.evidence_gate is None:
            message = (
                "Evidence gate result was missing; safe defaults were applied."
            )

        state.log(
            LEGAL_GATE_STEP,
            StepStatus.COMPLETED,
            message=message,
        )
    except Exception as exc:
        _log_failure(state, LEGAL_GATE_STEP, exc)
        return

    # Stage 3: Score only the gated legal result.
    try:
        score = scoring_engine.calculate(gated_result)
        if not isinstance(score, ComplianceScore):
            score = ComplianceScore.model_validate(score)

        state.compliance_score = score.model_dump(mode="json")
        state.log(SCORING_STEP, StepStatus.COMPLETED)
    except Exception as exc:
        _log_failure(state, SCORING_STEP, exc)
        return

    # Stage 4: Agent 5 receives the gated findings and the score.
    try:
        report_request = ReportGenerationRequest(
            assessment_id=state.request.assessment_id,
            legal_analysis=gated_result,
            score=score,
        )
        report = report_agent.generate(report_request)

        # Validate the returned object before storing it.
        from agents.report_agent.schemas import ComplianceReport

        if not isinstance(report, ComplianceReport):
            report = ComplianceReport.model_validate(report)

        state.compliance_report = report.model_dump(mode="json")
        state.log(REPORT_STEP, StepStatus.COMPLETED)
    except Exception as exc:
        _log_failure(state, REPORT_STEP, exc)
        return
