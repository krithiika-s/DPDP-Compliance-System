from agents.legal_rag.schemas import LegalAnalysisResult, LegalAssessmentStatus as S
from agents.report_agent.schemas import (
    ComplianceReport,
    ReportGenerationRequest,
    ReportRecommendation,
)


class ReportGenerationAgent:
    """Agent 5: builds a structured report from validated assessment results."""

    DISCLAIMER = (
        "This report is an automated informational assessment, not legal advice "
        "or a legal certification. Its provisional score is not an overall DPDP "
        "compliance percentage. Findings and source citations should be reviewed "
        "by qualified personnel."
    )

    def generate(self, request: ReportGenerationRequest) -> ComplianceReport:
        if not isinstance(request.legal_analysis, LegalAnalysisResult):
            raise TypeError("legal_analysis must be a LegalAnalysisResult.")

        analysis = request.legal_analysis
        recommendations = []

        for finding in analysis.findings:
            if finding.status == S.POTENTIAL_NON_COMPLIANCE:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="high",
                    action="Review the identified implementation gap and plan corrective measures.",
                    reason=finding.legal_reasoning,
                ))
            elif finding.status == S.INSUFFICIENT_EVIDENCE:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="medium",
                    action="Collect additional implementation evidence and repeat the assessment.",
                    reason=finding.legal_reasoning,
                ))
            elif finding.status == S.REQUIRES_REVIEW:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="high",
                    action="Refer this finding for qualified legal or compliance review.",
                    reason=finding.legal_reasoning,
                ))
            elif finding.status == S.POTENTIAL_COMPLIANCE:
                recommendations.append(ReportRecommendation(
                    finding_id=finding.alignment_finding_id,
                    priority="low",
                    action="Retain supporting evidence and periodically revalidate the implementation.",
                    reason=finding.legal_reasoning,
                ))

        score = request.score
        coverage_text = (
            f"{score.assessment_coverage_percent:.2f}%"
            if score.assessment_coverage_percent is not None
            else "unavailable"
        )

        failure_text = (
            f" Recorded failures: {len(analysis.failed_calls)} failed LLM calls "
            f"and {len(analysis.retrieval_failures)} retrieval failures."
        )

        if score.score is None:
            reason = score.score_withheld_reason or (
                "No provisional score was supplied."
            )
            summary = (
                f"The provisional score was withheld. Reason: {reason} "
                f"Assessment coverage: {coverage_text}. "
                f"{score.assessed_findings} findings were legally assessed; "
                f"{score.requires_review_count} require review and "
                f"{score.insufficient_evidence_count} have insufficient evidence."
                f"{failure_text}"
            )
        else:
            summary = (
                f"The provisional assessed-findings indicator is {score.score:.2f}%. "
                "It is not an overall DPDP compliance percentage. "
                f"Assessment coverage: {coverage_text}. "
                f"{score.assessed_findings} findings were legally assessed; "
                f"{score.requires_review_count} require review and "
                f"{score.insufficient_evidence_count} have insufficient evidence."
                f"{failure_text}"
            )

        return ComplianceReport(
            assessment_id=request.assessment_id,
            score=score,
            findings=analysis.findings,
            recommendations=recommendations,
            sources_consulted=analysis.sources_consulted,
            executive_summary=summary,
            disclaimer=self.DISCLAIMER,
            failed_calls=analysis.failed_calls,
            retrieval_failures=analysis.retrieval_failures,
        )
