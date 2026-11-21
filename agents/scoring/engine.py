from agents.legal_rag.schemas import LegalAnalysisResult, LegalAssessmentStatus
from agents.scoring.schemas import ComplianceScore


class ComplianceScoringEngine:
    """Calculate a provisional indicator with a minimum coverage safeguard."""

    def __init__(self, min_coverage_percent: float = 50.0) -> None:
        if not 0 <= min_coverage_percent <= 100:
            raise ValueError("min_coverage_percent must be between 0 and 100.")
        self.min_coverage_percent = float(min_coverage_percent)

    def calculate(self, result: LegalAnalysisResult) -> ComplianceScore:
        findings = result.findings

        counts = {
            status: sum(1 for finding in findings if finding.status == status)
            for status in LegalAssessmentStatus
        }

        compliant = counts[LegalAssessmentStatus.POTENTIAL_COMPLIANCE]
        non_compliant = counts[LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE]
        insufficient = counts[LegalAssessmentStatus.INSUFFICIENT_EVIDENCE]
        review = counts[LegalAssessmentStatus.REQUIRES_REVIEW]
        not_applicable = counts[LegalAssessmentStatus.NOT_APPLICABLE]

        assessed = compliant + non_compliant
        applicable = len(findings) - not_applicable

        coverage = (
            round((assessed / applicable) * 100, 2)
            if applicable else None
        )

        # Preserve the existing formula. Withhold its result when coverage
        # is too low to present a meaningful provisional indicator.
        raw_score = round((compliant / assessed) * 100, 2) if assessed else None

        if assessed == 0:
            score = None
            withheld_reason = "No findings have a legally assessable outcome."
        elif coverage is not None and coverage < self.min_coverage_percent:
            score = None
            withheld_reason = (
                f"Assessment coverage is {coverage:.2f}%, below the required "
                f"minimum of {self.min_coverage_percent:.2f}%."
            )
        else:
            score = raw_score
            withheld_reason = None

        return ComplianceScore(
            score=score,
            total_findings=len(findings),
            assessed_findings=assessed,
            potential_compliance_count=compliant,
            potential_non_compliance_count=non_compliant,
            insufficient_evidence_count=insufficient,
            requires_review_count=review,
            not_applicable_count=not_applicable,
            assessment_coverage_percent=coverage,
            minimum_coverage_percent=self.min_coverage_percent,
            score_withheld_reason=withheld_reason,
            score_description=(
                "This is a provisional ratio of potential-compliance findings "
                "to legally assessed findings only. It is not an overall DPDP "
                "compliance percentage, legal certification, or legal advice. "
                "Insufficient-evidence, requires-review, and not-applicable "
                "findings are excluded from the score."
            ),
        )
