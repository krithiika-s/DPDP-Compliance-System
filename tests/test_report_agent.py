from agents.legal_rag.schemas import (
    FailedLLMCall,
    FailedRetrieval,
    LegalAnalysisResult,
    LegalAssessmentStatus as S,
    LegalFinding,
)
from agents.scoring.schemas import ComplianceScore
from agents.report_agent.schemas import ReportGenerationRequest
from agents.report_agent.agent import ReportGenerationAgent


def test_report_contains_findings_recommendations_and_score():
    analysis = LegalAnalysisResult(
        findings=[
            LegalFinding(
                alignment_finding_id="A1",
                status=S.POTENTIAL_NON_COMPLIANCE,
                legal_reasoning="A potential implementation gap was identified.",
            ),
            LegalFinding(
                alignment_finding_id="A2",
                status=S.INSUFFICIENT_EVIDENCE,
                legal_reasoning="More evidence is needed.",
            ),
        ],
        sources_consulted=["dpdp_act.pdf"],
        failed_calls=[
            FailedLLMCall(
                alignment_finding_id="A1",
                attempt=1,
                error_type="RateLimitError",
            )
        ],
        retrieval_failures=[
            FailedRetrieval(
                alignment_finding_id="A1",
                error_type="ConnectionError",
            )
        ],
    )

    score = ComplianceScore(
        score=0,
        total_findings=2,
        assessed_findings=1,
        potential_compliance_count=0,
        potential_non_compliance_count=1,
        insufficient_evidence_count=1,
        requires_review_count=0,
        not_applicable_count=0,
        assessment_coverage_percent=50,
        score_description="Test score",
    )

    request = ReportGenerationRequest(
        assessment_id="TEST-001",
        legal_analysis=analysis,
        score=score,
    )

    report = ReportGenerationAgent().generate(request)

    assert report.assessment_id == "TEST-001"
    assert len(report.findings) == 2
    assert len(report.recommendations) == 2
    assert report.recommendations[0].priority == "high"
    assert report.recommendations[1].priority == "medium"
    assert report.sources_consulted == ["dpdp_act.pdf"]
    assert "not legal advice" in report.disclaimer
    assert "50.00%" in report.executive_summary
    assert "0 require review" in report.executive_summary
    assert len(report.failed_calls) == 1
    assert report.failed_calls[0].error_type == "RateLimitError"
    assert len(report.retrieval_failures) == 1
    assert report.retrieval_failures[0].error_type == "ConnectionError"


def test_report_handles_missing_score():
    analysis = LegalAnalysisResult(findings=[])
    score = ComplianceScore(
        score=None,
        total_findings=0,
        assessed_findings=0,
        potential_compliance_count=0,
        potential_non_compliance_count=0,
        insufficient_evidence_count=0,
        requires_review_count=0,
        not_applicable_count=0,
        assessment_coverage_percent=None,
        score_description="No score",
    )

    report = ReportGenerationAgent().generate(
        ReportGenerationRequest(
            assessment_id="EMPTY",
            legal_analysis=analysis,
            score=score,
        )
    )

    assert report.recommendations == []
    assert report.score.score is None
    assert "The provisional score was withheld" in report.executive_summary
    assert "Assessment coverage: unavailable" in report.executive_summary
    assert report.failed_calls == []



def test_not_applicable_finding_is_preserved_without_recommendation():
    not_applicable = LegalFinding(
        alignment_finding_id="A-NOT-APPLICABLE",
        status=S.NOT_APPLICABLE,
        legal_reasoning="This requirement does not apply to the organisation.",
    )
    analysis = LegalAnalysisResult(findings=[not_applicable])

    score = ComplianceScore(
        score=None,
        total_findings=1,
        assessed_findings=0,
        potential_compliance_count=0,
        potential_non_compliance_count=0,
        insufficient_evidence_count=0,
        requires_review_count=0,
        not_applicable_count=1,
        assessment_coverage_percent=None,
        score_description="No applicable findings were assessed.",
    )

    report = ReportGenerationAgent().generate(
        ReportGenerationRequest(
            assessment_id="NOT-APPLICABLE-TEST",
            legal_analysis=analysis,
            score=score,
        )
    )

    assert len(report.findings) == 1
    assert report.findings[0].status == S.NOT_APPLICABLE
    assert report.findings[0].alignment_finding_id == "A-NOT-APPLICABLE"
    assert report.recommendations == []
    assert report.score.not_applicable_count == 1
