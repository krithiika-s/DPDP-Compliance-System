
from types import SimpleNamespace

from agents.alignment_agent.schemas import AlignmentResult
from agents.implementation_agent.schemas import ImplementationReport
from agents.legal_rag.schemas import (
    LegalAnalysisResult,
    LegalAssessmentStatus,
    LegalFinding,
)
from agents.report_agent.schemas import ComplianceReport
from agents.scoring.schemas import ComplianceScore
from orchestrator.final_assessment_stage import run_final_assessment_stage
from orchestrator.models import (
    AssessmentRequest,
    AssessmentState,
    PolicyDocument,
    StepStatus,
)


def make_state(*, evidence_gate=None):
    implementation = ImplementationReport(
        project_path="tests/fixtures/apps/acmenotes_violating",
        files_scanned=0,
        coverage={},
        evidence=[],
        facts=[],
    )

    alignment = AlignmentResult(
        findings=[],
        summary={},
    )

    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="test-assessment",
            policy_documents=[
                PolicyDocument(
                    name="test-policy.txt",
                    text="We keep user data only for two years.",
                )
            ],
            project_path="tests/fixtures/apps/acmenotes_violating",
        )
    )

    state.implementation_report = implementation.model_dump(mode="json")
    state.alignment_result = alignment.model_dump(mode="json")
    state.evidence_gate = evidence_gate
    return state


def make_legal_result():
    finding = LegalFinding(
        alignment_finding_id="finding-1",
        status=LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE,
        applicable_provisions=[],
        legal_reasoning="A potential issue was identified.",
        evidence_fact_ids=["fact-1"],
        citations=[],
        confidence=0.8,
        limitations=[],
    )

    return LegalAnalysisResult(
        findings=[finding],
        summary={},
        sources_consulted=[],
        failed_calls=[],
        retrieval_failures=[],
    )


def make_score():
    return ComplianceScore(
        score=None,
        total_findings=1,
        assessed_findings=0,
        potential_compliance_count=0,
        potential_non_compliance_count=0,
        insufficient_evidence_count=1,
        requires_review_count=0,
        not_applicable_count=0,
        assessment_coverage_percent=0.0,
        minimum_coverage_percent=50.0,
        score_withheld_reason="Insufficient assessed findings.",
        score_description="Provisional assessment score.",
    )


class FakeLegalAgent:
    def __init__(self, result=None, error=None):
        self.result = result or make_legal_result()
        self.error = error
        self.called = False

    def run(self, request):
        self.called = True
        if self.error:
            raise self.error
        return self.result


class FakeScoringEngine:
    def __init__(self, error=None):
        self.error = error
        self.called = False
        self.received_result = None

    def calculate(self, result):
        self.called = True
        self.received_result = result
        if self.error:
            raise self.error
        return make_score()


class FakeReportAgent:
    def __init__(self, error=None):
        self.error = error
        self.called = False
        self.received_request = None

    def generate(self, request):
        self.called = True
        self.received_request = request

        if self.error:
            raise self.error

        return ComplianceReport(
            assessment_id=request.assessment_id,
            score=request.score,
            findings=request.legal_analysis.findings,
            recommendations=[],
            sources_consulted=[],
            executive_summary="Test assessment completed.",
            disclaimer="For informational purposes only; not legal advice.",
            failed_calls=[],
            retrieval_failures=[],
        )


def fake_gate_application(result, changes=None):
    return SimpleNamespace(
        result=result,
        changes=changes or [],
        to_dict=lambda: {"changes": changes or []},
    )


def test_successful_final_pipeline(monkeypatch):
    import orchestrator.final_assessment_stage as stage

    state = make_state()
    legal_agent = FakeLegalAgent()
    scoring_engine = FakeScoringEngine()
    report_agent = FakeReportAgent()

    monkeypatch.setattr(
        stage,
        "apply_gate",
        lambda legal_result, gate: fake_gate_application(legal_result),
    )

    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    assert legal_agent.called
    assert scoring_engine.called
    assert report_agent.called

    assert state.legal_analysis_original is not None
    assert state.legal_analysis_gated is not None
    assert state.compliance_score is not None
    assert state.compliance_report is not None

    assert scoring_engine.received_result.findings[0].status == (
        LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE
    )

    assert [entry.step for entry in state.step_log] == [
        "legal_analysis",
        "legal_gate",
        "compliance_scoring",
        "report_generation",
    ]

    assert all(
        entry.status == StepStatus.COMPLETED
        for entry in state.step_log
    )


def test_blocked_finding_is_downgraded_before_scoring(monkeypatch):
    import orchestrator.final_assessment_stage as stage

    state = make_state(evidence_gate={"decisions": []})
    legal_agent = FakeLegalAgent()
    scoring_engine = FakeScoringEngine()
    report_agent = FakeReportAgent()

    def blocked_gate(legal_result, gate):
        gated = dict(legal_result)
        gated["findings"] = [
            {
                **legal_result["findings"][0],
                "status": "insufficient_evidence",
                "limitations": ["Blocked by evidence gate."],
            }
        ]
        changes = [{"finding_id": "finding-1"}]
        return fake_gate_application(gated, changes)

    monkeypatch.setattr(stage, "apply_gate", blocked_gate)

    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    assert state.legal_analysis_original["findings"][0]["status"] == (
        "potential_non_compliance"
    )
    assert state.legal_analysis_gated["findings"][0]["status"] == (
        "insufficient_evidence"
    )

    assert scoring_engine.received_result.findings[0].status == (
        LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    )

    assert report_agent.received_request.legal_analysis.findings[0].status == (
        LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    )


def test_missing_evidence_gate_uses_safe_gate_call(monkeypatch):
    import orchestrator.final_assessment_stage as stage

    state = make_state(evidence_gate=None)
    legal_agent = FakeLegalAgent()
    scoring_engine = FakeScoringEngine()
    report_agent = FakeReportAgent()
    observed = {}

    def missing_gate(legal_result, gate):
        observed["gate"] = gate
        gated = dict(legal_result)
        gated["findings"] = [
            {
                **legal_result["findings"][0],
                "status": "requires_review",
                "limitations": ["Evidence gate was unavailable."],
            }
        ]
        return fake_gate_application(gated)

    monkeypatch.setattr(stage, "apply_gate", missing_gate)

    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    assert observed["gate"] is None
    assert state.legal_analysis_gated["findings"][0]["status"] == (
        "requires_review"
    )
    assert scoring_engine.called
    assert report_agent.called


def test_legal_agent_failure_stops_later_stages():
    state = make_state()
    legal_agent = FakeLegalAgent(
        error=RuntimeError("Simulated Agent 4 failure")
    )
    scoring_engine = FakeScoringEngine()
    report_agent = FakeReportAgent()

    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    assert legal_agent.called
    assert not scoring_engine.called
    assert not report_agent.called

    assert state.legal_analysis_original is None
    assert state.compliance_score is None
    assert state.compliance_report is None

    assert state.step_log[-1].step == "legal_analysis"
    assert state.step_log[-1].status == StepStatus.FAILED


def test_scoring_failure_stops_report_generation(monkeypatch):
    import orchestrator.final_assessment_stage as stage

    state = make_state()
    legal_agent = FakeLegalAgent()
    scoring_engine = FakeScoringEngine(
        error=RuntimeError("Simulated scoring failure")
    )
    report_agent = FakeReportAgent()

    monkeypatch.setattr(
        stage,
        "apply_gate",
        lambda legal_result, gate: fake_gate_application(legal_result),
    )

    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    assert legal_agent.called
    assert scoring_engine.called
    assert not report_agent.called

    assert state.compliance_score is None
    assert state.compliance_report is None
    assert state.step_log[-1].step == "compliance_scoring"
    assert state.step_log[-1].status == StepStatus.FAILED


def test_report_failure_is_recorded():
    state = make_state()
    legal_agent = FakeLegalAgent()
    scoring_engine = FakeScoringEngine()
    report_agent = FakeReportAgent(
        error=RuntimeError("Simulated Agent 5 failure")
    )

    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    assert legal_agent.called
    assert scoring_engine.called
    assert report_agent.called

    assert state.compliance_score is not None
    assert state.compliance_report is None
    assert state.step_log[-1].step == "report_generation"
    assert state.step_log[-1].status == StepStatus.FAILED


def test_real_evidence_gate_downgrades_before_scoring_and_reporting():
    state = make_state(
        evidence_gate={
            "decisions": [
                {
                    "finding_id": "finding-1",
                    "decision": "blocked",
                    "reasons": ["Required evidence was not available."],
                }
            ]
        }
    )

    legal_agent = FakeLegalAgent()
    scoring_engine = FakeScoringEngine()
    report_agent = FakeReportAgent()

    # Do not mock apply_gate here.
    # This test exercises the actual legal gate implementation.
    run_final_assessment_stage(
        state,
        legal_agent,
        scoring_engine,
        report_agent,
    )

    # Original Agent 4 result must remain unchanged.
    assert state.legal_analysis_original["findings"][0]["status"] == (
        "potential_non_compliance"
    )

    # The real evidence gate must downgrade the finding.
    assert state.legal_analysis_gated["findings"][0]["status"] == (
        "insufficient_evidence"
    )

    # Scoring must receive the downgraded result.
    assert scoring_engine.received_result.findings[0].status == (
        LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    )

    # Agent 5 must also receive the downgraded result.
    assert report_agent.received_request.legal_analysis.findings[0].status == (
        LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    )

    assert len(state.legal_gate_application["changes"]) == 1
    assert state.step_log[-1].status == StepStatus.COMPLETED
