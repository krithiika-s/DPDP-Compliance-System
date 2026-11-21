import json

from agents.alignment_agent.schemas import (
    AlignmentFinding,
    AlignmentResult,
    AlignmentStatus,
)
from agents.implementation_agent.schemas import (
    CodeLocation,
    EvidenceItem,
    ImplementationFact,
    ImplementationReport,
)
from agents.legal_rag.agent import LegalAnalysisAgent
from agents.legal_rag.schemas import (
    LegalAnalysisRequest,
    LegalAssessmentStatus,
    RetrievedLegalChunk,
)
from agents.policy_agent.schemas import PolicyCategory, PolicyClaim


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query, *, k=5):
        return self.chunks


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def generate(self, **kwargs):
        return json.dumps(self.response)


def make_request(status=AlignmentStatus.VIOLATED, include_fact=True, fact_type='partial'):
    claim = PolicyClaim(
        claim="We obtain consent before processing personal data.",
        category=PolicyCategory.CONSENT,
        evidence="Privacy policy",
        source="privacy.html",
    )

    alignment = AlignmentFinding(
        id="A1",
        kind="claim_check",
        category=PolicyCategory.CONSENT,
        status=status,
        claim=claim,
        fact_ids=["F1"] if include_fact else [],
        reasoning="Consent implementation needs review.",
        confidence=0.9,
    )

    evidence = EvidenceItem(
        id="E1",
        kind="consent_check",
        location=CodeLocation(file="app.py", line=10),
        snippet="user = create_user(email)",
    )

    fact = ImplementationFact(
        id="F1",
        category=PolicyCategory.CONSENT,
        statement="The consent control is present, but does not clearly record an affirmative consent action.",
        fact_type=fact_type,
        evidence_ids=["E1"],
        locations=[CodeLocation(file="app.py", line=10)],
    )

    report = ImplementationReport(
        project_path="sample",
        files_scanned=1,
        coverage={},
        evidence=[evidence],
        facts=[fact] if include_fact else [],
    )

    return LegalAnalysisRequest(
        alignment_result=AlignmentResult(
            findings=[alignment],
            summary={"violated": 1},
        ),
        implementation_report=report,
    )


def make_chunk():
    return RetrievedLegalChunk(
        source_id="LEGAL-act-P6-R1",
        text="The consent given by the Data Principal shall be free, specific, informed, unconditional and unambiguous with a clear affirmative action.",
        source_file="dpdp_act.pdf",
        document_type="act",
        authority="primary",
        pdf_page=6,
        provision="Section 6",
    )


def valid_response():
    return {
        "status": "potential_non_compliance",
        "applicable_provisions": ["Section 6"],
        "legal_reasoning": "Section 6 requires consent to be free, specific, informed, unconditional and unambiguous, with a clear affirmative action.",
        "evidence_fact_ids": ["F1"],
        "citations": [{
            "source_id": "LEGAL-act-P6-R1",
            "provision": "Section 6",
            "document_type": "act",
            "source_file": "dpdp_act.pdf",
            "pdf_page": 6,
            "excerpt": "The consent given by the Data Principal shall be free, specific, informed, unconditional and unambiguous with a clear affirmative action.",
        }],
        "confidence": 0.7,
        "limitations": ["The supplied implementation evidence does not establish every aspect of the consent process."],
    }


def test_legal_agent_returns_validated_finding():
    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(valid_response()),
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert len(result.findings) == 1
    assert result.findings[0].status == LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE
    assert result.findings[0].citations[0].provision == "Section 6"
    assert result.findings[0].evidence_fact_ids == ["F1"]
    assert result.sources_consulted == ["dpdp_act.pdf"]


def test_legal_agent_rejects_fabricated_citation():
    response = valid_response()
    response["citations"][0]["excerpt"] = "This text does not exist in the law."

    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(response),
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert "could not be validated" in result.findings[0].legal_reasoning


def test_legal_agent_handles_missing_implementation_facts():
    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(valid_response()),
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request(include_fact=False))

    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert "No verifiable implementation fact" in result.findings[0].legal_reasoning


def test_out_of_scope_finding_does_not_call_llm_or_retriever():
    class MustNotBeCalled:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Retriever should not be called")

    class MustNotGenerate:
        def generate(self, **kwargs):
            raise AssertionError("LLM should not be called")

    agent = LegalAnalysisAgent(
        llm_client=MustNotGenerate(),
        retriever=MustNotBeCalled(),
    )

    result = agent.run(make_request(status=AlignmentStatus.OUT_OF_SCOPE))

    assert result.findings[0].status == LegalAssessmentStatus.NOT_APPLICABLE



class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return json.dumps(response)


def test_forced_statuses_skip_llm_and_retriever():
    class MustNotBeCalled:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Retriever should not be called")

        def generate(self, **kwargs):
            raise AssertionError("LLM should not be called")

    forced_statuses = [
        AlignmentStatus.UNVERIFIABLE,
        AlignmentStatus.UNDECLARED,
        AlignmentStatus.INSUFFICIENT_EVIDENCE,
    ]

    for status in forced_statuses:
        agent = LegalAnalysisAgent(
            llm_client=MustNotBeCalled(),
            retriever=MustNotBeCalled(),
        )

        result = agent.run(make_request(status=status))

        expected = (
            LegalAssessmentStatus.REQUIRES_REVIEW
            if status == AlignmentStatus.UNDECLARED
            else LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
        )
        assert result.findings[0].status == expected
        assert result.failed_calls == []



def test_absence_only_fact_requires_review_without_llm_or_retrieval():
    class MustNotBeCalled:
        def retrieve(self, *args, **kwargs):
            raise AssertionError("Retriever must not be called for absence-only evidence")

        def generate(self, **kwargs):
            raise AssertionError("LLM must not be called for absence-only evidence")

    agent = LegalAnalysisAgent(
        llm_client=MustNotBeCalled(),
        retriever=MustNotBeCalled(),
    )

    result = agent.run(make_request(fact_type="absent"))

    assert result.findings[0].status == LegalAssessmentStatus.REQUIRES_REVIEW
    assert result.failed_calls == []
    assert result.retrieval_failures == []

def test_validation_failure_then_successful_retry():
    invalid_response = valid_response()
    invalid_response["citations"][0]["excerpt"] = "Fabricated legal text."

    llm = SequenceLLM([invalid_response, valid_response()])

    agent = LegalAnalysisAgent(
        llm_client=llm,
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert llm.calls == 2
    assert result.findings[0].status == LegalAssessmentStatus.POTENTIAL_NON_COMPLIANCE
    assert len(result.failed_calls) == 1
    assert result.failed_calls[0].alignment_finding_id == "A1"
    assert result.failed_calls[0].attempt == 1
    assert result.failed_calls[0].error_type == "ValueError"


def test_two_validation_failures_return_insufficient_evidence():
    invalid_response = valid_response()
    invalid_response["citations"][0]["excerpt"] = "Fabricated legal text."

    llm = SequenceLLM([invalid_response, invalid_response])

    agent = LegalAnalysisAgent(
        llm_client=llm,
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert llm.calls == 2
    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert [call.attempt for call in result.failed_calls] == [1, 2]
    assert all(call.error_type == "ValueError" for call in result.failed_calls)


def test_api_failure_is_recorded_without_retry():
    llm = SequenceLLM([RuntimeError("Network connection failed")])

    agent = LegalAnalysisAgent(
        llm_client=llm,
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert llm.calls == 1
    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert len(result.failed_calls) == 1
    assert result.failed_calls[0].alignment_finding_id == "A1"
    assert result.failed_calls[0].attempt == 1
    assert result.failed_calls[0].error_type == "RuntimeError"


def test_api_failure_stops_llm_calls_for_remaining_findings():
    request = make_request()
    second_finding = request.alignment_result.findings[0].model_copy(
        update={"id": "A2"}
    )
    request.alignment_result.findings.append(second_finding)

    llm = SequenceLLM([RuntimeError("API unavailable")])

    agent = LegalAnalysisAgent(
        llm_client=llm,
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(request)

    assert llm.calls == 1
    assert len(result.findings) == 2
    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert result.findings[1].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert len(result.failed_calls) == 1
    assert "stopped" in result.findings[1].legal_reasoning.lower()


def test_retrieval_failure_is_recorded():
    class FailingRetriever:
        def retrieve(self, query, *, k=5):
            raise ConnectionError("Legal database unavailable")

    agent = LegalAnalysisAgent(
        llm_client=FakeLLM(valid_response()),
        retriever=FailingRetriever(),
    )

    result = agent.run(make_request())

    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert len(result.retrieval_failures) == 1
    assert result.retrieval_failures[0].alignment_finding_id == "A1"
    assert result.retrieval_failures[0].error_type == "ConnectionError"


def test_legal_conclusion_without_linked_fact_is_rejected():
    response = valid_response()
    response["evidence_fact_ids"] = []

    llm = SequenceLLM([response, response])

    agent = LegalAnalysisAgent(
        llm_client=llm,
        retriever=FakeRetriever([make_chunk()]),
    )

    result = agent.run(make_request())

    assert llm.calls == 2
    assert result.findings[0].status == LegalAssessmentStatus.INSUFFICIENT_EVIDENCE
    assert len(result.failed_calls) == 2
    assert all(
        call.error_type == "ValueError"
        for call in result.failed_calls
    )
