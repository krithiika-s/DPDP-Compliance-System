
from collections import Counter
from pathlib import Path

from dashboard.data_loader import load_audit_json


SAMPLE_PATH = (
    Path(__file__).parent / "fixtures" / "sample_audit.json"
)


def load_sample():
    return load_audit_json(SAMPLE_PATH)


def test_sample_contains_six_final_legal_findings():
    data = load_sample()

    findings = data["legal_analysis_gated"]["findings"]

    assert len(findings) == 6


def test_final_legal_finding_status_counts():
    data = load_sample()

    findings = data["legal_analysis_gated"]["findings"]

    status_counts = Counter(
        finding["status"] for finding in findings
    )

    assert status_counts == {
        "not_applicable": 3,
        "insufficient_evidence": 2,
        "requires_review": 1,
    }


def test_compliance_score_is_withheld_not_zero():
    data = load_sample()

    score = data["compliance_score"]

    assert score["score"] is None
    assert (
        score["score_withheld_reason"]
        == "No findings have a legally assessable outcome."
    )


def test_alignment_and_final_legal_findings_are_separate():
    data = load_sample()

    alignment_findings = data["alignment_result"]["findings"]
    legal_findings = data["legal_analysis_gated"]["findings"]

    assert len(alignment_findings) == 6
    assert len(legal_findings) == 6
    assert alignment_findings is not legal_findings
