"""Tests for orchestrator/legal_gate.py. No model, key or network needed."""

import copy

from orchestrator.legal_gate import (
    INSUFFICIENT,
    NON_COMPLIANCE,
    REQUIRES_REVIEW,
    apply_gate,
)


def legal(*findings, summary=None):
    return {
        "findings": list(findings),
        "summary": summary if summary is not None else {},
        "failed_calls": [],
    }


def lf(fid="A-1", status=NON_COMPLIANCE, limitations=None):
    return {
        "alignment_finding_id": fid,
        "status": status,
        "legal_reasoning": "reasoning",
        "limitations": list(limitations or []),
    }


def gate(*entries, warnings=()):
    return {
        "decisions": [
            {"finding_id": fid, "decision": d, "reasons": list(r)}
            for fid, d, r in entries
        ],
        "warnings": list(warnings),
    }


def status_of(application, fid="A-1"):
    for f in application.result["findings"]:
        if f["alignment_finding_id"] == fid:
            return f["status"]
    raise AssertionError(f"{fid} not in result")


def test_blocked_caps_to_insufficient_evidence():
    out = apply_gate(legal(lf()), gate(("A-1", "blocked", ["Scan was cut."])))
    assert status_of(out) == INSUFFICIENT
    assert out.changes[0].old_status == NON_COMPLIANCE
    assert out.changes[0].new_status == INSUFFICIENT


def test_review_only_caps_to_requires_review():
    out = apply_gate(legal(lf()), gate(("A-1", "review_only", ["Absence only."])))
    assert status_of(out) == REQUIRES_REVIEW


def test_pass_and_not_gated_leave_the_status_alone():
    for decision in ("pass", "not_gated"):
        out = apply_gate(legal(lf()), gate(("A-1", decision, [])))
        assert status_of(out) == NON_COMPLIANCE
        assert out.changes == ()


def test_other_statuses_are_never_changed():
    findings = [
        lf("A-1", "potential_compliance"),
        lf("A-2", "not_applicable"),
        lf("A-3", INSUFFICIENT),
        lf("A-4", REQUIRES_REVIEW),
    ]
    decisions = [(f"A-{i}", "blocked", ["x"]) for i in range(1, 5)]
    out = apply_gate(legal(*findings), gate(*decisions))
    assert [f["status"] for f in out.result["findings"]] == [
        "potential_compliance",
        "not_applicable",
        INSUFFICIENT,
        REQUIRES_REVIEW,
    ]
    assert out.changes == ()


def test_gate_reasons_are_added_to_limitations_once():
    out = apply_gate(
        legal(lf(limitations=["Existing limit."])),
        gate(("A-1", "blocked", ["Scan was cut."])),
    )
    limits = out.result["findings"][0]["limitations"]
    assert limits[0] == "Existing limit."
    assert "Evidence gate: Scan was cut." in limits
    assert any("changed from potential_non_compliance" in x for x in limits)
    assert len(limits) == len(set(limits))


def test_already_cautious_finding_still_gets_the_reasons():
    out = apply_gate(
        legal(lf(status=REQUIRES_REVIEW)),
        gate(("A-1", "review_only", ["Absence only."])),
    )
    assert status_of(out) == REQUIRES_REVIEW
    assert "Evidence gate: Absence only." in out.result["findings"][0]["limitations"]
    assert out.changes == ()


def test_missing_gate_caps_non_compliance_to_requires_review():
    out = apply_gate(legal(lf("A-1"), lf("A-2", "potential_compliance")), None)
    assert status_of(out, "A-1") == REQUIRES_REVIEW
    assert status_of(out, "A-2") == "potential_compliance"
    assert "did not run" in out.result["findings"][0]["limitations"][0]


def test_finding_with_no_gate_decision_is_capped_to_requires_review():
    out = apply_gate(legal(lf("A-9")), gate(("A-1", "pass", [])))
    assert status_of(out, "A-9") == REQUIRES_REVIEW
    assert "no decision" in out.result["findings"][0]["limitations"][0]


def test_duplicate_decisions_use_the_worst_one():
    out = apply_gate(
        legal(lf()),
        gate(("A-1", "pass", []), ("A-1", "blocked", ["Cut."])),
    )
    assert status_of(out) == INSUFFICIENT


def test_inputs_are_not_changed():
    legal_in = legal(lf(), summary={NON_COMPLIANCE: 1})
    gate_in = gate(("A-1", "blocked", ["Cut."]))
    before = (copy.deepcopy(legal_in), copy.deepcopy(gate_in))
    apply_gate(legal_in, gate_in)
    assert (legal_in, gate_in) == before


def test_status_summary_is_recounted():
    out = apply_gate(
        legal(
            lf("A-1"),
            lf("A-2", "potential_compliance"),
            summary={NON_COMPLIANCE: 1, "potential_compliance": 1, INSUFFICIENT: 0},
        ),
        gate(("A-1", "blocked", ["Cut."]), ("A-2", "pass", [])),
    )
    assert out.result["summary"] == {
        NON_COMPLIANCE: 0,
        "potential_compliance": 1,
        INSUFFICIENT: 1,
    }


def test_unknown_summary_format_is_left_alone():
    summary = {"total": 3, "assessed": 2}
    out = apply_gate(
        legal(lf(), summary=summary), gate(("A-1", "blocked", ["Cut."]))
    )
    assert out.result["summary"] == summary


def test_empty_findings_and_to_dict():
    out = apply_gate(legal(), gate())
    assert out.result["findings"] == []
    assert out.to_dict() == {"changes": []}


def test_changes_to_dict():
    out = apply_gate(legal(lf()), gate(("A-1", "review_only", ["Absence only."])))
    entry = out.to_dict()["changes"][0]
    assert entry["finding_id"] == "A-1"
    assert entry["new_status"] == REQUIRES_REVIEW
    assert entry["reason"].startswith("Evidence gate: ")

def test_unknown_gate_decision_fails_safe_to_requires_review():
    out = apply_gate(
        legal(lf()),
        {
            "decisions": [
                {
                    "finding_id": "A-1",
                    "decision": "unknown_value",
                    "reasons": [],
                }
            ]
        },
    )
    assert status_of(out) == REQUIRES_REVIEW
    assert any(
        "invalid decision" in limitation
        for limitation in out.result["findings"][0]["limitations"]
    )


def test_missing_decision_field_fails_safe_to_requires_review():
    out = apply_gate(
        legal(lf()),
        {
            "decisions": [
                {
                    "finding_id": "A-1",
                    "reasons": [],
                }
            ]
        },
    )
    assert status_of(out) == REQUIRES_REVIEW


def test_empty_gate_decision_fails_safe_to_requires_review():
    out = apply_gate(
        legal(lf()),
        {
            "decisions": [
                {
                    "finding_id": "A-1",
                    "decision": "",
                    "reasons": [],
                }
            ]
        },
    )
    assert status_of(out) == REQUIRES_REVIEW


def test_none_gate_decision_fails_safe_to_requires_review():
    out = apply_gate(
        legal(lf()),
        {
            "decisions": [
                {
                    "finding_id": "A-1",
                    "decision": None,
                    "reasons": [],
                }
            ]
        },
    )
    assert status_of(out) == REQUIRES_REVIEW


def test_invalid_duplicate_decision_is_worse_than_pass():
    out = apply_gate(
        legal(lf()),
        {
            "decisions": [
                {
                    "finding_id": "A-1",
                    "decision": "pass",
                    "reasons": [],
                },
                {
                    "finding_id": "A-1",
                    "decision": "invalid",
                    "reasons": [],
                },
            ]
        },
    )
    assert status_of(out) == REQUIRES_REVIEW