"""Test that the fake run script works end to end. No model, no network."""

import json

from scripts.run_assessment import main


def test_fake_run_writes_a_valid_audit_record(tmp_path):
    path = main(tmp_path / "audit")

    assert path.exists()
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["assessment_id"] == "demo-run"
    assert record["policy_documents"][0]["name"] == "demo_policy.txt"

    # Verify Agent 3's important output fields.
    alignment = record["alignment_result"]
    assert alignment["findings"] == []
    assert alignment["llm_failures"] == 0
    assert alignment["failed_calls"] == []

    # Verify the final pipeline stages are included in the audit.
    assert record["legal_analysis_original"] is not None
    assert record["legal_analysis_gated"] is not None
    assert record["compliance_score"] is not None
    assert record["compliance_report"] is not None

    steps = {
        entry["step"]: entry["status"]
        for entry in record["step_log"]
    }

    assert steps["agent3_alignment"] == "completed"
    assert steps["legal_analysis"] == "completed"
    assert steps["compliance_scoring"] == "completed"
    assert steps["report_generation"] == "completed"

    # Raw policy text must not be stored in the document metadata.
    assert "We keep your data for two years." not in json.dumps(
        record["policy_documents"]
    )