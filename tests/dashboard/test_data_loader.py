
import io
import json

import pytest

from dashboard.data_loader import (
    load_audit_json,
    get_nested,
    safe_list,
    safe_dict,
    display_value,
)


SAMPLE_AUDIT = {
    "assessment_id": "test-001",
    "summary": {
        "is_complete": True,
        "outcome_counts": {
            "compliant": 1,
            "gap": 2,
        },
    },
}


def test_load_audit_json_from_file(tmp_path):
    file_path = tmp_path / "audit.json"
    file_path.write_text(json.dumps(SAMPLE_AUDIT), encoding="utf-8")

    result = load_audit_json(file_path)

    assert result["assessment_id"] == "test-001"


def test_load_audit_json_from_uploaded_file():
    uploaded_file = io.BytesIO(json.dumps(SAMPLE_AUDIT).encode("utf-8"))

    result = load_audit_json(uploaded_file)

    assert result["assessment_id"] == "test-001"


def test_load_audit_json_from_bytes():
    result = load_audit_json(json.dumps(SAMPLE_AUDIT).encode("utf-8"))

    assert result["assessment_id"] == "test-001"


def test_empty_json_raises_error():
    with pytest.raises(ValueError, match="empty"):
        load_audit_json("")


def test_invalid_json_raises_error():
    with pytest.raises(ValueError, match="Invalid audit JSON"):
        load_audit_json(b"{invalid json}")


def test_non_dictionary_json_raises_error():
    with pytest.raises(ValueError, match="JSON object"):
        load_audit_json(b'["item1", "item2"]')


def test_get_nested_existing_value():
    result = get_nested(SAMPLE_AUDIT, "summary", "outcome_counts", "gap")

    assert result == 2


def test_get_nested_missing_value():
    result = get_nested(
        SAMPLE_AUDIT,
        "summary",
        "missing_field",
        default="Not available",
    )

    assert result == "Not available"


def test_safe_list():
    assert safe_list([1, 2]) == [1, 2]
    assert safe_list(None) == []


def test_safe_dict():
    assert safe_dict({"a": 1}) == {"a": 1}
    assert safe_dict(None) == {}


def test_display_value():
    assert display_value(None) == "Not available"
    assert display_value("") == "Not available"
    assert display_value("Complete") == "Complete"
