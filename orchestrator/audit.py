
"""Audit record for a DPDP compliance assessment.

Builds one JSON-friendly record of the assessment and writes it to disk.
Deterministic Python, no LLM.

Rules:
- Policy document names and character counts are recorded, never raw text.
- All available pipeline results are preserved, including final stages.
- Missing results stay None.
- A file is never overwritten.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.models import AssessmentState
from orchestrator.summary import AssessmentSummary

AUDIT_VERSION = 1


def _safe_name(text: str) -> str:
    """Make text safe to use inside a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return cleaned or "assessment"


def build_audit_record(
    state: AssessmentState,
    summary: AssessmentSummary,
    *,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Return a plain dictionary suitable for JSON serialization."""

    created = created_at or datetime.now(timezone.utc)
    request = state.request

    documents = [
        {"name": document.name, "chars": len(document.text)}
        for document in request.policy_documents
    ]

    return {
        "audit_version": AUDIT_VERSION,
        "created_at": created.isoformat(),
        "assessment_id": request.assessment_id,
        "project_path": request.project_path,
        "policy_documents": documents,

        # Agent 1.
        "policy_claims": [
            claim.model_dump(mode="json")
            for claim in state.policy_claims
        ],
        "rejected_claims": [
            claim.model_dump(mode="json")
            for claim in state.rejected_claims
        ],

        # Agents 2 and 3.
        "implementation_report": state.implementation_report,
        "alignment_result": state.alignment_result,
        "evidence_gate": state.evidence_gate,

        # Agent 4 and legal evidence gate.
        "legal_analysis_original": state.legal_analysis_original,
        "legal_analysis_gated": state.legal_analysis_gated,
        "legal_gate_application": state.legal_gate_application,

        # Deterministic scoring and Agent 5.
        "compliance_score": state.compliance_score,
        "compliance_report": state.compliance_report,

        # Execution history and summary.
        "step_log": [
            record.model_dump(mode="json")
            for record in state.step_log
        ],
        "summary": summary.model_dump(mode="json"),
    }


def write_audit_record(
    record: dict[str, Any],
    directory: str | Path,
) -> Path:
    """Write an audit JSON file without overwriting an existing record."""

    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)

    name = (
        f"{_safe_name(str(record['assessment_id']))}"
        f"__{_safe_name(str(record['created_at']))}.json"
    )
    path = folder / name

    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return path
