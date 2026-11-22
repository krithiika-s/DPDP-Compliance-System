
"""Independent, read-only Streamlit dashboard for DPDP audit records."""

import json

import streamlit as st

from dashboard.data_loader import (
    display_value,
    get_nested,
    load_audit_json,
    safe_dict,
    safe_list,
)

from dashboard.pdf_report import make_pdf


st.set_page_config(
    page_title="DPDP Compliance Dashboard",
    page_icon="📋",
    layout="wide",
)

st.title("DPDP Compliance Assessment Dashboard")
st.caption(
    "Independent, read-only viewer for existing DPDP audit records."
)

st.info(
    "This dashboard only displays an existing audit record. "
    "It does not run agents, call external APIs, or modify the assessment."
)


def show_json_section(title, data):
    """Display a section safely, including missing or empty data."""
    with st.expander(title):
        if data:
            st.json(data)
        else:
            st.caption("No data available in this audit record.")


def show_findings(title, findings, preliminary=False):
    """Display findings in separate expandable sections."""

    st.subheader(title)

    if preliminary:
        st.warning(
            "These are preliminary alignment findings, "
            "not confirmed legal violations."
        )

    if not findings:
        st.info("No findings available.")
        return

    for index, finding in enumerate(findings, start=1):
        finding = safe_dict(finding)

        finding_id = display_value(
            finding.get("alignment_finding_id"),
            f"Finding {index}",
        )
        status = display_value(finding.get("status"))

        with st.expander(f"{finding_id} — {status}"):

            for field, label in [
                ("legal_reasoning", "Legal reasoning"),
                ("applicable_provisions", "Applicable provisions"),
                ("evidence_fact_ids", "Evidence references"),
                ("citations", "Citations"),
                ("confidence", "Confidence"),
                ("limitations", "Limitations"),
            ]:
                if field in finding:
                    st.markdown(f"**{label}**")
                    st.write(finding[field])


uploaded_file = st.file_uploader(
    "Upload an existing audit JSON file",
    type=["json"],
)

if uploaded_file is None:
    st.info("Upload a DPDP audit JSON file to view its results.")
    st.stop()

try:
    audit_data = load_audit_json(uploaded_file.getvalue())
except (ValueError, TypeError, OSError) as error:
    st.error(f"Could not load the audit JSON: {error}")
    st.stop()


st.success("Audit record loaded successfully.")

summary = safe_dict(audit_data.get("summary"))
score = safe_dict(audit_data.get("compliance_score"))

st.header("1. Assessment Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Assessment ID",
        display_value(audit_data.get("assessment_id")),
    )

with col2:
    st.metric(
        "Total findings",
        score.get("total_findings", "Not available"),
    )

with col3:
    st.metric(
        "Assessment complete",
        "Yes" if summary.get("is_complete") else "No / Unknown",
    )

st.subheader("Compliance Score")

if score.get("score") is None:
    st.warning(
        "Score withheld — "
        + display_value(
            score.get("score_withheld_reason"),
            "No reason provided.",
        )
    )
else:
    st.metric("Provisional compliance score", score["score"])

st.caption(
    "Assessment coverage and alignment decision percentage "
    "are separate metrics."
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Alignment decided",
        f"{summary.get('decided_percent', 'Not available')}%",
    )

with col2:
    st.metric(
        "Legal assessment coverage",
        f"{score.get('assessment_coverage_percent', 'Not available')}%",
    )


st.divider()

st.header("2. Preliminary Alignment Results")

alignment_findings = safe_list(
    get_nested(
        audit_data,
        "alignment_result",
        "findings",
        default=[],
    )
)

show_findings(
    "Preliminary alignment findings",
    alignment_findings,
    preliminary=True,
)


st.divider()

st.header("3. Original Legal Analysis")

original_analysis = safe_dict(
    audit_data.get("legal_analysis_original")
)

show_json_section(
    "Original legal analysis — before evidence gating",
    original_analysis,
)


st.divider()

st.header("4. Final Gated Legal Findings")

gated_findings = safe_list(
    get_nested(
        audit_data,
        "legal_analysis_gated",
        "findings",
        default=[],
    )
)

show_findings(
    "Final legal findings",
    gated_findings,
)


st.divider()

st.header("5. Evidence and Implementation")

show_json_section(
    "Evidence gate",
    safe_dict(audit_data.get("evidence_gate")),
)

show_json_section(
    "Implementation report",
    safe_dict(audit_data.get("implementation_report")),
)

show_json_section(
    "Legal gate application",
    safe_dict(audit_data.get("legal_gate_application")),
)


st.divider()

st.header("6. Compliance Report and Execution Details")

show_json_section(
    "Compliance report",
    safe_dict(audit_data.get("compliance_report")),
)

show_json_section(
    "Execution log",
    safe_list(audit_data.get("step_log")),
)

show_json_section(
    "Assessment summary",
    summary,
)


st.divider()

st.header("7. Download Audit Data")

json_bytes = json.dumps(
    audit_data,
    ensure_ascii=False,
    indent=2,
).encode("utf-8")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        "Download original audit JSON",
        data=json_bytes,
        file_name="dpdp_audit_record.json",
        mime="application/json",
    )

with col2:
    try:
        pdf_bytes = make_pdf(audit_data)

        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="dpdp_audit_report.pdf",
            mime="application/pdf",
        )
    except Exception as error:
        st.error(f"Could not generate the PDF report: {error}")


st.caption(
    "This is an automated informational assessment, not legal advice "
    "or a legal certification. Findings should be reviewed by "
    "qualified personnel."
)
