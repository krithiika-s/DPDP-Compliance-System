
"""PDF report generation for existing DPDP audit records."""


def make_pdf(data):
    """Create a PDF using only the loaded audit data."""

    from io import BytesIO

    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("reportlab"):
            raise RuntimeError(
                "PDF export requires ReportLab. "
                "Please install project dependencies using "
                "'pip install -r requirements.txt'."
            ) from error
        raise

    from dashboard.data_loader import get_nested, safe_dict, safe_list

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    story = []

    def add_paragraph(label, value):
        story.append(Paragraph(f"<b>{label}</b>", styles["Heading3"]))
        story.append(
            Paragraph(
                str(value).replace("&", "&amp;").replace("<", "&lt;"),
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            "DPDP Compliance Assessment Report",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 12))

    add_paragraph("Assessment ID", data.get("assessment_id", "Not available"))
    add_paragraph("Created at", data.get("created_at", "Not available"))

    score = safe_dict(data.get("compliance_score"))

    if score.get("score") is None:
        add_paragraph(
            "Compliance score",
            "Score withheld — "
            + str(score.get("score_withheld_reason", "Reason not available")),
        )
    else:
        add_paragraph("Compliance score", score.get("score"))

    findings = safe_list(
        get_nested(data, "legal_analysis_gated", "findings", default=[])
    )

    story.append(
        Paragraph("Final Legal Findings", styles["Heading2"])
    )

    table_data = [["Finding ID", "Final status"]]

    for index, finding in enumerate(findings, start=1):
        finding = safe_dict(finding)
        table_data.append([
            str(finding.get("alignment_finding_id", f"Finding {index}")),
            str(finding.get("status", "Not available")),
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 14))

    add_paragraph(
        "Disclaimer",
        get_nested(
            data,
            "compliance_report",
            "disclaimer",
            default="Automated assessment. Please review findings with qualified personnel.",
        ),
    )

    doc.build(story)

    return buffer.getvalue()
