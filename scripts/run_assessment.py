
"""Run the DPDP assessment pipeline.

Default mode uses fake Agents 1-4 and deterministic Agent 5/scoring.
It requires no model, API key, or network access.

The --real mode currently runs real Agents 1-3 only and makes API calls.
"""

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from agents.alignment_agent.schemas import AlignmentResult
from agents.implementation_agent.schemas import ImplementationReport
from agents.legal_rag.schemas import LegalAnalysisResult
from agents.policy_agent.schemas import (
    PolicyCategory,
    PolicyClaim,
    PolicyExtractionResult,
)
from agents.policy_agent.verification import ExtractionReport
from agents.report_agent.agent import ReportGenerationAgent
from agents.scoring.engine import ComplianceScoringEngine

from orchestrator.analysis_stage import run_analysis_stages
from orchestrator.audit import build_audit_record, write_audit_record
from orchestrator.final_assessment_stage import run_final_assessment_stage
from orchestrator.models import AssessmentRequest, AssessmentState, PolicyDocument
from orchestrator.summary import summarize_assessment


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = ROOT / "tests" / "fixtures" / "apps" / "acmenotes_violating"
DEFAULT_POLICY = ROOT / "tests" / "fixtures" / "synthetic_privacy_policy.txt"

POLICY_NAME = "demo_policy.txt"
POLICY_TEXT = "We keep your data for two years."


class FakeExtractor:
    """Fake Agent 1: returns one fixed policy claim."""

    def extract_with_report(self, *, document_text: str, source: str):
        claim = PolicyClaim(
            claim="Data is kept for two years.",
            category=PolicyCategory.OTHER,
            evidence="We keep your data for two years.",
            source=source,
        )
        return ExtractionReport(
            result=PolicyExtractionResult(claims=[claim])
        )


class FakeImplementationAgent:
    """Fake Agent 2: returns a valid, empty implementation report."""

    def run(self, project_path):
        return ImplementationReport(
            project_path=str(project_path),
            files_scanned=0,
            coverage={},
            evidence=[],
            facts=[],
            llm_retries=0,
            llm_failures=0,
            failed_categories=[],
        )


class FakeAlignmentAgent:
    """Fake Agent 3: returns a valid alignment result with no findings."""

    def run(self, policy, implementation, *, policy_text=None):
        return AlignmentResult(
            findings=[],
            summary={},
            llm_retries=0,
            llm_failures=0,
            failed_calls=[],
            policy_text_chars=0,
            policy_text_truncated=False,
        )


class FakeLegalAgent:
    """Fake Agent 4: returns a valid legal result without model calls."""

    def run(self, request):
        return LegalAnalysisResult(
            findings=[],
            summary={},
            sources_consulted=[],
            failed_calls=[],
            retrieval_failures=[],
        )


def _finish(
    state: AssessmentState,
    output_dir,
    heading: str,
    default_folder: str,
) -> Path:
    """Print the step log and summary, then write the audit record."""

    summary = summarize_assessment(state)
    record = build_audit_record(state, summary)

    folder = (
        Path(output_dir)
        if output_dir
        else Path(tempfile.gettempdir()) / default_folder
    )
    path = write_audit_record(record, folder)

    print(heading)
    print()

    for entry in state.step_log:
        print(
            f"{entry.step} [{entry.target}] "
            f"{entry.status.value} (attempt {entry.attempt})"
        )

    print()
    print(summary.model_dump_json(indent=2))
    print()
    print(f"Audit record written to: {path}")

    if state.compliance_score is not None:
        print()
        print("Provisional scoring result:")
        print(state.compliance_score)

    if state.compliance_report is not None:
        print()
        print("Report generated successfully.")

    return path


def main(output_dir: str | Path | None = None) -> Path:
    """Run the complete fake assessment pipeline."""

    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id="demo-run",
            policy_documents=[
                PolicyDocument(
                    name=POLICY_NAME,
                    text=POLICY_TEXT,
                )
            ],
            project_path=".",
        )
    )

    # Agents 1, 2 and 3.
    run_analysis_stages(
        state,
        FakeExtractor(),
        FakeImplementationAgent,
        FakeAlignmentAgent,
    )

    # Agent 4, evidence gate, deterministic scoring and Agent 5.
    run_final_assessment_stage(
        state=state,
        legal_agent=FakeLegalAgent(),
        scoring_engine=ComplianceScoringEngine(),
        report_agent=ReportGenerationAgent(),
    )

    return _finish(
        state,
        output_dir,
        "FAKE pipeline only. No model or API was called.",
        "dpdp_audit_demo",
    )


def _load_model():
    """Load the real chat model lazily, only for --real mode."""

    from llm.langchain_model import get_langchain_model

    return get_langchain_model()


def run_real(
    project: str | Path,
    policy: str | Path,
    output_dir=None,
) -> int:
    """Run real Agents 1-3. This mode makes API calls."""

    project_path = Path(project)
    policy_path = Path(policy)

    if not project_path.is_dir():
        print(f"Project folder not found: {project_path}")
        return 1

    if not policy_path.is_file():
        print(f"Policy file not found: {policy_path}")
        return 1

    policy_text = policy_path.read_text(encoding="utf-8")

    if not policy_text.strip():
        print(f"Policy file is empty: {policy_path}")
        return 1

    print("REAL model. This makes API calls (Agents 1, 2 and 3 only).")
    print(f"Project: {project_path}")
    print(f"Policy:  {policy_path} ({len(policy_text)} characters)")
    print()

    try:
        model = _load_model()
    except ValueError as error:
        print(f"Cannot start the real run: {error}")
        return 1

    from agents.alignment_agent.agent import AlignmentAgent
    from agents.implementation_agent.agent import ImplementationExtractionAgent
    from agents.policy_agent.agent import PolicyExtractionAgent

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    state = AssessmentState(
        request=AssessmentRequest(
            assessment_id=f"real-{stamp}",
            policy_documents=[
                PolicyDocument(
                    name=policy_path.name,
                    text=policy_text,
                )
            ],
            project_path=str(project_path),
        )
    )

    run_analysis_stages(
        state,
        PolicyExtractionAgent(model),
        lambda: ImplementationExtractionAgent(model),
        lambda: AlignmentAgent(model),
    )

    _finish(
        state,
        output_dir,
        "REAL Agents 1-3 run finished.",
        "dpdp_audit_real",
    )

    return 0


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the DPDP assessment pipeline."
    )

    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="folder for the audit file (default: system temp folder)",
    )

    parser.add_argument(
        "--real",
        action="store_true",
        help="use the real model for Agents 1-3 (makes API calls)",
    )

    parser.add_argument(
        "--project",
        default=str(DEFAULT_PROJECT),
        help="Django project folder to scan (only with --real)",
    )

    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY),
        help="policy text file (only with --real)",
    )

    args = parser.parse_args(argv)

    if not args.real:
        main(args.output_dir)
        return 0

    return run_real(
        args.project,
        args.policy,
        args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(cli())
