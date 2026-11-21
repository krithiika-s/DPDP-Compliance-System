"""Apply the evidence gate's decisions to Agent 4's legal result.

Deterministic Python. Reads Agent 4's result and the gate's result as dicts
(model_dump(mode="json") and GateResult.to_dict()) and returns a NEW legal
result. The inputs are never changed, so the audit record can keep both what
Agent 4 said and what was allowed to stand.

Only one status is ever changed: potential_non_compliance. It is capped:
- gate said blocked      -> insufficient_evidence
- gate said review_only  -> requires_review
- gate said pass, not_gated, or the finding is not potential_non_compliance
  -> left as it is

A status is never moved toward "more certain". Every other status is kept.

Why this runs before scoring: the scoring engine leaves insufficient_evidence
and requires_review findings out of the score, so a capped finding no longer
counts against the application. That is the honest reading (the scan could not
support it), but it can raise the score. The caller should show the
assessment coverage next to the score.

Safe defaults, so a missing check never looks like a clean one:
- If the gate result is missing, every potential_non_compliance finding is
  capped to requires_review.
- If the gate has no decision for a potential_non_compliance finding (an id
  that does not match), that finding is capped to requires_review.

Nothing here imports an agent, so it needs no model and no Person 3 code.
"""

import copy
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from orchestrator.evidence_gate import BLOCKED, PASS, REVIEW_ONLY

NON_COMPLIANCE = "potential_non_compliance"
INSUFFICIENT = "insufficient_evidence"
REQUIRES_REVIEW = "requires_review"

# Every Agent 4 status value (lower case, as in model_dump(mode="json")).
STATUS_VALUES = frozenset(
    {
        NON_COMPLIANCE,
        "potential_compliance",
        INSUFFICIENT,
        "not_applicable",
        REQUIRES_REVIEW,
    }
)

_SEVERITY = {PASS: 0, REVIEW_ONLY: 1, BLOCKED: 2}
_CAP_FOR = {BLOCKED: INSUFFICIENT, REVIEW_ONLY: REQUIRES_REVIEW}
_PREFIX = "Evidence gate: "


@dataclass(frozen=True)
class StatusChange:
    """One finding whose status was capped."""

    finding_id: str
    old_status: str
    new_status: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "finding_id": self.finding_id,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GateApplication:
    """The capped legal result plus a list of what changed."""

    result: dict[str, Any]
    changes: tuple[StatusChange, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"changes": [c.to_dict() for c in self.changes]}


def apply_gate(
    legal_result: Mapping[str, Any],
    gate: Mapping[str, Any] | None,
) -> GateApplication:
    """Return a copy of legal_result with gate caps applied."""
    result = copy.deepcopy(dict(legal_result))
    findings = result.get("findings") or []

    decisions = _decisions_by_id(gate)
    changes: list[StatusChange] = []

    for finding in findings:
        if not isinstance(finding, dict):
            continue
        finding_id = str(finding.get("alignment_finding_id", ""))
        status = finding.get("status")

        if gate is None:
            decision, reasons = None, ["The evidence gate did not run."]
        elif finding_id in decisions:
            decision, reasons = decisions[finding_id]
        else:
            decision = None
            reasons = ["The evidence gate has no decision for this finding."]

        if status == NON_COMPLIANCE:
            if decision in _CAP_FOR:
                new_status = _CAP_FOR[decision]
            elif decision is None:
                new_status = REQUIRES_REVIEW
            else:
                new_status = None
            if new_status is not None:
                reason = _PREFIX + (reasons[0] if reasons else "capped.")
                finding["status"] = new_status
                _add_limitations(finding, reasons)
                _add_limitation(
                    finding,
                    f"{_PREFIX}status changed from {status} to {new_status}.",
                )
                changes.append(StatusChange(finding_id, status, new_status, reason))
                continue

        # Status kept. Still carry the gate's reasons, so a report that shows
        # an already cautious finding also shows why it is cautious.
        if decision in _CAP_FOR:
            _add_limitations(finding, reasons)

    _refresh_summary(result)
    return GateApplication(result=result, changes=tuple(changes))

def _decisions_by_id(
    gate: Mapping[str, Any] | None,
) -> dict[str, tuple[str, list[str]]]:
    """Map finding id to (worst decision, reasons). Invalid decisions fail safe."""
    out: dict[str, tuple[str, list[str]]] = {}

    if gate is None:
        return out

    for entry in gate.get("decisions") or []:
        if not isinstance(entry, dict):
            continue

        finding_id = str(entry.get("finding_id", ""))
        raw_decision = entry.get("decision")
        reasons = [str(r) for r in (entry.get("reasons") or [])]

        if raw_decision == "not_gated":
            decision = PASS
        elif isinstance(raw_decision, str) and raw_decision in _SEVERITY:
            decision = raw_decision
        else:
            decision = REVIEW_ONLY
            invalid_reason = (
                "The evidence gate returned a missing or invalid decision."
            )
            if invalid_reason not in reasons:
                reasons.append(invalid_reason)

        previous = out.get(finding_id)
        if previous is None or _SEVERITY[decision] > _SEVERITY[previous[0]]:
            out[finding_id] = (decision, reasons)

    return out
def _add_limitation(finding: dict[str, Any], text: str) -> None:
    limitations = finding.get("limitations")
    if not isinstance(limitations, list):
        limitations = []
        finding["limitations"] = limitations
    if text not in limitations:
        limitations.append(text)


def _add_limitations(finding: dict[str, Any], reasons: list[str]) -> None:
    for reason in reasons:
        _add_limitation(finding, _PREFIX + reason)


def _refresh_summary(result: dict[str, Any]) -> None:
    """Recount statuses, but only if the summary is a status -> count map."""
    summary = result.get("summary")
    if not isinstance(summary, dict) or not summary:
        return
    if not set(summary) <= STATUS_VALUES:
        return
    counts = Counter(
        f.get("status") for f in (result.get("findings") or []) if isinstance(f, dict)
    )
    updated = {key: counts.get(key, 0) for key in summary}
    for status, count in counts.items():
        if status in STATUS_VALUES and status not in updated:
            updated[status] = count
    result["summary"] = updated