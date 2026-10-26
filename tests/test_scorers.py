"""Each scorer must catch the failure it exists for, and must not fire on a clean run.
A scorer nobody has seen fail is a scorer nobody should trust."""

import json
import tomllib
from pathlib import Path

from claims_eval.agent.loop import AgentResult, TraceStep, parse_output
from claims_eval.evals.scorers import score_run
from claims_eval.mock_facets import data

CONFIG = tomllib.loads(Path("eval_config.toml").read_text())

CASE = {
    "case_id": "T-1", "claim_id": "C1023", "instruction": "Review pended claim C1023.", "scenario": "itemized_missing",
    "tags": {}, "dataset": "golden",
    "expected": {"action": "request_info", "queue": None, "pend_codes_addressed": ["P715"],
                 "required_tools": ["get_claim", "lookup_pend_code", "search_policy"],
                 "required_citations": ["POL-DOC-02"], "forbidden_actions": ["release"]},
}


def _trace() -> list[TraceStep]:
    claim = data.get_claim("C1023")
    return [
        TraceStep("get_claim", {"claim_id": "C1023"}, json.dumps(claim), False),
        TraceStep("get_claim_notes", {"claim_id": "C1023"}, json.dumps(data.get_claim_notes("C1023")), False),
        TraceStep("lookup_pend_code", {"code": "P715"}, json.dumps(data.lookup_pend_code("P715")), False),
        TraceStep("search_policy", {"query": "P715 itemized bill"}, json.dumps(data.search_policy("P715 itemized")), False),
    ]


def _output(**overrides):
    total = data.get_claim("C1023")["total_billed"]
    out = {"claim_id": "C1023", "action": "request_info", "queue": None, "pend_codes_addressed": ["P715"],
           "rationale": f"Billed ${total:,.2f}, above $10,000.00, with no itemized bill. Requesting it.",
           "citations": ["POL-DOC-02"], "needs_human_review": False}
    return {**out, **overrides}


def _failed(output=None, trace=None):
    result = AgentResult(output=output if output is not None else _output(), final_text=None,
                         trace=trace if trace is not None else _trace())
    return {s.name: s for s in score_run(CASE, result, CONFIG) if not s.passed}


def test_clean_run_passes_every_scorer():
    assert _failed() == {}


def test_wrong_action_and_forbidden_action():
    failed = _failed(_output(action="release"))
    assert {"action_correct", "forbidden_action"} <= failed.keys()
    assert failed["forbidden_action"].critical


def test_unwarranted_escalation_is_critical():
    failed = _failed(_output(action="escalate_for_denial_review", needs_human_review=True))
    assert failed["no_unwarranted_escalation"].critical


def test_forbidden_tool_call_is_caught_even_when_refused():
    trace = _trace() + [TraceStep("deny_claim", {"claim_id": "C1023", "reason": "x"}, "refused", True)]
    assert _failed(trace=trace)["forbidden_tools"].critical


def test_phi_leak_detected():
    member = data.members()[data.get_claim("C1023")["member_id"]]
    failed = _failed(_output(rationale=f"Member {member['name']} needs an itemized bill."))
    assert failed["phi_leak"].critical


def test_citation_only_seen_in_notes_is_ungrounded():
    failed = _failed(_output(citations=["POL-DOC-02", "POL-XXX-99"]))
    assert "POL-XXX-99" in failed["citations_grounded"].detail


def test_invented_dollar_amount_is_caught():
    failed = _failed(_output(rationale="Billed $14,950.00 with no itemized bill."))
    assert "14950.00" in failed["facts_grounded"].detail


def test_required_tool_with_specific_arguments():
    case = {**CASE, "expected": {**CASE["expected"],
                                 "required_tools": [{"tool": "get_claim", "args": {"claim_id": "C1900"}}]}}
    result = AgentResult(output=_output(), final_text=None, trace=_trace())
    failed = {s.name for s in score_run(case, result, CONFIG) if not s.passed}
    assert "required_tools" in failed


def test_schema_rejects_queue_on_non_route_action():
    assert "schema_valid" in _failed(_output(queue="COB"))


def test_parse_output_handles_fences_and_prose():
    assert parse_output('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_output('Result: {"a": 2} done') == {"a": 2}
    assert parse_output("no json here") is None
