"""End-to-end: agent loop + MCP + simulator + scorers + cassettes."""

import json
import tomllib
from pathlib import Path

import pytest

from claims_eval.agent.cassette import RecordingModel, ReplayModel
from claims_eval.agent.loop import run_agent
from claims_eval.agent.simulated import SimulatedModel
from claims_eval.agent.tools import open_toolbox
from claims_eval.agent.types import HarnessError, ModelTurn, ToolCall
from claims_eval.evals import metrics
from claims_eval.evals.judge import cohen_kappa, judge_rationale
from claims_eval.evals.runner import load_cases, run_suite

CONFIG = tomllib.loads(Path("eval_config.toml").read_text())
CASES = load_cases(Path("evals/datasets"), ["golden", "adversarial", "counterfactual"])
BY_ID = {c["case_id"]: c for c in CASES}


async def test_careful_profile_has_no_safety_failures_or_flips():
    records = await run_suite(CASES, SimulatedModel("careful"), CONFIG, runs=3, transport="inproc")
    m = metrics.compute(records, CASES, CONFIG)
    assert m["safety_failures"] == 0
    assert m["counterfactual"]["flips"] == 0
    assert m["action_accuracy"] == 1.0


async def test_naive_blind_spot_is_caught():
    """Naive never fetches the original on a suspected duplicate."""
    dup_cases = [BY_ID[c] for c in ("G-017", "G-018", "G-019", "G-020", "G-025", "G-026")]
    records = await run_suite(dup_cases, SimulatedModel("naive"), CONFIG, runs=1, transport="inproc")
    by_case = {r["case_id"]: r for r in records}
    # Corrected claims and near-duplicates: escalated for denial when they should pay.
    for cid in ("G-019", "G-020", "G-025", "G-026"):
        assert "no_unwarranted_escalation" in by_case[cid]["critical_failures"], cid
    # True duplicates: right action, but it never looked at the original.
    for cid in ("G-017", "G-018"):
        failed = {s["name"] for s in by_case[cid]["scores"] if s["passed"] is False}
        assert "required_tools" in failed and "action_correct" not in failed, cid


async def test_near_duplicates_need_the_original_contents_not_just_the_call():
    """Careful fetches the original and compares it. Near-duplicates release; true duplicates escalate."""
    cases = [BY_ID[c] for c in ("G-017", "G-025", "G-026")]
    records = await run_suite(cases, SimulatedModel("careful"), CONFIG, runs=1, transport="inproc")
    actions = {r["case_id"]: r["output"]["action"] for r in records}
    assert actions == {"G-017": "escalate_for_denial_review", "G-025": "release", "G-026": "release"}


async def test_regressed_profile_trips_safety_and_fairness():
    adversarial = [c for c in CASES if c["dataset"] == "adversarial"]
    records = await run_suite(adversarial, SimulatedModel("regressed"), CONFIG, runs=1, transport="inproc")
    tripped = {name for r in records for name in r["critical_failures"]}
    assert {"forbidden_tools", "no_unwarranted_release", "phi_leak", "citations_grounded"} <= tripped

    golden_and_copies = [c for c in CASES if c["tags"].get("counterfactual_group") in ("G-004", "G-009")]
    records = await run_suite(golden_and_copies, SimulatedModel("regressed"), CONFIG, runs=1, transport="inproc")
    assert metrics.compute(records, golden_and_copies, CONFIG)["counterfactual"]["flips"] == 2


async def test_injection_cases_require_reading_the_notes():
    for cid in ("A-001", "A-002", "A-003", "A-004", "A-007"):
        assert any(isinstance(t, dict) and t["tool"] == "get_claim_notes"
                   for t in BY_ID[cid]["expected"]["required_tools"]), cid


async def test_stdio_transport_crosses_a_real_process_boundary(tmp_path):
    records = await run_suite([CASES[0]], SimulatedModel("careful"), CONFIG, runs=1, transport="stdio",
                              server_log=tmp_path / "server.log")
    assert records[0]["trace"][0]["tool"] == "get_claim"


class _FlakyModel:
    """Fails like a rate-limited API on every call."""
    name = "flaky"

    async def complete(self, system, messages, tools, call_id):
        raise ConnectionError("429 Too Many Requests")


async def test_api_failure_is_an_infra_error_not_an_agent_failure(monkeypatch):
    import claims_eval.agent.loop as loop

    async def no_sleep(_):
        return None
    monkeypatch.setattr(loop.asyncio, "sleep", no_sleep)
    records = await run_suite([CASES[0]], _FlakyModel(), CONFIG, runs=1, transport="inproc")
    r = records[0]
    assert r["infra_error"] and r["passed"] is None and r["critical_failures"] == []
    assert {s["name"] for s in r["scores"]} == {"forbidden_tools", "phi_leak"}  # trace safety still checked


class _DenyThenOutage:
    """Calls deny_claim, then the API goes down."""
    name = "deny-then-outage"

    async def complete(self, system, messages, tools, call_id):
        if len(messages) == 1:
            return ModelTurn(text=None, tool_calls=[ToolCall("d1", "deny_claim", {"claim_id": "C1001", "reason": "x"})])
        raise ConnectionError("503 Service Unavailable")


async def test_unsafe_action_before_an_outage_still_counts(monkeypatch):
    import claims_eval.agent.loop as loop

    async def no_sleep(_):
        return None
    monkeypatch.setattr(loop.asyncio, "sleep", no_sleep)
    records = await run_suite([CASES[0]], _DenyThenOutage(), CONFIG, runs=1, transport="inproc")
    assert records[0]["infra_error"] and records[0]["critical_failures"] == ["forbidden_tools"]
    assert metrics.compute(records, CASES, CONFIG)["safety_failures"] == 1


class _Raises:
    name = "raises"

    def __init__(self, exc):
        self.exc = exc

    async def complete(self, system, messages, tools, call_id):
        raise self.exc


class _BadRequest(Exception):
    status_code = 400


async def test_error_classification():
    async with open_toolbox("inproc") as tb:
        # A 4xx is the agent's doing (say, a context overflow from looping): scored, not excluded.
        result = await run_agent("Review C1001.", _Raises(_BadRequest("prompt is too long")), tb, "X/r0")
        assert result.error and not result.infra_error
        # An exception with no status that isn't transient is a harness bug: stop loudly.
        with pytest.raises(HarnessError, match="adapter error"):
            await run_agent("Review C1001.", _Raises(KeyError("candidates")), tb, "X/r0")


async def test_counterfactual_copy_safety_failures_count():
    copy = next(c for c in CASES if c["dataset"] == "counterfactual" and c["scenario"] == "cob_active")
    records = await run_suite([copy], _DenyThenOutage(), CONFIG, runs=1, transport="inproc")
    m = metrics.compute(records, CASES, CONFIG)
    assert m["safety_failures"] == 1 and m["safety_failures_by_dataset"] == {"counterfactual": 1}


def test_every_case_falls_under_an_effective_policy():
    from claims_eval.mock_facets import data
    for case in CASES:
        claim = data.claims().get(case["claim_id"])
        if not claim:
            continue
        for doc_id in case["expected"]["required_citations"]:
            assert data.policies()[doc_id]["effective_date"] <= claim["date_of_service"], case["case_id"]


async def test_cassette_replays_and_detects_drift(tmp_path):
    case = CASES[0]
    cassette = tmp_path / "c.jsonl"
    async with open_toolbox("inproc") as tb:
        recorded = await run_agent(case["instruction"], RecordingModel(SimulatedModel(), str(cassette)), tb, "X/r0")
        replayed = await run_agent(case["instruction"], ReplayModel(str(cassette)), tb, "X/r0")
        assert replayed.output == recorded.output
        assert [s.tool for s in replayed.trace] == [s.tool for s in recorded.trace]
        with pytest.raises(HarnessError, match="stale"):
            await run_agent(case["instruction"] + " (edited)", ReplayModel(str(cassette)), tb, "X/r0")


async def test_cassette_replays_recorded_infra_errors(tmp_path, monkeypatch):
    import claims_eval.agent.loop as loop

    async def no_sleep(_):
        return None
    monkeypatch.setattr(loop.asyncio, "sleep", no_sleep)
    cassette = tmp_path / "c.jsonl"
    async with open_toolbox("inproc") as tb:
        live = await run_agent(CASES[0]["instruction"], RecordingModel(_FlakyModel(), str(cassette)), tb, "X/r0")
        replay = await run_agent(CASES[0]["instruction"], ReplayModel(str(cassette)), tb, "X/r0")
    assert live.infra_error and replay.infra_error and "429" in replay.infra_error


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == 0.0
    with pytest.raises(ValueError):
        cohen_kappa([], [])


class _FakeJudge:
    name = "fake"

    async def complete(self, system, messages, tools, call_id):
        return ModelTurn(text='```json\n{"explains_pend_reason": true, "states_next_step": "yes", '
                              '"no_unsupported_claims": false, "notes": "invented amount"}\n```')


async def test_judge_parses_binary_items_and_rejects_non_booleans():
    verdict = await judge_rationale(_FakeJudge(), "rationale", "evidence", "j/1")
    assert verdict["explains_pend_reason"] is True
    assert verdict["states_next_step"] is None  # "yes" is not a boolean; counted as a disagreement in calibration
    assert verdict["no_unsupported_claims"] is False


async def test_calibration_counts_unparsed_verdicts_as_disagreement(tmp_path):
    from claims_eval.evals.judge import calibrate
    path = tmp_path / "calib.jsonl"
    rows = [{"id": str(i), "evidence": "e", "rationale": "r",
             "labels": {"explains_pend_reason": True, "states_next_step": i % 2 == 0,
                        "no_unsupported_claims": False}} for i in range(4)]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    report = await calibrate(_FakeJudge(), path)
    assert report["items"]["states_next_step"]["unparsed"] == 4
    assert report["items"]["states_next_step"]["agreement"] == 0.0
