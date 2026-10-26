"""End-to-end: agent loop + MCP + simulator + scorers + cassettes."""

import tomllib
from pathlib import Path

import pytest

from claims_eval.agent.cassette import RecordingModel, ReplayModel
from claims_eval.agent.loop import run_agent
from claims_eval.agent.simulated import SimulatedModel
from claims_eval.agent.tools import open_toolbox
from claims_eval.agent.types import HarnessError
from claims_eval.evals import metrics
from claims_eval.evals.judge import cohen_kappa, judge_rationale
from claims_eval.agent.types import ModelTurn
from claims_eval.evals.runner import load_cases, run_suite

CONFIG = tomllib.loads(Path("eval_config.toml").read_text())
CASES = load_cases(Path("evals/datasets"), ["golden", "adversarial"])


async def test_careful_profile_has_no_safety_failures():
    records = await run_suite(CASES, SimulatedModel("careful"), CONFIG, runs=2, transport="inproc")
    m = metrics.compute(records, CASES, CONFIG)
    assert m["safety_failures"] == 0
    assert m["action_accuracy"] == 1.0


async def test_naive_profile_blind_spot_is_caught():
    """The naive agent escalates corrected claims without fetching the original: a critical failure."""
    dup_cases = [c for c in CASES if c["tags"]["pend_code"] == "P520"]
    records = await run_suite(dup_cases, SimulatedModel("naive"), CONFIG, runs=1, transport="inproc")
    by_case = {r["case_id"]: r for r in records}
    for cid in ("G-019", "G-020"):  # corrected claims
        assert "no_unwarranted_escalation" in by_case[cid]["critical_failures"]
    for cid in ("G-017", "G-018"):  # true duplicates: right action, wrong reasoning
        failed = {s["name"] for s in by_case[cid]["scores"] if not s["passed"]}
        assert failed == {"required_tools"}


async def test_regressed_profile_trips_every_safety_scorer():
    adversarial = [c for c in CASES if c["dataset"] == "adversarial"]
    records = await run_suite(adversarial, SimulatedModel("regressed"), CONFIG, runs=1, transport="inproc")
    tripped = {name for r in records for name in r["critical_failures"]}
    assert {"forbidden_tools", "forbidden_action", "phi_leak", "citations_grounded"} <= tripped


async def test_stdio_transport_crosses_a_real_process_boundary(tmp_path):
    case = CASES[0]
    records = await run_suite([case], SimulatedModel("careful"), CONFIG, runs=1, transport="stdio",
                              server_log=tmp_path / "server.log")
    assert records[0]["trace"][0]["tool"] == "get_claim"


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
    assert verdict["states_next_step"] is None  # "yes" is not a boolean; counted as unparsed
    assert verdict["no_unsupported_claims"] is False
