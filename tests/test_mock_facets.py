"""Tool contract tests. Deterministic, no model involved: the MCP layer must hold its own guarantees
regardless of what the agent does."""

import json
from pathlib import Path

import pytest
from mcp import Client

from claims_eval.mock_facets import data
from claims_eval.mock_facets.server import mcp as server

READ_TOOLS = {"get_claim", "get_claim_notes", "get_member_eligibility", "get_provider", "lookup_pend_code",
              "search_policy"}
PHI_FIELDS = {"name", "dob", "ssn", "address"}
CASES = {json.loads(line)["case_id"]: json.loads(line)
         for name in ("golden", "adversarial") for line in Path(f"evals/datasets/{name}.jsonl").open()}
MEMBER_ID = data.get_claim(CASES["G-001"]["claim_id"])["member_id"]


async def call(tool: str, args: dict):
    # A client per call keeps anyio cancel scopes inside one task (async fixtures would split them).
    async with Client(server) as c:
        return await c.call_tool(tool, args)


async def test_tool_inventory_and_annotations():
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == READ_TOOLS | {"deny_claim"}
    for name in READ_TOOLS:
        assert tools[name].annotations.read_only_hint is True, name
    assert tools["deny_claim"].annotations.destructive_hint is True


async def test_eligibility_returns_minimum_necessary():
    result = await call("get_member_eligibility", {"member_id": MEMBER_ID, "date_of_service": "2026-01-10"})
    payload = json.loads(result.content[0].text)
    assert not result.is_error
    assert PHI_FIELDS.isdisjoint(payload)
    member = data.members()[MEMBER_ID]
    assert member["name"] not in result.content[0].text
    assert member["dob"] not in result.content[0].text


async def test_notes_redact_ssn_but_not_names():
    """DLP catches the pattern it knows. The name survives, which is why the agent is tested for leaks too."""
    claim_id = CASES["A-003"]["claim_id"]  # notes with PHI
    text = (await call("get_claim_notes", {"claim_id": claim_id})).content[0].text
    member = data.members()[data.claims()[claim_id]["member_id"]]
    assert member["ssn"] not in text
    assert "[REDACTED-SSN]" in text
    assert member["name"] in text


async def test_deny_claim_always_refuses():
    result = await call("deny_claim", {"claim_id": CASES["G-001"]["claim_id"], "reason": "test", "approval_token": "x"})
    assert result.is_error
    assert "approval token" in result.content[0].text


@pytest.mark.parametrize("tool,args,needle", [
    ("get_claim", {"claim_id": "C9999"}, "not found"),
    ("get_provider", {"npi": "0000000000"}, "not on provider file"),
    ("lookup_pend_code", {"code": "P999"}, "not in the pend code table"),
])
async def test_missing_records_are_tool_errors_the_model_can_read(tool, args, needle):
    result = await call(tool, args)
    assert result.is_error
    assert needle in result.content[0].text


@pytest.mark.parametrize("bad_date", ["10/19/2026", "2026-1-5", "2026-02-30", "20260105"])
async def test_eligibility_rejects_non_iso_dates(bad_date):
    """A malformed date must be an error, never a confident 'not eligible'."""
    result = await call("get_member_eligibility", {"member_id": MEMBER_ID, "date_of_service": bad_date})
    assert result.is_error
    assert "YYYY-MM-DD" in result.content[0].text


def test_coverage_termination_date_is_inclusive():
    member_id = next(mid for mid, m in data.members().items()
                     if m["coverage"][0]["termination"] and m["coverage"][0]["termination"] >= "2026-01-01")
    last_day = data.members()[member_id]["coverage"][0]["termination"]
    assert data.get_member_eligibility(member_id, last_day)["eligible_on_dos"] is True


async def test_tool_descriptions_have_no_docstring_indentation():
    async with Client(server) as c:
        for t in (await c.list_tools()).tools:
            assert "\n    " not in t.description, t.name


async def test_schema_rejects_missing_arguments():
    result = await call("get_member_eligibility", {"member_id": MEMBER_ID})
    assert result.is_error


def test_policy_search_finds_governing_doc_for_every_pend_code():
    for code, entry in data.pend_codes().items():
        hits = [r["doc_id"] for r in data.search_policy(f"{code} {entry['description']}")["results"]]
        assert entry["policy_ref"] in hits, code
