"""Offline tests for the live adapters: request shape in, neutral turn out. No network, no keys.
Skipped when the live extras aren't installed (uv sync --extra live)."""

from types import SimpleNamespace

import pytest

from claims_eval.agent.models import _to_anthropic
from claims_eval.agent.types import ToolSpec

TOOLS = [ToolSpec("get_claim", "Get a claim.", {"type": "object", "properties": {"claim_id": {"type": "string"}}})]
CONVERSATION = [
    {"role": "user", "content": "Review C1001."},
    {"role": "assistant", "text": None, "tool_calls": [{"id": "a1", "name": "get_claim", "args": {"claim_id": "C1001"}},
                                                       {"id": "a2", "name": "get_claim", "args": {"claim_id": "C5001"}}]},
    {"role": "tool", "tool_call_id": "a1", "name": "get_claim", "content": "{}", "is_error": False},
    {"role": "tool", "tool_call_id": "a2", "name": "get_claim", "content": "not found", "is_error": True},
]


def test_anthropic_groups_parallel_tool_results_in_one_user_message():
    msgs = _to_anthropic(CONVERSATION)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["a1", "a2"]
    assert msgs[2]["content"][1]["is_error"] is True


async def test_anthropic_turn_parsing(monkeypatch):
    pytest.importorskip("anthropic")
    from claims_eval.agent.models import AnthropicModel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    model = AnthropicModel("claude-test")
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Checking."),
                     SimpleNamespace(type="tool_use", id="tu1", name="get_claim", input={"claim_id": "C1001"})],
            usage=SimpleNamespace(input_tokens=100, output_tokens=20), stop_reason="tool_use")

    model._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    turn = await model.complete("sys", CONVERSATION[:1], TOOLS, "X/r0/t0")
    assert sent["tools"][0]["input_schema"] == TOOLS[0].input_schema
    assert turn.tool_calls[0].id == "tu1" and turn.text == "Checking." and turn.stop_reason == "tool_use"

    sent.clear()
    await model.complete("judge", CONVERSATION[:1], [], "j/r0/t0")
    assert "tools" not in sent  # the judge sends no tools


async def test_gemini_local_ids_are_never_sent_back_and_thinking_tokens_count(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import types as t

    from claims_eval.agent.models import GeminiModel

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    model = GeminiModel("gemini-test")
    sent = []

    async def generate_content(model, contents, config):
        sent.append(contents)
        return t.GenerateContentResponse(
            candidates=[t.Candidate(finish_reason="STOP", content=t.Content(role="model", parts=[
                t.Part(function_call=t.FunctionCall(name="get_claim", args={"claim_id": "C1001"}),
                       thought_signature=b"sig")]))],
            usage_metadata=t.GenerateContentResponseUsageMetadata(
                prompt_token_count=50, candidates_token_count=10, thoughts_token_count=30))

    model._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    turn = await model.complete("sys", CONVERSATION[:1], TOOLS, "X/r0/t0")
    assert turn.tool_calls[0].id.startswith("local:")  # Gemini gave no id
    assert turn.usage.output_tokens == 40

    history = CONVERSATION[:1] + [
        {"role": "assistant", "text": None, "raw": turn.raw,
         "tool_calls": [{"id": turn.tool_calls[0].id, "name": "get_claim", "args": {"claim_id": "C1001"}}]},
        {"role": "tool", "tool_call_id": turn.tool_calls[0].id, "name": "get_claim", "content": "{}", "is_error": False},
    ]
    await model.complete("sys", history, TOOLS, "X/r0/t1")
    echoed, response = sent[1][1], sent[1][2]
    assert echoed.parts[0].thought_signature == b"sig"  # native turn echoed verbatim
    assert response.parts[0].function_response.id is None  # our local id stays local


async def test_gemini_empty_candidate_is_a_scored_non_answer(monkeypatch):
    pytest.importorskip("google.genai")
    from google.genai import types as t

    from claims_eval.agent.models import GeminiModel

    monkeypatch.setenv("GEMINI_API_KEY", "test")
    model = GeminiModel("gemini-test")

    async def generate_content(model, contents, config):
        return t.GenerateContentResponse(candidates=[t.Candidate(finish_reason="SAFETY")])

    model._client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    turn = await model.complete("sys", CONVERSATION[:1], TOOLS, "X/r0/t0")
    assert turn.text is None and not turn.tool_calls and "SAFETY" in turn.stop_reason
