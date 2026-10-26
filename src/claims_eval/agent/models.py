"""Live model adapters and the factory that picks a model from a CLI spec.

Each adapter translates the neutral message format (see types.py) to one vendor's API and back.
Nothing outside this file knows which vendor is in use.

Specs:
    sim[:naive|careful|regressed]    scripted stand-in (default: careful)
    anthropic[:<model id>]           Claude via the Anthropic API (ANTHROPIC_API_KEY)
    gemini[:<model id>]              Gemini via google-genai (GEMINI_API_KEY, or Vertex AI env vars)
    replay:<cassette.jsonl>          replay a recorded run, no network
"""

from __future__ import annotations

import os
from typing import Any

from claims_eval.agent.types import HarnessError, Model, ModelTurn, ToolCall, ToolSpec, Usage

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
MAX_OUTPUT_TOKENS = 2048


class AnthropicModel:
    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise HarnessError("install live extras: uv sync --extra live") from exc
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.name = f"anthropic:{self.model}"
        self._client = anthropic.AsyncAnthropic()

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec],
                       call_id: str) -> ModelTurn:
        kwargs: dict[str, Any] = {}
        if tools:  # the judge calls with no tools
            kwargs["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                               for t in tools]
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=system,
            messages=_to_anthropic(messages),
            **kwargs,
        )
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input)))
        return ModelTurn(
            text="\n".join(text_parts) or None,
            tool_calls=calls,
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
            # "max_tokens" mid tool call or "refusal" show up in the report when the answer is unusable.
            stop_reason=resp.stop_reason,
        )


def _to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.get("text"):
                blocks.append({"type": "text", "text": m["text"]})
            blocks += [{"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]}
                       for tc in m["tool_calls"]]
            out.append({"role": "assistant", "content": blocks})
        elif m["role"] == "tool":
            result = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"],
                      "is_error": m["is_error"]}
            # All results for one assistant turn go back in a single user message.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
    return out


class GeminiModel:
    def __init__(self, model: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise HarnessError("install live extras: uv sync --extra live") from exc
        self.model = model or os.environ.get("GEMINI_MODEL")
        if not self.model:
            raise HarnessError("set a Gemini model id: --model gemini:<id> or GEMINI_MODEL")
        self.name = f"gemini:{self.model}"
        self._types = types
        # Reads GEMINI_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI + GOOGLE_CLOUD_PROJECT/LOCATION for Vertex AI.
        self._client = genai.Client()

    async def complete(self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec],
                       call_id: str) -> ModelTurn:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            tools=[t.Tool(function_declarations=[
                t.FunctionDeclaration(name=s.name, description=s.description, parameters_json_schema=s.input_schema)
                for s in tools
            ])] if tools else None,
            # The harness runs the loop itself so every tool call lands in the trace.
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = await self._client.aio.models.generate_content(
            model=self.model, contents=self._to_gemini(messages), config=config)
        usage = resp.usage_metadata
        tokens = Usage(usage.prompt_token_count or 0,
                       # Thinking tokens are billed as output.
                       (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else Usage()
        candidate = resp.candidates[0] if resp.candidates else None
        finish = str(candidate.finish_reason) if candidate and candidate.finish_reason else None
        content = candidate.content if candidate else None
        if content is None or not content.parts:
            # A safety stop or malformed call: the model gave no usable answer. Scored as the agent's failure.
            return ModelTurn(text=None, usage=tokens, stop_reason=finish or "no candidate")
        text_parts, calls = [], []
        for i, part in enumerate(content.parts):
            if part.function_call:
                fc = part.function_call
                # Gemini may omit call ids. Use a local one for our trace, marked so it is never sent back.
                calls.append(ToolCall(id=fc.id or f"{LOCAL_ID}{call_id}-{i}", name=fc.name, args=dict(fc.args or {})))
            elif part.text and not part.thought:
                text_parts.append(part.text)
        return ModelTurn(
            text="\n".join(text_parts) or None,
            tool_calls=calls,
            usage=tokens,
            # Echo the native turn back verbatim; Gemini rejects function calls stripped of thought signatures.
            raw={"gemini_content": content.model_dump(mode="json", exclude_none=True)},
            stop_reason=finish,
        )

    def _to_gemini(self, messages: list[dict[str, Any]]) -> list[Any]:
        t = self._types
        out: list[Any] = []
        for m in messages:
            if m["role"] == "user":
                out.append(t.Content(role="user", parts=[t.Part(text=m["content"])]))
            elif m["role"] == "assistant":
                raw = (m.get("raw") or {}).get("gemini_content")
                if raw:
                    out.append(t.Content.model_validate(raw))
                else:
                    parts = [t.Part(text=m["text"])] if m.get("text") else []
                    parts += [t.Part(function_call=t.FunctionCall(id=_vendor_id(tc["id"]), name=tc["name"],
                                                                  args=tc["args"]))
                              for tc in m["tool_calls"]]
                    out.append(t.Content(role="model", parts=parts))
            elif m["role"] == "tool":
                key = "error" if m["is_error"] else "result"
                part = t.Part.from_function_response(name=m["name"], response={key: m["content"]})
                part.function_response.id = _vendor_id(m["tool_call_id"])
                if out and out[-1].role == "user" and out[-1].parts and out[-1].parts[0].function_response:
                    out[-1].parts.append(part)
                else:
                    out.append(t.Content(role="user", parts=[part]))
        return out


LOCAL_ID = "local:"


def _vendor_id(call_id: str) -> str | None:
    """Only echo ids the vendor issued. Ids we made up for the trace stay local."""
    return None if call_id.startswith(LOCAL_ID) else call_id


def make_model(spec: str, record_to: str | None = None) -> Model:
    from claims_eval.agent.cassette import RecordingModel, ReplayModel
    from claims_eval.agent.simulated import SimulatedModel

    kind, _, arg = spec.partition(":")
    if kind == "sim":
        model: Model = SimulatedModel(arg or "careful")
    elif kind == "anthropic":
        model = AnthropicModel(arg or None)
    elif kind == "gemini":
        model = GeminiModel(arg or None)
    elif kind == "replay":
        if not arg:
            raise HarnessError("replay needs a cassette path: replay:cassettes/run.jsonl")
        return ReplayModel(arg)
    else:
        raise HarnessError(f"unknown model spec {spec!r}")
    return RecordingModel(model, record_to) if record_to else model
