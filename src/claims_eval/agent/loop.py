"""The agent loop: model proposes tool calls, MCP executes them, repeat until a final answer.

The loop records a trace of every tool call. Scorers judge the trace as much as the answer,
because an agent can reach the right action for the wrong reason.

Failures are sorted into two kinds. If the model or the MCP transport breaks (rate limit, bad key,
dead server), the run is an infrastructure error and is excluded from scoring. If the model answers
badly (no JSON, too many turns), that is the agent's failure and gets scored.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from claims_eval.agent.prompts import SYSTEM_PROMPT
from claims_eval.agent.tools import ToolBox
from claims_eval.agent.types import HarnessError, InfraError, Model, ModelTurn, ToolSpec, Usage

MAX_TURNS = 15
MODEL_ATTEMPTS = 3


@dataclass
class TraceStep:
    tool: str
    args: dict[str, Any]
    result: str
    is_error: bool


@dataclass
class AgentResult:
    output: dict[str, Any] | None
    final_text: str | None
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    latency_ms: int = 0
    error: str | None = None        # the agent failed: scored
    infra_error: str | None = None  # the infrastructure failed: not scored
    stop_reason: str | None = None


def parse_output(text: str | None) -> dict[str, Any] | None:
    """Pull the JSON object out of the final message. Tolerates code fences and stray prose."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _complete_with_retry(model: Model, messages: list[dict[str, Any]], tools: list[ToolSpec],
                               call_id: str) -> ModelTurn:
    for attempt in range(MODEL_ATTEMPTS):
        try:
            return await model.complete(SYSTEM_PROMPT, messages, tools, call_id=call_id)
        except (HarnessError, InfraError):
            raise
        except Exception as exc:  # noqa: BLE001 - vendor SDKs raise many types; all are infrastructure here
            if attempt == MODEL_ATTEMPTS - 1:
                raise InfraError(f"model call failed after {MODEL_ATTEMPTS} attempts: "
                                 f"{type(exc).__name__}: {exc}") from exc
            await asyncio.sleep(2 ** attempt)
    raise AssertionError("unreachable")


async def run_agent(instruction: str, model: Model, toolbox: ToolBox, call_prefix: str) -> AgentResult:
    tools = await toolbox.specs()
    messages: list[dict[str, Any]] = [{"role": "user", "content": instruction}]
    result = AgentResult(output=None, final_text=None)
    started = time.perf_counter()

    try:
        for turn_idx in range(MAX_TURNS):
            turn = await _complete_with_retry(model, messages, tools, f"{call_prefix}/t{turn_idx}")
            result.turns += 1
            result.usage.add(turn.usage)
            result.stop_reason = turn.stop_reason
            messages.append({
                "role": "assistant",
                "text": turn.text,
                "tool_calls": [{"id": tc.id, "name": tc.name, "args": tc.args} for tc in turn.tool_calls],
                **({"raw": turn.raw} if turn.raw else {}),
            })
            if not turn.tool_calls:
                result.final_text = turn.text
                result.output = parse_output(turn.text)
                if result.output is None and turn.stop_reason:
                    result.error = f"no JSON answer (model stop reason: {turn.stop_reason})"
                break
            for tc in turn.tool_calls:
                try:
                    text, is_error = await toolbox.call(tc.name, tc.args)
                except Exception as exc:  # noqa: BLE001 - tool errors come back as results; this is transport
                    raise InfraError(f"MCP transport failed on {tc.name}: {type(exc).__name__}: {exc}") from exc
                result.trace.append(TraceStep(tc.name, tc.args, text, is_error))
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                 "content": text, "is_error": is_error})
        else:
            result.error = f"no final answer after {MAX_TURNS} turns"
    except HarnessError:
        raise
    except InfraError as exc:
        result.infra_error = str(exc)

    result.latency_ms = int((time.perf_counter() - started) * 1000)
    return result
