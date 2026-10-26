"""Provider-neutral types shared by the agent loop and every model adapter.

Messages use one small format so the loop, the simulator, and the cassettes never
care which vendor produced a turn:

    {"role": "user", "content": "..."}
    {"role": "assistant", "text": "..." | None, "tool_calls": [{"id", "name", "args"}], "raw": {...}?}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "...", "is_error": bool}

"raw" holds a vendor's native turn when the vendor needs it echoed back (Gemini thought signatures).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


class HarnessError(Exception):
    """A problem with the harness itself (stale cassette, bad config). Aborts the run instead of
    being scored as an agent failure."""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = False  # True when tokens are approximated (the simulator)

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.estimated = self.estimated or other.estimated


@dataclass
class ModelTurn:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelTurn:
        return cls(
            text=d.get("text"),
            tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
            usage=Usage(**d.get("usage", {})),
            raw=d.get("raw"),
        )


class Model(Protocol):
    """Anything that can take one turn in the agent loop.

    call_id identifies the turn as "<case_id>/r<run>/t<turn>". Live models ignore it; the simulator
    uses it to seed its faults and the cassettes use it as the lookup key.
    """

    name: str

    async def complete(
        self, system: str, messages: list[dict[str, Any]], tools: list[ToolSpec], call_id: str
    ) -> ModelTurn: ...
