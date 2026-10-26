"""Record a live run once, replay it for free.

A cassette is JSONL: one line per model turn, keyed by call_id ("<case>/r<run>/t<turn>"), plus a
hash of exactly what the model was sent. On replay, if the prompt, tool definitions, or prior
messages changed, the hash will not match and replay stops with an error instead of silently
scoring stale answers. That is the cassette equivalent of a snapshot test going red.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from claims_eval.agent.types import HarnessError, InfraError, Model, ModelTurn, ToolSpec


def request_hash(system: str, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> str:
    # Vendor-native "raw" payloads are excluded: they are outputs, not inputs we control.
    clean = [{k: v for k, v in m.items() if k != "raw"} for m in messages]
    payload = json.dumps({"system": system, "messages": clean,
                          "tools": [[t.name, t.description, t.input_schema] for t in tools]}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class RecordingModel:
    def __init__(self, inner: Model, path: str):
        self.inner = inner
        self.name = inner.name
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # a recording replaces any previous cassette at this path
        header = {"model": inner.name}
        with self.path.open("a") as f:
            f.write(json.dumps({"header": header}) + "\n")

    async def complete(self, system, messages, tools, call_id) -> ModelTurn:
        entry: dict[str, Any] = {"call_id": call_id, "request_hash": request_hash(system, messages, tools)}
        try:
            turn = await self.inner.complete(system, messages, tools, call_id)
        except Exception as exc:
            # Record failures too, so replay reproduces an infra error instead of aborting on a missing turn.
            # If a retry succeeds, its entry comes later in the file and wins on replay.
            entry["error"] = f"{type(exc).__name__}: {exc}"
            self._write(entry)
            raise
        entry["turn"] = turn.to_dict()
        self._write(entry)
        return turn

    def _write(self, entry: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")


class ReplayModel:
    def __init__(self, path: str):
        p = Path(path)
        if not p.exists():
            raise HarnessError(f"cassette not found: {path}")
        lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        header = lines[0].get("header", {}) if lines else {}
        self.name = f"replay:{header.get('model', 'unknown')}"
        self._entries = {e["call_id"]: e for e in lines if "call_id" in e}

    async def complete(self, system, messages, tools, call_id) -> ModelTurn:
        entry = self._entries.get(call_id)
        if entry is None:
            raise HarnessError(f"cassette has no turn {call_id}; re-record (did --runs or the dataset change?)")
        if entry["request_hash"] != request_hash(system, messages, tools):
            raise HarnessError(f"cassette is stale at {call_id}: the prompt, tools, or tool results changed. "
                               "Re-record with a live model.")
        if "error" in entry:
            raise InfraError(f"recorded failure: {entry['error']}")
        return ModelTurn.from_dict(entry["turn"])
