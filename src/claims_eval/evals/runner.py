"""Runs every case N times against one model and scores each run."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from claims_eval.agent.loop import AgentResult, run_agent
from claims_eval.agent.tools import ToolBox, Transport, open_toolbox
from claims_eval.agent.types import Model
from claims_eval.evals.scorers import score_run


def load_cases(dataset_dir: Path, names: list[str]) -> list[dict[str, Any]]:
    cases = []
    for name in names:
        path = dataset_dir / f"{name}.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                case = json.loads(line)
                case["dataset"] = name
                cases.append(case)
    return cases


async def _run_one(case, run_idx, model: Model, toolbox: ToolBox, config, sem) -> dict[str, Any]:
    async with sem:
        result: AgentResult = await run_agent(case["instruction"], model, toolbox,
                                              call_prefix=f"{case['case_id']}/r{run_idx}")
    scores = score_run(case, result, config)
    return {
        "case_id": case["case_id"],
        "run": run_idx,
        "passed": all(s.passed for s in scores),
        "critical_failures": [s.name for s in scores if s.critical and not s.passed],
        "scores": [asdict(s) for s in scores],
        "output": result.output,
        "final_text": result.final_text if result.output is None else None,
        "trace": [asdict(step) for step in result.trace],
        "usage": asdict(result.usage),
        "turns": result.turns,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }


async def run_suite(cases: list[dict[str, Any]], model: Model, config: dict[str, Any], runs: int,
                    transport: Transport = "stdio", concurrency: int = 4,
                    server_log: Path | None = None) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    async with open_toolbox(transport, server_log) as toolbox:
        jobs = [_run_one(case, r, model, toolbox, config, sem) for case in cases for r in range(runs)]
        return await asyncio.gather(*jobs)
