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
from claims_eval.evals.scorers import TRACE_SAFETY_SCORERS, run_passed, score_run


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
    if result.infra_error:
        # The run didn't finish, so it isn't scored for quality. But anything unsafe the agent already did
        # before the outage (a deny_claim call, PHI sent to a tool) still counts.
        scores = score_run(case, result, config, only=TRACE_SAFETY_SCORERS)
        passed = None
        critical = [s.name for s in scores if s.passed is False]
    else:
        scores = score_run(case, result, config)
        passed = run_passed(scores)
        critical = [s.name for s in scores if s.critical and s.passed is False]
    return {
        "case_id": case["case_id"],
        "run": run_idx,
        "passed": passed,
        "critical_failures": critical,
        "scores": [asdict(s) for s in scores],
        "output": result.output,
        "final_text": result.final_text,
        "trace": [asdict(step) for step in result.trace],
        "usage": asdict(result.usage),
        "turns": result.turns,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "infra_error": result.infra_error,
        "stop_reason": result.stop_reason,
    }


async def run_suite(cases: list[dict[str, Any]], model: Model, config: dict[str, Any], runs: int,
                    transport: Transport = "stdio", concurrency: int = 4,
                    server_log: Path | None = None) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    async with open_toolbox(transport, server_log) as toolbox:
        jobs = [_run_one(case, r, model, toolbox, config, sem) for case in cases for r in range(runs)]
        return await asyncio.gather(*jobs)
