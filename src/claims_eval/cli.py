"""Command line entry point.

    claims-eval run       --model sim:careful         run the suite, write results/run.json + report.html
    claims-eval gate      --results results/run.json  compare to the baseline; exit 1 if blocked
    claims-eval baseline  --results results/run.json  promote a run's metrics to the new baseline
    claims-eval calibrate --judge anthropic           measure judge agreement with human labels
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claims_eval.agent.models import make_model
from claims_eval.agent.types import HarnessError
from claims_eval.evals import gate, metrics, report
from claims_eval.evals.runner import load_cases, run_suite

ROOT = Path.cwd()
DATASETS = ROOT / "evals" / "datasets"


def load_config(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def run_meta(model_name: str, cases: list[dict[str, Any]], runs: int, datasets: list[str],
             config_path: Path) -> dict[str, Any]:
    """What the gate needs to know two runs are comparable, plus what a reader needs to reproduce one."""
    # The case set includes the fixtures behind the cases: changing a policy or a claim changes the test.
    fixtures = sorted((Path(__file__).parent / "mock_facets" / "fixtures").glob("*.json"))
    case_set = json.dumps(sorted([[c["case_id"], c["instruction"], c["expected"]] for c in cases]), sort_keys=True)
    case_set += "".join(f.read_text() for f in fixtures)
    try:
        sha = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True,
                             check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = None
    return {
        "model": model_name,
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": runs,
        "datasets": datasets,
        "case_set_hash": hashlib.sha256(case_set.encode()).hexdigest()[:12],
        "config_hash": hashlib.sha256(config_path.read_bytes()).hexdigest()[:12],
        "git_sha": sha,
        "mcp_sdk": version("mcp"),
        "sampling": "vendor defaults",
    }


def _print_checks(checks: list[gate.Check]) -> None:
    width = max(len(c.name) for c in checks)
    for c in checks:
        base = "" if c.baseline is None else f"  (baseline {report._fmt(c.name, c.baseline)})"
        print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name:<{width}}  {report._fmt(c.name, c.current):>6}"
              f"{base}  limit {c.limit}  {c.note}")


async def _judge_records(spec: str, records: list[dict[str, Any]]) -> None:
    from claims_eval.evals.judge import evidence_from_trace, judge_rationale

    judge_model = make_model(spec)
    for r in records:
        if r["output"] and r["output"].get("rationale"):
            r["judge"] = await judge_rationale(judge_model, r["output"]["rationale"], evidence_from_trace(r["trace"]),
                                               call_id=f"judge/{r['case_id']}/r{r['run']}")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    if not DATASETS.exists():
        raise HarnessError(f"no datasets at {DATASETS}; run claims-eval from the repo root")
    datasets = args.datasets.split(",") if args.datasets else config["run"]["datasets"]
    runs = args.runs or config["run"]["runs"]
    cases = load_cases(DATASETS, datasets)
    model = make_model(args.model, record_to=args.record)
    if args.judge and args.judge.split(":")[0] == args.model.split(":")[0]:
        print(f"warning: judge and agent are both {args.judge.split(':')[0]}; a model family tends to grade "
              "its own habits as correct. Prefer a different family.", file=sys.stderr)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = run_meta(model.name, cases, runs, datasets, Path(args.config))
    print(f"running {len(cases)} cases x {runs} runs against {model.name} ...", flush=True)
    records = asyncio.run(run_suite(cases, model, config, runs, transport=args.transport,
                                    concurrency=args.concurrency, server_log=out / "server.log"))
    if args.judge:
        asyncio.run(_judge_records(args.judge, records))

    m = metrics.compute(records, cases, config)
    if args.judge:
        judged = [r["judge"] for r in records if "judge" in r]
        m["judge"] = {"model": args.judge, "advisory": True, "items": {
            item: round(sum(1 for j in judged if j.get(item)) / len(judged), 4) if judged else None
            for item in ("explains_pend_reason", "states_next_step", "no_unsupported_claims")}}

    baseline = _load_json(Path(args.baseline))
    checks = None
    if baseline:
        try:
            gate.require_comparable(meta, baseline["meta"])
            checks = gate.evaluate(m, baseline["metrics"], config)
        except HarnessError as exc:
            print(f"not gated: {exc}")

    (out / "run.json").write_text(json.dumps({"meta": meta, "metrics": m, "records": records}, indent=2))
    (out / "report.html").write_text(report.html_report(meta, m, records, cases, checks))
    (out / "summary.md").write_text(report.markdown_summary(meta, m, checks))

    print(f"pass rate {m['pass_rate']:.1%} · pass^{runs} {m['pass_hat_k']:.1%} · "
          f"action accuracy {m['action_accuracy']:.1%} · safety failures {m['safety_failures']} · "
          f"counterfactual flips {m['counterfactual']['flips']} · infra errors {m['infra_errors']}")
    if checks:
        print(f"gate vs {args.baseline}: {'PASS' if gate.passed(checks) else 'BLOCKED'}")
        _print_checks(checks)
    print(f"report: {out / 'report.html'}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    run = _load_json(Path(args.results))
    if run is None:
        raise HarnessError(f"no results at {args.results}; run `claims-eval run` first")
    baseline = _load_json(Path(args.baseline))
    if baseline:
        gate.require_comparable(run["meta"], baseline["meta"])
    checks = gate.evaluate(run["metrics"], baseline["metrics"] if baseline else None, config)
    ok = gate.passed(checks)
    print(f"gate: {'PASS' if ok else 'BLOCKED'}  ({run['meta']['model']} vs {args.baseline})")
    _print_checks(checks)
    if args.summary:
        with open(args.summary, "a") as f:
            f.write(report.markdown_summary(run["meta"], run["metrics"], checks))
    return 0 if ok else 1


def cmd_baseline(args: argparse.Namespace) -> int:
    run = _load_json(Path(args.results))
    if run is None:
        raise HarnessError(f"no results at {args.results}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": run["meta"], "metrics": run["metrics"]}, indent=2) + "\n")
    print(f"baseline written to {args.out} from {run['meta']['model']}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from claims_eval.evals.judge import calibrate

    config = load_config(Path(args.config))
    result = asyncio.run(calibrate(make_model(args.judge), DATASETS / "judge_calibration.jsonl"))
    min_kappa = config["judge"]["min_kappa"]
    result["meets_bar"] = all((v["kappa"] or 0) >= min_kappa for v in result["items"].values())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    for item, v in result["items"].items():
        print(f"  {item:<24} agreement {v['agreement']}  kappa {v['kappa']}  unparsed {v['unparsed']}  "
              f"human-false examples {v['human_false']}")
    verdict = "meets" if result["meets_bar"] else "does NOT meet"
    print(f"judge {verdict} the kappa bar ({min_kappa}) on {result['examples']} examples; that is too few to "
          f"rely on, so judge results stay advisory either way. Wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="claims-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="eval_config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the eval suite")
    r.add_argument("--model", default="sim:careful",
                   help="sim[:naive|careful|regressed], anthropic[:id], gemini[:id], replay:<cassette>")
    r.add_argument("--record", help="also write a cassette of every model turn to this path")
    r.add_argument("--runs", type=int, help="repeats per case (default from config)")
    r.add_argument("--datasets", help="comma-separated, e.g. golden,adversarial")
    r.add_argument("--out", default="results")
    r.add_argument("--baseline", default="baselines/baseline.json")
    r.add_argument("--transport", choices=["stdio", "inproc"], default="stdio")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--judge", help="model spec for the advisory LLM judge, e.g. gemini:<id>")
    r.set_defaults(fn=cmd_run)

    g = sub.add_parser("gate", help="compare a run to the baseline; exit 1 if blocked")
    g.add_argument("--results", default="results/run.json")
    g.add_argument("--baseline", default="baselines/baseline.json")
    g.add_argument("--summary", help="append a Markdown summary here (e.g. $GITHUB_STEP_SUMMARY)")
    g.set_defaults(fn=cmd_gate)

    b = sub.add_parser("baseline", help="promote a run to the baseline")
    b.add_argument("--results", default="results/run.json")
    b.add_argument("--out", default="baselines/baseline.json")
    b.set_defaults(fn=cmd_baseline)

    c = sub.add_parser("calibrate", help="measure LLM judge agreement with human labels")
    c.add_argument("--judge", required=True)
    c.add_argument("--out", default="results/judge_calibration.json")
    c.set_defaults(fn=cmd_calibrate)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except HarnessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
