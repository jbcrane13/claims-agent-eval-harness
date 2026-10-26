import tomllib
from pathlib import Path

from claims_eval.evals import gate, metrics

CONFIG = tomllib.loads(Path("eval_config.toml").read_text())


def _rec(case_id, run, passed, critical=()):
    return {"case_id": case_id, "run": run, "passed": passed, "critical_failures": list(critical),
            "scores": [{"name": "action_correct", "passed": passed}],
            "latency_ms": 10, "usage": {"input_tokens": 1, "output_tokens": 1, "estimated": True}}


def _cases(n, lob=lambda i: "commercial"):
    return [{"case_id": f"C{i}", "dataset": "golden",
             "tags": {"line_of_business": lob(i), "provider_type": "professional", "region": "urban"}}
            for i in range(n)]


def test_pass_hat_k_punishes_flakiness_that_pass_rate_hides():
    cases = _cases(4)
    # Every case passes 2 of 3 runs: pass_rate looks decent, pass^3 is zero.
    records = [_rec(f"C{i}", r, r != 0) for i in range(4) for r in range(3)]
    m = metrics.compute(records, cases, CONFIG)
    assert m["pass_rate"] == round(8 / 12, 4)
    assert m["pass_hat_k"] == 0.0
    assert m["flaky_cases"] == ["C0", "C1", "C2", "C3"]


def test_small_slices_are_not_compared():
    cases = _cases(8, lob=lambda i: "commercial" if i < 6 else "medicaid")
    records = [_rec(f"C{i}", 0, i < 6) for i in range(8)]  # medicaid (2 cases) all fail
    s = metrics.compute(records, cases, CONFIG)["slices"]["line_of_business"]
    assert s["compared_groups"] == 1 and s["max_gap"] == 0.0


def _metrics(**over):
    base = {"safety_failures": 0, "safety_failures_by_scorer": {}, "pass_rate": 0.95, "pass_hat_k": 0.9,
            "action_accuracy": 0.98, "slices": {"region": {"max_gap": 0.05, "compared_groups": 2}}}
    return {**base, **over}


def test_gate_passes_within_tolerance():
    checks = gate.evaluate(_metrics(pass_rate=0.93), _metrics(), CONFIG)
    assert gate.passed(checks)


def test_gate_blocks_on_metric_drop():
    checks = gate.evaluate(_metrics(pass_hat_k=0.80), _metrics(), CONFIG)
    assert not gate.passed(checks)
    assert [c.name for c in checks if not c.passed] == ["pass_hat_k"]


def test_one_safety_failure_blocks_even_if_baseline_had_it():
    checks = gate.evaluate(_metrics(safety_failures=1), _metrics(safety_failures=1), CONFIG)
    assert not gate.passed(checks)


def test_gate_blocks_on_slice_gap():
    current = _metrics(slices={"region": {"max_gap": 0.25, "compared_groups": 2}})
    assert not gate.passed(gate.evaluate(current, _metrics(), CONFIG))
