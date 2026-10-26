import tomllib
from pathlib import Path

import pytest

from claims_eval.agent.types import HarnessError
from claims_eval.evals import gate, metrics

CONFIG = tomllib.loads(Path("eval_config.toml").read_text())


def _rec(case_id, run, passed, critical=(), infra=None, action=("release", None)):
    return {"case_id": case_id, "run": run, "passed": None if infra else passed,
            "critical_failures": list(critical), "infra_error": infra,
            "scores": [] if infra else [{"name": "action_correct", "passed": passed, "critical": False}],
            "output": None if infra else {"action": action[0], "queue": action[1]},
            "latency_ms": 10, "usage": {"input_tokens": 1, "output_tokens": 1, "estimated": True}}


def _cases(n, dataset="golden", lob=lambda i: "commercial"):
    return [{"case_id": f"C{i}", "dataset": dataset,
             "tags": {"line_of_business": lob(i), "provider_type": "professional", "counterfactual_group": f"C{i}"}}
            for i in range(n)]


def test_pass_hat_k_punishes_flakiness_that_pass_rate_hides():
    cases = _cases(4)
    records = [_rec(f"C{i}", r, r != 0) for i in range(4) for r in range(3)]  # every case passes 2 of 3
    m = metrics.compute(records, cases, CONFIG)
    assert m["pass_rate"] == round(8 / 12, 4)
    assert m["pass_hat_k"] == 0.0
    assert m["flaky_cases"] == ["C0", "C1", "C2", "C3"]


def test_infra_errors_are_excluded_not_counted_as_clean():
    cases = _cases(2)
    records = [_rec("C0", 0, True), _rec("C1", 0, None, infra="RateLimitError")]
    m = metrics.compute(records, cases, CONFIG)
    assert m["scored_runs"] == 1 and m["infra_errors"] == 1 and m["infra_error_rate"] == 0.5
    assert "C1" not in m["per_case"]


def test_counterfactual_flip_detected_on_usual_decision():
    base = {"case_id": "G", "dataset": "golden",
            "tags": {"line_of_business": "commercial", "provider_type": "professional", "counterfactual_group": "G"}}
    copy = {"case_id": "G-MCD", "dataset": "counterfactual",
            "tags": {"line_of_business": "medicaid", "provider_type": "professional", "counterfactual_group": "G"}}
    records = [_rec("G", r, True) for r in range(3)]
    # One noisy run on the copy doesn't flip the usual decision...
    noisy = [_rec("G-MCD", 0, False, action=("request_info", None))] + [_rec("G-MCD", r, True) for r in (1, 2)]
    assert metrics.compute(records + noisy, [base, copy], CONFIG)["counterfactual"]["flips"] == 0
    # ...but a consistent difference does.
    biased = [_rec("G-MCD", r, False, action=("request_info", None)) for r in range(3)]
    assert metrics.compute(records + biased, [base, copy], CONFIG)["counterfactual"]["flips"] == 1
    # A 1-1 split has no usual decision: inconclusive, not a flip.
    split = [_rec("G-MCD", 0, False, action=("request_info", None)), _rec("G-MCD", 1, True)]
    cf = metrics.compute(records + split, [base, copy], CONFIG)["counterfactual"]
    assert cf["flips"] == 0 and cf["inconclusive_groups"] == ["G"]


def _metrics(per_case, **over):
    base = {"safety_failures": 0, "safety_failures_by_scorer": {}, "safety_rate_upper_95": None,
            "infra_errors": 0, "infra_error_rate": 0.0,
            "counterfactual": {"flips": 0, "groups_compared": 0, "flipped_groups": {}},
            "per_case": per_case}
    rows = per_case.values()
    base["pass_rate"] = sum(r["passes"] for r in rows) / sum(r["runs"] for r in rows)
    base["pass_hat_k"] = sum(r["passes"] == r["runs"] for r in rows) / len(per_case)
    base["action_accuracy"] = sum(r["action_correct"] for r in rows) / sum(r["runs"] for r in rows)
    return {**base, **over}


def _per_case(n=40, failing=()):
    return {f"C{i}": {"runs": 3, "passes": 1 if i in failing else 3, "action_correct": 1 if i in failing else 3}
            for i in range(n)}


def test_gate_tolerates_noise_on_a_single_case():
    current = _metrics(_per_case(failing={7}))
    checks = gate.evaluate(current, _metrics(_per_case()), CONFIG)
    assert gate.passed(checks), [c for c in checks if not c.passed]


def test_gate_blocks_a_real_regression():
    current = _metrics(_per_case(failing=set(range(8))))
    failed = {c.name for c in gate.evaluate(current, _metrics(_per_case()), CONFIG) if not c.passed}
    assert {"pass_rate", "pass_hat_k", "action_accuracy"} <= failed


def test_one_safety_failure_blocks_even_if_baseline_had_it():
    checks = gate.evaluate(_metrics(_per_case(), safety_failures=1), _metrics(_per_case(), safety_failures=1), CONFIG)
    assert not gate.passed(checks)


def test_gate_blocks_on_infra_error_rate():
    checks = gate.evaluate(_metrics(_per_case(), infra_error_rate=0.2, infra_errors=24), _metrics(_per_case()), CONFIG)
    assert [c.name for c in checks if not c.passed] == ["infra_error_rate"]


def test_gate_refuses_incomparable_runs():
    base = {"case_set_hash": "a", "runs": 3, "config_hash": "x"}
    gate.require_comparable(dict(base), base)
    for change in ({"runs": 1}, {"case_set_hash": "b"}, {"config_hash": "y"}):
        with pytest.raises(HarnessError, match="not comparable"):
            gate.require_comparable({**base, **change}, base)
