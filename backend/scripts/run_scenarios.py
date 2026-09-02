"""Run fraud-lab API scenarios and write comparison logs."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parents[2] / "test-logs" / "fraud-lab-scenarios.json"


def post(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = Request(
        f"{BASE}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=120) as resp:
            return {"status": resp.status, "body": json.loads(resp.read())}
    except HTTPError as exc:
        detail = exc.read().decode()
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            pass
        return {"status": exc.code, "error": detail}


def get(path: str) -> dict:
    with urlopen(f"{BASE}{path}", timeout=30) as resp:
        return {"status": resp.status, "body": json.loads(resp.read())}


def summarize_generate(body: dict) -> dict:
    txs = body["transactions"]
    frauds = sum(1 for t in txs if t.get("is_fraud") == 1)
    return {
        "count": len(txs),
        "users": body["meta"]["number_of_users"],
        "requested_fraud_rate": body["meta"]["requested_fraud_rate"],
        "realized_fraud_rate": round(body["meta"]["realized_fraud_rate"], 4),
        "fraud_count": frauds,
        "sample_tx": txs[0]["transaction_id"],
    }


def summarize_evaluate(body: dict) -> dict:
    ev = body["evaluation"]
    cm = ev["confusion_matrix"]
    return {
        "pr_auc": round(ev["pr_auc"], 4),
        "precision": round(ev["precision"], 4),
        "recall": round(ev["recall"], 4),
        "f1": round(ev["f1"], 4),
        "brier": round(ev["brier_score"], 4),
        "expected_cost": round(ev["expected_cost"], 2),
        "allow_rate": round(ev["allow_rate"], 4),
        "review_rate": round(ev["review_rate"], 4),
        "block_rate": round(ev["block_rate"], 4),
        "confusion": cm,
        "holdout_n": body["split"]["n_test"],
        "train_n": body["split"]["n_train"],
        "top_risk": max(body["scored_transactions"], key=lambda r: r["risk_probability"])[
            "transaction_id"
        ],
        "top_risk_score": round(
            max(r["risk_probability"] for r in body["scored_transactions"]), 4
        ),
    }


def summarize_analyze(body: dict) -> dict:
    e = body["evidence"]
    top = e.get("top_contributors") or e.get("feature_contributions", [])[:3]
    return {
        "transaction_id": e["transaction_id"],
        "decision": e["decision"],
        "risk_probability": round(e["risk_probability"], 4),
        "top_features": [
            {"feature": c["feature"], "contribution": round(c["contribution"], 4)}
            for c in top[:3]
        ],
    }


def run_scenario(name: str, fn) -> dict:
    started = time.perf_counter()
    try:
        result = fn()
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001
        result = None
        ok = False
        err = str(exc)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    return {"name": name, "ok": ok, "elapsed_ms": elapsed_ms, "error": err, "result": result}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    log: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "api_base": BASE,
        "scenarios": [],
    }

    # Health + policy
    log["scenarios"].append(
        run_scenario("health_check", lambda: {"health": get("/health")["body"]})
    )
    log["scenarios"].append(
        run_scenario("default_policy", lambda: {"policy": get("/fraud/policy")["body"]})
    )

    # Generate variants
    gen_default = post(
        "/fraud/generate",
        {
            "number_of_users": 300,
            "number_of_transactions": 2500,
            "fraud_rate": 0.04,
            "time_range_days": 30,
            "seed": 42,
            "hard_fraud_rate": 0.25,
            "suspicious_legit_rate": 0.18,
        },
    )
    gen_low_fraud = post(
        "/fraud/generate",
        {
            "number_of_users": 200,
            "number_of_transactions": 1500,
            "fraud_rate": 0.01,
            "time_range_days": 30,
            "seed": 42,
        },
    )
    gen_high_fraud = post(
        "/fraud/generate",
        {
            "number_of_users": 200,
            "number_of_transactions": 1500,
            "fraud_rate": 0.15,
            "time_range_days": 30,
            "seed": 42,
        },
    )
    gen_seed_repeat = post(
        "/fraud/generate",
        {
            "number_of_users": 300,
            "number_of_transactions": 2500,
            "fraud_rate": 0.04,
            "time_range_days": 30,
            "seed": 42,
        },
    )
    gen_seed_diff = post(
        "/fraud/generate",
        {
            "number_of_users": 300,
            "number_of_transactions": 2500,
            "fraud_rate": 0.04,
            "time_range_days": 30,
            "seed": 99,
        },
    )
    gen_small = post(
        "/fraud/generate",
        {
            "number_of_users": 50,
            "number_of_transactions": 500,
            "fraud_rate": 0.04,
            "time_range_days": 14,
            "seed": 7,
        },
    )

    txs_default = gen_default["body"]["transactions"]
    txs_small = gen_small["body"]["transactions"]

    log["scenarios"].append(
        run_scenario(
            "generate_default",
            lambda: summarize_generate(gen_default["body"]),
        )
    )
    log["scenarios"].append(
        run_scenario("generate_low_fraud_rate", lambda: summarize_generate(gen_low_fraud["body"]))
    )
    log["scenarios"].append(
        run_scenario("generate_high_fraud_rate", lambda: summarize_generate(gen_high_fraud["body"]))
    )
    log["scenarios"].append(
        run_scenario(
            "generate_same_seed_reproducible",
            lambda: {
                "first_sample": gen_default["body"]["transactions"][0]["transaction_id"],
                "repeat_sample": gen_seed_repeat["body"]["transactions"][0]["transaction_id"],
                "identical": gen_default["body"]["transactions"]
                == gen_seed_repeat["body"]["transactions"],
            },
        )
    )
    log["scenarios"].append(
        run_scenario(
            "generate_different_seed",
            lambda: {
                "seed_42_sample": gen_default["body"]["transactions"][0]["transaction_id"],
                "seed_99_sample": gen_seed_diff["body"]["transactions"][0]["transaction_id"],
                "identical": gen_default["body"]["transactions"]
                == gen_seed_diff["body"]["transactions"],
            },
        )
    )

    # Evaluate variants on default dataset
    eval_default = post(
        "/fraud/evaluate",
        {
            "transactions": txs_default,
            "seed": 42,
            "test_size": 0.3,
            "policy": {
                "allow_threshold": 0.2,
                "review_threshold": 0.55,
                "block_threshold": 0.55,
            },
            "costs": {
                "false_positive_cost": 5,
                "false_negative_cost": 100,
                "review_cost": 2,
            },
        },
    )
    eval_strict = post(
        "/fraud/evaluate",
        {
            "transactions": txs_default,
            "seed": 42,
            "test_size": 0.3,
            "policy": {
                "allow_threshold": 0.05,
                "review_threshold": 0.25,
                "block_threshold": 0.25,
            },
            "costs": {
                "false_positive_cost": 5,
                "false_negative_cost": 100,
                "review_cost": 2,
            },
        },
    )
    eval_loose = post(
        "/fraud/evaluate",
        {
            "transactions": txs_default,
            "seed": 42,
            "test_size": 0.3,
            "policy": {
                "allow_threshold": 0.5,
                "review_threshold": 0.85,
                "block_threshold": 0.85,
            },
            "costs": {
                "false_positive_cost": 5,
                "false_negative_cost": 100,
                "review_cost": 2,
            },
        },
    )
    eval_fn_heavy = post(
        "/fraud/evaluate",
        {
            "transactions": txs_default,
            "seed": 42,
            "test_size": 0.3,
            "policy": {
                "allow_threshold": 0.2,
                "review_threshold": 0.55,
                "block_threshold": 0.55,
            },
            "costs": {
                "false_positive_cost": 5,
                "false_negative_cost": 500,
                "review_cost": 2,
            },
        },
    )

    log["scenarios"].append(
        run_scenario("evaluate_default_policy", lambda: summarize_evaluate(eval_default["body"]))
    )
    log["scenarios"].append(
        run_scenario("evaluate_strict_policy", lambda: summarize_evaluate(eval_strict["body"]))
    )
    log["scenarios"].append(
        run_scenario("evaluate_loose_policy", lambda: summarize_evaluate(eval_loose["body"]))
    )
    log["scenarios"].append(
        run_scenario(
            "evaluate_high_fn_cost",
            lambda: {
                **summarize_evaluate(eval_fn_heavy["body"]),
                "best_thresholds": eval_fn_heavy["body"]["evaluation"]["threshold_cost_analysis"][
                    "best_thresholds"
                ],
            },
        )
    )

    scored = eval_default["body"]["scored_transactions"]
    highest = max(scored, key=lambda r: r["risk_probability"])
    lowest = min(scored, key=lambda r: r["risk_probability"])

    analyze_high = post(
        "/fraud/analyze",
        {
            "transactions": txs_default,
            "transaction_id": highest["transaction_id"],
            "seed": 42,
            "test_size": 0.3,
            "policy": {
                "allow_threshold": 0.2,
                "review_threshold": 0.55,
                "block_threshold": 0.55,
            },
        },
    )
    analyze_low = post(
        "/fraud/analyze",
        {
            "transactions": txs_default,
            "transaction_id": lowest["transaction_id"],
            "seed": 42,
            "test_size": 0.3,
            "policy": {
                "allow_threshold": 0.2,
                "review_threshold": 0.55,
                "block_threshold": 0.55,
            },
        },
    )

    log["scenarios"].append(
        run_scenario("analyze_highest_risk_tx", lambda: summarize_analyze(analyze_high["body"]))
    )
    log["scenarios"].append(
        run_scenario("analyze_lowest_risk_tx", lambda: summarize_analyze(analyze_low["body"]))
    )

    # Error / edge cases
    log["scenarios"].append(
        run_scenario(
            "error_evaluate_too_few_tx",
            lambda: post("/fraud/evaluate", {"transactions": txs_small[:40], "seed": 7}),
        )
    )
    log["scenarios"].append(
        run_scenario(
            "error_generate_invalid_users",
            lambda: post(
                "/fraud/generate",
                {
                    "number_of_users": 5,
                    "number_of_transactions": 500,
                    "fraud_rate": 0.04,
                    "seed": 1,
                },
            ),
        )
    )
    log["scenarios"].append(
        run_scenario(
            "error_analyze_missing_tx",
            lambda: post(
                "/fraud/analyze",
                {
                    "transactions": txs_small,
                    "transaction_id": "TX-DOES-NOT-EXIST",
                    "seed": 7,
                },
            ),
        )
    )

    # Comparison block
    log["comparison"] = {
        "fraud_rate_effect": {
            "low_1pct": summarize_generate(gen_low_fraud["body"]),
            "default_4pct": summarize_generate(gen_default["body"]),
            "high_15pct": summarize_generate(gen_high_fraud["body"]),
        },
        "policy_effect_on_decisions": {
            "strict": {
                "allow": summarize_evaluate(eval_strict["body"])["allow_rate"],
                "review": summarize_evaluate(eval_strict["body"])["review_rate"],
                "block": summarize_evaluate(eval_strict["body"])["block_rate"],
            },
            "default": {
                "allow": summarize_evaluate(eval_default["body"])["allow_rate"],
                "review": summarize_evaluate(eval_default["body"])["review_rate"],
                "block": summarize_evaluate(eval_default["body"])["block_rate"],
            },
            "loose": {
                "allow": summarize_evaluate(eval_loose["body"])["allow_rate"],
                "review": summarize_evaluate(eval_loose["body"])["review_rate"],
                "block": summarize_evaluate(eval_loose["body"])["block_rate"],
            },
        },
        "risk_extremes": {
            "highest": summarize_analyze(analyze_high["body"]),
            "lowest": summarize_analyze(analyze_low["body"]),
        },
    }

    OUT.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")
    passed = sum(1 for s in log["scenarios"] if s["ok"])
    print(f"Scenarios: {passed}/{len(log['scenarios'])} completed without exception")
    return 0


if __name__ == "__main__":
    sys.exit(main())
