from __future__ import annotations

from fastapi.testclient import TestClient

from app import app
from fraud.generator import generate_fraud_dataset
from fraud.schema import FraudGenerateRequest

client = TestClient(app)


def test_fraud_policy_endpoint():
    response = client.get("/fraud/policy")
    assert response.status_code == 200
    body = response.json()
    assert body["policy"]["version"] == "policy-v1"
    assert "allow_threshold" in body["policy"]


def test_fraud_generate_schema_and_seed():
    payload = {
        "number_of_users": 50,
        "number_of_transactions": 500,
        "fraud_rate": 0.04,
        "time_range_days": 14,
        "seed": 123,
    }
    r1 = client.post("/fraud/generate", json=payload)
    r2 = client.post("/fraud/generate", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    body = r1.json()
    assert len(body["transactions"]) == 500
    assert "transaction_id" in body["transactions"][0]
    assert "amount" in body["transactions"][0]
    assert body["transactions"] == r2.json()["transactions"]


def test_fraud_generate_rejects_invalid():
    response = client.post(
        "/fraud/generate",
        json={"number_of_users": 1, "number_of_transactions": 10, "fraud_rate": 0.05},
    )
    assert response.status_code == 422


def test_fraud_evaluate_and_analyze():
    df = generate_fraud_dataset(
        FraudGenerateRequest(
            number_of_users=60,
            number_of_transactions=600,
            fraud_rate=0.07,
            seed=3,
        )
    )
    transactions = df.to_dict(orient="records")

    eval_resp = client.post(
        "/fraud/evaluate",
        json={
            "transactions": transactions,
            "seed": 3,
            "policy": {"allow_threshold": 0.2, "review_threshold": 0.55, "block_threshold": 0.55},
            "costs": {"false_positive_cost": 5, "false_negative_cost": 100, "review_cost": 2},
        },
    )
    assert eval_resp.status_code == 200, eval_resp.text
    eval_body = eval_resp.json()
    evaluation = eval_body["evaluation"]
    assert eval_body["evaluation_mode"] == "holdout"
    assert evaluation["evaluation_mode"] == "holdout"
    assert "pr_auc" in evaluation
    assert "brier_score" in evaluation
    assert "threshold_cost_analysis" in evaluation
    assert "confusion_matrix" in evaluation
    assert "n" in evaluation["calibration_curve"][0] or evaluation["calibration_curve"] == []

    scored = eval_body["scored_transactions"]
    assert "split_role" in scored[0]
    target_row = next(r for r in scored if r["split_role"] == "holdout")
    target = target_row["transaction_id"]
    analyze_resp = client.post(
        "/fraud/analyze",
        json={
            "transactions": transactions,
            "transaction_id": target,
            "seed": 3,
            "policy": {"allow_threshold": 0.2, "review_threshold": 0.55, "block_threshold": 0.55},
        },
    )
    assert analyze_resp.status_code == 200, analyze_resp.text
    analyze_body = analyze_resp.json()
    evidence = analyze_body["evidence"]
    assert evidence["transaction_id"] == target
    assert evidence["decision"] == target_row["decision"]
    assert abs(evidence["risk_probability"] - target_row["risk_probability"]) < 1e-9
    assert analyze_body["evaluation_mode"] == "holdout"
    assert "top_contributors" in evidence
    assert "features" in evidence
    assert "calibration" in evidence
    assert "positive_contributions" in evidence
    assert "negative_contributions" in evidence


def test_existing_health_still_works():
    assert client.get("/health").json()["status"] == "ok"
