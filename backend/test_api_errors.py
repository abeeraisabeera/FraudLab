"""Fraud API error contracts + temporal split gate."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app import app
from fraud.generator import generate_fraud_dataset
from fraud.schema import FraudGenerateRequest

client = TestClient(app)


def _labeled_txs(n: int = 120, seed: int = 5):
    count = max(n, 100)
    df = generate_fraud_dataset(
        FraudGenerateRequest(
            number_of_users=40,
            number_of_transactions=count,
            fraud_rate=0.08,
            seed=seed,
        )
    )
    rows = df.to_dict(orient="records")
    return rows[:n] if n < count else rows


def test_fraud_evaluate_unlabeled_returns_400():
    txs = _labeled_txs(120, seed=9)
    for row in txs:
        row["is_fraud"] = None
    response = client.post("/fraud/evaluate", json={"transactions": txs, "seed": 9})
    assert response.status_code == 400
    assert "label" in response.json()["detail"].lower()


def test_fraud_analyze_unknown_id_returns_404():
    txs = _labeled_txs(100, seed=4)
    response = client.post(
        "/fraud/analyze",
        json={"transactions": txs, "transaction_id": "does-not-exist", "seed": 4},
    )
    assert response.status_code == 404


def test_fraud_policy_threshold_order_422():
    response = client.post(
        "/fraud/evaluate",
        json={
            "transactions": _labeled_txs(80, seed=2),
            "seed": 2,
            "policy": {"allow_threshold": 0.8, "review_threshold": 0.2, "block_threshold": 0.2},
        },
    )
    assert response.status_code == 422


def test_fraud_evaluate_too_few_transactions_422():
    txs = _labeled_txs(100, seed=6)[:40]
    response = client.post("/fraud/evaluate", json={"transactions": txs, "seed": 6})
    assert response.status_code == 422


def test_fraud_generate_over_transaction_cap_422():
    response = client.post(
        "/fraud/generate",
        json={
            "number_of_users": 50,
            "number_of_transactions": 20_001,
            "fraud_rate": 0.05,
            "seed": 1,
        },
    )
    assert response.status_code == 422


def test_fraud_evaluate_reports_temporal_split():
    response = client.post(
        "/fraud/evaluate",
        json={"transactions": _labeled_txs(200, seed=17), "seed": 17, "test_size": 0.3},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["split"]["split_method"] == "temporal"
    assert body["evaluation_mode"] == "holdout"
    roles = {row["split_role"] for row in body["scored_transactions"]}
    assert roles == {"train", "holdout"}
