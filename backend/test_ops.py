"""Ops / calibration / generator polish tests for Fraud Lab."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import app
from fraud.calibration import choose_calibration_method
from fraud.evaluation import expected_cost_from_counts, policy_confusion
from fraud.features import compute_features, feature_matrix
from fraud.generator import generate_fraud_dataset
from fraud.model import _classify_term
from fraud.schema import CostConfig, FraudGenerateRequest, PolicyConfig

client = TestClient(app)


def _tx(**overrides):
    base = {
        "transaction_id": "TX-1",
        "user_id": "U-1",
        "timestamp": datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc),
        "amount": 50.0,
        "currency": "USD",
        "merchant_id": "M-1",
        "merchant_category": "retail",
        "payment_method": "card_online",
        "device_id": "DEV-1",
        "ip_address": "10.0.0.1",
        "country": "US",
        "billing_country": "US",
        "account_age_days": 100,
        "failed_attempts": 0,
        "previous_chargebacks": 0,
        "shipping_country": "US",
    }
    base.update(overrides)
    return base


def test_duplicate_transaction_id_rejected():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        _tx(transaction_id="DUP", timestamp=t0),
        _tx(transaction_id="DUP", timestamp=t0 + timedelta(minutes=1), amount=80.0),
    ]
    with pytest.raises(ValueError, match="Duplicate transaction_id"):
        compute_features(rows)


def test_missingness_indicator_and_fill_policy():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        _tx(transaction_id="TX-0", timestamp=t0, amount=40.0),
        _tx(transaction_id="TX-1", timestamp=t0 + timedelta(hours=1), amount=80.0),
    ]
    feats = compute_features(rows)
    assert feats.loc[0, "amount_history_missing"] == 1.0
    X, names = feature_matrix(feats)
    assert "shipping_country_mismatch" in names
    assert np.isfinite(X).all()


def test_forced_isotonic_downgraded_on_tiny_fold():
    assert choose_calibration_method(50, forced="isotonic") == "sigmoid"


def test_review_fn_cost_option():
    costs_default = CostConfig(false_positive_cost=5, false_negative_cost=100, review_cost=2)
    costs_charge = CostConfig(
        false_positive_cost=5,
        false_negative_cost=100,
        review_cost=2,
        charge_false_negative_on_review=True,
    )
    base = expected_cost_from_counts(tp=0, fp=0, tn=0, fn=0, reviews=1, costs=costs_default, reviewed_fraud=1)
    charged = expected_cost_from_counts(tp=0, fp=0, tn=0, fn=0, reviews=1, costs=costs_charge, reviewed_fraud=1)
    assert base["expected_cost"] == 2.0
    assert charged["expected_cost"] == 102.0


def test_policy_confusion_tracks_reviewed_fraud():
    y = np.array([1, 0, 1])
    p = np.array([0.3, 0.3, 0.9])
    policy = PolicyConfig(allow_threshold=0.2, review_threshold=0.55, block_threshold=0.55)
    conf = policy_confusion(y, p, policy)
    assert conf["reviewed_fraud"] == 1


def test_generator_hard_fraud_knob_changes_overlap():
    calm = generate_fraud_dataset(
        FraudGenerateRequest(
            number_of_users=80,
            number_of_transactions=800,
            fraud_rate=0.1,
            seed=11,
            hard_fraud_rate=0.0,
            suspicious_legit_rate=0.0,
        )
    )
    noisy = generate_fraud_dataset(
        FraudGenerateRequest(
            number_of_users=80,
            number_of_transactions=800,
            fraud_rate=0.1,
            seed=11,
            hard_fraud_rate=1.0,
            suspicious_legit_rate=1.0,
        )
    )
    assert not np.allclose(calm["amount"].to_numpy(), noisy["amount"].to_numpy())


def test_term_type_classification():
    assert _classify_term("amount_zscore_user")["term_type"] == "main"
    interaction = _classify_term("amount_zscore_user & new_device")
    assert interaction["term_type"] == "interaction"


def test_api_key_optional_by_default():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["limits"]["api_key_required"] is False


def test_api_key_enforced(monkeypatch):
    monkeypatch.setattr("app.API_KEY", "secret-test-key")
    denied = client.post(
        "/fraud/generate",
        json={"number_of_users": 40, "number_of_transactions": 120, "fraud_rate": 0.05, "seed": 1},
    )
    assert denied.status_code == 401
    ok = client.post(
        "/fraud/generate",
        json={"number_of_users": 40, "number_of_transactions": 120, "fraud_rate": 0.05, "seed": 1},
        headers={"X-API-Key": "secret-test-key"},
    )
    assert ok.status_code == 200
