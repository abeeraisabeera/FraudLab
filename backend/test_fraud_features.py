from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fraud.features import compute_features, provenance_for_row


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


def test_velocity_and_amount_deviation():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        _tx(transaction_id="TX-0", timestamp=t0, amount=40.0),
        _tx(transaction_id="TX-1", timestamp=t0 + timedelta(minutes=2), amount=42.0),
        _tx(transaction_id="TX-2", timestamp=t0 + timedelta(minutes=5), amount=200.0),
        _tx(transaction_id="TX-3", timestamp=t0 + timedelta(hours=2), amount=45.0),
    ]
    feats = compute_features(rows)
    by_id = feats.set_index("transaction_id")

    assert by_id.loc["TX-0", "transactions_10m"] == 1
    assert by_id.loc["TX-2", "transactions_10m"] == 3
    assert by_id.loc["TX-3", "transactions_10m"] == 1
    assert by_id.loc["TX-3", "transactions_1h"] == 1
    assert by_id.loc["TX-2", "transactions_1h"] == 3

    # First tx: insufficient history for amount stats
    assert np.isnan(by_id.loc["TX-0", "amount_vs_user_median"])
    assert by_id.loc["TX-2", "amount_vs_user_median"] > 1.5
    assert by_id.loc["TX-1", "time_since_previous_transaction"] == 120.0


def test_novelty_and_geo_mismatch():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        _tx(transaction_id="TX-0", timestamp=t0, device_id="DEV-1", country="US", billing_country="US"),
        _tx(
            transaction_id="TX-1",
            timestamp=t0 + timedelta(hours=1),
            device_id="DEV-2",
            country="DE",
            billing_country="US",
            shipping_country="FR",
            ip_address="10.0.0.9",
        ),
    ]
    feats = compute_features(rows).set_index("transaction_id")
    assert feats.loc["TX-0", "new_device"] == 1.0
    assert feats.loc["TX-1", "new_device"] == 1.0
    assert feats.loc["TX-1", "new_country"] == 1.0
    assert feats.loc["TX-1", "new_ip"] == 1.0
    assert feats.loc["TX-1", "billing_country_mismatch"] == 1.0
    assert feats.loc["TX-1", "shipping_country_mismatch"] == 1.0


def test_entity_reuse_and_missing_history_provenance():
    t0 = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    rows = [
        _tx(transaction_id="TX-a", user_id="U-1", timestamp=t0, device_id="SHARED", ip_address="9.9.9.9"),
        _tx(transaction_id="TX-b", user_id="U-2", timestamp=t0 + timedelta(minutes=1), device_id="SHARED", ip_address="9.9.9.9"),
        _tx(transaction_id="TX-c", user_id="U-3", timestamp=t0 + timedelta(minutes=2), device_id="SHARED", ip_address="1.1.1.1"),
    ]
    feats = compute_features(rows)
    by_id = feats.set_index("transaction_id")
    assert by_id.loc["TX-a", "device_user_count"] == 1
    assert by_id.loc["TX-b", "device_user_count"] == 2
    assert by_id.loc["TX-c", "device_user_count"] == 3
    assert by_id.loc["TX-b", "ip_user_count"] == 2

    prov = provenance_for_row(feats, "TX-a")
    amount_feat = next(item for item in prov if item["feature"] == "amount_vs_user_median")
    assert amount_feat["value"] is None
    assert amount_feat["note"] == "insufficient history"
    assert amount_feat["version"] == "v2"
    assert by_id.loc["TX-a", "amount_history_missing"] == 1.0


def test_boundary_single_transaction():
    feats = compute_features([_tx()])
    assert len(feats) == 1
    assert feats.iloc[0]["transactions_24h"] == 1
    assert feats.iloc[0]["new_device"] == 1.0
