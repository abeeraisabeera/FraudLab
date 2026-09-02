"""Synthetic banking/fintech fraud generator.

Fraud labels are assigned from a latent risk process, then observable fields are
drawn from *overlapping* class-conditional distributions.

Statistical validity constraints:
- No feature deterministically encodes fraud (every elevated pattern appears in both classes).
- Legitimate and fraudulent marginals overlap.
- Fraud prevalence is configurable; generation is seed-reproducible.
- Some fraud looks mostly normal; some legit looks suspicious (label noise / hard cases).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from fraud.schema import FraudGenerateRequest

MERCHANT_CATEGORIES = [
    ("grocery", 0.18, 45.0),
    ("fuel", 0.08, 55.0),
    ("restaurant", 0.14, 38.0),
    ("retail", 0.16, 72.0),
    ("travel", 0.07, 220.0),
    ("electronics", 0.06, 280.0),
    ("utilities", 0.09, 95.0),
    ("subscription", 0.08, 18.0),
    ("atm", 0.06, 120.0),
    ("transfer", 0.08, 350.0),
]

PAYMENT_METHODS = ["card_chip", "card_online", "wallet", "ach", "wire"]
COUNTRIES = ["US", "DE", "UK", "FR", "CA", "NL", "ES", "IT"]
SHARED_DEVICES = [f"DEV-SHARED-{i:03d}" for i in range(12)]
SHARED_IPS = [f"203.0.113.{i}" for i in range(12)]


def generate_fraud_dataset(request: FraudGenerateRequest) -> pd.DataFrame:
    rng = np.random.default_rng(request.seed)
    n_users = request.number_of_users
    n_tx = request.number_of_transactions
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(days=request.time_range_days)
    span_seconds = max(int((end - start).total_seconds()), 1)

    cat_names = [c[0] for c in MERCHANT_CATEGORIES]
    cat_weights = np.array([c[1] for c in MERCHANT_CATEGORIES], dtype=float)
    cat_weights /= cat_weights.sum()
    cat_base_amount = {c[0]: c[2] for c in MERCHANT_CATEGORIES}

    users = []
    for i in range(n_users):
        home = str(rng.choice(COUNTRIES, p=[0.45, 0.12, 0.12, 0.08, 0.08, 0.05, 0.05, 0.05]))
        users.append(
            {
                "user_id": f"U-{i:05d}",
                "home_country": home,
                "account_age_days": int(rng.integers(7, 3600)),
                "median_amount": float(rng.lognormal(mean=3.6, sigma=0.45)),
                "device_id": f"DEV-{i:05d}",
                "ip_address": f"198.51.100.{i % 250}",
                "tx_rate_per_day": float(rng.uniform(0.2, 2.5)),
                "risk_propensity": float(rng.beta(1.4, 7.0)),
            }
        )

    user_weights = np.array([u["tx_rate_per_day"] for u in users], dtype=float)
    user_weights /= user_weights.sum()
    chosen_users = rng.choice(n_users, size=n_tx, p=user_weights)

    # Soft labels from latent propensity + noise (not from observed features).
    latent = np.array([users[int(u)]["risk_propensity"] for u in chosen_users], dtype=float)
    score = 1.8 * (latent - latent.mean()) / (latent.std() + 1e-9) + rng.normal(0, 0.7, size=n_tx)
    order = np.argsort(-score)
    target_fraud = max(1, int(round(n_tx * request.fraud_rate)))
    fraud_flags = np.zeros(n_tx, dtype=int)
    fraud_flags[order[:target_fraud]] = 1

    records: list[dict] = []
    user_tx_times: dict[int, list[datetime]] = {i: [] for i in range(n_users)}

    for i in range(n_tx):
        uidx = int(chosen_users[i])
        user = users[uidx]
        is_fraud = bool(fraud_flags[i])

        # Appearance channel: how suspicious the *observables* look.
        # Fraud usually elevated; legit usually calm; deliberate crossover for overlap.
        if is_fraud:
            # hard_fraud_rate → camouflaged; remainder elevated fraud look
            camouflage = rng.random() < request.hard_fraud_rate
            appearance = float(rng.beta(1.5, 3.5)) if camouflage else float(rng.beta(4.5, 1.6))
        else:
            # suspicious_legit_rate → suspicious-looking legit; remainder calm
            suspicious = rng.random() < request.suspicious_legit_rate
            appearance = float(rng.beta(3.5, 1.8)) if suspicious else float(rng.beta(1.3, 5.0))
        appearance = float(np.clip(appearance + rng.normal(0, 0.04), 0.0, 1.0))

        category = str(rng.choice(cat_names, p=cat_weights))
        base = cat_base_amount[category] * (user["median_amount"] / 40.0)
        offset = int(rng.integers(0, span_seconds))
        ts = start + timedelta(seconds=offset)

        amount_scale = 1.0 + appearance * rng.uniform(0.5, 4.0)
        amount = float(np.clip(rng.lognormal(np.log(max(base * amount_scale, 1.0)), 0.48), 1.0, 30000.0))

        device_id = user["device_id"]
        ip_address = user["ip_address"]
        country = user["home_country"]
        billing_country = user["home_country"]
        shipping_country = user["home_country"] if rng.random() < 0.55 else None
        payment_method = str(rng.choice(PAYMENT_METHODS, p=[0.35, 0.3, 0.2, 0.1, 0.05]))
        failed_attempts = int(min(rng.poisson(0.12 + 1.1 * appearance), 8))
        previous_chargebacks = int(min(rng.poisson(0.04 + 0.4 * appearance), 5))
        merchant_id = f"M-{category[:3].upper()}-{int(rng.integers(1, 80)):03d}"

        if rng.random() < (0.05 + 0.30 * appearance):
            device_id = str(rng.choice(SHARED_DEVICES))
        if rng.random() < (0.05 + 0.28 * appearance):
            ip_address = str(rng.choice(SHARED_IPS))
        if rng.random() < (0.06 + 0.42 * appearance):
            device_id = f"DEV-NEW-{int(rng.integers(0, 8000)):04d}"
        if rng.random() < (0.04 + 0.32 * appearance):
            country = str(rng.choice([c for c in COUNTRIES if c != user["home_country"]] or COUNTRIES))
        if rng.random() < (0.04 + 0.30 * appearance):
            ip_address = f"185.220.{int(rng.integers(0, 50))}.{int(rng.integers(1, 250))}"
        if rng.random() < (0.03 + 0.35 * appearance):
            billing_country = str(rng.choice([c for c in COUNTRIES if c != country] or COUNTRIES))
        if rng.random() < (0.04 + 0.36 * appearance):
            shipping_country = str(rng.choice([c for c in COUNTRIES if c != country] or COUNTRIES))
        if rng.random() < (0.04 + 0.45 * appearance) and user_tx_times[uidx]:
            ts = user_tx_times[uidx][-1] + timedelta(seconds=int(rng.integers(5, 480)))
        if rng.random() < (0.03 + 0.24 * appearance):
            day = start + timedelta(days=int(rng.integers(0, request.time_range_days)))
            ts = day.replace(hour=int(rng.integers(0, 5)), minute=int(rng.integers(0, 60)))
            if rng.random() < 0.55:
                category = "transfer"
                merchant_id = f"M-TRF-{int(rng.integers(1, 40)):03d}"
                payment_method = str(rng.choice(["wire", "ach", "card_online"]))

        if ts < start:
            ts = start + timedelta(seconds=int(rng.integers(0, 3600)))
        if ts > end:
            ts = end - timedelta(seconds=int(rng.integers(1, 3600)))

        user_tx_times[uidx].append(ts)
        records.append(
            {
                "transaction_id": f"TX-{i:06d}",
                "user_id": user["user_id"],
                "timestamp": ts.isoformat(),
                "amount": round(amount, 2),
                "currency": "USD" if user["home_country"] in {"US", "CA"} else "EUR",
                "merchant_id": merchant_id,
                "merchant_category": category,
                "payment_method": payment_method,
                "device_id": device_id,
                "ip_address": ip_address,
                "country": country,
                "billing_country": billing_country,
                "account_age_days": user["account_age_days"],
                "failed_attempts": failed_attempts,
                "previous_chargebacks": previous_chargebacks,
                "shipping_country": shipping_country,
                "is_fraud": int(is_fraud),
            }
        )

    return pd.DataFrame.from_records(records).sort_values("timestamp").reset_index(drop=True)


def realized_fraud_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["is_fraud"].mean())
