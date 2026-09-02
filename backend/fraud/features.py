"""Deterministic, leakage-safe fraud feature engine.

Every derived feature exposes name, definition, calculation, and window/population.
Missing or insufficient history is represented explicitly (nulls / zeros with
provenance notes) — we do not invent statistical meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURE_VERSION = "v2"

FEATURE_DEFINITIONS: dict[str, dict[str, str]] = {
    "amount_vs_user_median": {
        "definition": "transaction amount divided by the user's historical median amount",
        "calculation": "amount / median(prior amounts for user)",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "amount_percentile_user": {
        "definition": "empirical percentile of amount among the user's prior amounts",
        "calculation": "mean(prior_amount <= current_amount)",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "amount_zscore_user": {
        "definition": "z-score of amount vs user's prior mean and std",
        "calculation": "(amount - mean_prior) / std_prior",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "time_since_previous_transaction": {
        "definition": "seconds since the user's previous transaction",
        "calculation": "timestamp - previous_timestamp (seconds)",
        "window": "immediate prior transaction for user",
        "population": "user",
    },
    "transactions_10m": {
        "definition": "number of transactions by user in trailing 10 minutes",
        "calculation": "count(user transactions in (t-10m, t])",
        "window": "trailing 10 minutes",
        "population": "user",
    },
    "transactions_1h": {
        "definition": "number of transactions by user in trailing 1 hour",
        "calculation": "count(user transactions in (t-1h, t])",
        "window": "trailing 1 hour",
        "population": "user",
    },
    "transactions_24h": {
        "definition": "number of transactions by user in trailing 24 hours",
        "calculation": "count(user transactions in (t-24h, t])",
        "window": "trailing 24 hours",
        "population": "user",
    },
    "new_device": {
        "definition": "1 if device_id not seen before for this user",
        "calculation": "device_id not in prior user devices",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "new_country": {
        "definition": "1 if country not seen before for this user",
        "calculation": "country not in prior user countries",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "new_ip": {
        "definition": "1 if ip_address not seen before for this user",
        "calculation": "ip_address not in prior user IPs",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "billing_country_mismatch": {
        "definition": "1 if billing_country differs from country",
        "calculation": "billing_country != country",
        "window": "current transaction",
        "population": "transaction",
    },
    "shipping_country_mismatch": {
        "definition": "1 if shipping_country is present and differs from transaction country",
        "calculation": "shipping_country is not null and shipping_country != country",
        "window": "current transaction",
        "population": "transaction",
        "note": "Renamed from ip_country_mismatch — this is a shipping/geo proxy, not IP geolocation.",
    },
    "amount_history_missing": {
        "definition": "1 if the user has no prior amounts (amount stats would be undefined)",
        "calculation": "prior_transaction_count == 0",
        "window": "all prior transactions for user",
        "population": "user",
    },
    "account_age_days": {
        "definition": "account age in days at transaction time",
        "calculation": "passthrough from transaction.account_age_days",
        "window": "account lifetime",
        "population": "account",
    },
    "failed_attempts": {
        "definition": "recent failed payment attempts associated with the account",
        "calculation": "passthrough from transaction.failed_attempts (0 if missing)",
        "window": "recent account history (provided)",
        "population": "account",
    },
    "previous_chargebacks": {
        "definition": "count of prior chargebacks on the account",
        "calculation": "passthrough from transaction.previous_chargebacks (0 if missing)",
        "window": "account lifetime (provided)",
        "population": "account",
    },
    "device_user_count": {
        "definition": "number of distinct users observed on this device up to this transaction",
        "calculation": "nunique(user_id | device_id, timestamp <= t)",
        "window": "all transactions on device up to t",
        "population": "device",
    },
    "ip_user_count": {
        "definition": "number of distinct users observed on this IP up to this transaction",
        "calculation": "nunique(user_id | ip_address, timestamp <= t)",
        "window": "all transactions on IP up to t",
        "population": "ip",
    },
}

MODEL_FEATURE_NAMES = list(FEATURE_DEFINITIONS.keys())


@dataclass(frozen=True)
class FeatureProvenance:
    feature: str
    value: float | None
    baseline: float | None
    definition: str
    source: str
    version: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature": self.feature,
            "value": self.value,
            "baseline": self.baseline,
            "definition": self.definition,
            "source": self.source,
            "version": self.version,
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


def _ensure_frame(transactions: pd.DataFrame | list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(transactions) if not isinstance(transactions, pd.DataFrame) else transactions.copy()
    if df.empty:
        raise ValueError("transactions must be non-empty")
    required = {
        "transaction_id",
        "user_id",
        "timestamp",
        "amount",
        "device_id",
        "ip_address",
        "country",
        "billing_country",
        "account_age_days",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    dup_mask = df["transaction_id"].duplicated(keep=False)
    if dup_mask.any():
        sample = sorted({str(v) for v in df.loc[dup_mask, "transaction_id"].head(5)})
        raise ValueError(f"Duplicate transaction_id values are not allowed: {sample}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["amount"] = df["amount"].astype(float)
    df["account_age_days"] = df["account_age_days"].astype(int)
    if "failed_attempts" not in df.columns:
        df["failed_attempts"] = 0
    if "previous_chargebacks" not in df.columns:
        df["previous_chargebacks"] = 0
    if "shipping_country" not in df.columns:
        df["shipping_country"] = None
    df["failed_attempts"] = df["failed_attempts"].fillna(0).astype(int)
    df["previous_chargebacks"] = df["previous_chargebacks"].fillna(0).astype(int)
    return df.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def _user_window_counts(df: pd.DataFrame, window: pd.Timedelta) -> np.ndarray:
    """Count user transactions in (t - window, t] using per-user searchsorted."""
    counts = np.ones(len(df), dtype=float)
    ts_ns = df["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    window_ns = int(window.to_timedelta64().astype("timedelta64[ns]").astype(np.int64))
    for _, idx in df.groupby("user_id", sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        times = ts_ns[positions]
        # positions are in global time order because df is globally sorted
        for j, pos in enumerate(positions):
            left = np.searchsorted(times, times[j] - window_ns, side="left")
            counts[pos] = float(j - left + 1)
    return counts


def compute_features(transactions: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Compute leakage-safe features for every transaction.

    Uses only information available at or before each transaction timestamp.
    Vectorized groupby/rolling where practical; avoids per-row Python scans.
    """
    df = _ensure_frame(transactions)
    out = pd.DataFrame({"transaction_id": df["transaction_id"].values})

    # --- Amount / time relative to user history (shift = prior only) ---
    g = df.groupby("user_id", sort=False)
    # Expanding stats on prior amounts: use cum* on shifted series
    prior_count = g.cumcount()  # number of prior txs
    expanding_sum = g["amount"].cumsum() - df["amount"]
    expanding_sum_sq = g["amount"].transform(lambda s: (s**2).cumsum()) - df["amount"] ** 2

    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, expanding_sum / prior_count, np.nan)
        prior_var = np.where(
            prior_count > 1,
            (expanding_sum_sq - (expanding_sum**2) / prior_count) / (prior_count - 1),
            np.nan,
        )
        prior_std = np.sqrt(np.maximum(prior_var, 0.0))

    # Median of prior amounts: approximate with expanding median via transform (acceptable for v1).
    # Exact prior median requires shift; compute via rolling apply on shifted amounts.
    prior_median = g["amount"].transform(lambda s: s.shift(1).expanding().median())

    amount = df["amount"].to_numpy(dtype=float)
    out["amount_vs_user_median"] = np.where(
        prior_count.to_numpy() > 0,
        amount / np.maximum(prior_median.to_numpy(dtype=float), 1e-9),
        np.nan,
    )
    out["amount_zscore_user"] = np.where(
        (prior_count.to_numpy() > 1) & (prior_std > 1e-9),
        (amount - prior_mean) / prior_std,
        np.nan,
    )

    # Empirical prior percentile via searchsorted on growing history (O(n log n) per user).
    out["amount_percentile_user"] = _prior_percentiles(df)

    prev_ts = g["timestamp"].shift(1)
    delta = (df["timestamp"] - prev_ts).dt.total_seconds()
    out["time_since_previous_transaction"] = delta.to_numpy(dtype=float)
    out["amount_history_missing"] = (prior_count.to_numpy() == 0).astype(float)

    # --- Velocity (include current) ---
    out["transactions_10m"] = _user_window_counts(df, pd.Timedelta(minutes=10))
    out["transactions_1h"] = _user_window_counts(df, pd.Timedelta(hours=1))
    out["transactions_24h"] = _user_window_counts(df, pd.Timedelta(hours=24))

    # --- Novelty (first occurrence for user) ---
    out["new_device"] = (~g["device_id"].transform(lambda s: s.duplicated())).astype(float).to_numpy()
    out["new_country"] = (~g["country"].transform(lambda s: s.duplicated())).astype(float).to_numpy()
    out["new_ip"] = (~g["ip_address"].transform(lambda s: s.duplicated())).astype(float).to_numpy()

    # --- Consistency ---
    out["billing_country_mismatch"] = (
        df["billing_country"].astype(str).str.upper() != df["country"].astype(str).str.upper()
    ).astype(float)
    shipping = df["shipping_country"]
    out["shipping_country_mismatch"] = (
        shipping.notna()
        & (shipping.astype(str).str.upper() != df["country"].astype(str).str.upper())
    ).astype(float)

    # --- Account / history passthrough ---
    out["account_age_days"] = df["account_age_days"].astype(float).to_numpy()
    out["failed_attempts"] = df["failed_attempts"].astype(float).to_numpy()
    out["previous_chargebacks"] = df["previous_chargebacks"].astype(float).to_numpy()

    # --- Entity reuse (distinct users on device/IP up to t) ---
    # Cumulative unique users: for each entity, track seen users in time order.
    out["device_user_count"] = _cumulative_unique_users(df, "device_id")
    out["ip_user_count"] = _cumulative_unique_users(df, "ip_address")

    # Baselines for provenance (dataset-level means of non-null values)
    for name in MODEL_FEATURE_NAMES:
        if name not in out.columns:
            out[name] = np.nan

    return out


def _prior_percentiles(df: pd.DataFrame) -> np.ndarray:
    """Empirical P(prior_amount <= current) with a sorted prior list (O(n log n) per user)."""
    result = np.full(len(df), np.nan, dtype=float)
    amounts_all = df["amount"].to_numpy(dtype=float)
    for _, idx in df.groupby("user_id", sort=False).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        amounts = amounts_all[positions]
        sorted_prior: list[float] = []
        for j, pos in enumerate(positions):
            if j > 0:
                insert_at = int(np.searchsorted(sorted_prior, amounts[j], side="right"))
                result[pos] = float(insert_at / j)
            idx_ins = int(np.searchsorted(sorted_prior, amounts[j], side="left"))
            sorted_prior.insert(idx_ins, float(amounts[j]))
    return result


def _cumulative_unique_users(df: pd.DataFrame, entity_col: str) -> np.ndarray:
    """Count distinct users observed on each entity up to and including current row."""
    counts = np.empty(len(df), dtype=float)
    seen: dict[str, set[str]] = {}
    for i, (entity, user) in enumerate(zip(df[entity_col].astype(str), df["user_id"].astype(str))):
        bucket = seen.setdefault(entity, set())
        bucket.add(user)
        counts[i] = float(len(bucket))
    return counts


def feature_matrix(feature_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return X matrix with NaNs filled for model input (explicit fill documented).

    Fill policy (v2):
    - Continuous amount/time stats: NaN → 0.0 ("no elevation"), with
      ``amount_history_missing`` carrying the missingness signal.
    - Binary / count features: already defined; NaN → 0.0.
    Provenance retains nulls for analyst inspection.
    """
    cols = MODEL_FEATURE_NAMES
    X = feature_df[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, cols


def provenance_for_row(
    feature_df: pd.DataFrame,
    transaction_id: str,
    baselines: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    row = feature_df.loc[feature_df["transaction_id"] == transaction_id]
    if row.empty:
        raise KeyError(f"Unknown transaction_id: {transaction_id}")
    row = row.iloc[0]
    if baselines is None:
        baselines = dataset_baselines(feature_df)
    items: list[dict[str, Any]] = []
    for name in MODEL_FEATURE_NAMES:
        meta = FEATURE_DEFINITIONS[name]
        raw = row[name]
        value = None if pd.isna(raw) else float(raw)
        note = None
        if value is None:
            note = "insufficient history"
        items.append(
            FeatureProvenance(
                feature=name,
                value=value,
                baseline=baselines.get(name),
                definition=meta["definition"],
                source="transactions",
                version=FEATURE_VERSION,
                note=note,
            ).to_dict()
        )
    return items


def dataset_baselines(feature_df: pd.DataFrame) -> dict[str, float]:
    return {
        name: float(np.nanmean(feature_df[name].to_numpy(dtype=float)))
        if name in feature_df.columns and feature_df[name].notna().any()
        else 0.0
        for name in MODEL_FEATURE_NAMES
    }
