"""Probability calibration for fraud scores.

Choice rule (documented for analysts):
- Prefer **isotonic regression** when the calibration fold has >= 200 rows:
  fraud scores are often poorly fit by a single sigmoid, and isotonic is flexible.
- Fall back to **sigmoid (Platt) scaling** on smaller folds to avoid isotonic
  overfitting / step artifacts.

Raw classifier scores are never treated as trustworthy probabilities.
Calibration is always fit on a dedicated calibration fold carved from the
training partition — never on the holdout evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

CalibrationMethod = Literal["isotonic", "sigmoid"]

ISOTONIC_MIN_ROWS = 200


@dataclass
class Calibrator:
    method: CalibrationMethod
    _iso: IsotonicRegression | None = None
    _platt: LogisticRegression | None = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "Calibrator":
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        y = np.asarray(y_true, dtype=int).reshape(-1)
        if len(y) == 0:
            raise ValueError("Calibration fold is empty.")
        if len(np.unique(y)) < 2:
            # Degenerate fold: identity map clipped to empirical rate.
            rate = float(y.mean()) if len(y) else 0.0
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(scores, np.full_like(scores, rate, dtype=float))
            self.method = "isotonic"
            return self
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(scores, y)
        else:
            self._platt = LogisticRegression(solver="lbfgs", max_iter=500)
            self._platt.fit(scores.reshape(-1, 1), y)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if self.method == "isotonic":
            if self._iso is None:
                raise RuntimeError("Isotonic calibrator is not fitted.")
            return np.clip(self._iso.predict(scores), 0.0, 1.0)
        if self._platt is None:
            raise RuntimeError("Sigmoid calibrator is not fitted.")
        return self._platt.predict_proba(scores.reshape(-1, 1))[:, 1]


def choose_calibration_method(n_calibration_rows: int, forced: CalibrationMethod | None = None) -> CalibrationMethod:
    """Select calibrator. Forced isotonic below ``ISOTONIC_MIN_ROWS`` is downgraded to sigmoid."""
    if forced == "isotonic" and n_calibration_rows < ISOTONIC_MIN_ROWS:
        return "sigmoid"
    if forced is not None:
        return forced
    return "isotonic" if n_calibration_rows >= ISOTONIC_MIN_ROWS else "sigmoid"


def calibration_safety_notes(
    *,
    n_calibration: int,
    n_model_fit: int,
    reused_train_for_cal: bool,
    requested_method: CalibrationMethod | None,
    chosen_method: CalibrationMethod,
) -> list[str]:
    notes: list[str] = []
    if reused_train_for_cal:
        notes.append(
            "Calibration fold reused the full training partition (n_train < 80). "
            "Treat calibrated probabilities as provisional."
        )
    if n_calibration < 80:
        notes.append(f"Small calibration fold (n={n_calibration}). Prefer larger datasets before policy tuning.")
    if requested_method == "isotonic" and chosen_method != "isotonic":
        notes.append(
            f"Requested isotonic but n_cal={n_calibration} < {ISOTONIC_MIN_ROWS}; used sigmoid instead."
        )
    if n_model_fit < 40:
        notes.append("Model fit fold is small; EBM explanations may be unstable.")
    return notes


def fit_calibrator(
    raw_scores: np.ndarray,
    y_true: np.ndarray,
    method: CalibrationMethod | None = None,
) -> Calibrator:
    chosen = choose_calibration_method(len(y_true), forced=method)
    return Calibrator(method=chosen).fit(raw_scores, y_true)


def brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return 0.0
    return float(brier_score_loss(y, p))


def calibration_curve_points(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 8,
) -> list[dict[str, float | int]]:
    """Equal-frequency calibration bins with sample counts.

    Each bin reports:
    - mean_predicted_probability
    - fraction_positive (empirical)
    - n (sample count)
    - bin_lower / bin_upper (probability edges)
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return []
    n_bins = int(max(2, min(n_bins, len(y))))
    # Quantile edges; duplicates collapsed.
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        # Fallback: uniform bins
        edges = np.linspace(0.0, 1.0, min(n_bins, len(y)) + 1)
    bins: list[dict[str, float | int]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi < edges[-1]:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        bins.append(
            {
                "mean_predicted_probability": float(p[mask].mean()),
                "fraction_positive": float(y[mask].mean()),
                "n": n,
                "bin_lower": float(lo),
                "bin_upper": float(hi),
            }
        )
    return bins


def calibration_payload(
    method: CalibrationMethod,
    raw_score: float,
    calibrated_probability: float,
) -> dict[str, Any]:
    return {
        "raw_score": float(raw_score),
        "calibrated_probability": float(calibrated_probability),
        "calibration_method": method,
    }
