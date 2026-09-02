"""Explainable Boosting Machine (InterpretML) fraud model.

EBM is chosen because it remains a glass-box model: additive feature functions
with optional pairwise interactions learned from data, plus local contributions
for every prediction. v1 deliberately excludes XGBoost / neural nets / GNNs.

Contributions are taken directly from InterpretML ``explain_local`` — never
fabricated. Positive and negative terms are returned as-is; additivity against
the model's raw logit is checked when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fraud.features import MODEL_FEATURE_NAMES

MODEL_VERSION = "ebm-v1"
# Soft CI gate for explanation reconstruction vs model logit.
ADDITIVITY_MAX_ERROR = 0.20


def _classify_term(label: str) -> dict[str, Any]:
    """Tag InterpretML term names as main effect vs pairwise interaction."""
    name = str(label)
    if " & " in name or " x " in name.lower():
        members = [part.strip() for part in name.replace(" x ", " & ").split("&")]
        return {
            "feature": name,
            "term_type": "interaction",
            "members": members,
        }
    return {"feature": name, "term_type": "main", "members": [name]}


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x_arr, -50, 50)))


@dataclass
class FraudEBMModel:
    model_version: str = MODEL_VERSION
    feature_names: list[str] = field(default_factory=lambda: list(MODEL_FEATURE_NAMES))
    _ebm: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> "FraudEBMModel":
        from interpret.glassbox import ExplainableBoostingClassifier

        y_arr = np.asarray(y, dtype=int)
        if len(np.unique(y_arr)) < 2:
            raise ValueError("Need both fraud and non-fraud labels to train the EBM.")

        self._ebm = ExplainableBoostingClassifier(
            feature_names=self.feature_names,
            interactions="2x",
            max_bins=64,
            max_interaction_bins=32,
            outer_bags=4,
            inner_bags=0,
            learning_rate=0.02,
            max_rounds=120,
            early_stopping_rounds=20,
            validation_size=0.15,
            random_state=seed,
            n_jobs=1,
        )
        self._ebm.fit(np.asarray(X, dtype=float), y_arr)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._ebm is not None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._ebm is None:
            raise RuntimeError("EBM is not fitted.")
        proba = self._ebm.predict_proba(np.asarray(X, dtype=float))
        return proba[:, 1]

    def decision_logit(self, X: np.ndarray) -> np.ndarray:
        """Raw additive logit when available; otherwise logit of predict_proba."""
        if self._ebm is None:
            raise RuntimeError("EBM is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        if hasattr(self._ebm, "decision_function"):
            return np.asarray(self._ebm.decision_function(X_arr), dtype=float)
        p = np.clip(self.predict_proba(X_arr), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def feature_contributions(self, X: np.ndarray) -> list[dict[str, Any]]:
        """Per-row explanation packages from InterpretML local explanations.

        Returns dicts with:
        - contributions: all terms (main effects + interactions + intercept)
        - positive_contributions / negative_contributions
        - intercept
        - contribution_sum / model_logit / additivity_error
        """
        if self._ebm is None:
            raise RuntimeError("EBM is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        local = self._ebm.explain_local(X_arr)
        logits = self.decision_logit(X_arr)
        rows: list[dict[str, Any]] = []
        for i in range(X_arr.shape[0]):
            data = local.data(i)
            names = list(data.get("names") or [])
            scores = list(data.get("scores") or [])
            intercept = 0.0
            # InterpretML may place intercept in extra
            extra = data.get("extra") or {}
            if isinstance(extra, dict):
                extra_names = list(extra.get("names") or [])
                extra_scores = list(extra.get("scores") or [])
                for en, es in zip(extra_names, extra_scores):
                    if str(en).lower() in {"intercept", "bias"}:
                        intercept = float(es)
            contribs: list[dict[str, Any]] = []
            for name, score in zip(names, scores):
                label = str(name)
                if label.lower() in {"intercept", "bias"}:
                    intercept = float(score)
                    continue
                term = _classify_term(label)
                contribs.append({**term, "contribution": float(score)})
            contrib_sum = float(sum(float(c["contribution"]) for c in contribs) + intercept)
            model_logit = float(logits[i])
            additivity_error = abs(contrib_sum - model_logit)
            rows.append(
                {
                    "contributions": sorted(
                        contribs,
                        key=lambda item: abs(float(item["contribution"])),
                        reverse=True,
                    ),
                    "positive_contributions": sorted(
                        [c for c in contribs if float(c["contribution"]) > 0],
                        key=lambda item: float(item["contribution"]),
                        reverse=True,
                    ),
                    "negative_contributions": sorted(
                        [c for c in contribs if float(c["contribution"]) < 0],
                        key=lambda item: float(item["contribution"]),
                    ),
                    "intercept": intercept,
                    "contribution_sum": contrib_sum,
                    "model_logit": model_logit,
                    "additivity_error": additivity_error,
                }
            )
        return rows

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw classifier score used for calibration (fraud class probability)."""
        return self.predict_proba(X)
