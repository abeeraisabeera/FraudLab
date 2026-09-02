from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


Decision = Literal["ALLOW", "REVIEW", "BLOCK"]
CalibrationMethod = Literal["isotonic", "sigmoid"]


class FraudTransaction(BaseModel):
    """Canonical fraud transaction schema.

    Required fields are the minimum needed for feature engineering.
    Optional history fields default when absent rather than inventing signal.
    """

    transaction_id: str
    user_id: str
    timestamp: datetime
    amount: float = Field(..., ge=0)
    currency: str = Field(..., min_length=3, max_length=3)
    merchant_id: str
    merchant_category: str
    payment_method: str
    device_id: str
    ip_address: str
    country: str = Field(..., min_length=2, max_length=2)
    billing_country: str = Field(..., min_length=2, max_length=2)
    account_age_days: int = Field(..., ge=0)

    failed_attempts: int = Field(0, ge=0)
    previous_chargebacks: int = Field(0, ge=0)
    shipping_country: str | None = None
    is_fraud: int | None = Field(None, description="Label for synthetic/eval data only (0/1).")

    @field_validator("currency", "country", "billing_country", "shipping_country", mode="before")
    @classmethod
    def uppercase_codes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).strip().upper()

    @field_validator("is_fraud")
    @classmethod
    def binary_label(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value not in (0, 1):
            raise ValueError("is_fraud must be 0 or 1 when provided.")
        return value


class PolicyConfig(BaseModel):
    """Versioned decision thresholds. Model predicts risk; policy decides action.

    Mapping:
    - probability < allow_threshold → ALLOW (low risk)
    - allow_threshold <= probability < review_threshold → REVIEW (medium risk)
    - probability >= review_threshold → BLOCK (high risk)

    block_threshold is kept explicit for UI/audit and equals review_threshold in v1.
    """

    version: str = "policy-v1"
    allow_threshold: float = Field(0.20, ge=0.0, le=1.0)
    review_threshold: float = Field(0.55, ge=0.0, le=1.0)
    block_threshold: float = Field(0.55, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "PolicyConfig":
        if self.allow_threshold > self.review_threshold:
            raise ValueError("allow_threshold must be <= review_threshold.")
        # Align block boundary with review in v1 unless caller already matched them.
        object.__setattr__(self, "block_threshold", self.review_threshold)
        return self


class CostConfig(BaseModel):
    """Operational/financial cost assumptions for threshold selection."""

    false_positive_cost: float = Field(5.0, ge=0.0)
    false_negative_cost: float = Field(100.0, ge=0.0)
    review_cost: float = Field(2.0, ge=0.0)
    # When True, REVIEW on fraud also incurs false_negative_cost (in addition to review_cost).
    charge_false_negative_on_review: bool = False


class FraudGenerateRequest(BaseModel):
    number_of_users: int = Field(400, ge=20, le=5000)
    number_of_transactions: int = Field(4000, ge=100, le=20000)
    fraud_rate: float = Field(0.03, ge=0.001, le=0.4)
    time_range_days: int = Field(30, ge=1, le=365)
    seed: int = Field(42, ge=0)
    # Fraction of fraud labels drawn from the camouflaged (hard) appearance channel.
    hard_fraud_rate: float = Field(0.25, ge=0.0, le=1.0)
    # Fraction of legit labels drawn from the suspicious appearance channel.
    suspicious_legit_rate: float = Field(0.18, ge=0.0, le=1.0)


class FraudAnalyzeRequest(BaseModel):
    transactions: list[FraudTransaction] = Field(..., min_length=1, max_length=20000)
    transaction_id: str | None = None
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    seed: int = Field(42, ge=0)
    test_size: float = Field(0.3, ge=0.15, le=0.5)
    calibration_method: CalibrationMethod | None = None


class FraudEvaluateRequest(BaseModel):
    transactions: list[FraudTransaction] = Field(..., min_length=50, max_length=20000)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    seed: int = Field(42, ge=0)
    test_size: float = Field(0.3, ge=0.15, le=0.5)
    calibration_method: CalibrationMethod | None = None
