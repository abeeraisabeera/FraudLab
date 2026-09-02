"""Fraud Lab HTTP routes — mounted on the main FastAPI app."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from fraud import pipeline
from fraud.policy import DEFAULT_POLICY, policy_payload
from fraud.schema import (
    CostConfig,
    FraudAnalyzeRequest,
    FraudEvaluateRequest,
    FraudGenerateRequest,
    PolicyConfig,
)

router = APIRouter(prefix="/fraud", tags=["fraud"])


@router.get("/policy")
def get_default_policy() -> dict:
    return {
        "policy": policy_payload(DEFAULT_POLICY),
        "costs": CostConfig().model_dump(),
        "notes": {
            "principle": "Evidence before automation.",
            "model_role": "Predict risk probability.",
            "policy_role": "Map probability to ALLOW / REVIEW / BLOCK.",
        },
    }


@router.post("/generate")
def fraud_generate(request: FraudGenerateRequest) -> dict:
    return pipeline.generate(request)


@router.post("/evaluate")
def fraud_evaluate(request: FraudEvaluateRequest) -> dict:
    try:
        return pipeline.evaluate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/analyze")
def fraud_analyze(request: FraudAnalyzeRequest) -> dict:
    try:
        return pipeline.analyze(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
