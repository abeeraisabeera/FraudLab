"""Kasauti Fraud Lab API — standalone FastAPI service."""

from __future__ import annotations

import os
import time
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fraud.routes import router as fraud_router

API_KEY = os.getenv("FRAUD_LAB_API_KEY", os.getenv("KASAUTI_API_KEY", "")).strip()
RATE_LIMIT_PER_MIN = int(os.getenv("FRAUD_LAB_RATE_LIMIT_PER_MIN", os.getenv("KASAUTI_RATE_LIMIT_PER_MIN", "120")))

app = FastAPI(title="Kasauti Fraud Lab API", version="1.0.0")

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRAUD_LAB_CORS_ORIGINS",
        os.getenv("KASAUTI_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"),
    ).split(",")
    if origin.strip()
]
_cors_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins if _cors_origins else ["http://localhost:3000"],
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rate_buckets: dict[str, list[float]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    path = request.url.path
    open_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/"}
    if request.method == "OPTIONS" or path in open_paths:
        return await call_next(request)

    if API_KEY:
        provided = request.headers.get("x-api-key", "")
        if provided != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized. Provide X-API-Key."})

    if request.method in {"POST", "PUT", "PATCH"} and RATE_LIMIT_PER_MIN > 0:
        ip = _client_ip(request)
        now = time.time()
        bucket = [t for t in _rate_buckets[ip] if now - t < 60.0]
        if len(bucket) >= RATE_LIMIT_PER_MIN:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again shortly."})
        bucket.append(now)
        _rate_buckets[ip] = bucket

    return await call_next(request)


app.include_router(fraud_router)


@app.get("/")
def root() -> dict:
    return {
        "service": "kasauti-fraud-lab",
        "docs": "/docs",
        "health": "/health",
        "routes": ["/fraud/policy", "/fraud/generate", "/fraud/evaluate", "/fraud/analyze"],
    }


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "version": app.version,
        "limits": {
            "rate_limit_per_min": RATE_LIMIT_PER_MIN,
            "api_key_required": bool(API_KEY),
        },
    }
