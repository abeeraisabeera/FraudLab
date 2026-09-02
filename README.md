# Kasauti Fraud Lab

Standalone fraud evidence workstation extracted from [Kasauti](https://github.com/abeeraisabeera/Kasauti-).

**Principle:** Evidence before automation.

```text
generate → deterministic features → EBM → calibration → policy → evidence
```

## Structure

| Path | Role |
|------|------|
| `backend/` | FastAPI + `fraud/` package (Vercel Python) |
| `frontend/` | Next.js Fraud Lab UI |

## Local run

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app:app --reload --port 8000
```

### Frontend

```bash
cd frontend
pnpm install
# copy .env.example → .env.local
pnpm dev
```

UI → http://localhost:3000 · API docs → http://127.0.0.1:8000/docs

## API

- `GET /health`
- `GET /fraud/policy`
- `POST /fraud/generate`
- `POST /fraud/evaluate`
- `POST /fraud/analyze`

## Ops

| Env | Purpose |
|-----|---------|
| `FRAUD_LAB_CORS_ORIGINS` | Allowed browser origins |
| `FRAUD_LAB_API_KEY` | Optional `X-API-Key` gate |
| `FRAUD_LAB_RATE_LIMIT_PER_MIN` | POST rate limit (default 120) |
| `NEXT_PUBLIC_API_BASE_URL` | Frontend → API origin |

Python on Vercel: **3.12** (see `.python-version`). Uses **`interpret-core`** for EBM only.
