# Fraud Lab

**Evidence before automation.**

Fraud Lab is a small learning / demo app that answers one question:

> “Is this payment risky — and *why*?”

It is **not** a real bank system. It makes **fake** transactions, trains a model on them, then shows you scores, decisions, and evidence in a dashboard.

Extracted from [Kasauti](https://github.com/abeeraisabeera/Kasauti-) as a standalone app.

---

## What problem does this solve?

Real fraud systems often say “blocked” with no explanation. This lab teaches the safer idea:

1. **Predict risk** (a number from 0% to 100%)
2. **Apply clear rules** (ALLOW / REVIEW / BLOCK)
3. **Show evidence** (which signals pushed the score up or down)

So you can change rules and *see* what happens — without guessing.

---

## Big picture (two apps talking)

```
You (browser)  →  Frontend (Next.js)  →  Backend (FastAPI)  →  fraud/ Python code
                       UI                     API                 brain of the lab
```

| Piece | Folder | Job |
| --- | --- | --- |
| **Frontend** | `frontend/` | Pretty dashboard. Buttons call the API. Charts show results. |
| **Backend** | `backend/` | Does the real work: generate data, train model, score, explain. |
| **Fraud package** | `backend/fraud/` | Pipeline steps (features, model, policy, evidence). |

You must run **both**. The UI alone cannot train a model.

---

## How one full run works (beginner walkthrough)

Think of it as cooking:

| Your action in the UI | What the code does |
| --- | --- |
| Click **Generate Fraud Dataset** | Backend invents fake users + payments. Some are labeled fraud (`is_fraud = 1`). |
| Set thresholds / costs | You choose how strict the rules are and how expensive mistakes are. |
| Click **Train · Calibrate · Evaluate** | Backend learns from part of the data, tests on the rest, scores every tx. |
| Click **Inspect Evidence** / Open | Backend explains one transaction: risk %, decision, top features. |

### The pipeline (what happens under the hood)

```
generate → features → EBM → calibration → policy → evidence
```

1. **Generate** (`fraud/generator.py`)  
   Creates synthetic transactions (amount, country, device, etc.).  
   Same **seed** → same dataset every time (good for experiments).

2. **Features** (`fraud/features.py`)  
   Turns raw fields into numbers the model can use  
   (e.g. “is this amount weird for this user?”, “new account?”, “failed attempts?”).  
   Same input → same features (deterministic).

3. **EBM** (`fraud/model.py`)  
   **Explainable Boosting Machine** — a model that scores risk *and* can say  
   “feature X added +0.4 to the score”. Like a smart checklist, not a black box.

4. **Calibration** (`fraud/calibration.py`)  
   Raw model scores are adjusted so a reported **80%** roughly means  
   “about 80 out of 100 similar cases are fraud”.

5. **Policy** (`fraud/policy.py` + `fraud/decision.py`)  
   Maps probability to an action with two knobs:

   | Risk level | Decision |
   | --- | --- |
   | Below **allow threshold** | **ALLOW** (auto approve) |
   | Between allow and review | **REVIEW** (human should check) |
   | At/above **review threshold** | **BLOCK** (auto reject) |

6. **Evidence** (`fraud/evidence.py`)  
   For one transaction: calibrated risk, decision, and feature contributions.

**Orchestrator:** `fraud/pipeline.py` runs these steps in order for `/fraud/evaluate` and `/fraud/analyze`.

**Important training rule:** data is split by time. The model trains on earlier txs and is graded on a later **holdout** set it never saw. Dashboard metrics come from that holdout only — so numbers are honest.

---

## Using the dashboard (what each screen is for)

| Area | What you do there |
| --- | --- |
| **Sidebar** | Set users, #transactions, fraud rate, seed, thresholds, FP/FN/review costs |
| **Dashboard** | See KPIs, holdout metrics, charts after evaluate |
| **Transactions** | Browse scored rows; open one to inspect |
| **Evidence** | Read feature contributions for the inspected tx |
| **Hover tooltips** | Short explanations on labels and metrics |

### Reading holdout metrics without panic

| Metric | Plain meaning |
| --- | --- |
| **PR-AUC** | Can the model put fraud above normal txs? Higher is better. |
| **Precision** | Of auto-**blocks**, how many were real fraud? |
| **Recall** | Of fraud that was auto-decided (allow/block), how many did we block? |
| **F1** | Balance of precision + recall |
| **Brier** | Are probabilities trustworthy? Lower is better |
| **FPR** | Legit txs wrongly blocked (among allow/block) |
| **FNR** | Fraud wrongly allowed (among allow/block) |
| **Expected cost** | Business cost = missed fraud + false blocks + reviews |
| **Review rate** | % sent to humans |
| **Prevalence** | True fraud % in the holdout set |

**Gotcha:** Precision / Recall only count **BLOCK** as “caught”. **REVIEW does not count**.  
If your policy never blocks (default is strict about blocking), precision/recall can be **0%** even when PR-AUC looks okay. Tighten thresholds and re-run evaluate.

---

## Where the code lives

```
backend/
  app.py                 Starts the API server
  fraud/
    routes.py            HTTP endpoints (/fraud/generate, evaluate, analyze)
    pipeline.py          Glue: runs the full workflow
    generator.py         Fake transactions
    features.py          Feature engineering
    model.py             EBM train / score
    calibration.py       Probability calibration
    policy.py            Threshold rules
    decision.py          Score → ALLOW/REVIEW/BLOCK (one place only)
    evaluation.py        Holdout metrics + cost
    evidence.py          Per-tx explanation
  scripts/run_scenarios.py   Optional: many test inputs → logs
  test_*.py              Automated backend tests

frontend/
  app/                   Next.js pages
  components/
    fraud-lab.tsx        Main UI (buttons, tabs, state)
    fraud-charts.tsx     Charts
    ui/                  Buttons, cards, inputs, tooltips
  lib/
    api.ts               Calls the backend
    fraud-types.ts       TypeScript shapes for API data
    fraud-tooltips.ts    Hover help text
```

**Frontend flow in one sentence:**  
`fraud-lab.tsx` keeps state → calls `lib/api.ts` → shows charts/metrics from the JSON response.

---

## Quick start

### 1. Backend

```bash
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app:app --reload --port 8000
```

API docs: http://127.0.0.1:8000/docs

### 2. Frontend (new terminal)

```bash
cd frontend
pnpm install
cp .env.example .env.local
pnpm dev
```

UI: http://localhost:3000

`.env.local` should point at the API, e.g.:

```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

### First experiment (suggested)

1. Leave defaults (or use ~2500 txs, fraud rate `0.04`, seed `42`)
2. Generate → Evaluate
3. Open a high-risk transaction
4. Lower **review / block threshold** (e.g. `0.25`), evaluate again — watch review/block rates and cost change

---

## API (what the UI calls)

| Endpoint | In plain English |
| --- | --- |
| `GET /health` | “Is the server up?” |
| `GET /fraud/policy` | Default rules + cost assumptions |
| `POST /fraud/generate` | Make fake data |
| `POST /fraud/evaluate` | Train + score + holdout report |
| `POST /fraud/analyze` | Explain one transaction |

Typical JSON flow:

1. Generate → get `transactions[]`  
2. Evaluate with those transactions + `policy` + `costs`  
3. Analyze with same transactions + a `transaction_id`

---

## Config

| Variable | What it does |
| --- | --- |
| `FRAUD_LAB_CORS_ORIGINS` | Which browser origins may call the API |
| `FRAUD_LAB_API_KEY` | Optional API key header |
| `FRAUD_LAB_RATE_LIMIT_PER_MIN` | Cap on POSTs per minute (default 120) |
| `NEXT_PUBLIC_API_BASE_URL` | Where the frontend finds the API |

---

## Tests

```bash
# Backend
cd backend && python -m pytest -q

# Frontend
cd frontend && pnpm typecheck && pnpm lint && pnpm test
```

Optional scenario sweep (API must be running):

```bash
cd backend
python scripts/run_scenarios.py
```

Writes `test-logs/fraud-lab-scenarios.json` and `test-logs/fraud-lab-summary.md`.

---

## Stack

| Layer | Tech | Role |
| --- | --- | --- |
| Backend | Python 3.12, FastAPI, Uvicorn | API + pipeline |
| Model | interpret-core (EBM) | Explainable risk scoring |
| Frontend | Next.js, React, Tailwind, Recharts | Dashboard |
| Tests | pytest, Vitest | Catch regressions |

---

## Mental model to remember

```
Fake data  →  Model risk %  →  Policy decision  →  Evidence
(generate)    (evaluate)       (ALLOW/REVIEW/BLOCK)  (analyze)
```

- **Model** answers: “How risky?”  
- **Policy** answers: “What do we do?”  
- **Evidence** answers: “Why?”  

Change policy without retraining the whole worldview — then re-evaluate to refresh metrics.
