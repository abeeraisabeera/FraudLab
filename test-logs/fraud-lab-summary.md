# Fraud Lab — scenario run (2026-09-02)

Full machine-readable log: `fraud-lab-scenarios.json`

## Quick comparison

| Scenario | Key result |
|---|---|
| Generate 1% fraud | 15 / 1500 txs flagged fraud |
| Generate 4% fraud (default) | 100 / 2500 |
| Generate 15% fraud | 225 / 1500 |
| Same seed twice | Identical dataset ✓ |
| Different seed | Different dataset ✓ |
| Evaluate default policy | 95% ALLOW, 5% REVIEW, 0% BLOCK |
| Evaluate strict policy | 71% ALLOW, 27% REVIEW, 2% BLOCK |
| Evaluate loose policy | 99% ALLOW, 1% REVIEW |
| Highest-risk tx | TX-001824 → REVIEW @ 50% risk |
| Lowest-risk tx | TX-001667 → ALLOW @ 0% risk |
| 40 txs evaluate | Rejected (need ≥50) |
| 5 users generate | Rejected (need ≥20) |
| Fake tx id analyze | 404 not found |

## Policy comparison (same model, different thresholds)

| Policy | Allow | Review | Block | Expected cost | Fraud caught (recall) |
|---|---|---|---|---|---|
| Strict (0.05 / 0.25) | 70.8% | 27.3% | 1.9% | 955 | 50% |
| Default (0.2 / 0.55) | 95.3% | 4.7% | 0% | 1670 | 0% |
| Loose (0.5 / 0.85) | 98.5% | 1.5% | 0% | 2422 | 0% |

Model quality (PR-AUC) stayed **0.2424** across all three — only the decision rules changed.

## Risk extremes

**Highest risk — TX-001824**
- Risk: 50% → REVIEW
- Top signals: young account, failed attempts, chargeback history

**Lowest risk — TX-001667**
- Risk: 0% → ALLOW
- Normal spending pattern for that user
