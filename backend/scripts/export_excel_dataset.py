"""Export synthetic Fraud Lab transactions to Excel for anomaly detection.

Usage (from backend/):
  python scripts/export_excel_dataset.py
  python scripts/export_excel_dataset.py --transactions 10000 --users 800 --fraud-rate 0.05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud.generator import generate_fraud_dataset
from fraud.schema import FraudGenerateRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Fraud Lab synthetic data to Excel")
    parser.add_argument("--users", type=int, default=800)
    parser.add_argument("--transactions", type=int, default=10000)
    parser.add_argument("--fraud-rate", type=float, default=0.05)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_fraud_dataset(
        FraudGenerateRequest(
            number_of_users=args.users,
            number_of_transactions=args.transactions,
            fraud_rate=args.fraud_rate,
            time_range_days=args.days,
            seed=args.seed,
        )
    )
    export = df.sort_values("timestamp").reset_index(drop=True).copy()
    export["timestamp"] = export["timestamp"].astype(str)
    export["is_fraud"] = export["is_fraud"].astype(int)

    stem = f"fraud_lab_transactions_{args.transactions // 1000}k"
    xlsx_path = args.out_dir / f"{stem}.xlsx"
    csv_path = args.out_dir / f"{stem}.csv"

    legit = export[export["is_fraud"] == 0]
    fraud = export[export["is_fraud"] == 1]

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name="all_transactions", index=False)
        legit.to_excel(writer, sheet_name="legitimate", index=False)
        fraud.to_excel(writer, sheet_name="fraud", index=False)
        pd.DataFrame(
            [
                {"metric": "total_transactions", "value": len(export)},
                {"metric": "unique_users", "value": export["user_id"].nunique()},
                {"metric": "fraud_count", "value": int(len(fraud))},
                {"metric": "legit_count", "value": int(len(legit))},
                {"metric": "fraud_rate", "value": round(len(fraud) / max(len(export), 1), 4)},
                {"metric": "seed", "value": args.seed},
                {
                    "metric": "note",
                    "value": "For unsupervised anomaly detection, train on legitimate sheet (or drop is_fraud).",
                },
            ]
        ).to_excel(writer, sheet_name="summary", index=False)

    export.to_csv(csv_path, index=False)
    print(f"Wrote {xlsx_path}")
    print(f"Wrote {csv_path}")
    print(f"Rows={len(export)} fraud={len(fraud)} users={export['user_id'].nunique()}")


if __name__ == "__main__":
    main()
