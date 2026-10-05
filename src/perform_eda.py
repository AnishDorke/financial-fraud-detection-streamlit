
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = Path("data/raw")
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("PaySim CSV not found in data/raw.")

file_path = csv_files[0]
reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

chunk_size = 100_000

type_counts = {}
type_fraud = {}
step_fraud = {}
amount_stats = []
balance_errors = {
    "origin_balance_mismatch": 0,
    "destination_balance_mismatch": 0
}
fraud_amounts = []
nonfraud_amounts = []
total_rows = 0
total_fraud = 0
total_nonfraud = 0
flagged_fraud_count = 0
flagged_nonfraud_count = 0

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    total_rows += len(chunk)

    fraud = chunk["isFraud"].eq(1)
    nonfraud = chunk["isFraud"].eq(0)

    total_fraud += int(fraud.sum())
    total_nonfraud += int(nonfraud.sum())

    flagged_fraud_count += int(
        ((chunk["isFlaggedFraud"] == 1) & fraud).sum()
    )
    flagged_nonfraud_count += int(
        ((chunk["isFlaggedFraud"] == 1) & nonfraud).sum()
    )

    grouped = chunk.groupby("type")["isFraud"].agg(["count", "sum"])
    for transaction_type, row in grouped.iterrows():
        type_counts[transaction_type] = (
            type_counts.get(transaction_type, 0) + int(row["count"])
        )
        type_fraud[transaction_type] = (
            type_fraud.get(transaction_type, 0) + int(row["sum"])
        )

    step_group = chunk.groupby("step")["isFraud"].agg(["count", "sum"])
    for step, row in step_group.iterrows():
        if step not in step_fraud:
            step_fraud[step] = [0, 0]
        step_fraud[step][0] += int(row["count"])
        step_fraud[step][1] += int(row["sum"])

    origin_expected = chunk["oldbalanceOrg"] - chunk["amount"]
    origin_error = (
        chunk["newbalanceOrig"] - origin_expected
    ).abs() > 0.01

    destination_expected = chunk["oldbalanceDest"] + chunk["amount"]
    destination_error = (
        chunk["newbalanceDest"] - destination_expected
    ).abs() > 0.01

    balance_errors["origin_balance_mismatch"] += int(origin_error.sum())
    balance_errors["destination_balance_mismatch"] += int(destination_error.sum())

    amount_stats.append(
        chunk.groupby("isFraud")["amount"]
        .agg(["count", "mean", "median", "min", "max"])
    )

    fraud_amounts.extend(chunk.loc[fraud, "amount"].tolist())
    nonfraud_amounts.extend(chunk.loc[nonfraud, "amount"].tolist())

    print(f"EDA processed {total_rows:,} records")

type_summary = pd.DataFrame([
    {
        "type": transaction_type,
        "transactions": count,
        "fraud_count": type_fraud[transaction_type],
        "fraud_rate_pct": type_fraud[transaction_type] / count * 100
    }
    for transaction_type, count in type_counts.items()
]).sort_values("transactions", ascending=False)

step_summary = pd.DataFrame([
    {
        "step": step,
        "transactions": values[0],
        "fraud_count": values[1],
        "fraud_rate_pct": values[1] / values[0] * 100
    }
    for step, values in step_fraud.items()
]).sort_values("step")

type_summary.to_csv(reports_dir / "fraud_by_type.csv", index=False)
step_summary.to_csv(reports_dir / "fraud_by_step.csv", index=False)

amount_summary = pd.concat(amount_stats).groupby(level=0).sum(numeric_only=True)
amount_summary.to_csv(reports_dir / "amount_summary_aggregated.csv")

balance_summary = pd.DataFrame([
    {"check": key, "mismatch_count": value, "percentage_of_rows": value / total_rows * 100}
    for key, value in balance_errors.items()
])
balance_summary.to_csv(reports_dir / "balance_consistency.csv", index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=type_summary, x="type", y="transactions")
plt.title("Transaction Volume by Type")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(reports_dir / "transaction_volume_by_type.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sns.barplot(data=type_summary, x="type", y="fraud_rate_pct")
plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()
plt.savefig(reports_dir / "fraud_rate_by_type.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sns.lineplot(data=step_summary, x="step", y="fraud_count")
plt.title("Fraud Count Across Simulation Steps")
plt.xlabel("Simulation Step")
plt.ylabel("Fraudulent Transactions")
plt.tight_layout()
plt.savefig(reports_dir / "fraud_over_time.png", dpi=150)
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=pd.DataFrame({
        "amount": fraud_amounts + nonfraud_amounts,
        "label": ["Fraud"] * len(fraud_amounts) + ["Non-Fraud"] * len(nonfraud_amounts)
    }),
    x="label",
    y="amount",
    showfliers=False
)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Amount")
plt.tight_layout()
plt.savefig(reports_dir / "transaction_amount_distribution.png", dpi=150)
plt.close()

print("\n========== EDA SUMMARY ==========")
print(f"Total records: {total_rows:,}")
print(f"Fraud records: {total_fraud:,}")
print(f"Non-fraud records: {total_nonfraud:,}")
print(f"Fraud percentage: {total_fraud / total_rows * 100:.4f}%")
print(f"Existing flagged fraud records: {flagged_fraud_count:,}")
print(f"Flagged non-fraud records: {flagged_nonfraud_count:,}")

print("\nTransaction type summary:")
print(type_summary.to_string(index=False))

print("\nBalance consistency checks:")
print(balance_summary.to_string(index=False))

print("\nReports and charts saved in reports/")