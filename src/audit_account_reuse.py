import pandas as pd
import numpy as np

RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"

print("=" * 70)
print("PAYSIM ACCOUNT REUSE AUDIT")
print("=" * 70)

columns = [
    "step",
    "type",
    "nameOrig",
    "nameDest",
    "amount",
    "isFraud"
]

sender_counts = {}
destination_counts = {}
sender_fraud = {}
destination_fraud = {}

total_rows = 0
total_fraud = 0

for chunk in pd.read_csv(
    RAW_PATH,
    usecols=columns,
    chunksize=250000
):

    total_rows += len(chunk)
    total_fraud += int(chunk["isFraud"].sum())

    sender_group = (
        chunk.groupby("nameOrig")
        .agg(
            transaction_count=("nameOrig", "size"),
            fraud_count=("isFraud", "sum")
        )
    )

    destination_group = (
        chunk.groupby("nameDest")
        .agg(
            transaction_count=("nameDest", "size"),
            fraud_count=("isFraud", "sum")
        )
    )

    for account, row in sender_group.iterrows():

        if account not in sender_counts:
            sender_counts[account] = 0
            sender_fraud[account] = 0

        sender_counts[account] += int(row["transaction_count"])
        sender_fraud[account] += int(row["fraud_count"])

    for account, row in destination_group.iterrows():

        if account not in destination_counts:
            destination_counts[account] = 0
            destination_fraud[account] = 0

        destination_counts[account] += int(row["transaction_count"])
        destination_fraud[account] += int(row["fraud_count"])

print("\nTotal rows:", f"{total_rows:,}")
print("Total fraud:", f"{total_fraud:,}")

sender_counts_series = pd.Series(sender_counts, dtype="int64")
destination_counts_series = pd.Series(
    destination_counts,
    dtype="int64"
)

print("\n" + "=" * 70)
print("SENDER ACCOUNT REUSE")
print("=" * 70)

print(
    "Unique senders:",
    f"{len(sender_counts_series):,}"
)

print(
    "Transactions per sender - statistics:"
)

print(
    sender_counts_series.describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    ).to_string()
)

sender_once = int(
    (sender_counts_series == 1).sum()
)

sender_multiple = int(
    (sender_counts_series > 1).sum()
)

print(
    "\nSenders appearing once:",
    f"{sender_once:,}",
    f"({sender_once / len(sender_counts_series):.2%})"
)

print(
    "Senders appearing more than once:",
    f"{sender_multiple:,}",
    f"({sender_multiple / len(sender_counts_series):.2%})"
)

print("\n" + "=" * 70)
print("DESTINATION ACCOUNT REUSE")
print("=" * 70)

print(
    "Unique destinations:",
    f"{len(destination_counts_series):,}"
)

print(
    "Transactions per destination - statistics:"
)

print(
    destination_counts_series.describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            0.999
        ]
    ).to_string()
)

destination_once = int(
    (destination_counts_series == 1).sum()
)

destination_multiple = int(
    (destination_counts_series > 1).sum()
)

print(
    "\nDestinations appearing once:",
    f"{destination_once:,}",
    f"({destination_once / len(destination_counts_series):.2%})"
)

print(
    "Destinations appearing more than once:",
    f"{destination_multiple:,}",
    f"({destination_multiple / len(destination_counts_series):.2%})"
)

print("\n" + "=" * 70)
print("SENDER TRANSACTION-COUNT DISTRIBUTION")
print("=" * 70)

sender_distribution = (
    sender_counts_series
    .value_counts()
    .sort_index()
    .head(20)
)

print(sender_distribution.to_string())

print("\n" + "=" * 70)
print("DESTINATION TRANSACTION-COUNT DISTRIBUTION")
print("=" * 70)

destination_distribution = (
    destination_counts_series
    .value_counts()
    .sort_index()
    .head(20)
)

print(destination_distribution.to_string())

print("\n" + "=" * 70)
print("FRAUD BY SENDER REUSE")
print("=" * 70)

sender_fraud_table = pd.DataFrame({
    "transaction_count": sender_counts_series,
    "fraud_count": pd.Series(sender_fraud)
})

sender_fraud_table["fraud_rate"] = (
    sender_fraud_table["fraud_count"]
    / sender_fraud_table["transaction_count"]
)

sender_buckets = pd.cut(
    sender_fraud_table["transaction_count"],
    bins=[0, 1, 2, 5, 10, 50, np.inf],
    labels=[
        "1",
        "2",
        "3-5",
        "6-10",
        "11-50",
        "51+"
    ]
)

sender_summary = (
    sender_fraud_table
    .groupby(sender_buckets, observed=False)
    .agg(
        accounts=("transaction_count", "size"),
        transactions=("transaction_count", "sum"),
        frauds=("fraud_count", "sum")
    )
)

sender_summary["fraud_rate"] = (
    sender_summary["frauds"]
    / sender_summary["transactions"]
)

print(sender_summary.to_string())

print("\n" + "=" * 70)
print("FRAUD BY DESTINATION REUSE")
print("=" * 70)

destination_fraud_table = pd.DataFrame({
    "transaction_count": destination_counts_series,
    "fraud_count": pd.Series(destination_fraud)
})

destination_fraud_table["fraud_rate"] = (
    destination_fraud_table["fraud_count"]
    / destination_fraud_table["transaction_count"]
)

destination_buckets = pd.cut(
    destination_fraud_table["transaction_count"],
    bins=[0, 1, 2, 5, 10, 50, np.inf],
    labels=[
        "1",
        "2",
        "3-5",
        "6-10",
        "11-50",
        "51+"
    ]
)

destination_summary = (
    destination_fraud_table
    .groupby(destination_buckets, observed=False)
    .agg(
        accounts=("transaction_count", "size"),
        transactions=("transaction_count", "sum"),
        frauds=("fraud_count", "sum")
    )
)

destination_summary["fraud_rate"] = (
    destination_summary["frauds"]
    / destination_summary["transactions"]
)

print(destination_summary.to_string())

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)