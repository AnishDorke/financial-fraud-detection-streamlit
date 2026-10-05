import os
import pandas as pd
import numpy as np

BASE_PATH = "data/processed_historical"

FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "sender_txn_count_before",
    "sender_amount_sum_before",
    "sender_amount_mean_before",
    "sender_amount_max_before",
    "sender_time_since_last",
    "destination_txn_count_before",
    "destination_amount_sum_before",
    "destination_amount_mean_before",
    "destination_amount_max_before",
    "destination_time_since_last",
    "target"
]

print("=" * 70)
print("HISTORICAL FEATURE AUDIT")
print("=" * 70)

for split in ["train", "validation", "test"]:

    print("\n" + "=" * 70)
    print(split.upper())
    print("=" * 70)

    path = os.path.join(BASE_PATH, split)

    files = sorted(
        f for f in os.listdir(path)
        if f.endswith(".parquet")
    )

    print(f"Parquet files: {len(files)}")

    df = pd.read_parquet(
        os.path.join(path, files[0])
    )

    print("\nColumns:")
    print(list(df.columns))

    missing_columns = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_columns:
        print("\nMISSING COLUMNS:")
        print(missing_columns)
        continue

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df[FEATURES].isna().sum())

    numeric_features = [
        feature for feature in FEATURES
        if feature not in ["type"]
    ]

    print("\nInfinite values:")

    for feature in numeric_features:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        infinite_count = np.isinf(values).sum()

        if infinite_count > 0:
            print(
                f"{feature}: {infinite_count}"
            )

    print("\nSample rows:")
    print(df.head(5).to_string(index=False))

    print("\nHistorical feature statistics:")

    history_features = [
        "sender_txn_count_before",
        "sender_amount_sum_before",
        "sender_amount_mean_before",
        "sender_amount_max_before",
        "sender_time_since_last",
        "destination_txn_count_before",
        "destination_amount_sum_before",
        "destination_amount_mean_before",
        "destination_amount_max_before",
        "destination_time_since_last"
    ]

    print(
        df[history_features]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T
        .to_string()
    )

    print("\nHistory availability:")

    print(
        "Sender has no previous transaction:",
        (df["sender_txn_count_before"] == 0).sum()
    )

    print(
        "Destination has no previous transaction:",
        (df["destination_txn_count_before"] == 0).sum()
    )

    print("\nFraud vs legitimate historical behavior:")

    grouped = (
        df.groupby("target")[
            history_features
        ]
        .mean()
        .T
    )

    grouped.columns = [
        "Legitimate",
        "Fraud"
    ]

    print(grouped.to_string())

    print("\nFraud count:")
    print(int(df["target"].sum()))

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)