import os
import glob
import numpy as np
import pandas as pd



BASE_PATH = "data/processed_historical_v3_1"

SPLITS = ["train", "validation", "test"]

EXPECTED_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
    "destination_txn_count_before",
    "destination_amount_sum_before",
    "destination_amount_mean_before",
    "destination_amount_max_before",
    "destination_time_since_last",
    "destination_txn_count_prev_1",
    "destination_txn_count_prev_5",
    "destination_txn_count_prev_24",
    "destination_amount_sum_prev_5",
    "destination_amount_sum_prev_24",
    "target"
]

VELOCITY_COLUMNS = [
    "destination_txn_count_prev_1",
    "destination_txn_count_prev_5",
    "destination_txn_count_prev_24",
    "destination_amount_sum_prev_5",
    "destination_amount_sum_prev_24"
]

HISTORY_COLUMNS = [
    "destination_txn_count_before",
    "destination_amount_sum_before",
    "destination_amount_mean_before",
    "destination_amount_max_before",
    "destination_time_since_last"
]


def audit_split(split):

    path = os.path.join(BASE_PATH, split)
    files = sorted(glob.glob(os.path.join(path, "*.parquet")))

    print("\n" + "=" * 80)
    print(f"AUDITING {split.upper()}")
    print("=" * 80)

    print(f"Parquet files: {len(files)}")

    total_rows = 0
    total_fraud = 0

    min_step = None
    max_step = None

    missing_total = {}
    negative_total = {}

    fraud_rows = []
    legit_rows = []

    velocity_nonzero = {
        column: 0
        for column in VELOCITY_COLUMNS
    }

    for file in files:

        df = pd.read_parquet(file)

        total_rows += len(df)

        total_fraud += int(df["target"].sum())

        current_min = int(df["step"].min())
        current_max = int(df["step"].max())

        if min_step is None:
            min_step = current_min
        else:
            min_step = min(min_step, current_min)

        if max_step is None:
            max_step = current_max
        else:
            max_step = max(max_step, current_max)

        for column in EXPECTED_COLUMNS:

            missing = int(df[column].isna().sum())

            missing_total[column] = (
                missing_total.get(column, 0) + missing
            )

        for column in EXPECTED_COLUMNS:

            if pd.api.types.is_numeric_dtype(df[column]):

                negative_count = int(
                    (df[column] < 0).sum()
                )

                negative_total[column] = (
                    negative_total.get(column, 0)
                    + negative_count
                )

        for column in VELOCITY_COLUMNS:

            velocity_nonzero[column] += int(
                (df[column] > 0).sum()
            )

        fraud_part = df[df["target"] == 1]

        legit_part = df[df["target"] == 0]

        if len(fraud_part) > 0:
            fraud_rows.append(fraud_part)

        if len(legit_part) > 0:
            legit_rows.append(legit_part)

    fraud_df = pd.concat(
        fraud_rows,
        ignore_index=True
    )

    legit_df = pd.concat(
        legit_rows,
        ignore_index=True
    )

    print(f"Rows: {total_rows:,}")
    print(f"Fraud: {total_fraud:,}")
    print(
        f"Fraud rate: "
        f"{total_fraud / total_rows * 100:.6f}%"
    )

    print(f"Step range: {min_step} - {max_step}")

    print("\nColumn check:")

    sample = pd.read_parquet(files[0])

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in sample.columns
    ]

    unexpected_columns = [
        column
        for column in sample.columns
        if column not in EXPECTED_COLUMNS
    ]

    print(
        "Missing expected columns:",
        missing_columns
    )

    print(
        "Unexpected columns:",
        unexpected_columns
    )

    print("\nMissing values:")

    for column in EXPECTED_COLUMNS:

        print(
            f"{column}: "
            f"{missing_total[column]:,}"
        )

    print("\nNegative values:")

    for column in EXPECTED_COLUMNS:

        if column in negative_total:

            print(
                f"{column}: "
                f"{negative_total[column]:,}"
            )

    print("\nVelocity feature availability:")

    for column in VELOCITY_COLUMNS:

        count = velocity_nonzero[column]

        percentage = (
            count / total_rows * 100
        )

        print(
            f"{column}: "
            f"{count:,} non-zero "
            f"({percentage:.4f}%)"
        )

    print("\nFraud vs legitimate means:")

    comparison_columns = (
        HISTORY_COLUMNS
        + VELOCITY_COLUMNS
    )

    for column in comparison_columns:

        fraud_mean = fraud_df[column].mean()
        legit_mean = legit_df[column].mean()

        print(
            f"{column}: "
            f"fraud={fraud_mean:.6f} | "
            f"legit={legit_mean:.6f}"
        )

    print("\nFraud percentiles:")

    for column in VELOCITY_COLUMNS:

        values = fraud_df[column]

        print(f"\n{column}")

        print(
            values.quantile(
                [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            )
        )

    print("\nLegitimate percentiles:")

    for column in VELOCITY_COLUMNS:

        values = legit_df[column]

        print(f"\n{column}")

        print(
            values.quantile(
                [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            )
        )


print("=" * 80)
print("FULL HISTORICAL FEATURES V3 AUDIT")
print("=" * 80)

for split in SPLITS:

    audit_split(split)

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)