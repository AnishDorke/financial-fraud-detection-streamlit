import os
import glob
import pandas as pd

BASE_PATH = "data/processed_historical_v3"

columns = [
    "destination_txn_count_before",
    "destination_amount_sum_before",
    "destination_amount_max_before",
    "destination_txn_count_prev_1",
    "destination_txn_count_prev_5",
    "destination_txn_count_prev_24",
    "destination_amount_sum_prev_5",
    "destination_amount_sum_prev_24"
]

for split in ["train", "validation", "test"]:

    print("\n" + "=" * 80)
    print(split.upper())
    print("=" * 80)

    files = glob.glob(
        os.path.join(
            BASE_PATH,
            split,
            "*.parquet"
        )
    )

    global_max = {
        column: 0
        for column in columns
    }

    global_min = {
        column: float("inf")
        for column in columns
    }

    max_rows = {}

    for file in files:

        df = pd.read_parquet(file)

        for column in columns:

            current_max = df[column].max()
            current_min = df[column].min()

            if current_max > global_max[column]:

                global_max[column] = current_max

                max_row = df.loc[
                    df[column].idxmax()
                ]

                max_rows[column] = {
                    "file": file,
                    "value": current_max,
                    "step": max_row["step"],
                    "amount": max_row["amount"],
                    "target": max_row["target"]
                }

            if current_min < global_min[column]:
                global_min[column] = current_min

    for column in columns:

        print(f"\n{column}")

        print(
            f"Minimum: {global_min[column]}"
        )

        print(
            f"Maximum: {global_max[column]}"
        )

        print(
            f"Max row: {max_rows[column]}"
        )