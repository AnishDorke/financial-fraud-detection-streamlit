import os
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict

RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"
OUTPUT_BASE = "data/processed_historical_v3_1"

TRAIN_END_STEP = 323
VALIDATION_END_STEP = 378

CHUNK_SIZE = 250000

OUTPUT_COLUMNS = [
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

if os.path.exists(OUTPUT_BASE):
    raise RuntimeError(
        f"Output directory already exists: {OUTPUT_BASE}\n"
        "Delete it manually only if you want to regenerate V3.1."
    )

for split in ["train", "validation", "test"]:
    os.makedirs(
        os.path.join(OUTPUT_BASE, split),
        exist_ok=True
    )


def get_split(step):

    if step <= TRAIN_END_STEP:
        return "train"

    if step <= VALIDATION_END_STEP:
        return "validation"

    return "test"


destination_state = {}

recent_by_step = {}

rolling_5_count = defaultdict(int)
rolling_5_amount = defaultdict(float)

rolling_24_count = defaultdict(int)
rolling_24_amount = defaultdict(float)

part_counters = {
    "train": 0,
    "validation": 0,
    "test": 0
}


def process_step(step_df, step_value):

    global destination_state

    step_df = step_df.copy()

    step_df["nameDest"] = (
        step_df["nameDest"]
        .astype(str)
    )

    destination_groups = (
        step_df
        .groupby("nameDest", sort=False)
        .agg(
            current_count=("amount", "size"),
            current_sum=("amount", "sum"),
            current_max=("amount", "max")
        )
    )

    previous_step_destinations = recent_by_step.get(
        step_value - 1,
        {}
    )

    records = []

    for row in step_df.itertuples(index=False):

        destination = str(row.nameDest)

        state = destination_state.get(
            destination
        )

        if state is None:

            previous_count = 0
            previous_sum = 0.0
            previous_max = 0.0
            previous_last_step = None

        else:

            (
                previous_count,
                previous_sum,
                previous_max,
                previous_last_step
            ) = state

        if previous_last_step is None:
            time_since_last = -1
        else:
            time_since_last = (
                step_value - previous_last_step
            )

        previous_1 = previous_step_destinations.get(
            destination,
            (0, 0.0)
        )

        count_prev_1 = previous_1[0]

        count_prev_5 = rolling_5_count.get(
            destination,
            0
        )

        count_prev_24 = rolling_24_count.get(
            destination,
            0
        )

        amount_prev_5 = rolling_5_amount.get(
            destination,
            0.0
        )

        amount_prev_24 = rolling_24_amount.get(
            destination,
            0.0
        )

        records.append({
            "step": int(row.step),
            "type": row.type,
            "amount": float(row.amount),
            "oldbalanceOrg": float(row.oldbalanceOrg),
            "oldbalanceDest": float(row.oldbalanceDest),
            "log_amount": float(
                np.log1p(row.amount)
            ),
            "destination_txn_count_before": int(
                previous_count
            ),
            "destination_amount_sum_before": float(
                previous_sum
            ),
            "destination_amount_mean_before": float(
                previous_sum / previous_count
                if previous_count > 0
                else 0.0
            ),
            "destination_amount_max_before": float(
                previous_max
            ),
            "destination_time_since_last": int(
                time_since_last
            ),
            "destination_txn_count_prev_1": int(
                count_prev_1
            ),
            "destination_txn_count_prev_5": int(
                count_prev_5
            ),
            "destination_txn_count_prev_24": int(
                count_prev_24
            ),
            "destination_amount_sum_prev_5": float(
                amount_prev_5
            ),
            "destination_amount_sum_prev_24": float(
                amount_prev_24
            ),
            "target": int(row.target)
        })

    output = pd.DataFrame(
        records,
        columns=OUTPUT_COLUMNS
    )

    split = get_split(step_value)

    part_number = part_counters[split]

    output_path = os.path.join(
        OUTPUT_BASE,
        split,
        f"part_{part_number:05d}.parquet"
    )

    output.to_parquet(
        output_path,
        index=False
    )

    part_counters[split] += 1

    current_step_history = {}

    for destination, group in destination_groups.iterrows():

        current_count = int(
            group["current_count"]
        )

        current_sum = float(
            group["current_sum"]
        )

        current_max = float(
            group["current_max"]
        )

        current_step_history[destination] = (
            current_count,
            current_sum
        )

        state = destination_state.get(
            destination
        )

        if state is None:

            previous_count = 0
            previous_sum = 0.0
            previous_max = 0.0

        else:

            (
                previous_count,
                previous_sum,
                previous_max,
                _
            ) = state

        destination_state[destination] = (
            previous_count + current_count,
            previous_sum + current_sum,
            max(previous_max, current_max),
            step_value
        )

        rolling_5_count[destination] += current_count
        rolling_5_amount[destination] += current_sum

        rolling_24_count[destination] += current_count
        rolling_24_amount[destination] += current_sum

    recent_by_step[step_value] = (
        current_step_history
    )

    remove_5_step = step_value - 5

    if remove_5_step in recent_by_step:

        old_step = recent_by_step[
            remove_5_step
        ]

        for destination, values in old_step.items():

            old_count, old_amount = values

            rolling_5_count[destination] -= old_count
            rolling_5_amount[destination] -= old_amount

            if rolling_5_count[destination] <= 0:
                rolling_5_count.pop(
                    destination,
                    None
                )

            if rolling_5_amount[destination] <= 0:
                rolling_5_amount.pop(
                    destination,
                    None
                )

    remove_24_step = step_value - 24

    if remove_24_step in recent_by_step:

        old_step = recent_by_step[
            remove_24_step
        ]

        for destination, values in old_step.items():

            old_count, old_amount = values

            rolling_24_count[destination] -= old_count
            rolling_24_amount[destination] -= old_amount

            if rolling_24_count[destination] <= 0:
                rolling_24_count.pop(
                    destination,
                    None
                )

            if rolling_24_amount[destination] <= 0:
                rolling_24_amount.pop(
                    destination,
                    None
                )

        del recent_by_step[
            remove_24_step
        ]


    max_count = output[
        "destination_txn_count_before"
    ].max()

    if max_count > 1000:

        raise RuntimeError(
            f"Sanity check failed at step {step_value}: "
            f"destination_txn_count_before reached "
            f"{max_count}. "
            "This indicates corrupted destination state."
        )

    max_prev_1 = output[
        "destination_txn_count_prev_1"
    ].max()

    max_prev_5 = output[
        "destination_txn_count_prev_5"
    ].max()

    max_prev_24 = output[
        "destination_txn_count_prev_24"
    ].max()

    if max_prev_1 > 1000:
        raise RuntimeError(
            f"Sanity check failed at step {step_value}: "
            f"prev_1 count = {max_prev_1}"
        )

    if max_prev_5 > 1000:
        raise RuntimeError(
            f"Sanity check failed at step {step_value}: "
            f"prev_5 count = {max_prev_5}"
        )

    if max_prev_24 > 1000:
        raise RuntimeError(
            f"Sanity check failed at step {step_value}: "
            f"prev_24 count = {max_prev_24}"
        )

    print(
        f"Step {step_value:3d} | "
        f"Rows {len(step_df):7,d} | "
        f"Destinations {step_df['nameDest'].nunique():6,d} | "
        f"Max historical count {int(max_count):3d}"
    )


print("=" * 80)
print("PREPARING HISTORICAL FEATURES V3.1")
print("=" * 80)

reader = pd.read_csv(
    RAW_PATH,
    usecols=[
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "oldbalanceDest",
        "nameDest",
        "isFraud"
    ],
    chunksize=CHUNK_SIZE
)

current_step = None
step_frames = []

total_rows = 0
total_fraud = 0

for chunk in reader:

    chunk = chunk.rename(
        columns={
            "isFraud": "target"
        }
    )

    chunk["nameDest"] = (
        chunk["nameDest"]
        .astype(str)
    )

    for step_value, step_part in chunk.groupby(
        "step",
        sort=False
    ):

        step_value = int(step_value)

        if current_step is None:
            current_step = step_value

        if step_value != current_step:

            full_step = pd.concat(
                step_frames,
                ignore_index=True
            )

            process_step(
                full_step,
                current_step
            )

            step_frames = []
            current_step = step_value

        step_frames.append(
            step_part[
                [
                    "step",
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "oldbalanceDest",
                    "nameDest",
                    "target"
                ]
            ].copy()
        )

        total_rows += len(step_part)

        total_fraud += int(
            step_part["target"].sum()
        )

if step_frames:

    full_step = pd.concat(
        step_frames,
        ignore_index=True
    )

    process_step(
        full_step,
        current_step
    )

print("\n" + "=" * 80)
print("V3.1 DATASET CREATION COMPLETE")
print("=" * 80)

print(
    f"Total rows: {total_rows:,}"
)

print(
    f"Total fraud: {total_fraud:,}"
)

print(
    f"Train parts: {part_counters['train']}"
)

print(
    f"Validation parts: {part_counters['validation']}"
)

print(
    f"Test parts: {part_counters['test']}"
)

print("\nStrict temporal rule:")
print("Current step is never included in its own history.")

print("\nTarget leakage rule:")
print("Target values are never used to construct features.")

print("\nOutput:")
print(OUTPUT_BASE)

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)