
from pathlib import Path

import pandas as pd

from inference_pipeline import FraudInferencePipeline


RAW_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
FEATURE_DIR = Path("data/processed_historical_v3_1/train")
STEPS_TO_VERIFY = 10

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "nameDest",
]

FEATURE_COLUMNS = [
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
]

pipeline = FraudInferencePipeline()

for step in range(1, STEPS_TO_VERIFY + 1):
    reference_path = FEATURE_DIR / f"part_{step - 1:05d}.parquet"

    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference file: {reference_path}")

    reference = pd.read_parquet(reference_path)

    if reference["step"].nunique() != 1:
        raise ValueError(f"Expected one step in {reference_path.name}")

    reference_step = int(reference["step"].iloc[0])

    if reference_step != step:
        raise ValueError(
            f"Expected step {step}, found step {reference_step}"
        )

    raw = pd.read_csv(
        RAW_PATH,
        usecols=RAW_COLUMNS,
        nrows=1,
    ) if step == 1 else None

    if step == 1:
        raw_batches = pd.read_csv(
            RAW_PATH,
            usecols=RAW_COLUMNS,
            chunksize=250_000,
        )
        pending = pd.DataFrame(columns=RAW_COLUMNS)
        raw_steps = {}

        for chunk in raw_batches:
            combined = pd.concat([pending, chunk], ignore_index=True)
            combined["step"] = combined["step"].astype(int)

            for current_step, group in combined.groupby("step", sort=True):
                if current_step <= STEPS_TO_VERIFY:
                    raw_steps[int(current_step)] = group.copy()
                elif current_step >= STEPS_TO_VERIFY:
                    break

            max_step = int(combined["step"].max())
            pending = combined.loc[combined["step"] == max_step].copy()

            if max_step >= STEPS_TO_VERIFY:
                break

        for current_step in range(1, STEPS_TO_VERIFY + 1):
            if current_step not in raw_steps:
                if current_step == STEPS_TO_VERIFY:
                    continue
                raise ValueError(f"Raw transactions for step {current_step} not found")

    batch = raw_steps.get(step)

    if batch is None:
        print(f"Step {step}: skipped because raw step was incomplete")
        break

    generated = pipeline._build_features(batch, step)

    expected = reference[FEATURE_COLUMNS].reset_index(drop=True)
    actual = generated[FEATURE_COLUMNS].reset_index(drop=True)

    if len(actual) != len(expected):
        raise AssertionError(
            f"Step {step}: row count mismatch "
            f"(generated={len(actual)}, reference={len(expected)})"
        )

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_categorical=False,
        check_exact=False,
        rtol=1e-8,
        atol=1e-7,
    )

    pipeline._update_history(batch, step)
    pipeline.last_processed_step = step

    print(f"Step {step}: passed ({len(actual):,} transactions)")

print("\nMulti-step feature verification completed.")