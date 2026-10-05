from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from inference_pipeline import FraudInferencePipeline


RAW_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
REFERENCE_ROOT = Path("data/processed_historical_v3_1")

COLUMNS = [
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


def reference_path(step):
    if step <= 323:
        split = "train"
        part = step - 1
    elif step <= 378:
        split = "validation"
        part = step - 324
    else:
        split = "test"
        part = step - 379

    return (
        REFERENCE_ROOT
        / split
        / f"part_{part:05d}.parquet"
    )


pipeline = FraudInferencePipeline()

raw_chunks = pd.read_csv(
    RAW_PATH,
    usecols=COLUMNS,
    chunksize=250_000,
)

pending = None
steps_tested = 0
rows_tested = 0
max_score_diff = 0.0

for chunk in raw_chunks:
    chunk["step"] = chunk["step"].astype(int)

    if pending is not None:
        chunk = pd.concat([pending, chunk], ignore_index=True)
        pending = None

    max_step = int(chunk["step"].max())

    complete = chunk.loc[
        chunk["step"] < max_step,
        COLUMNS,
    ].copy()

    pending = chunk.loc[
        chunk["step"] == max_step,
        COLUMNS,
    ].copy()

    for step in sorted(complete["step"].unique()):
        if steps_tested >= 743:
            break

        step_batch = complete.loc[
            complete["step"] == step,
            COLUMNS,
        ].copy()

        reference_file = reference_path(step)
        reference = pd.read_parquet(reference_file)

        expected_features = reference[FEATURE_COLUMNS].copy()
        expected_features["type"] = expected_features["type"].astype("category")

        actual_features = pipeline._build_features(
            step_batch,
            step,
        )

        actual_features["type"] = actual_features["type"].astype(
            expected_features["type"].dtype
        )

        assert_frame_equal(
            actual_features[FEATURE_COLUMNS].reset_index(drop=True),
            expected_features.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-8,
            atol=1e-7,
        )

        results = pipeline.predict_step(step_batch)

        expected_scores = pipeline.model.predict(
            expected_features[FEATURE_COLUMNS]
        )

        actual_scores = results["fraud_score"].to_numpy()

        np.testing.assert_allclose(
            actual_scores,
            expected_scores,
            rtol=1e-12,
            atol=1e-12,
        )

        assert len(results) == len(step_batch)
        assert results["fraud_score"].notna().all()
        assert results["fraud_score"].between(0, 1).all()

        score_diff = float(
            np.max(np.abs(actual_scores - expected_scores))
        )

        max_score_diff = max(max_score_diff, score_diff)
        rows_tested += len(results)
        steps_tested += 1

        print(
            f"Step {step:3d} PASSED | "
            f"Rows: {len(results):,} | "
            f"Max score diff: {score_diff:.3e}"
        )

    if steps_tested >= 743:
        break

if pending is not None and steps_tested < 743:
    step = int(pending["step"].iloc[0])

    reference_file = reference_path(step)
    reference = pd.read_parquet(reference_file)

    expected_features = reference[FEATURE_COLUMNS].copy()
    expected_features["type"] = expected_features["type"].astype("category")

    actual_features = pipeline._build_features(
        pending,
        step,
    )

    actual_features["type"] = actual_features["type"].astype(
        expected_features["type"].dtype
    )

    assert_frame_equal(
        actual_features[FEATURE_COLUMNS].reset_index(drop=True),
        expected_features.reset_index(drop=True),
        check_dtype=False,
        rtol=1e-8,
        atol=1e-7,
    )

    results = pipeline.predict_step(pending)

    expected_scores = pipeline.model.predict(
        expected_features[FEATURE_COLUMNS]
    )

    actual_scores = results["fraud_score"].to_numpy()

    np.testing.assert_allclose(
        actual_scores,
        expected_scores,
        rtol=1e-12,
        atol=1e-12,
    )

    assert len(results) == len(pending)
    assert results["fraud_score"].notna().all()
    assert results["fraud_score"].between(0, 1).all()

    score_diff = float(
        np.max(np.abs(actual_scores - expected_scores))
    )

    max_score_diff = max(max_score_diff, score_diff)
    rows_tested += len(results)
    steps_tested += 1

    print(
        f"Step {step:3d} PASSED | "
        f"Rows: {len(results):,} | "
        f"Max score diff: {score_diff:.3e}"
    )

assert steps_tested == 743

print()
print("PREDICT_STEP API VERIFICATION PASSED")
print(f"Steps verified       : {steps_tested}")
print(f"Rows verified        : {rows_tested:,}")
print(f"Maximum score diff   : {max_score_diff:.3e}")
print("Reference            : V3.1 historical features")
print("Inference method     : FraudInferencePipeline.predict_step()")