from pathlib import Path

import pandas as pd

from inference_pipeline import FEATURE_COLUMNS, FraudInferencePipeline


RAW_PATH = Path("data/raw/PS_20174392719_1491204439457_log.csv")
FEATURE_DIR = Path("data/processed_historical_v3_1/train")

TARGET_STEPS = {1, 2, 3, 4, 5, 6, 10, 23, 24, 25, 26, 30}

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "nameDest",
]


def verify_step(pipeline, batch, step):
    reference_path = FEATURE_DIR / f"part_{step - 1:05d}.parquet"
    reference = pd.read_parquet(reference_path)

    if reference["step"].nunique() != 1:
        raise AssertionError(
            f"Reference file contains multiple steps: {reference_path}"
        )

    if int(reference["step"].iloc[0]) != step:
        raise AssertionError(
            f"Reference step mismatch for step {step}"
        )

    generated = pipeline._build_features(batch, step)

    expected = reference[FEATURE_COLUMNS].reset_index(drop=True)
    actual = generated[FEATURE_COLUMNS].reset_index(drop=True)

    if len(batch) != len(reference):
        raise AssertionError(
            f"Step {step}: row count mismatch "
            f"(raw={len(batch)}, reference={len(reference)})"
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

    print(
        f"Step {step}: feature parity passed "
        f"({len(batch):,} rows)"
    )


pipeline = FraudInferencePipeline()

pending = None
verified_steps = set()

with pd.read_csv(
    RAW_PATH,
    usecols=RAW_COLUMNS,
    chunksize=250_000,
) as reader:

    for chunk in reader:
        chunk["step"] = chunk["step"].astype(int)

        if pending is not None:
            chunk = pd.concat(
                [pending, chunk],
                ignore_index=True,
            )

        max_step_in_chunk = int(chunk["step"].max())

        complete_steps = sorted(
            step
            for step in chunk["step"].unique()
            if int(step) < max_step_in_chunk
        )

        for step in complete_steps:
            step = int(step)

            if step in verified_steps:
                continue

            batch = chunk.loc[
                chunk["step"] == step
            ].copy()

            if step in TARGET_STEPS:
                verify_step(
                    pipeline,
                    batch,
                    step,
                )
                verified_steps.add(step)
            else:
                pipeline._update_history(
                    batch,
                    step,
                )
                pipeline.last_processed_step = step

        pending = chunk.loc[
            chunk["step"] == max_step_in_chunk
        ].copy()

        if (
            max(TARGET_STEPS) < max_step_in_chunk
            and verified_steps == TARGET_STEPS
        ):
            break


if pending is not None:
    final_step = int(pending["step"].iloc[0])

    if final_step in TARGET_STEPS and final_step not in verified_steps:
        verify_step(
            pipeline,
            pending,
            final_step,
        )
        verified_steps.add(final_step)


missing = TARGET_STEPS - verified_steps

if missing:
    raise AssertionError(
        f"Steps not verified: {sorted(missing)}"
    )

print("\nAll selected history-boundary checks passed.")