from pathlib import Path

import pandas as pd

from inference_pipeline import FEATURE_COLUMNS, FraudInferencePipeline


RAW_PATH = Path(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "nameDest",
]


pipeline = FraudInferencePipeline()

verified_steps = 0
total_rows = 0
pending = None


def reference_info_for_step(step):
    if 1 <= step <= 323:
        reference_dir = Path(
            "data/processed_historical_v3_1/train"
        )
        part_index = step - 1

    elif 324 <= step <= 378:
        reference_dir = Path(
            "data/processed_historical_v3_1/validation"
        )
        part_index = step - 324

    elif 379 <= step <= 743:
        reference_dir = Path(
            "data/processed_historical_v3_1/test"
        )
        part_index = step - 379

    else:
        raise ValueError(
            f"Invalid step: {step}. "
            "Expected a step between 1 and 743."
        )

    return reference_dir, part_index


def verify_step(batch, step):
    reference_dir, part_index = reference_info_for_step(step)

    reference_path = (
        reference_dir
        / f"part_{part_index:05d}.parquet"
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing reference file for step {step}: "
            f"{reference_path}"
        )

    reference = pd.read_parquet(reference_path)

    if reference["step"].nunique() != 1:
        raise AssertionError(
            f"Reference file for step {step} "
            "contains multiple steps."
        )

    reference_step = int(
        reference["step"].iloc[0]
    )

    if reference_step != step:
        raise AssertionError(
            f"Reference step mismatch: "
            f"expected {step}, "
            f"found {reference_step}"
        )

    if len(batch) != len(reference):
        raise AssertionError(
            f"Step {step}: row count mismatch. "
            f"Raw={len(batch)}, "
            f"reference={len(reference)}"
        )

    generated = pipeline._build_features(
        batch,
        step,
    )

    actual = generated[
        FEATURE_COLUMNS
    ].reset_index(drop=True)

    expected = reference[
        FEATURE_COLUMNS
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_categorical=False,
        check_exact=False,
        rtol=1e-8,
        atol=1e-7,
    )

    pipeline._update_history(
        batch,
        step,
    )

    pipeline.last_processed_step = step

    return len(batch)


with pd.read_csv(
    RAW_PATH,
    usecols=RAW_COLUMNS,
    chunksize=250_000,
) as reader:

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):

        chunk["step"] = chunk["step"].astype(int)

        if pending is not None:
            chunk = pd.concat(
                [
                    pending,
                    chunk,
                ],
                ignore_index=True,
            )

        max_step = int(
            chunk["step"].max()
        )

        complete_steps = sorted(
            int(step)
            for step in chunk["step"].unique()
            if int(step) < max_step
        )

        for step in complete_steps:

            batch = chunk.loc[
                chunk["step"] == step
            ].copy()

            rows = verify_step(
                batch,
                step,
            )

            verified_steps += 1
            total_rows += rows

            if (
                verified_steps <= 10
                or verified_steps % 25 == 0
            ):
                print(
                    f"Step {step:3d}: passed "
                    f"({rows:,} rows) | "
                    f"verified={verified_steps}/743"
                )

        pending = chunk.loc[
            chunk["step"] == max_step
        ].copy()


if pending is not None and not pending.empty:

    final_step = int(
        pending["step"].iloc[0]
    )

    rows = verify_step(
        pending,
        final_step,
    )

    verified_steps += 1
    total_rows += rows

    print(
        f"Step {final_step:3d}: passed "
        f"({rows:,} rows) | "
        f"verified={verified_steps}/743"
    )


if verified_steps != 743:
    raise AssertionError(
        f"Expected 743 verified steps, "
        f"but verified {verified_steps}"
    )


print()
print("=" * 60)
print("FULL INFERENCE FEATURE VERIFICATION PASSED")
print("=" * 60)
print(f"Steps verified : {verified_steps:,}")
print(f"Rows verified  : {total_rows:,}")
print("Reference      : V3.1 historical features")
print("Splits         : train + validation + test")
print("=" * 60)