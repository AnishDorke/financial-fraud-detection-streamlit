
import pandas as pd

from inference_pipeline import FraudInferencePipeline


VALID_TRANSACTION = {
    "step": 1,
    "type": "PAYMENT",
    "amount": 100.0,
    "oldbalanceOrg": 500.0,
    "oldbalanceDest": 200.0,
    "nameDest": "C_TEST_001",
}


def expect_rejection(label, transaction):
    pipeline = FraudInferencePipeline()

    try:
        pipeline.predict_step(pd.DataFrame([transaction]))
    except (ValueError, TypeError):
        print(f"PASS: {label} rejected")
        return

    raise AssertionError(f"FAIL: {label} was accepted")


expect_rejection(
    "missing destination",
    {key: value for key, value in VALID_TRANSACTION.items()
     if key != "nameDest"},
)

expect_rejection(
    "negative amount",
    {**VALID_TRANSACTION, "amount": -10.0},
)

expect_rejection(
    "unknown transaction type",
    {**VALID_TRANSACTION, "type": "UNKNOWN"},
)

expect_rejection(
    "missing amount",
    {**VALID_TRANSACTION, "amount": None},
)

expect_rejection(
    "non-finite amount",
    {**VALID_TRANSACTION, "amount": float("inf")},
)

expect_rejection(
    "multiple steps in one batch",
    pd.DataFrame([
        VALID_TRANSACTION,
        {**VALID_TRANSACTION, "step": 2},
    ]),
)

print("\nInput validation tests completed.")