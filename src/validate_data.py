
from pathlib import Path
import pandas as pd

data_dir = Path("data/raw")
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("PaySim CSV not found in data/raw.")

file_path = csv_files[0]
chunk_size = 100_000

total_rows = 0
fraud_counts = {0: 0, 1: 0}
missing_counts = None
invalid_numeric_counts = {}
transaction_types = {}
minimum_step = None
maximum_step = None
minimum_amount = None
maximum_amount = None
negative_amounts = 0
invalid_labels = 0
duplicate_rows_within_chunks = 0

numeric_columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    total_rows += len(chunk)

    if missing_counts is None:
        missing_counts = pd.Series(0, index=chunk.columns, dtype="int64")
        invalid_numeric_counts = {col: 0 for col in numeric_columns}

    missing_counts = missing_counts.add(
        chunk.isna().sum(), fill_value=0
    ).astype("int64")

    for col in numeric_columns:
        values = pd.to_numeric(chunk[col], errors="coerce")
        invalid_numeric_counts[col] += int(values.isna().sum())
        invalid_numeric_counts[col] += int(
            (~values.isna() & ~values.map(lambda x: pd.notna(x) and abs(x) != float("inf"))).sum()
        )

    fraud_counts.update({
        0: fraud_counts[0] + int((chunk["isFraud"] == 0).sum()),
        1: fraud_counts[1] + int((chunk["isFraud"] == 1).sum()),
    })

    invalid_labels += int((~chunk["isFraud"].isin([0, 1])).sum())
    negative_amounts += int((chunk["amount"] < 0).sum())
    duplicate_rows_within_chunks += int(chunk.duplicated().sum())

    transaction_types.update({
        key: transaction_types.get(key, 0) + value
        for key, value in chunk["type"].value_counts(dropna=False).items()
    })

    current_min_step = chunk["step"].min()
    current_max_step = chunk["step"].max()
    current_min_amount = chunk["amount"].min()
    current_max_amount = chunk["amount"].max()

    minimum_step = current_min_step if minimum_step is None else min(minimum_step, current_min_step)
    maximum_step = current_max_step if maximum_step is None else max(maximum_step, current_max_step)
    minimum_amount = current_min_amount if minimum_amount is None else min(minimum_amount, current_min_amount)
    maximum_amount = current_max_amount if maximum_amount is None else max(maximum_amount, current_max_amount)

    print(f"Processed {total_rows:,} rows")

print("\n========== DATASET VALIDATION REPORT ==========")
print(f"File: {file_path.name}")
print(f"Total records: {total_rows:,}")
print(f"Total columns: {len(missing_counts)}")

print("\nFraud distribution:")
for label, count in sorted(fraud_counts.items()):
    print(f"Label {label}: {count:,} ({count / total_rows * 100:.4f}%)")

print("\nMissing values:")
print(missing_counts)

print("\nInvalid numeric values:")
print(invalid_numeric_counts)

print("\nTransaction types:")
for transaction_type, count in sorted(transaction_types.items()):
    print(f"{transaction_type}: {count:,}")

print("\nOther validation checks:")
print(f"Invalid fraud labels: {invalid_labels:,}")
print(f"Negative transaction amounts: {negative_amounts:,}")
print(f"Duplicate rows within individual chunks: {duplicate_rows_within_chunks:,}")
print(f"Minimum step: {minimum_step}")
print(f"Maximum step: {maximum_step}")
print(f"Minimum amount: {minimum_amount}")
print(f"Maximum amount: {maximum_amount}")

report_path = Path("reports")
report_path.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame({
    "metric": [
        "total_records",
        "fraud_records",
        "non_fraud_records",
        "fraud_percentage",
        "invalid_fraud_labels",
        "negative_amounts",
        "duplicate_rows_within_chunks",
    ],
    "value": [
        total_rows,
        fraud_counts[1],
        fraud_counts[0],
        fraud_counts[1] / total_rows * 100 if total_rows else 0,
        invalid_labels,
        negative_amounts,
        duplicate_rows_within_chunks,
    ],
})

summary.to_csv(report_path / "validation_summary.csv", index=False)
missing_counts.rename("missing_count").to_csv(
    report_path / "missing_values.csv"
)

print("\nReports saved to reports/")