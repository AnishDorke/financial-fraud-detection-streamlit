
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "train"
VALIDATION_DIR = PROJECT_ROOT / "data" / "processed" / "validation"
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"

for split_name, split_dir in [
    ("TRAIN", TRAIN_DIR),
    ("VALIDATION", VALIDATION_DIR),
    ("TEST", TEST_DIR),
]:
    files = sorted(split_dir.glob("part-*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {split_dir}")

    df = pd.read_parquet(files[0])

    print(f"\n{'=' * 60}")
    print(f"{split_name} FEATURE AUDIT")
    print(f"{'=' * 60}")
    print("Sample file:", files[0].name)
    print("Columns:")
    for column in df.columns:
        print(f"  {column}: {df[column].dtype}")

    if "target" in df.columns:
        print("\nFraud rate in sample part:")
        print(f"{df['target'].mean() * 100:.6f}%")

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nSample rows:")
    print(df.head(3).to_string(index=False))