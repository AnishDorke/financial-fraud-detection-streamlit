
from pathlib import Path
import pandas as pd

data_dir = Path("data/raw")
csv_files = list(data_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in data/raw. Copy the PaySim CSV there first.")

file_path = csv_files[0]

print("Dataset:", file_path.name)
print("File size (MB):", round(file_path.stat().st_size / (1024 * 1024), 2))

df = pd.read_csv(file_path, nrows=10000)

print("\nSample shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values in sample:")
print(df.isnull().sum())

if "isFraud" in df.columns:
    print("\nFraud label distribution in sample:")
    print(df["isFraud"].value_counts(dropna=False))
    print("\nFraud percentage in sample:")
    print((df["isFraud"].value_counts(normalize=True) * 100).round(4))