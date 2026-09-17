from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FILE = ROOT / "data" / "raw" / "meal_records.csv"


df = pd.read_csv(FILE)


print("=" * 70)
print("DATASET VERIFICATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nBasic statistics:")
print(df.describe())

print("\nMeal types:")
print(df["meal_type"].value_counts())

print("\nMesses:")
print(df["mess_name"].value_counts())

print("\nDate range:")
print(df["date"].min())
print(df["date"].max())

print("=" * 70)