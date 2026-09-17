from pathlib import Path

import pandas as pd
import numpy as np

from src.constants import PORTION_KG


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data"
    / "raw"
    / "meal_records.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "clean_meal_data.csv"
)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("Original shape:", df.shape)


# ---------------------------------------------------------
# DATE
# ---------------------------------------------------------

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# Remove rows where date could not be parsed
df = df.dropna(
    subset=["date"]
)


# ---------------------------------------------------------
# DUPLICATES
# ---------------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(
    "Duplicates removed:",
    before_duplicates - after_duplicates
)


# ---------------------------------------------------------
# NUMERIC COLUMNS
# ---------------------------------------------------------

numeric_columns = [
    "hostel_occupancy_pct",
    "exam_day",
    "holiday",
    "special_event",
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "students_expected",
    "students_present",
    "food_prepared_kg",
    "food_consumed_kg",
    "food_wasted_kg",
    "food_cost_per_kg"
]


for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# HANDLE MISSING NUMERIC VALUES
# ---------------------------------------------------------

for column in numeric_columns:

    if df[column].isnull().any():
        df[column] = df[column].fillna(
            df[column].median()
        )


# ---------------------------------------------------------
# HANDLE MISSING CATEGORICAL VALUES
# ---------------------------------------------------------

categorical_columns = [
    "mess_name",
    "meal_type",
    "menu_name"
]


for column in categorical_columns:

    df[column] = df[column].fillna(
        "Unknown"
    )


# ---------------------------------------------------------
# CORRECT IMPOSSIBLE VALUES
# ---------------------------------------------------------

df["students_expected"] = (
    df["students_expected"]
    .clip(lower=0)
    .round()
    .astype(int)
)

df["students_present"] = (
    df["students_present"]
    .clip(
        lower=0,
        upper=df["students_expected"]
    )
    .round()
    .astype(int)
)

for column in [
    "food_prepared_kg",
    "food_consumed_kg",
    "food_wasted_kg",
    "rainfall_mm"
]:
    df[column] = df[column].clip(
        lower=0
    )


# ---------------------------------------------------------
# RECALCULATE WASTE
# ---------------------------------------------------------

df["food_wasted_kg"] = (
    df["food_prepared_kg"]
    - df["food_consumed_kg"]
).clip(lower=0)


# ---------------------------------------------------------
# CALENDAR FEATURES
# ---------------------------------------------------------

df["day_of_week"] = (
    df["date"]
    .dt
    .day_name()
)

df["day_of_week_num"] = (
    df["date"]
    .dt
    .dayofweek
)

df["month"] = (
    df["date"]
    .dt
    .month
)

df["is_weekend"] = (
    df["day_of_week_num"] >= 5
).astype(int)


def get_semester(month):

    if 1 <= month <= 5:
        return 4

    if 7 <= month <= 11:
        return 5

    return 0


df["semester"] = df["month"].apply(
    get_semester
)


# ---------------------------------------------------------
# FOOD PLANNING FEATURES
# ---------------------------------------------------------

df["portion_kg"] = (
    df["meal_type"]
    .map(PORTION_KG)
    .fillna(0.4)
)

df["planned_preparation_kg"] = (
    df["students_expected"]
    * df["portion_kg"]
    * 1.08
)

df["waste_rate"] = np.where(
    df["food_prepared_kg"] > 0,
    (
        df["food_wasted_kg"]
        / df["food_prepared_kg"]
    ),
    0
)


# ---------------------------------------------------------
# SORT
# ---------------------------------------------------------

df = df.sort_values(
    [
        "date",
        "mess_name",
        "meal_type"
    ]
).reset_index(drop=True)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("=" * 70)
print("DATA CLEANING COMPLETE")
print("=" * 70)

print("Final shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSaved to:")
print(OUTPUT_FILE)

print("=" * 70)