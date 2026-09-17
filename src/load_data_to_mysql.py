from pathlib import Path

import pandas as pd

from src.db import get_engine


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "clean_meal_data.csv"
)


df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"]
)


df = df.rename(
    columns={
        "date": "meal_date"
    }
)


engine = get_engine()


print(
    "Loading rows into MySQL..."
)


df.to_sql(
    "meal_records",
    engine,
    if_exists="append",
    index=False
)


print(
    "Data loaded successfully!"
)

print(
    "Rows loaded:",
    len(df)
)