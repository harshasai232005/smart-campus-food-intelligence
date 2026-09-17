from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "clean_meal_data.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "anomaly_model.pkl"
)

OUTPUT_FILE = (
    ROOT
    / "reports"
    / "anomaly_results.csv"
)


df = pd.read_csv(DATA_FILE)


df["waste_rate"] = (
    df["food_wasted_kg"]
    / df["food_prepared_kg"]
).fillna(0)


feature_columns = [
    "students_present",
    "food_wasted_kg",
    "waste_rate"
]


X = df[feature_columns]


pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            IsolationForest(
                contamination=0.05,
                random_state=42
            )
        )
    ]
)


print("Training anomaly detector...")

pipeline.fit(X)


df["anomaly"] = pipeline.predict(X)

df["anomaly_score"] = (
    pipeline
    .named_steps["model"]
    .decision_function(
        pipeline
        .named_steps["scaler"]
        .transform(X)
    )
)


joblib.dump(
    {
        "model": pipeline,
        "features": feature_columns
    },
    MODEL_FILE
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


anomaly_count = (
    df["anomaly"] == -1
).sum()


print("=" * 70)
print("ANOMALY DETECTION COMPLETE")
print("=" * 70)

print(
    "Total records:",
    len(df)
)

print(
    "Anomalies:",
    anomaly_count
)

print(
    "Anomaly percentage:",
    round(
        anomaly_count
        / len(df)
        * 100,
        2
    ),
    "%"
)

print(
    "Model saved:",
    MODEL_FILE
)

print("=" * 70)