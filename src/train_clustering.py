from pathlib import Path

import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


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
    / "cluster_model.pkl"
)

ASSIGNMENT_FILE = (
    ROOT
    / "reports"
    / "segment_assignments.csv"
)

PROFILE_FILE = (
    ROOT
    / "reports"
    / "cluster_profiles.csv"
)


df = pd.read_csv(DATA_FILE)


df["waste_rate"] = (
    df["food_wasted_kg"]
    / df["food_prepared_kg"]
).fillna(0)


# ---------------------------------------------------------
# AGGREGATE MESS + MEAL
# ---------------------------------------------------------

grouped = (
    df.groupby(
        [
            "mess_name",
            "meal_type"
        ]
    )
    .agg(
        avg_demand=(
            "students_present",
            "mean"
        ),
        avg_waste_rate=(
            "waste_rate",
            "mean"
        ),
        avg_occupancy=(
            "hostel_occupancy_pct",
            "mean"
        )
    )
    .reset_index()
)


feature_columns = [
    "avg_demand",
    "avg_waste_rate",
    "avg_occupancy"
]


X = grouped[
    feature_columns
]


# ---------------------------------------------------------
# CLUSTER
# ---------------------------------------------------------

pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            KMeans(
                n_clusters=4,
                random_state=42,
                n_init=20
            )
        )
    ]
)


pipeline.fit(X)


clusters = pipeline.predict(X)


grouped["cluster"] = clusters


# ---------------------------------------------------------
# SILHOUETTE SCORE
# ---------------------------------------------------------

scaled_X = (
    pipeline
    .named_steps["scaler"]
    .transform(X)
)


score = silhouette_score(
    scaled_X,
    clusters
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

joblib.dump(
    {
        "model": pipeline,
        "features": feature_columns
    },
    MODEL_FILE
)


grouped.to_csv(
    ASSIGNMENT_FILE,
    index=False
)


profiles = (
    grouped.groupby("cluster")
    .agg(
        avg_demand=(
            "avg_demand",
            "mean"
        ),
        avg_waste_rate=(
            "avg_waste_rate",
            "mean"
        ),
        avg_occupancy=(
            "avg_occupancy",
            "mean"
        ),
        combinations=(
            "cluster",
            "count"
        )
    )
    .reset_index()
)


profiles.to_csv(
    PROFILE_FILE,
    index=False
)


print("=" * 70)
print("CLUSTERING COMPLETE")
print("=" * 70)

print(
    "Silhouette Score:",
    round(score, 4)
)

print("\nCluster assignments:")
print(grouped)

print("\nCluster profiles:")
print(profiles)

print("=" * 70)