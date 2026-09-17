from pathlib import Path

import joblib
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.features import (
    add_calendar_features,
    add_demand_lags
)


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
    / "demand_xgboost.pkl"
)

OUTPUT_FILE = (
    ROOT
    / "reports"
    / "shap_summary.png"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

bundle = joblib.load(
    MODEL_FILE
)

pipeline = bundle["model"]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(
    df["date"]
)

df = add_calendar_features(df)

df = add_demand_lags(df)

df = df.dropna(
    subset=[
        "lag_1",
        "lag_7",
        "rolling_7_demand"
    ]
)


categorical_columns = [
    "mess_name",
    "meal_type",
    "menu_name"
]

numeric_columns = [
    "hostel_occupancy_pct",
    "temperature_c",
    "rainfall_mm",
    "humidity_pct",
    "exam_day",
    "holiday",
    "special_event",
    "students_expected",
    "day_of_week_num",
    "month",
    "semester",
    "lag_1",
    "lag_7",
    "rolling_7_demand"
]

feature_columns = (
    categorical_columns
    + numeric_columns
)


X = df[
    feature_columns
].tail(500)


# ---------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------

preprocessor = (
    pipeline
    .named_steps[
        "preprocessor"
    ]
)

tree_model = (
    pipeline
    .named_steps[
        "model"
    ]
)


X_transformed = (
    preprocessor
    .transform(X)
)


feature_names = (
    preprocessor
    .get_feature_names_out()
)


# ---------------------------------------------------------
# SHAP
# ---------------------------------------------------------

explainer = shap.TreeExplainer(
    tree_model
)

shap_values = (
    explainer
    .shap_values(
        X_transformed
    )
)


# ---------------------------------------------------------
# PLOT
# ---------------------------------------------------------

plt.figure(
    figsize=(12, 8)
)

shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)

plt.title(
    "SHAP Feature Importance - Demand Model"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_FILE
)

print("=" * 70)