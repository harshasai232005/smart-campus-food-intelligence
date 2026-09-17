from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from src.features import (
    add_calendar_features,
    add_demand_lags
)

from src.model_utils import (
    create_preprocessor,
    calculate_regression_metrics
)


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "clean_meal_data.csv"
)

MODEL_DIR = ROOT / "models"

REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

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

target_column = "students_present"


# ---------------------------------------------------------
# TIME-BASED SPLIT
# ---------------------------------------------------------

split_date = (
    df["date"].max()
    - pd.Timedelta(days=60)
)

train_df = df[
    df["date"] < split_date
].copy()

test_df = df[
    df["date"] >= split_date
].copy()


X_train = train_df[
    feature_columns
]

y_train = train_df[
    target_column
]

X_test = test_df[
    feature_columns
]

y_test = test_df[
    target_column
]


# ---------------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------------

preprocessor = create_preprocessor(
    categorical_columns,
    numeric_columns
)


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=4
        )
}


results = {}

fitted_models = {}

predictions = {}


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

for model_name, model in models.items():

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print(
        f"\nTraining: {model_name}"
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predicted = pipeline.predict(
        X_test
    )

    metrics = calculate_regression_metrics(
        y_test,
        predicted
    )

    results[model_name] = metrics

    fitted_models[model_name] = pipeline

    predictions[model_name] = predicted

    print(metrics)


# ---------------------------------------------------------
# BASELINE
# ---------------------------------------------------------

baseline_pred = test_df["lag_7"]

baseline_metrics = (
    calculate_regression_metrics(
        y_test,
        baseline_pred
    )
)

results["Lag-7 Baseline"] = (
    baseline_metrics
)


# ---------------------------------------------------------
# BEST MODEL
# ---------------------------------------------------------

model_names = list(
    fitted_models.keys()
)

best_model_name = min(
    model_names,
    key=lambda name:
    results[name]["MAE"]
)

best_pipeline = fitted_models[
    best_model_name
]


# ---------------------------------------------------------
# SAVE BEST MODEL
# ---------------------------------------------------------

joblib.dump(
    {
        "model": best_pipeline,
        "features": feature_columns
    },
    MODEL_DIR / "demand_model.pkl"
)


# ---------------------------------------------------------
# SAVE XGBOOST MODEL
# ---------------------------------------------------------

xgb_pipeline = fitted_models[
    "XGBoost"
]

joblib.dump(
    {
        "model": xgb_pipeline,
        "features": feature_columns
    },
    MODEL_DIR / "demand_xgboost.pkl"
)


# ---------------------------------------------------------
# TEST PREDICTIONS
# ---------------------------------------------------------

best_predictions = predictions[
    best_model_name
]

prediction_output = test_df[
    [
        "date",
        "mess_name",
        "meal_type",
        "students_present"
    ]
].copy()

prediction_output[
    "predicted_demand"
] = best_predictions

prediction_output.to_csv(
    REPORT_DIR
    / "demand_test_predictions.csv",
    index=False
)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

metrics_output = {
    "best_model": best_model_name,
    "split_date": str(split_date.date()),
    "models": results
}

with open(
    REPORT_DIR / "demand_metrics.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics_output,
        file,
        indent=4
    )


# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

best_estimator = (
    best_pipeline
    .named_steps["model"]
)

best_preprocessor = (
    best_pipeline
    .named_steps["preprocessor"]
)

if hasattr(
    best_estimator,
    "feature_importances_"
):

    feature_names = (
        best_preprocessor
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance":
                best_estimator
                .feature_importances_
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    importance_df.to_csv(
        REPORT_DIR
        / "feature_importance.csv",
        index=False
    )


print("\n" + "=" * 70)
print("DEMAND MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    "Best model:",
    best_model_name
)

print("\nModel comparison:")

for name, metrics in results.items():

    print(
        f"\n{name}"
    )

    print(
        f"MAE  : {metrics['MAE']:.2f}"
    )

    print(
        f"RMSE : {metrics['RMSE']:.2f}"
    )

    print(
        f"MAPE : {metrics['MAPE']:.2f}%"
    )

    print(
        f"R2   : {metrics['R2']:.4f}"
    )

print("=" * 70)