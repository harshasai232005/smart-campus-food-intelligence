from functools import lru_cache
from pathlib import Path
from src.db import get_engine
from sqlalchemy import text
from datetime import datetime

import json
import joblib
import pandas as pd
import numpy as np

from src.constants import (
    PORTION_KG
)

from src.recommendations import (
    calculate_recommended_meals,
    generate_recommendation
)


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "clean_meal_data.csv"
)

DEMAND_MODEL_FILE = (
    ROOT
    / "models"
    / "demand_model.pkl"
)

WASTE_MODEL_FILE = (
    ROOT
    / "models"
    / "waste_model.pkl"
)

ANOMALY_MODEL_FILE = (
    ROOT
    / "models"
    / "anomaly_model.pkl"
)

METRICS_FILE = (
    ROOT
    / "reports"
    / "demand_metrics.json"
)


@lru_cache(maxsize=1)
def load_history():

    df = pd.read_csv(
        DATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


@lru_cache(maxsize=1)
def load_demand_model():

    return joblib.load(
        DEMAND_MODEL_FILE
    )


@lru_cache(maxsize=1)
def load_waste_model():

    return joblib.load(
        WASTE_MODEL_FILE
    )


@lru_cache(maxsize=1)
def load_anomaly_model():

    return joblib.load(
        ANOMALY_MODEL_FILE
    )


@lru_cache(maxsize=1)
def load_demand_metrics():

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def semester_from_month(month):

    if 1 <= month <= 5:
        return 4

    if 7 <= month <= 11:
        return 5

    return 0


def get_history_values(
    meal_date,
    mess_name,
    meal_type
):

    history = load_history()

    history = history[
        (history["date"] < meal_date)
        &
        (history["mess_name"] == mess_name)
        &
        (history["meal_type"] == meal_type)
    ].sort_values("date")

    if len(history) == 0:

        return {
            "lag_1": 0,
            "lag_7": 0,
            "rolling_7_demand": 0
        }

    last_values = (
        history["students_present"]
        .tolist()
    )

    lag_1 = last_values[-1]

    if len(last_values) >= 7:
        lag_7 = last_values[-7]
    else:
        lag_7 = (
            float(
                np.mean(last_values)
            )
        )

    rolling_7 = (
        float(
            np.mean(
                last_values[-7:]
            )
        )
    )

    return {
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_7_demand": rolling_7
    }


def build_demand_row(payload):

    meal_date = pd.Timestamp(
        payload["meal_date"]
    )

    history_values = (
        get_history_values(
            meal_date,
            payload["mess_name"],
            payload["meal_type"]
        )
    )

    day_of_week_num = (
        meal_date.dayofweek
    )

    month = meal_date.month

    row = {
        "mess_name":
            payload["mess_name"],

        "meal_type":
            payload["meal_type"],

        "menu_name":
            payload["menu_name"],

        "hostel_occupancy_pct":
            payload["hostel_occupancy_pct"],

        "temperature_c":
            payload["temperature_c"],

        "rainfall_mm":
            payload["rainfall_mm"],

        "humidity_pct":
            payload["humidity_pct"],

        "exam_day":
            payload["exam_day"],

        "holiday":
            payload["holiday"],

        "special_event":
            payload["special_event"],

        "students_expected":
            payload["students_expected"],

        "day_of_week_num":
            day_of_week_num,

        "month":
            month,

        "semester":
            semester_from_month(
                month
            ),

        **history_values
    }

    return pd.DataFrame(
        [row]
    )


def build_waste_row(
    payload,
    recommended_meals
):

    meal_date = pd.Timestamp(
        payload["meal_date"]
    )

    portion_kg = PORTION_KG[
        payload["meal_type"]
    ]

    planned_preparation_kg = (
        recommended_meals
        * portion_kg
    )

    row = {
        "mess_name":
            payload["mess_name"],

        "meal_type":
            payload["meal_type"],

        "menu_name":
            payload["menu_name"],

        "hostel_occupancy_pct":
            payload["hostel_occupancy_pct"],

        "temperature_c":
            payload["temperature_c"],

        "rainfall_mm":
            payload["rainfall_mm"],

        "humidity_pct":
            payload["humidity_pct"],

        "exam_day":
            payload["exam_day"],

        "holiday":
            payload["holiday"],

        "special_event":
            payload["special_event"],

        "students_expected":
            payload["students_expected"],

        "planned_preparation_kg":
            planned_preparation_kg,

        "day_of_week_num":
            meal_date.dayofweek,

        "month":
            meal_date.month,

        "semester":
            semester_from_month(
                meal_date.month
            )
    }

    return (
        pd.DataFrame([row]),
        planned_preparation_kg
    )


def make_prediction(payload):

    # -----------------------------------------------------
    # LOAD MODELS
    # -----------------------------------------------------

    demand_bundle = (
        load_demand_model()
    )

    waste_bundle = (
        load_waste_model()
    )

    anomaly_bundle = (
        load_anomaly_model()
    )

    demand_model = (
        demand_bundle["model"]
    )

    waste_model = (
        waste_bundle["model"]
    )

    anomaly_model = (
        anomaly_bundle["model"]
    )


    # -----------------------------------------------------
    # DEMAND
    # -----------------------------------------------------

    demand_input = (
        build_demand_row(
            payload
        )
    )

    predicted_demand = float(
        demand_model.predict(
            demand_input
        )[0]
    )

    predicted_demand = max(
        0,
        predicted_demand
    )


    # -----------------------------------------------------
    # PREPARATION QUANTITY
    # -----------------------------------------------------

    demand_metrics = (
        load_demand_metrics()
    )

    best_model = (
        demand_metrics["best_model"]
    )

    demand_mae = (
        demand_metrics["models"]
        [best_model]["MAE"]
    )

    recommended_meals = (
        calculate_recommended_meals(
            predicted_demand,
            demand_mae
        )
    )


    # -----------------------------------------------------
    # WASTE
    # -----------------------------------------------------

    waste_input, planned_kg = (
        build_waste_row(
            payload,
            recommended_meals
        )
    )

    predicted_waste_kg = float(
        waste_model.predict(
            waste_input
        )[0]
    )

    predicted_waste_kg = max(
        0,
        predicted_waste_kg
    )


    # -----------------------------------------------------
    # ANOMALY
    # -----------------------------------------------------

    waste_rate = (
        predicted_waste_kg
        / planned_kg
        if planned_kg > 0
        else 0
    )

    anomaly_input = pd.DataFrame(
        [
            {
                "students_present":
                    predicted_demand,

                "food_wasted_kg":
                    predicted_waste_kg,

                "waste_rate":
                    waste_rate
            }
        ]
    )

    anomaly_prediction = (
        anomaly_model.predict(
            anomaly_input
        )[0]
    )

    anomaly_detected = (
        anomaly_prediction == -1
    )


    # -----------------------------------------------------
    # RECOMMENDATION
    # -----------------------------------------------------

    recommendation = (
        generate_recommendation(
            payload["meal_type"],
            predicted_demand,
            recommended_meals,
            predicted_waste_kg,
            anomaly_detected
        )
    )


    # -----------------------------------------------------
    # RISK
    # -----------------------------------------------------

    if (
        anomaly_detected
        or recommendation["waste_rate"] >= 0.12
    ):
        risk_level = "HIGH"

    elif recommendation[
        "waste_rate"
    ] >= 0.07:

        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"


    return {
        "meal_date":
            str(payload["meal_date"]),

        "mess_name":
            payload["mess_name"],

        "meal_type":
            payload["meal_type"],

        "predicted_demand":
            round(
                predicted_demand,
                2
            ),

        "recommended_preparation":
            recommended_meals,

        "planned_preparation_kg":
            round(
                planned_kg,
                2
            ),

        "predicted_waste_kg":
            round(
                predicted_waste_kg,
                2
            ),

        "predicted_waste_rate":
            round(
                recommendation["waste_rate"]
                * 100,
                2
            ),

        "estimated_waste_cost":
            round(
                recommendation[
                    "estimated_waste_cost"
                ],
                2
            ),

        "anomaly_detected":
            bool(anomaly_detected),

        "risk_level":
            risk_level,

        "recommendation":
            recommendation[
                "recommendation"
            ]
    }
def save_prediction_to_db(result):

    engine = get_engine()

    with engine.begin() as connection:

        insert_prediction = text(
            """
            INSERT INTO predictions (
                meal_date,
                mess_name,
                meal_type,
                predicted_demand,
                recommended_preparation,
                planned_preparation_kg,
                predicted_waste_kg,
                predicted_waste_rate,
                estimated_waste_cost,
                anomaly_detected,
                risk_level
            )
            VALUES (
                :meal_date,
                :mess_name,
                :meal_type,
                :predicted_demand,
                :recommended_preparation,
                :planned_preparation_kg,
                :predicted_waste_kg,
                :predicted_waste_rate,
                :estimated_waste_cost,
                :anomaly_detected,
                :risk_level
            )
            """
        )

        db_result = connection.execute(
            insert_prediction,
            result
        )

        prediction_id = (
            db_result.lastrowid
        )

        connection.execute(
            text(
                """
                INSERT INTO recommendations (
                    prediction_id,
                    recommendation_text
                )
                VALUES (
                    :prediction_id,
                    :recommendation_text
                )
                """
            ),
            {
                "prediction_id":
                    prediction_id,

                "recommendation_text":
                    result["recommendation"]
            }
        )

    return prediction_id