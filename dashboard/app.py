from pathlib import Path
import os
import json
from datetime import date, timedelta

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

from dotenv import load_dotenv


# =========================================================
# PATH
# =========================================================

ROOT = Path(__file__).resolve().parents[1]


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# DATA FILES
# =========================================================

DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "clean_meal_data.csv"
)

METRICS_FILE = (
    ROOT
    / "reports"
    / "demand_metrics.json"
)

ANOMALY_FILE = (
    ROOT
    / "reports"
    / "anomaly_results.csv"
)

SEGMENT_FILE = (
    ROOT
    / "reports"
    / "segment_assignments.csv"
)

SHAP_FILE = (
    ROOT
    / "reports"
    / "shap_summary.png"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Campus Food Intelligence",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


df = load_data()


# =========================================================
# API URL
# =========================================================

def get_api_url():

    try:

        return st.secrets[
            "API_URL"
        ]

    except Exception:

        return os.getenv(
            "API_URL",
            "https://smart-campus-food-api.vercel.app"
        )


API_URL = get_api_url().rstrip("/")


# =========================================================
# TITLE
# =========================================================

st.title(
    "🍽️ Smart Campus Food Demand & Waste Intelligence"
)

st.caption(
    "Machine-learning based meal demand forecasting, "
    "waste intelligence and preparation recommendations."
)


# =========================================================
# SIDEBAR
# =========================================================

page = st.sidebar.selectbox(
    "Select Page",
    [
        "Executive Overview",
        "Demand Forecast",
        "Waste Analytics",
        "Anomaly Detection",
        "Meal Segmentation",
        "Model Performance"
    ]
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.header(
        "Executive Overview"
    )

    total_records = len(df)

    total_prepared = (
        df["food_prepared_kg"]
        .sum()
    )

    total_consumed = (
        df["food_consumed_kg"]
        .sum()
    )

    total_waste = (
        df["food_wasted_kg"]
        .sum()
    )

    if total_prepared > 0:

        waste_rate = (
            total_waste
            / total_prepared
            * 100
        )

    else:

        waste_rate = 0


    average_daily_demand = (
        df.groupby("date")
        ["students_present"]
        .sum()
        .mean()
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Meal Records",
        f"{total_records:,}"
    )


    col2.metric(
        "Average Daily Demand",
        f"{average_daily_demand:,.0f}"
    )


    col3.metric(
        "Total Waste",
        f"{total_waste:,.0f} kg"
    )


    col4.metric(
        "Waste Rate",
        f"{waste_rate:.2f}%"
    )


    st.subheader(
        "Daily Meal Demand"
    )


    daily_demand = (
        df.groupby("date")
        ["students_present"]
        .sum()
        .reset_index()
    )


    fig = px.line(
        daily_demand,
        x="date",
        y="students_present",
        title="Daily Campus Meal Demand"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# DEMAND FORECAST
# =========================================================

elif page == "Demand Forecast":

    st.header(
        "🔮 Demand Forecast"
    )


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # INPUT SECTION
    # -----------------------------------------------------

    with col1:

        selected_date = st.date_input(
            "Meal Date",
            value=(
                date.today()
                + timedelta(days=1)
            )
        )


        mess_name = st.selectbox(
            "Mess",
            sorted(
                df["mess_name"]
                .unique()
                .tolist()
            )
        )


        meal_type = st.selectbox(
            "Meal Type",
            [
                "Breakfast",
                "Lunch",
                "Dinner"
            ]
        )


        menu_options = sorted(
            df[
                df["meal_type"]
                == meal_type
            ]["menu_name"]
            .unique()
            .tolist()
        )


        menu_name = st.selectbox(
            "Menu",
            menu_options
        )


    # -----------------------------------------------------
    # ENVIRONMENT SECTION
    # -----------------------------------------------------

    with col2:

        occupancy = st.number_input(
            "Hostel Occupancy (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0
        )


        temperature = st.number_input(
            "Temperature (°C)",
            value=28.0
        )


        rainfall = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=2.0
        )


        humidity = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )


        exam_day = st.checkbox(
            "Exam Day"
        )


        holiday = st.checkbox(
            "Holiday"
        )


        special_event = st.checkbox(
            "Special Event"
        )


        students_expected = st.number_input(
            "Expected Students",
            min_value=0,
            value=1200
        )


    # -----------------------------------------------------
    # PREDICTION BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚀 Predict Demand",
        type="primary"
    ):


        # -------------------------------------------------
        # CREATE API PAYLOAD
        # -------------------------------------------------

        payload = {

            "meal_date":
                str(selected_date),

            "mess_name":
                mess_name,

            "meal_type":
                meal_type,

            "menu_name":
                menu_name,

            "hostel_occupancy_pct":
                occupancy,

            "temperature_c":
                temperature,

            "rainfall_mm":
                rainfall,

            "humidity_pct":
                humidity,

            "exam_day":
                int(exam_day),

            "holiday":
                int(holiday),

            "special_event":
                int(special_event),

            "students_expected":
                int(students_expected)
        }


        # -------------------------------------------------
        # CALL DEPLOYED FASTAPI
        # -------------------------------------------------

        try:

            response = httpx.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )


            response.raise_for_status()


            result = response.json()


            st.success(
                "Prediction generated successfully through the deployed FastAPI."
            )


        except httpx.HTTPStatusError as error:

            st.error(
                "The prediction API returned an error."
            )

            st.code(
                f"HTTP Status: "
                f"{error.response.status_code}\n\n"
                f"Response:\n"
                f"{error.response.text}"
            )

            st.stop()


        except httpx.RequestError as error:

            st.error(
                "Unable to connect to the deployed prediction API."
            )

            st.code(
                str(error)
            )

            st.stop()


        except Exception as error:

            st.error(
                "An unexpected error occurred while "
                "calling the prediction API."
            )

            st.code(
                str(error)
            )

            st.stop()


        # -------------------------------------------------
        # DISPLAY PREDICTION
        # -------------------------------------------------

        st.subheader(
            "Prediction Result"
        )


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Predicted Demand",
            f"{result['predicted_demand']:.0f}"
        )


        col2.metric(
            "Recommended Preparation",
            f"{result['recommended_preparation']}"
        )


        col3.metric(
            "Predicted Waste",
            f"{result['predicted_waste_kg']:.1f} kg"
        )


        col4.metric(
            "Waste Rate",
            f"{result['predicted_waste_rate']:.2f}%"
        )


        # -------------------------------------------------
        # RISK
        # -------------------------------------------------

        st.subheader(
            "Risk"
        )


        st.write(
            result["risk_level"]
        )


        # -------------------------------------------------
        # RECOMMENDATION
        # -------------------------------------------------

        st.subheader(
            "Recommendation"
        )


        st.info(
            result["recommendation"]
        )


        # -------------------------------------------------
        # WASTE COST
        # -------------------------------------------------

        st.write(
            "Estimated Waste Cost:",
            f"₹{result['estimated_waste_cost']:,.2f}"
        )


        # -------------------------------------------------
        # DATABASE STATUS
        # -------------------------------------------------

        st.subheader(
            "Database Status"
        )


        if result.get(
            "database_saved",
            False
        ):

            st.success(
                "Prediction successfully saved to Railway MySQL."
            )

        else:

            st.warning(
                "Prediction generated, but it was not saved "
                "to the database."
            )


        if result.get(
            "database_error"
        ):

            st.error(
                result["database_error"]
            )


# =========================================================
# WASTE ANALYTICS
# =========================================================

elif page == "Waste Analytics":

    st.header(
        "♻️ Waste Analytics"
    )


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # WASTE BY MEAL
    # -----------------------------------------------------

    with col1:

        waste_by_meal = (
            df.groupby("meal_type")
            ["food_wasted_kg"]
            .mean()
            .reset_index()
        )


        fig1 = px.bar(
            waste_by_meal,
            x="meal_type",
            y="food_wasted_kg",
            title="Average Waste by Meal"
        )


        st.plotly_chart(
            fig1,
            use_container_width=True
        )


    # -----------------------------------------------------
    # WASTE BY MENU
    # -----------------------------------------------------

    with col2:

        waste_by_menu = (
            df.groupby("menu_name")
            ["food_wasted_kg"]
            .mean()
            .reset_index()
            .sort_values(
                "food_wasted_kg",
                ascending=False
            )
        )


        fig2 = px.bar(
            waste_by_menu,
            x="menu_name",
            y="food_wasted_kg",
            title="Average Waste by Menu"
        )


        st.plotly_chart(
            fig2,
            use_container_width=True
        )


    # -----------------------------------------------------
    # DAILY WASTE
    # -----------------------------------------------------

    daily_waste = (
        df.groupby("date")
        ["food_wasted_kg"]
        .sum()
        .reset_index()
    )


    fig3 = px.line(
        daily_waste,
        x="date",
        y="food_wasted_kg",
        title="Daily Food Waste"
    )


    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# =========================================================
# ANOMALY DETECTION
# =========================================================

elif page == "Anomaly Detection":

    st.header(
        "🚨 Anomaly Detection"
    )


    anomaly_df = pd.read_csv(
        ANOMALY_FILE
    )


    anomaly_df["date"] = pd.to_datetime(
        anomaly_df["date"]
    )


    anomalies = anomaly_df[
        anomaly_df["anomaly"] == -1
    ]


    st.metric(
        "Detected Anomalies",
        len(anomalies)
    )


    st.dataframe(
        anomalies[
            [
                "date",
                "mess_name",
                "meal_type",
                "students_present",
                "food_wasted_kg",
                "waste_rate",
                "anomaly_score"
            ]
        ].sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True
    )


# =========================================================
# MEAL SEGMENTATION
# =========================================================

elif page == "Meal Segmentation":

    st.header(
        "📊 Meal/Mess Segmentation"
    )


    segment_df = pd.read_csv(
        SEGMENT_FILE
    )


    st.dataframe(
        segment_df,
        use_container_width=True
    )


    fig = px.scatter(
        segment_df,
        x="avg_demand",
        y="avg_waste_rate",
        color="cluster",
        hover_data=[
            "mess_name",
            "meal_type",
            "avg_occupancy"
        ],
        title="Meal/Mess Segments"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.header(
        "🤖 Model Performance"
    )


    # -----------------------------------------------------
    # LOAD METRICS
    # -----------------------------------------------------

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metrics = json.load(
            file
        )


    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    st.subheader(
        "Best Demand Model"
    )


    st.write(
        metrics["best_model"]
    )


    # -----------------------------------------------------
    # MODEL METRICS TABLE
    # -----------------------------------------------------

    rows = []


    for name, values in (
        metrics["models"].items()
    ):

        rows.append(
            {
                "Model":
                    name,

                "MAE":
                    values["MAE"],

                "RMSE":
                    values["RMSE"],

                "MAPE":
                    values["MAPE"],

                "R²":
                    values["R2"]
            }
        )


    metrics_df = pd.DataFrame(
        rows
    )


    st.dataframe(
        metrics_df,
        use_container_width=True
    )


    # -----------------------------------------------------
    # SHAP
    # -----------------------------------------------------

    if SHAP_FILE.exists():

        st.subheader(
            "SHAP Explainability"
        )


        st.image(
            str(SHAP_FILE),
            caption=(
                "SHAP summary for the demand model"
            )
        )