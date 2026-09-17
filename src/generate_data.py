from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    BASE_DEMAND,
    PORTION_KG,
    FOOD_COST_PER_KG,
    MESS_FACTORS,
    MENU_OPTIONS,
    MENU_FACTORS
)


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = ROOT / "data" / "raw" / "meal_records.csv"


# ---------------------------------------------------------
# RANDOM GENERATOR
# ---------------------------------------------------------

rng = np.random.default_rng(42)


# ---------------------------------------------------------
# DATE RANGE
# ---------------------------------------------------------

dates = pd.date_range(
    start="2025-01-01",
    end="2026-08-31",
    freq="D"
)


rows = []


# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

for current_date in dates:

    day_of_week_num = current_date.dayofweek
    month = current_date.month
    day_of_year = current_date.dayofyear

    is_weekend = int(day_of_week_num >= 5)

    # Holiday simulation
    is_holiday = int(rng.random() < 0.045)

    # Exam periods are slightly more common around May and December
    if month in [5, 12]:
        exam_probability = 0.18
    else:
        exam_probability = 0.04

    is_exam_day = int(rng.random() < exam_probability)

    # Special events
    special_event = int(rng.random() < 0.025)

    # Semester
    if 1 <= month <= 5:
        semester = 4
    elif 7 <= month <= 11:
        semester = 5
    else:
        semester = 0

    # -----------------------------------------------------
    # WEATHER
    # -----------------------------------------------------

    temperature = (
        27
        + 7 * np.sin(
            2 * np.pi * (day_of_year / 365)
        )
        + rng.normal(0, 1.8)
    )

    # More rainfall during monsoon months
    if month in [6, 7, 8]:
        rainfall = rng.gamma(shape=1.8, scale=5.0)
    else:
        rainfall = rng.gamma(shape=0.8, scale=1.5)

    if rng.random() < 0.55:
        rainfall = 0

    humidity = (
        60
        + rainfall * 1.4
        + rng.normal(0, 5)
    )

    humidity = float(np.clip(humidity, 35, 98))

    # -----------------------------------------------------
    # LOOP THROUGH MESSES
    # -----------------------------------------------------

    for mess_name, mess_factor in MESS_FACTORS.items():

        # Occupancy changes over time
        seasonal_occupancy = (
            4 * np.sin(
                2 * np.pi * (day_of_year / 365)
            )
        )

        occupancy = (
            82
            + seasonal_occupancy
            + rng.normal(0, 5)
        )

        if is_holiday:
            occupancy -= 18

        occupancy = float(
            np.clip(occupancy, 45, 98)
        )

        # -------------------------------------------------
        # LOOP THROUGH MEALS
        # -------------------------------------------------

        for meal_type in ["Breakfast", "Lunch", "Dinner"]:

            base_demand = BASE_DEMAND[meal_type]

            menu_name = rng.choice(
                MENU_OPTIONS[meal_type]
            )

            menu_factor = MENU_FACTORS[menu_name]

            # Weekend effect
            weekend_factor = 0.78 if is_weekend else 1.0

            # Holiday effect
            holiday_factor = 0.52 if is_holiday else 1.0

            # Exam effect
            exam_factor = 0.88 if is_exam_day else 1.0

            # Event effect
            event_factor = 1.07 if special_event else 1.0

            # Occupancy effect
            occupancy_factor = (
                0.65
                + 0.35 * (occupancy / 100)
            )

            # Heavy rain can reduce physical attendance
            if rainfall > 25:
                weather_factor = 0.94
            else:
                weather_factor = 1.0

            # Add realistic noise
            random_factor = rng.normal(
                1.0,
                0.035
            )

            expected_students = (
                base_demand
                * mess_factor
                * menu_factor
                * weekend_factor
                * holiday_factor
                * exam_factor
                * event_factor
                * occupancy_factor
                * weather_factor
                * random_factor
            )

            expected_students = max(
                100,
                int(round(expected_students))
            )

            # Attendance rate
            attendance_rate = rng.uniform(
                0.90,
                0.99
            )

            students_present = int(
                round(
                    expected_students
                    * attendance_rate
                )
            )

            students_present = min(
                students_present,
                expected_students
            )

            # Portion size
            portion_kg = PORTION_KG[meal_type]

            # Food prepared based mainly on expected demand
            preparation_buffer = rng.uniform(
                1.05,
                1.14
            )

            food_prepared_kg = (
                expected_students
                * portion_kg
                * preparation_buffer
            )

            # Actual consumption
            consumption_factor = rng.uniform(
                0.92,
                0.99
            )

            possible_consumption = (
                students_present
                * portion_kg
                * consumption_factor
            )

            food_consumed_kg = min(
                food_prepared_kg,
                possible_consumption
            )

            food_wasted_kg = max(
                0,
                food_prepared_kg
                - food_consumed_kg
            )

            cost_per_kg = (
                FOOD_COST_PER_KG[meal_type]
            )

            rows.append(
                {
                    "date": current_date,
                    "mess_name": mess_name,
                    "meal_type": meal_type,
                    "menu_name": menu_name,
                    "hostel_occupancy_pct": round(
                        occupancy,
                        2
                    ),
                    "exam_day": is_exam_day,
                    "holiday": is_holiday,
                    "special_event": special_event,
                    "temperature_c": round(
                        temperature,
                        2
                    ),
                    "rainfall_mm": round(
                        rainfall,
                        2
                    ),
                    "humidity_pct": round(
                        humidity,
                        2
                    ),
                    "students_expected": expected_students,
                    "students_present": students_present,
                    "food_prepared_kg": round(
                        food_prepared_kg,
                        2
                    ),
                    "food_consumed_kg": round(
                        food_consumed_kg,
                        2
                    ),
                    "food_wasted_kg": round(
                        food_wasted_kg,
                        2
                    ),
                    "food_cost_per_kg": cost_per_kg
                }
            )


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(rows)

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
print("DATA GENERATION COMPLETE")
print("=" * 70)

print(f"Output file: {OUTPUT_FILE}")
print(f"Rows      : {len(df):,}")
print(f"Columns   : {len(df.columns)}")

print("\nFirst 5 rows:")
print(df.head())

print("\nMeal distribution:")
print(df["meal_type"].value_counts())

print("\nDate range:")
print(df["date"].min(), "to", df["date"].max())

print("=" * 70)