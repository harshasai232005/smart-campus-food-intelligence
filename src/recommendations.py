import math

from src.constants import (
    PORTION_KG,
    FOOD_COST_PER_KG
)


def calculate_recommended_meals(
    predicted_demand,
    demand_mae
):

    predicted_demand = max(
        1,
        predicted_demand
    )

    error_buffer = (
        demand_mae
        / predicted_demand
    )

    error_buffer = max(
        0.02,
        min(
            error_buffer,
            0.08
        )
    )

    recommended = math.ceil(
        predicted_demand
        * (1 + error_buffer)
    )

    return recommended


def generate_recommendation(
    meal_type,
    predicted_demand,
    recommended_meals,
    predicted_waste_kg,
    anomaly_detected
):

    portion_kg = PORTION_KG[
        meal_type
    ]

    prepared_kg = (
        recommended_meals
        * portion_kg
    )

    waste_rate = (
        predicted_waste_kg
        / prepared_kg
        if prepared_kg > 0
        else 0
    )

    if anomaly_detected:

        recommendation = (
            "An unusual demand/waste pattern "
            "was detected. Verify attendance, "
            "events and campus conditions before "
            "finalizing the preparation quantity."
        )

    elif waste_rate >= 0.10:

        recommendation = (
            "Predicted waste is relatively high. "
            "Use a smaller preparation buffer and "
            "consider preparing food in smaller batches."
        )

    elif predicted_demand >= 1200:

        recommendation = (
            "Demand is high. Prepare the recommended "
            "quantity and ensure sufficient serving "
            "capacity during the meal period."
        )

    else:

        recommendation = (
            "Use the recommended preparation quantity "
            "and monitor actual consumption for future "
            "forecast improvement."
        )

    estimated_waste_cost = (
        predicted_waste_kg
        * FOOD_COST_PER_KG[
            meal_type
        ]
    )

    return {
        "waste_rate": waste_rate,
        "estimated_waste_cost": estimated_waste_cost,
        "recommendation": recommendation
    }