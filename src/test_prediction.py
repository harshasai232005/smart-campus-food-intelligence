from src.predictor import make_prediction


sample = {
    "meal_date": "2026-09-18",
    "mess_name": "Central Mess",
    "meal_type": "Lunch",
    "menu_name": "Biryani",
    "hostel_occupancy_pct": 90,
    "temperature_c": 29,
    "rainfall_mm": 2,
    "humidity_pct": 72,
    "exam_day": 0,
    "holiday": 0,
    "special_event": 0,
    "students_expected": 1250
}


result = make_prediction(
    sample
)


print("=" * 70)
print("PREDICTION TEST")
print("=" * 70)

for key, value in result.items():
    print(
        f"{key}: {value}"
    )

print("=" * 70)