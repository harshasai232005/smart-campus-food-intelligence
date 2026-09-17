CREATE DATABASE IF NOT EXISTS campus_food_intelligence;

USE campus_food_intelligence;


CREATE TABLE IF NOT EXISTS meal_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    meal_date DATE,
    mess_name VARCHAR(100),
    meal_type VARCHAR(50),
    menu_name VARCHAR(100),
    day_of_week VARCHAR(20),
    day_of_week_num INT,
    month INT,
    semester INT,
    is_weekend BOOLEAN,
    hostel_occupancy_pct FLOAT,
    exam_day BOOLEAN,
    holiday BOOLEAN,
    special_event BOOLEAN,
    temperature_c FLOAT,
    rainfall_mm FLOAT,
    humidity_pct FLOAT,
    students_expected INT,
    students_present INT,
    food_prepared_kg FLOAT,
    food_consumed_kg FLOAT,
    food_wasted_kg FLOAT,
    food_cost_per_kg FLOAT,
    portion_kg FLOAT,
    planned_preparation_kg FLOAT,
    waste_rate FLOAT
);


CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    meal_date DATE,
    mess_name VARCHAR(100),
    meal_type VARCHAR(50),
    predicted_demand FLOAT,
    recommended_preparation INT,
    planned_preparation_kg FLOAT,
    predicted_waste_kg FLOAT,
    predicted_waste_rate FLOAT,
    estimated_waste_cost FLOAT,
    anomaly_detected BOOLEAN,
    risk_level VARCHAR(20)
);


CREATE TABLE IF NOT EXISTS recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT,
    recommendation_text TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (
        prediction_id
    )
    REFERENCES predictions(id)
);