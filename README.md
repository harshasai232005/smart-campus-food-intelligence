# 🍽️ Smart Campus Food Demand & Waste Intelligence

An end-to-end **Data Science + Machine Learning + FastAPI + Streamlit +
MySQL** system designed to help campus messes make data-driven food
preparation decisions.

The system forecasts meal demand, recommends preparation quantities,
predicts food waste, detects unusual demand/waste patterns, segments
mess-meal combinations, and provides an interactive analytics dashboard.

## 🚀 Live Demo

**Streamlit Dashboard:** "https://smart-campus-food-intelligence-auqbpq4p4ggxufvvd3wik5.streamlit.app/"

> Replace `YOUR_STREAMLIT_URL` with your current Streamlit Community
> Cloud URL.

**FastAPI Backend:** https://smart-campus-food-api.vercel.app

**FastAPI Docs:** https://smart-campus-food-api.vercel.app/docs

**GitHub:**
https://github.com/harshasai232005/smart-campus-food-intelligence

------------------------------------------------------------------------

## 📌 Project Overview

Campus messes need to decide how much food should be prepared for
breakfast, lunch, and dinner. Demand can change because of student
attendance, hostel occupancy, meal type, menu, weather, examinations,
holidays, special events, and recent attendance patterns.

Preparing too much food can increase waste and cost, while preparing too
little can result in shortages.

This project uses historical meal data and machine learning to provide
an intelligent decision-support system for campus food planning.

### Core Workflow

``` text
Historical Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Model Training
       ↓
Saved ML Models
       ↓
Prediction Engine
       ↓
FastAPI
       ↓
MySQL
       ↑
       |
Streamlit Dashboard
```

------------------------------------------------------------------------

# 🎯 Objectives

-   Forecast student meal demand.
-   Recommend an appropriate food preparation quantity.
-   Predict food waste in kilograms.
-   Calculate predicted waste rate.
-   Estimate food waste cost.
-   Detect unusual demand and waste patterns.
-   Segment mess and meal combinations using clustering.
-   Provide an interactive analytics dashboard.
-   Store predictions and recommendations in MySQL.
-   Deploy the application using cloud services.

------------------------------------------------------------------------

# ✨ Key Features

## 📈 1. Demand Forecasting

The system predicts the expected number of students for a particular
meal.

Features include:

-   Mess
-   Meal type
-   Menu
-   Hostel occupancy
-   Temperature
-   Rainfall
-   Humidity
-   Examination status
-   Holiday status
-   Special events
-   Expected students
-   Historical demand features

Target:

``` text
students_present
```

## 🍛 2. Preparation Recommendation

The system calculates a controlled preparation buffer using the demand
model's historical error.

The buffer is constrained between **2% and 8%**.

``` text
error_buffer = demand_MAE / predicted_demand

error_buffer = bounded between 2% and 8%

recommended_meals =
ceil(predicted_demand × (1 + error_buffer))
```

## 🗑️ 3. Food Waste Prediction

The waste model predicts:

``` text
food_wasted_kg
```

The system calculates:

``` text
Waste Rate =
Predicted Waste / Planned Preparation
```

## 💰 4. Waste Cost Estimation

``` text
Estimated Waste Cost =
Predicted Waste (kg) × Food Cost per kg
```

Current project constants:

  Meal          Portion   Cost/kg
  ----------- --------- ---------
  Breakfast     0.30 kg       ₹90
  Lunch         0.45 kg      ₹110
  Dinner        0.40 kg      ₹105

## 🚨 5. Anomaly Detection

Isolation Forest is used to detect unusual combinations of:

-   Student attendance
-   Food waste
-   Waste rate

Pipeline:

``` text
StandardScaler
      ↓
IsolationForest
```

Configuration:

``` text
contamination = 0.05
```

Output:

``` text
-1 → Anomaly
 1 → Normal
```

Model:

``` text
models/anomaly_model.pkl
```

## 🧩 6. Meal/Mess Segmentation

K-Means clustering groups similar mess + meal combinations using:

-   Average demand
-   Average waste rate
-   Average occupancy

Configuration:

``` text
n_clusters = 4
random_state = 42
n_init = 20
```

A silhouette score is also calculated.

------------------------------------------------------------------------

# 🧠 Machine Learning Models

## Demand Forecasting

Models evaluated:

1.  Linear Regression
2.  Random Forest
3.  XGBoost
4.  Lag-7 Baseline

A time-based final 60-day test period is used.

### Demand Model Results

  Model                     MAE       RMSE       MAPE        R²
  ------------------- --------- ---------- ---------- ---------
  Linear Regression     13.2469    16.3005    2.3769%   0.99255
  Random Forest         13.5312    16.8027    2.4297%   0.99208
  XGBoost               13.6370    16.8674    2.4890%   0.99202
  Lag-7 Baseline        74.8852   116.4058   16.3116%   0.62006

The current report identifies Linear Regression as the saved demand
model based on MAE.

## Waste Prediction

Models evaluated:

-   Linear Regression
-   Random Forest
-   XGBoost

### Waste Model Results

  Model                    MAE      RMSE       MAPE        R²
  ------------------- -------- --------- ---------- ---------
  Linear Regression     8.0914   10.9430   19.7984%   0.77426
  Random Forest         8.1911   11.0187   20.0983%   0.77113
  XGBoost               8.2964   11.2510   20.2111%   0.76138

The current report identifies Linear Regression as the saved waste model
based on MAE.

------------------------------------------------------------------------

# 📊 Evaluation Metrics

### MAE

Average absolute prediction error.

``` text
MAE = Average(|Actual - Predicted|)
```

### RMSE

Root Mean Squared Error, giving greater influence to larger errors.

``` text
RMSE = √Mean((Actual - Predicted)²)
```

### MAPE

Mean Absolute Percentage Error.

``` text
MAPE = Average(|Actual - Predicted| / Actual) × 100
```

### R²

Measures the amount of target variation explained by the model compared
with predicting the mean.

------------------------------------------------------------------------

# 🧹 Data Preprocessing

The preprocessing pipeline performs:

1.  Date conversion using `pd.to_datetime()`.
2.  Removal of invalid dates.
3.  Duplicate removal.
4.  Numeric type conversion.
5.  Missing numerical values filled with column medians.
6.  Missing categorical values filled with `"Unknown"`.
7.  Expected attendance constrained to non-negative values.
8.  Present attendance constrained between zero and expected attendance.
9.  Food quantities and rainfall constrained to non-negative values.
10. Waste recalculated as:

``` text
food_wasted_kg =
food_prepared_kg - food_consumed_kg
```

11. Calendar features added:

-   Day
-   Day number
-   Month
-   Weekend
-   Semester

12. Planning features added:

-   Portion size
-   Planned preparation
-   Waste rate

------------------------------------------------------------------------

# ⚙️ Feature Engineering

Important features:

``` text
day_of_week_num
month
is_weekend
semester
lag_1
lag_7
rolling_7_demand
```

### Lag Features

`lag_1` = previous attendance value within the same mess + meal group.

`lag_7` = seven records back within the same mess + meal group.

`rolling_7_demand` = mean of the previous seven attendance values.

Data is sorted by:

``` text
mess_name
meal_type
date
```

and grouped by:

``` text
mess_name + meal_type
```

------------------------------------------------------------------------

# 🏗️ System Architecture

``` text
                    ┌──────────────────────┐
                    │       USER           │
                    │   Web / Mobile       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    │   dashboard/app.py   │
                    └──────────┬───────────┘
                               │
                         HTTPS + JSON
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Vercel FastAPI    │
                    │      api/main.py     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Prediction Engine   │
                    │  src/predictor.py    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │   Demand   │   │   Waste    │   │  Anomaly   │
       │   Model    │   │   Model    │   │   Model    │
       │ Linear Reg │   │ Linear Reg │   │ Isolation  │
       │            │   │            │   │   Forest   │
       └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                   ┌─────────────────────┐
                   │ Recommendation +    │
                   │ Risk Calculation    │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   Railway MySQL     │
                   │      Database       │
                   └─────────────────────┘
```

------------------------------------------------------------------------

# 🔄 End-to-End Prediction Flow

``` text
1. User enters meal information
            ↓
2. Streamlit creates JSON payload
            ↓
3. Payload sent to FastAPI /predict
            ↓
4. Pydantic validates request
            ↓
5. Prediction engine prepares features
            ↓
6. Demand model predicts attendance
            ↓
7. Preparation buffer is calculated
            ↓
8. Waste model predicts food waste
            ↓
9. Waste rate and cost are calculated
            ↓
10. Isolation Forest checks anomaly
            ↓
11. Risk level is generated
            ↓
12. Recommendation is generated
            ↓
13. Result is saved to MySQL
            ↓
14. Dashboard displays results
```

------------------------------------------------------------------------

# 🚦 Risk Logic

``` text
IF anomaly_detected == True
OR waste_rate >= 12%
        ↓
      HIGH

ELSE IF waste_rate >= 7%
        ↓
     MEDIUM

ELSE
        ↓
      LOW
```

------------------------------------------------------------------------

# 📁 Project Structure

``` text
smart-campus-food-intelligence/
│
├── api/
│   ├── main.py
│   └── requirements.txt
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── meal_records.csv
│   └── processed/
│       ├── clean_meal_data.csv
│       └── features.csv
│
├── database/
│   └── schema.sql
│
├── models/
│   ├── demand_model.pkl
│   ├── demand_xgboost.pkl
│   ├── waste_model.pkl
│   ├── anomaly_model.pkl
│   └── cluster_model.pkl
│
├── notebooks/
│   ├── 01_environment_test.ipynb
│   ├── 03_eda.ipynb
│   └── 04_feature_engineering.ipynb
│
├── reports/
│   ├── demand_metrics.json
│   ├── waste_metrics.json
│   ├── demand_test_predictions.csv
│   ├── waste_test_predictions.csv
│   ├── anomaly_results.csv
│   ├── segment_assignments.csv
│   ├── cluster_profiles.csv
│   └── shap_summary.png
│
├── src/
│   ├── constants.py
│   ├── db.py
│   ├── explain_model.py
│   ├── features.py
│   ├── generate_data.py
│   ├── load_data_to_mysql.py
│   ├── model_utils.py
│   ├── predictor.py
│   ├── preprocess.py
│   ├── recommendations.py
│   ├── train_anomaly.py
│   ├── train_clustering.py
│   ├── train_demand.py
│   ├── train_waste.py
│   └── verify_data.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
└── verify_environment.py
```

------------------------------------------------------------------------

# 🗄️ MySQL Database

Database:

``` text
campus_food_intelligence
```

Main tables:

``` text
meal_records
predictions
recommendations
```

### meal_records

Stores historical meal information.

### predictions

Stores live prediction and risk information.

### recommendations

Stores recommendation text linked to a prediction.

Relationship:

``` text
predictions
     │
     │ prediction_id
     ▼
recommendations
```

### Useful SQL

``` sql
SHOW DATABASES;

USE campus_food_intelligence;

SHOW TABLES;

SELECT *
FROM predictions
ORDER BY id DESC
LIMIT 10;

SELECT *
FROM recommendations
ORDER BY id DESC
LIMIT 10;
```

------------------------------------------------------------------------

# ⚡ FastAPI Backend

The backend is implemented using FastAPI.

  Endpoint           Method   Purpose
  ------------------ -------- -------------------------------
  `/`                GET      Basic API status
  `/health`          GET      Health check
  `/predict`         POST     Generate a single prediction
  `/batch-predict`   POST     Generate multiple predictions
  `/docs`            GET      Interactive API documentation

### Example API Request

``` json
{
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
```

### Example Prediction Response

``` json
{
  "predicted_demand": 1179.44,
  "recommended_preparation": 1204,
  "predicted_waste_kg": 98.12,
  "waste_rate": 0.1811,
  "estimated_waste_cost": 10793.36,
  "anomaly_detected": true,
  "risk": "HIGH",
  "database_saved": true,
  "database_error": null
}
```

------------------------------------------------------------------------

# 📊 Streamlit Dashboard

The dashboard contains:

### Executive Overview

-   Number of records
-   Average demand
-   Total food waste
-   Waste rate
-   Demand visualizations

### Demand Forecast

User input and live prediction through FastAPI.

### Waste Analytics

Historical food-waste analysis.

### Anomaly Detection

Historical unusual demand/waste patterns.

### Meal Segmentation

K-Means clustering results.

### Model Performance

Demand and waste model metrics.

------------------------------------------------------------------------

# 🛠️ Technologies Used

  Technology                  Purpose
  --------------------------- ---------------------------
  Python                      Main programming language
  Pandas                      Data processing
  NumPy                       Numerical calculations
  scikit-learn                Machine learning
  XGBoost                     Gradient boosting
  Joblib                      Model serialization
  FastAPI                     Backend API
  Pydantic                    Request validation
  SQLAlchemy                  Database connectivity
  PyMySQL                     MySQL driver
  MySQL                       Persistent database
  Streamlit                   Interactive dashboard
  Plotly                      Data visualization
  Git                         Version control
  GitHub                      Source-code hosting
  Railway                     MySQL cloud deployment
  Vercel                      FastAPI deployment
  Streamlit Community Cloud   Dashboard deployment
  Docker                      Containerization

------------------------------------------------------------------------

# 💻 Local Installation

## 1. Clone Repository

``` bash
git clone https://github.com/harshasai232005/smart-campus-food-intelligence.git
cd smart-campus-food-intelligence
```

## 2. Create Virtual Environment

Windows:

``` powershell
python -m venv venv
```

Activate:

``` powershell
venv\Scripts\activate
```

Verify:

``` powershell
python --version
where python
```

## 3. Install Dependencies

``` powershell
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🔐 Environment Variables

Create `.env` in the project root:

``` env
MYSQL_PUBLIC_URL=your_railway_mysql_connection_string
API_URL=https://smart-campus-food-api.vercel.app
```

Never commit `.env` to GitHub.

Never expose:

-   Database username
-   Database password
-   Railway connection string
-   API keys
-   Cloud credentials

------------------------------------------------------------------------

# ▶️ Run Streamlit

``` powershell
venv\Scripts\activate
streamlit run dashboard/app.py
```

Open the local URL printed by Streamlit, for example:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# ▶️ Run FastAPI

``` powershell
venv\Scripts\activate
uvicorn api.main:app --reload
```

Open:

``` text
http://127.0.0.1:8000
```

Documentation:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

# 🧪 Run Tests

``` powershell
pytest
```

Tests are located in:

``` text
tests/test_api.py
```

------------------------------------------------------------------------

# 🌐 Deployment

The project uses:

``` text
GitHub
   │
   ├── Vercel
   │      └── FastAPI Backend
   │
   ├── Streamlit Community Cloud
   │      └── Dashboard
   │
   └── Railway
          └── MySQL Database
```

## 🚂 Railway MySQL

Railway hosts the MySQL database.

Database:

``` text
campus_food_intelligence
```

Tables:

``` text
meal_records
predictions
recommendations
```

## ▲ Vercel FastAPI

Live API:

https://smart-campus-food-api.vercel.app

Health:

https://smart-campus-food-api.vercel.app/health

Docs:

https://smart-campus-food-api.vercel.app/docs

Backend:

``` text
api/main.py
```

Prediction engine:

``` text
src/predictor.py
```

## ☁️ Streamlit Community Cloud

Dashboard entry point:

``` text
dashboard/app.py
```

Deployment settings:

``` text
Repository:
harshasai232005/smart-campus-food-intelligence

Branch:
main

Main file:
dashboard/app.py
```

Streamlit secret:

``` toml
API_URL = "https://smart-campus-food-api.vercel.app"
```

------------------------------------------------------------------------

# 📱 Mobile Access

The deployed application can be opened from a smartphone using the
public Streamlit URL.

``` text
Mobile Browser
      ↓
Streamlit Cloud
      ↓
Vercel FastAPI
      ↓
ML Prediction Engine
      ↓
Railway MySQL
```

The local PC does not need to remain switched on for the cloud
deployment to work.

------------------------------------------------------------------------

# 📈 Dataset

The project contains raw and processed meal data.

The deployed dashboard contained:

``` text
10,944 meal records
```

Important data categories:

### Meal Information

``` text
date
mess_name
meal_type
menu_name
```

### Attendance

``` text
students_expected
students_present
```

### Food Quantities

``` text
food_prepared_kg
food_consumed_kg
food_wasted_kg
```

### Environmental Information

``` text
hostel_occupancy_pct
temperature_c
rainfall_mm
humidity_pct
```

### Events

``` text
exam_day
holiday
special_event
```

### Planning/Economic Information

``` text
food_cost_per_kg
portion_kg
planned_preparation_kg
waste_rate
```

------------------------------------------------------------------------

# 📊 Reports

``` text
reports/
│
├── demand_metrics.json
├── waste_metrics.json
├── demand_test_predictions.csv
├── waste_test_predictions.csv
├── anomaly_results.csv
├── segment_assignments.csv
├── cluster_profiles.csv
└── shap_summary.png
```

------------------------------------------------------------------------

# 📂 Important Files

### `src/preprocess.py`

Data cleaning, validation, missing values, waste calculation and
calendar features.

### `src/features.py`

Calendar, lag and rolling demand features.

### `src/train_demand.py`

Demand model training, comparison, evaluation and saving.

### `src/train_waste.py`

Waste model training, comparison, evaluation and saving.

### `src/train_anomaly.py`

Isolation Forest anomaly detection.

### `src/train_clustering.py`

K-Means mess/meal segmentation.

### `src/recommendations.py`

Preparation recommendation, error buffer, waste rate, waste cost and
risk logic.

### `src/predictor.py`

Central prediction engine connecting:

``` text
Demand Model
Waste Model
Anomaly Model
Recommendation Logic
Database
```

### `src/db.py`

MySQL connection and SQLAlchemy database engine.

### `api/main.py`

FastAPI application, endpoints, request validation and prediction
handling.

### `dashboard/app.py`

Streamlit user interface, analytics, charts and live API predictions.

------------------------------------------------------------------------

# 🧠 Why Time-Based Split?

This is a forecasting problem, so the final 60 days are used as the test
period.

``` text
Past Data
─────────────────────────┬──────────────
                         │
                      Training
                         │
                         ▼
                       Testing
                       Future
```

A random split can mix future observations into training data. A
time-based split better represents future prediction usage.

------------------------------------------------------------------------

# 🧠 Why Lag Features?

Meal demand can depend on recent attendance patterns.

Important features include:

``` text
lag_1
lag_7
rolling_7_demand
```

These represent recent and weekly historical attendance behavior.

------------------------------------------------------------------------

# 🧠 Why Compare Multiple Models?

The project compares:

``` text
Linear Regression
Random Forest
XGBoost
```

using:

``` text
MAE
RMSE
MAPE
R²
```

This provides a consistent basis for evaluating different regression
approaches.

------------------------------------------------------------------------

# 🧠 Why Isolation Forest?

Isolation Forest provides an unsupervised approach for identifying
observations that differ from normal patterns.

The project uses:

``` text
students_present
food_wasted_kg
waste_rate
```

for anomaly detection.

------------------------------------------------------------------------

# 🧠 Why K-Means?

K-Means groups mess + meal combinations with similar:

``` text
Average Demand
Average Waste Rate
Average Occupancy
```

------------------------------------------------------------------------

# 🧠 Why StandardScaler?

Demand, waste rate and occupancy have different numerical ranges.
StandardScaler puts these features on a comparable scale before
clustering and anomaly detection.

------------------------------------------------------------------------

# 🧪 API Health Checks

### Root

https://smart-campus-food-api.vercel.app/

### Health

https://smart-campus-food-api.vercel.app/health

### Documentation

https://smart-campus-food-api.vercel.app/docs

A successful `/predict` request means demand inference, waste inference,
anomaly/risk logic and recommendation logic executed successfully.

If:

``` text
database_saved = true
```

the prediction was also saved successfully in MySQL.

------------------------------------------------------------------------

# 🔧 Troubleshooting

### Streamlit does not start

``` powershell
venv\Scripts\activate
streamlit run dashboard/app.py
```

### Dashboard cannot connect to API

Check `API_URL` and open:

https://smart-campus-food-api.vercel.app/health

### Vercel `/predict` returns 500

Check Vercel logs for:

``` text
PREDICTION_ERROR
DATABASE_SAVE_ERROR
BATCH_PREDICTION_ERROR
BATCH_DATABASE_SAVE_ERROR
```

### Database save fails

Check `MYSQL_PUBLIC_URL` and Railway MySQL public connection settings.

### Streamlit says application does not exist

Open Streamlit Community Cloud → My Apps and verify:

``` text
GitHub account
Repository
Branch
Main file
```

The intended entry point is:

``` text
dashboard/app.py
```

------------------------------------------------------------------------

# 📚 Recommended Code Study Order

``` text
1. src/preprocess.py
2. src/features.py
3. src/model_utils.py
4. src/train_demand.py
5. src/train_waste.py
6. src/train_anomaly.py
7. src/train_clustering.py
8. src/recommendations.py
9. src/predictor.py
10. src/db.py
11. database/schema.sql
12. api/main.py
13. dashboard/app.py
```

The central business flow is implemented in:

``` text
src/predictor.py
```

------------------------------------------------------------------------

# 🎓 Viva Explanation

## 30-Second Answer

> My project is a Smart Campus Food Demand and Waste Intelligence
> System. It uses historical meal data to forecast demand, recommend a
> preparation quantity, predict food waste, detect unusual patterns and
> provide operational recommendations. I built the machine-learning
> pipeline in Python, exposed the prediction system through FastAPI,
> stored prediction results in MySQL, and created an interactive
> Streamlit dashboard. The backend is deployed on Vercel and the
> dashboard is deployed using Streamlit Community Cloud.

## 2-Minute Answer

> I developed an end-to-end machine-learning system for campus food
> planning. The dataset contains information such as meal date, mess,
> meal type, menu, occupancy, weather, events, expected attendance,
> actual attendance and food quantities.
>
> First, I cleaned the raw data by handling missing values, duplicates,
> invalid values and inconsistent attendance or food quantities. I also
> recalculated food waste and generated calendar and planning features.
>
> For demand forecasting, I engineered historical lag features and
> compared Linear Regression, Random Forest and XGBoost using a
> time-based final 60-day test period. I also compared the models
> against a Lag-7 baseline.
>
> For food waste prediction, I trained and evaluated regression models
> using features related to attendance, preparation, occupancy, weather
> and events.
>
> I used Isolation Forest for anomaly detection and K-Means clustering
> to segment mess and meal combinations.
>
> During prediction, FastAPI receives the input, validates it, loads the
> trained models and sends the input through the prediction pipeline.
> The system predicts demand, calculates a preparation recommendation,
> predicts waste, calculates waste rate and cost, checks for anomalies
> and generates a risk level.
>
> Finally, the prediction and recommendation are stored in MySQL, while
> Streamlit provides the user interface and visualization dashboard.

------------------------------------------------------------------------

# 💼 Resume Description

## Smart Campus Food Demand & Waste Intelligence

**Technologies:**

``` text
Python, Pandas, NumPy, Scikit-learn, XGBoost,
FastAPI, Streamlit, MySQL, SQLAlchemy,
Git, GitHub, Vercel, Railway
```

### Resume Points

-   Developed an end-to-end machine-learning system for campus meal
    demand forecasting, food-waste prediction, anomaly detection and
    preparation recommendations.
-   Engineered calendar, occupancy, weather, event and historical lag
    features for meal demand forecasting.
-   Evaluated Linear Regression, Random Forest and XGBoost using a
    time-based test period.
-   Built a FastAPI inference service integrating demand forecasting,
    preparation recommendations, waste prediction and anomaly/risk
    analysis.
-   Implemented MySQL persistence for prediction and recommendation
    records.
-   Developed a Streamlit dashboard for executive analytics, demand
    forecasting, waste analysis, anomaly detection, segmentation and
    model performance.
-   Deployed the backend on Vercel, database on Railway and dashboard
    through Streamlit Community Cloud.

------------------------------------------------------------------------

# ❓ Common Interview Questions

### Why did you use a time-based split?

Because the project is a forecasting problem and future data should be
evaluated as future data rather than randomly mixed with historical
training data.

### Why did you use lag features?

Recent attendance and weekly historical patterns can provide useful
information for predicting upcoming meal demand.

### Why compare multiple regression models?

To compare different model behaviors on the same prediction task using
defined evaluation metrics.

### What is MAE?

MAE is the average absolute difference between actual and predicted
values.

### What is RMSE?

RMSE is the square root of mean squared error and gives greater
influence to larger errors.

### What is R²?

R² measures how much variation in the target is explained by the model
compared with predicting the mean.

### Why Isolation Forest?

It provides an unsupervised method for identifying unusual observations.

### What is contamination?

It represents the expected proportion of anomalies configured for
Isolation Forest.

This project uses:

``` text
0.05
```

### Why K-Means?

To group mess and meal combinations with similar demand, waste and
occupancy characteristics.

### Why FastAPI?

FastAPI exposes the machine-learning prediction pipeline as an HTTP API
and provides request validation.

### Why Streamlit?

Streamlit provides an interactive data-science dashboard.

### Why MySQL?

MySQL provides structured persistent storage for prediction and
recommendation records.

### What happens if MySQL fails?

The API catches the database-save error and can return the prediction
result with:

``` text
database_saved = false
```

------------------------------------------------------------------------

# 📌 Project Status

  Component             Status
  --------------------- ---------------
  Dataset               ✅ Available
  Data preprocessing    ✅ Complete
  Feature engineering   ✅ Complete
  Demand model          ✅ Trained
  Waste model           ✅ Trained
  Anomaly model         ✅ Trained
  Clustering model      ✅ Trained
  Reports               ✅ Available
  Prediction engine     ✅ Complete
  MySQL database        ✅ Configured
  FastAPI               ✅ Deployed
  Streamlit             ✅ Deployed
  Live prediction       ✅ Tested
  GitHub                ✅ Available

------------------------------------------------------------------------

# ⭐ Project Highlights

``` text
✔ End-to-end ML pipeline
✔ Demand forecasting
✔ Food waste prediction
✔ Anomaly detection
✔ K-Means segmentation
✔ Preparation recommendation
✔ Risk classification
✔ MySQL persistence
✔ FastAPI REST API
✔ Interactive Streamlit dashboard
✔ Cloud deployment
✔ Docker configuration
✔ Automated testing
```

------------------------------------------------------------------------

# 👨‍💻 Author

**Harsha Vardhan Satapathi**

B.Tech Computer Science & Engineering

Lovely Professional University

------------------------------------------------------------------------

## 📜 License

This project is intended for educational, academic, portfolio and
demonstration purposes.

------------------------------------------------------------------------

## ⭐ If you found this project useful

Consider giving the repository a ⭐ on GitHub.

------------------------------------------------------------------------

## 📌 Project in One Line

> **Data → Features → ML Models → Prediction Engine → FastAPI → MySQL,
> presented through Streamlit.**
