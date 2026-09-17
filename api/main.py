from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predictor import (
    make_prediction,
    save_prediction_to_db
)


app = FastAPI(
    title="Smart Campus Food Intelligence API",
    description=(
        "API for campus meal demand forecasting, "
        "food waste prediction and preparation recommendations."
    ),
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class PredictionInput(BaseModel):

    meal_date: date

    mess_name: str

    meal_type: str

    menu_name: str

    hostel_occupancy_pct: float = Field(
        ge=0,
        le=100
    )

    temperature_c: float

    rainfall_mm: float = Field(
        ge=0
    )

    humidity_pct: float = Field(
        ge=0,
        le=100
    )

    exam_day: int = Field(
        ge=0,
        le=1
    )

    holiday: int = Field(
        ge=0,
        le=1
    )

    special_event: int = Field(
        ge=0,
        le=1
    )

    students_expected: int = Field(
        ge=0
    )


@app.get("/")
def home():

    return {
        "message":
            "Smart Campus Food Intelligence API",
        "status":
            "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
    payload: PredictionInput
):

    payload_dict = (
        payload.model_dump()
    )

    result = make_prediction(
        payload_dict
    )

    database_saved = False

    database_error = None

    try:

        save_prediction_to_db(
            result
        )

        database_saved = True

    except Exception as error:

        database_error = str(error)

    result[
        "database_saved"
    ] = database_saved

    result[
        "database_error"
    ] = database_error

    return result


@app.post("/batch-predict")
def batch_predict(
    payloads: list[PredictionInput]
):

    results = []

    for payload in payloads:

        payload_dict = (
            payload.model_dump()
        )

        result = make_prediction(
            payload_dict
        )

        try:

            save_prediction_to_db(
                result
            )

            result[
                "database_saved"
            ] = True

        except Exception as error:

            result[
                "database_saved"
            ] = False

            result[
                "database_error"
            ] = str(error)

        results.append(result)

    return {
        "count": len(results),
        "results": results
    }