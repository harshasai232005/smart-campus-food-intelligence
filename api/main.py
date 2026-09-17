from datetime import date
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predictor import (
    make_prediction,
    save_prediction_to_db
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Smart Campus Food Intelligence API",
    description=(
        "API for campus meal demand forecasting, "
        "food waste prediction and preparation recommendations."
    ),
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# REQUEST MODEL
# =========================================================

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


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message":
            "Smart Campus Food Intelligence API",

        "status":
            "running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# =========================================================
# SINGLE PREDICTION
# =========================================================

@app.post("/predict")
def predict(
    payload: PredictionInput
):

    # -----------------------------------------------------
    # Convert Pydantic model to dictionary
    # -----------------------------------------------------

    payload_dict = (
        payload.model_dump()
    )


    # -----------------------------------------------------
    # RUN MACHINE LEARNING PREDICTION
    # -----------------------------------------------------

    try:

        result = make_prediction(
            payload_dict
        )

    except Exception as error:

        # Print the actual error into Vercel runtime logs
        print(
            "PREDICTION_ERROR:",
            repr(error),
            flush=True
        )

        # Print complete traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed on the server. "
                "Check Vercel runtime logs."
            )
        ) from error


    # -----------------------------------------------------
    # SAVE RESULT TO DATABASE
    # -----------------------------------------------------

    database_saved = False

    database_error = None

    try:

        save_prediction_to_db(
            result
        )

        database_saved = True

    except Exception as error:

        database_error = str(
            error
        )

        print(
            "DATABASE_SAVE_ERROR:",
            repr(error),
            flush=True
        )

        traceback.print_exc()


    # -----------------------------------------------------
    # ADD DATABASE STATUS TO RESPONSE
    # -----------------------------------------------------

    result[
        "database_saved"
    ] = database_saved

    result[
        "database_error"
    ] = database_error


    # -----------------------------------------------------
    # RETURN RESULT
    # -----------------------------------------------------

    return result


# =========================================================
# BATCH PREDICTION
# =========================================================

@app.post("/batch-predict")
def batch_predict(
    payloads: list[PredictionInput]
):

    results = []


    for payload in payloads:

        # -------------------------------------------------
        # Convert payload
        # -------------------------------------------------

        payload_dict = (
            payload.model_dump()
        )


        # -------------------------------------------------
        # Run prediction
        # -------------------------------------------------

        try:

            result = make_prediction(
                payload_dict
            )

        except Exception as error:

            print(
                "BATCH_PREDICTION_ERROR:",
                repr(error),
                flush=True
            )

            traceback.print_exc()

            raise HTTPException(
                status_code=500,
                detail=(
                    "Batch prediction failed. "
                    "Check Vercel runtime logs."
                )
            ) from error


        # -------------------------------------------------
        # Save prediction
        # -------------------------------------------------

        try:

            save_prediction_to_db(
                result
            )

            result[
                "database_saved"
            ] = True

            result[
                "database_error"
            ] = None

        except Exception as error:

            result[
                "database_saved"
            ] = False

            result[
                "database_error"
            ] = str(
                error
            )

            print(
                "BATCH_DATABASE_SAVE_ERROR:",
                repr(error),
                flush=True
            )

            traceback.print_exc()


        # -------------------------------------------------
        # Add result
        # -------------------------------------------------

        results.append(
            result
        )


    # -----------------------------------------------------
    # Return all results
    # -----------------------------------------------------

    return {
        "count": len(results),
        "results": results
    }