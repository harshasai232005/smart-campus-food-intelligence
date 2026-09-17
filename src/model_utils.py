import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def create_preprocessor(
    categorical_columns,
    numeric_columns
):

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns
            )
        ]
    )


def calculate_regression_metrics(
    actual,
    predicted
):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mae = np.mean(
        np.abs(actual - predicted)
    )

    rmse = np.sqrt(
        np.mean(
            (actual - predicted) ** 2
        )
    )

    denominator = np.where(
        np.abs(actual) < 1,
        1,
        np.abs(actual)
    )

    mape = np.mean(
        np.abs(
            (actual - predicted)
            / denominator
        )
    ) * 100

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = (
        1 - (ss_res / ss_tot)
        if ss_tot != 0
        else 0
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2)
    }