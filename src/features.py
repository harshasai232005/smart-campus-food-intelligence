import pandas as pd


def add_calendar_features(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["day_of_week_num"] = (
        df["date"].dt.dayofweek
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week_num"] >= 5
    ).astype(int)

    def semester_from_month(month):

        if 1 <= month <= 5:
            return 4

        if 7 <= month <= 11:
            return 5

        return 0

    df["semester"] = (
        df["month"]
        .apply(semester_from_month)
    )

    return df


def add_demand_lags(df):

    df = df.copy()

    df = df.sort_values(
        [
            "mess_name",
            "meal_type",
            "date"
        ]
    )

    group = df.groupby(
        [
            "mess_name",
            "meal_type"
        ],
        group_keys=False
    )

    df["lag_1"] = (
        group["students_present"]
        .shift(1)
    )

    df["lag_7"] = (
        group["students_present"]
        .shift(7)
    )

    df["rolling_7_demand"] = (
        group["students_present"]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(7)
            .mean()
        )
    )

    return df