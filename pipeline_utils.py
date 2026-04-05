import numpy as np
import pandas as pd


def get_sri_lanka_season(month):
    if month in [5, 6, 7, 8, 9]:
        return "South-west monsoon"
    elif month in [10, 11]:
        return "Intermonsoon after South-west monsoon"
    elif month in [12, 1, 2]:
        return "North-east monsoon"
    elif month in [3, 4]:
        return "Intermonsoon after North-east monsoon"
    return "Unknown"


def add_time_features(df, date_col):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day_of_year"] = df[date_col].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["season"] = df["month"].apply(get_sri_lanka_season)

    return df


def aggregate_to_seasonal(df, location_col, temp_col, rain_col, sun_col):
    seasonal_df = df.groupby(
        ["year", "season", location_col],
        as_index=False
    ).agg({
        temp_col: "mean",
        rain_col: "sum",
        sun_col: "mean",
        "month_sin": "mean",
        "month_cos": "mean"
    })

    seasonal_df = seasonal_df.rename(columns={
        temp_col: "avg_temperature_C",
        rain_col: "total_rainfall_mm",
        sun_col: "avg_sunshine_h"
    })

    seasonal_df["log_total_rainfall_mm"] = np.log1p(seasonal_df["total_rainfall_mm"])
    return seasonal_df