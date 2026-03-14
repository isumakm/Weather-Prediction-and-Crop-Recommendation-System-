import os
import sys
import numpy as np
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline_utils import get_sri_lanka_season, add_time_features, aggregate_to_seasonal


def test_get_sri_lanka_season():
    assert get_sri_lanka_season(1) == "North-east monsoon"
    assert get_sri_lanka_season(4) == "Intermonsoon after North-east monsoon"
    assert get_sri_lanka_season(6) == "South-west monsoon"
    assert get_sri_lanka_season(10) == "Intermonsoon after South-west monsoon"
    assert get_sri_lanka_season(99) == "Unknown"


def test_add_time_features():
    df = pd.DataFrame({
        "date": ["2026-01-15", "2026-06-20"]
    })

    result = add_time_features(df, "date")

    assert "year" in result.columns
    assert "month" in result.columns
    assert "day_of_year" in result.columns
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns
    assert "season" in result.columns

    assert result.loc[0, "season"] == "North-east monsoon"
    assert result.loc[1, "season"] == "South-west monsoon"


def test_aggregate_to_seasonal():
    df = pd.DataFrame({
        "year": [2026, 2026],
        "season": ["North-east monsoon", "North-east monsoon"],
        "location_id": ["1", "1"],
        "temperature": [28.0, 30.0],
        "rainfall": [100.0, 150.0],
        "sunshine_h": [6.0, 8.0],
        "month_sin": [0.5, 0.5],
        "month_cos": [0.8, 0.8]
    })

    result = aggregate_to_seasonal(
        df,
        location_col="location_id",
        temp_col="temperature",
        rain_col="rainfall",
        sun_col="sunshine_h"
    )

    assert len(result) == 1
    assert result.loc[0, "avg_temperature_C"] == 29.0
    assert result.loc[0, "total_rainfall_mm"] == 250.0
    assert result.loc[0, "avg_sunshine_h"] == 7.0
    assert np.isclose(result.loc[0, "log_total_rainfall_mm"], np.log1p(250.0))


def test_add_time_features_invalid_date():
    df = pd.DataFrame({
        "date": ["invalid-date"]
    })

    result = add_time_features(df, "date")
    assert pd.isna(result.loc[0, "year"])