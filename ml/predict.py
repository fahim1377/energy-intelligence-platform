from datetime import datetime
from typing import Any

import pandas as pd

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, prepare_hourly_prices


def forecast_prices(
    artifact: dict[str, Any],
    prices: pd.DataFrame,
    *,
    hours: int,
) -> list[tuple[datetime, float]]:
    if not 1 <= hours <= 168:
        raise ValueError("hours must be between 1 and 168")

    hourly = prepare_hourly_prices(prices)
    history = hourly[TARGET_COLUMN]
    if len(history) < 168 or history.iloc[-168:].isna().any():
        raise ValueError("At least 168 continuous hourly prices are required")

    model = artifact["model"]
    feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    predictions: list[tuple[datetime, float]] = []

    for _ in range(hours):
        timestamp = history.index[-1] + pd.Timedelta(hours=1)
        recent_day = history.iloc[-24:]
        feature_values = {
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "month": timestamp.month,
            "is_weekend": int(timestamp.dayofweek >= 5),
            "lag_1h": history.iloc[-1],
            "lag_24h": history.iloc[-24],
            "lag_168h": history.iloc[-168],
            "rolling_mean_24h": recent_day.mean(),
            "rolling_std_24h": recent_day.std(),
        }
        feature_frame = pd.DataFrame([feature_values], columns=feature_columns)
        predicted_price = float(model.predict(feature_frame)[0])
        history.loc[timestamp] = predicted_price
        predictions.append((timestamp.to_pydatetime(), predicted_price))

    return predictions
