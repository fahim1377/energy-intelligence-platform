import pandas as pd


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
]
TARGET_COLUMN = "price_eur_per_mwh"


def build_hourly_features(prices: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"timestamp_utc", TARGET_COLUMN}
    missing_columns = required_columns.difference(prices.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    hourly = prices.loc[:, ["timestamp_utc", TARGET_COLUMN]].copy()
    hourly["timestamp_utc"] = pd.to_datetime(hourly["timestamp_utc"], utc=True)
    hourly[TARGET_COLUMN] = pd.to_numeric(hourly[TARGET_COLUMN], errors="coerce")
    hourly = (
        hourly.dropna()
        .drop_duplicates(subset="timestamp_utc", keep="last")
        .set_index("timestamp_utc")
        .sort_index()
        .resample("1h")
        .mean()
    )

    features = hourly.copy()
    features["hour"] = features.index.hour
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["is_weekend"] = (features.index.dayofweek >= 5).astype(int)
    features["lag_1h"] = features[TARGET_COLUMN].shift(1)
    features["lag_24h"] = features[TARGET_COLUMN].shift(24)
    features["lag_168h"] = features[TARGET_COLUMN].shift(168)
    features["rolling_mean_24h"] = features[TARGET_COLUMN].shift(1).rolling(24).mean()
    features["rolling_std_24h"] = features[TARGET_COLUMN].shift(1).rolling(24).std()

    return features.dropna(subset=[TARGET_COLUMN, *FEATURE_COLUMNS])
