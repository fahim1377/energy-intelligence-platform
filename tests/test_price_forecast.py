from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_COLUMNS, build_hourly_features
from ml.train_price_forecast import save_model, train_model


def sample_prices(days: int = 45) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=days * 24, freq="1h", tz="UTC")
    hourly_pattern = np.sin(2 * np.pi * timestamps.hour.to_numpy() / 24) * 20
    weekly_pattern = timestamps.dayofweek.to_numpy() * 1.5
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "price_eur_per_mwh": 60 + hourly_pattern + weekly_pattern,
        }
    )


def test_build_hourly_features_uses_only_previous_prices() -> None:
    prices = sample_prices()

    features = build_hourly_features(prices)

    assert list(features.columns) == ["price_eur_per_mwh", *FEATURE_COLUMNS]
    assert features.index.min() == prices["timestamp_utc"].iloc[168]
    assert features.iloc[0]["lag_168h"] == pytest.approx(prices["price_eur_per_mwh"].iloc[0])


def test_train_and_save_model(tmp_path: Path) -> None:
    model, metrics = train_model(sample_prices())
    model_path = tmp_path / "forecast.joblib"

    save_model(model, metrics, country_code="de", model_path=model_path)
    artifact = joblib.load(model_path)

    assert metrics.train_rows > metrics.test_rows > 0
    assert metrics.model_mae >= 0
    assert artifact["country_code"] == "DE"
    assert artifact["feature_columns"] == FEATURE_COLUMNS


def test_train_model_rejects_too_little_data() -> None:
    with pytest.raises(ValueError, match="Not enough data"):
        train_model(sample_prices(days=9))
