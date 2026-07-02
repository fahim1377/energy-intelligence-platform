from unittest.mock import MagicMock

import pandas as pd
import pytest

from dashboard.app import build_forecast_view
from ml.features import FEATURE_COLUMNS


def sample_dashboard_prices(days: int = 9) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=days * 24, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "country_code": "DE",
            "price_eur_per_mwh": 50.0,
        }
    )


def test_build_forecast_view_combines_history_and_predictions() -> None:
    model = MagicMock()
    model.predict.return_value = [55.0]
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "country_code": "DE",
    }

    forecast, chart = build_forecast_view(
        sample_dashboard_prices(),
        artifact,
        country_code="DE",
        hours=24,
    )

    assert len(forecast) == 24
    assert len(chart) == 168 + 24
    assert chart["Historical EUR/MWh"].notna().sum() == 168
    assert chart["Forecast EUR/MWh"].notna().sum() == 25


def test_build_forecast_view_rejects_country_without_model() -> None:
    with pytest.raises(ValueError, match="Model is trained for DE, not FR"):
        build_forecast_view(
            sample_dashboard_prices(),
            {"country_code": "DE"},
            country_code="FR",
            hours=24,
        )
