from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app, get_database_connection, get_forecast_artifact


@pytest.fixture
def database_connection() -> MagicMock:
    connection = MagicMock()
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.fetchall.return_value = [
        {
            "source": "energy-charts",
            "country_code": "DE",
            "timestamp_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "price_eur_per_mwh": Decimal("42.50"),
        }
    ]
    return connection


@pytest.fixture
def client(database_connection: MagicMock) -> Iterator[TestClient]:
    def override_database_connection() -> Iterator[MagicMock]:
        yield database_connection

    app.dependency_overrides[get_database_connection] = override_database_connection
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_check(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_prices_filters_by_country_and_date_range(
    client: TestClient,
    database_connection: MagicMock,
) -> None:
    response = client.get(
        "/prices",
        params={"country_code": "de", "start": "2026-01-01", "end": "2026-01-31"},
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "source": "energy-charts",
            "country_code": "DE",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "price_eur_per_mwh": 42.5,
        }
    ]

    cursor = database_connection.cursor.return_value.__enter__.return_value
    parameters = cursor.execute.call_args.args[1]
    assert parameters == [
        "DE",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    ]


def test_list_prices_rejects_reversed_date_range(client: TestClient) -> None:
    response = client.get(
        "/prices",
        params={"start": "2026-02-01", "end": "2026-01-01"},
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "start must be on or before end"}


def test_get_price_forecast(
    client: TestClient,
    database_connection: MagicMock,
) -> None:
    model = MagicMock()
    model.predict.return_value = [55.25]
    artifact = {
        "model": model,
        "feature_columns": [
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "lag_1h",
            "lag_24h",
            "lag_168h",
            "rolling_mean_24h",
            "rolling_std_24h",
        ],
        "country_code": "DE",
        "trained_at_utc": "2026-07-02T10:00:00+00:00",
    }
    app.dependency_overrides[get_forecast_artifact] = lambda: artifact

    cursor = database_connection.cursor.return_value.__enter__.return_value
    timestamps = [
        datetime(2026, 1, 8, 23, tzinfo=timezone.utc) - timedelta(hours=offset)
        for offset in range(192)
    ]
    cursor.fetchall.return_value = [(timestamp, Decimal("50.00")) for timestamp in timestamps]

    response = client.get("/forecast", params={"country_code": "de", "hours": 2})

    assert response.status_code == 200
    assert response.json() == {
        "country_code": "DE",
        "model_trained_at_utc": "2026-07-02T10:00:00Z",
        "predictions": [
            {
                "timestamp_utc": "2026-01-09T00:00:00Z",
                "predicted_price_eur_per_mwh": 55.25,
            },
            {
                "timestamp_utc": "2026-01-09T01:00:00Z",
                "predicted_price_eur_per_mwh": 55.25,
            },
        ],
    }


def test_get_price_forecast_rejects_unsupported_country(client: TestClient) -> None:
    app.dependency_overrides[get_forecast_artifact] = lambda: {
        "country_code": "DE",
        "trained_at_utc": "2026-07-02T10:00:00+00:00",
    }

    response = client.get("/forecast", params={"country_code": "FR"})

    assert response.status_code == 422
    assert response.json() == {"detail": "Model is trained for country_code=DE"}
