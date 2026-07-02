import os
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from psycopg.rows import dict_row
from pydantic import BaseModel

from ml.predict import forecast_prices
from ml.train_price_forecast import DEFAULT_MODEL_PATH


load_dotenv()


class HealthResponse(BaseModel):
    status: Literal["ok"]


class EnergyPriceResponse(BaseModel):
    source: str
    country_code: str
    timestamp_utc: datetime
    price_eur_per_mwh: float | None


class ForecastPoint(BaseModel):
    timestamp_utc: datetime
    predicted_price_eur_per_mwh: float


class ForecastResponse(BaseModel):
    country_code: str
    model_trained_at_utc: datetime
    predictions: list[ForecastPoint]


def get_database_connection() -> Iterator[psycopg.Connection[Any]]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="Database is not configured")

    try:
        with psycopg.connect(database_url) as connection:
            yield connection
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Database is unavailable") from error


@lru_cache
def get_forecast_artifact() -> dict[str, Any]:
    model_path = Path(os.getenv("FORECAST_MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        artifact = joblib.load(model_path)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Forecast model is unavailable") from error

    required_keys = {"model", "feature_columns", "country_code", "trained_at_utc"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise HTTPException(status_code=503, detail="Forecast model is invalid")
    return artifact


app = FastAPI(
    title="Energy Intelligence Platform API",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/prices", response_model=list[EnergyPriceResponse], tags=["prices"])
def list_prices(
    country_code: Annotated[str, Query(min_length=2, max_length=10)] = "DE",
    start: date | None = None,
    end: date | None = None,
    connection: psycopg.Connection[Any] = Depends(get_database_connection),
) -> list[EnergyPriceResponse]:
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")

    conditions = ["country_code = %s"]
    parameters: list[object] = [country_code.upper()]

    if start:
        conditions.append("timestamp_utc >= %s")
        parameters.append(datetime.combine(start, time.min, tzinfo=timezone.utc))
    if end:
        conditions.append("timestamp_utc < %s")
        parameters.append(datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc))

    query = f"""
        SELECT source, country_code, timestamp_utc, price_eur_per_mwh
        FROM raw.energy_prices
        WHERE {" AND ".join(conditions)}
        ORDER BY timestamp_utc
    """

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()

    return [EnergyPriceResponse.model_validate(row) for row in rows]


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def get_price_forecast(
    connection: Annotated[
        psycopg.Connection[Any],
        Depends(get_database_connection),
    ],
    country_code: Annotated[str, Query(min_length=2, max_length=10)] = "DE",
    hours: Annotated[int, Query(ge=1, le=168)] = 24,
    artifact: dict[str, Any] = Depends(get_forecast_artifact),
) -> ForecastResponse:
    normalized_country_code = country_code.upper()
    if artifact["country_code"] != normalized_country_code:
        raise HTTPException(
            status_code=422,
            detail=f"Model is trained for country_code={artifact['country_code']}",
        )

    query = """
        SELECT timestamp_utc, price_eur_per_mwh
        FROM raw.energy_prices
        WHERE country_code = %s
          AND price_eur_per_mwh IS NOT NULL
        ORDER BY timestamp_utc DESC
        LIMIT 4032
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (normalized_country_code,))
        rows = cursor.fetchall()

    prices = pd.DataFrame(rows, columns=["timestamp_utc", "price_eur_per_mwh"])
    try:
        predictions = forecast_prices(artifact, prices, hours=hours)
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ForecastResponse(
        country_code=normalized_country_code,
        model_trained_at_utc=artifact["trained_at_utc"],
        predictions=[
            ForecastPoint(timestamp_utc=timestamp, predicted_price_eur_per_mwh=price)
            for timestamp, price in predictions
        ],
    )
