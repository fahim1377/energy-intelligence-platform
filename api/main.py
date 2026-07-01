import os
from collections.abc import Iterator
from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated, Any, Literal

import psycopg
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from psycopg.rows import dict_row
from pydantic import BaseModel


load_dotenv()


class HealthResponse(BaseModel):
    status: Literal["ok"]


class EnergyPriceResponse(BaseModel):
    source: str
    country_code: str
    timestamp_utc: datetime
    price_eur_per_mwh: float | None


def get_database_connection() -> Iterator[psycopg.Connection[Any]]:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise HTTPException(status_code=503, detail="Database is not configured")

    try:
        with psycopg.connect(database_url) as connection:
            yield connection
    except psycopg.Error as error:
        raise HTTPException(status_code=503, detail="Database is unavailable") from error


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
        WHERE {' AND '.join(conditions)}
        ORDER BY timestamp_utc
    """

    with connection.cursor(row_factory=dict_row) as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()

    return [EnergyPriceResponse.model_validate(row) for row in rows]
