from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import httpx
import psycopg
from dotenv import load_dotenv


LOGGER = logging.getLogger(__name__)
SOURCE = "energy_charts"
DEFAULT_BASE_URL = "https://api.energy-charts.info"
DEFAULT_BIDDING_ZONE = "DE-LU"
DEFAULT_COUNTRY_CODE = "DE"


@dataclass(frozen=True)
class EnergyPrice:
    source: str
    country_code: str
    timestamp_utc: datetime
    price_eur_per_mwh: Decimal | None


def fetch_price_payload(
    *,
    base_url: str,
    bidding_zone: str,
    start: str | None = None,
    end: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    params = {"bzn": bidding_zone}
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    url = f"{base_url.rstrip('/')}/price"
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def parse_price_payload(
    payload: dict[str, Any],
    *,
    country_code: str = DEFAULT_COUNTRY_CODE,
    source: str = SOURCE,
) -> list[EnergyPrice]:
    timestamps = first_present(payload, "unix_seconds", "time", "timestamp")
    prices = first_present(payload, "price", "prices")

    if not isinstance(timestamps, list) or not isinstance(prices, list):
        raise ValueError("Energy-Charts response must contain timestamp and price arrays.")
    if len(timestamps) != len(prices):
        raise ValueError("Energy-Charts timestamp and price arrays must have the same length.")

    records: list[EnergyPrice] = []
    for timestamp, price in zip(timestamps, prices, strict=True):
        records.append(
            EnergyPrice(
                source=source,
                country_code=country_code,
                timestamp_utc=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
                price_eur_per_mwh=Decimal(str(price)) if price is not None else None,
            )
        )
    return records


def first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def upsert_prices(database_url: str, records: list[EnergyPrice]) -> int:
    if not records:
        return 0

    query = """
        INSERT INTO raw.energy_prices (
            source,
            country_code,
            timestamp_utc,
            price_eur_per_mwh
        )
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source, country_code, timestamp_utc)
        DO UPDATE SET
            price_eur_per_mwh = EXCLUDED.price_eur_per_mwh
    """
    values = [
        (
            record.source,
            record.country_code,
            record.timestamp_utc,
            record.price_eur_per_mwh,
        )
        for record in records
    ]

    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.executemany(query, values)
        connection.commit()

    return len(records)


def run_ingestion(
    *,
    database_url: str,
    base_url: str,
    bidding_zone: str,
    country_code: str,
    start: str | None,
    end: str | None,
) -> int:
    payload = fetch_price_payload(
        base_url=base_url,
        bidding_zone=bidding_zone,
        start=start,
        end=end,
    )
    records = parse_price_payload(payload, country_code=country_code)
    inserted_count = upsert_prices(database_url, records)
    LOGGER.info("Upserted %s Energy-Charts price records.", inserted_count)
    return inserted_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Energy-Charts day-ahead electricity prices.")
    parser.add_argument("--bidding-zone", default=os.getenv("ENERGY_CHARTS_BIDDING_ZONE", DEFAULT_BIDDING_ZONE))
    parser.add_argument("--country-code", default=os.getenv("ENERGY_COUNTRY_CODE", DEFAULT_COUNTRY_CODE))
    parser.add_argument("--start", help="Optional start date, for example 2026-01-01.")
    parser.add_argument("--end", help="Optional end date, for example 2026-01-31.")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required. Create a .env file from .env.example.")

    run_ingestion(
        database_url=database_url,
        base_url=os.getenv("ENERGY_CHARTS_BASE_URL", DEFAULT_BASE_URL),
        bidding_zone=args.bidding_zone,
        country_code=args.country_code,
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()
