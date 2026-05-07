from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ingestion.energy_charts_prices import parse_price_payload


def test_parse_price_payload_normalizes_energy_charts_response() -> None:
    payload = {
        "unix_seconds": [1_704_067_200, 1_704_070_800],
        "price": [65.12, None],
        "unit": "EUR/MWh",
    }

    records = parse_price_payload(payload, country_code="DE")

    assert len(records) == 2
    assert records[0].source == "energy_charts"
    assert records[0].country_code == "DE"
    assert records[0].timestamp_utc == datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    assert records[0].price_eur_per_mwh == Decimal("65.12")
    assert records[1].price_eur_per_mwh is None


def test_parse_price_payload_rejects_mismatched_arrays() -> None:
    payload = {
        "unix_seconds": [1_704_067_200],
        "price": [65.12, 66.42],
    }

    with pytest.raises(ValueError, match="same length"):
        parse_price_payload(payload)
