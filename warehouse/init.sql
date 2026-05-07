CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE IF NOT EXISTS raw.energy_prices (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    country_code TEXT NOT NULL,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    price_eur_per_mwh NUMERIC,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, country_code, timestamp_utc)
);

