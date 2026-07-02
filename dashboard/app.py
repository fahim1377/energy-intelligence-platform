from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv

from ml.features import TARGET_COLUMN, prepare_hourly_prices
from ml.predict import forecast_prices
from ml.train_price_forecast import DEFAULT_MODEL_PATH


load_dotenv()


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        st.error("DATABASE_URL is missing. Create a .env file from .env.example.")
        st.stop()
    return database_url


@st.cache_data(ttl=300)
def load_energy_prices(database_url: str) -> pd.DataFrame:
    query = """
        SELECT
            timestamp_utc,
            country_code,
            price_eur_per_mwh
        FROM raw.energy_prices
        ORDER BY timestamp_utc
    """
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


@st.cache_resource
def load_forecast_artifact(model_path: str) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    required_keys = {"model", "feature_columns", "country_code", "trained_at_utc", "metrics"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError("Forecast model artifact is invalid")
    return artifact


def filter_prices(
    df: pd.DataFrame, selected_country: str, start_date: date, end_date: date
) -> pd.DataFrame:
    filtered = df[df["country_code"] == selected_country].copy()
    filtered["date"] = filtered["timestamp_utc"].dt.date
    return filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]


def build_forecast_view(
    prices: pd.DataFrame,
    artifact: dict[str, Any],
    *,
    country_code: str,
    hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if artifact["country_code"] != country_code:
        raise ValueError(f"Model is trained for {artifact['country_code']}, not {country_code}")

    country_prices = prices.loc[
        prices["country_code"] == country_code,
        ["timestamp_utc", TARGET_COLUMN],
    ]
    predictions = forecast_prices(artifact, country_prices, hours=hours)
    forecast = pd.DataFrame(
        predictions,
        columns=["timestamp_utc", "Forecast EUR/MWh"],
    ).set_index("timestamp_utc")

    history = (
        prepare_hourly_prices(country_prices)
        .tail(168)
        .rename(columns={TARGET_COLUMN: "Historical EUR/MWh"})
    )
    bridge_timestamp = history.index[-1]
    forecast_with_bridge = pd.concat(
        [
            pd.DataFrame(
                {"Forecast EUR/MWh": [history.iloc[-1]["Historical EUR/MWh"]]},
                index=[bridge_timestamp],
            ),
            forecast,
        ]
    )
    chart = history.join(forecast_with_bridge, how="outer")
    return forecast, chart


def main() -> None:
    st.set_page_config(page_title="Energy Intelligence", layout="wide")
    st.title("Energy Intelligence Platform")
    st.caption("Day-ahead electricity prices from Energy-Charts")

    database_url = get_database_url()
    df = load_energy_prices(database_url)

    if df.empty:
        st.warning("No price records found in raw.energy_prices.")
        st.stop()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["price_eur_per_mwh"] = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")

    min_date = df["timestamp_utc"].dt.date.min()
    max_date = df["timestamp_utc"].dt.date.max()
    countries = sorted(df["country_code"].dropna().unique())

    with st.sidebar:
        st.header("Filters")
        selected_country = st.selectbox("Country", countries)
        selected_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        forecast_hours = st.slider("Forecast horizon", min_value=1, max_value=168, value=24)

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range

    filtered = filter_prices(df, selected_country, start_date, end_date)
    if filtered.empty:
        st.warning("No records found for the selected filters.")
        st.stop()

    avg_price = filtered["price_eur_per_mwh"].mean()
    min_price = filtered["price_eur_per_mwh"].min()
    max_price = filtered["price_eur_per_mwh"].max()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Records", f"{len(filtered):,}")
    metric_cols[1].metric("Average price", f"{avg_price:.2f} EUR/MWh")
    metric_cols[2].metric("Lowest price", f"{min_price:.2f} EUR/MWh")
    metric_cols[3].metric("Highest price", f"{max_price:.2f} EUR/MWh")

    chart_data = filtered.set_index("timestamp_utc")["price_eur_per_mwh"]
    st.line_chart(chart_data, height=420, y_label="EUR/MWh")

    st.subheader("Price forecast")
    model_path = os.getenv("FORECAST_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    try:
        artifact = load_forecast_artifact(str(Path(model_path)))
        forecast, forecast_chart = build_forecast_view(
            df,
            artifact,
            country_code=selected_country,
            hours=forecast_hours,
        )
    except (FileNotFoundError, OSError, KeyError, ValueError) as error:
        st.info(str(error))
    else:
        metrics = artifact["metrics"]
        forecast_metric_cols = st.columns(4)
        forecast_metric_cols[0].metric("Forecast hours", forecast_hours)
        forecast_metric_cols[1].metric("Model MAE (EUR/MWh)", f"{metrics['model_mae']:.2f}")
        forecast_metric_cols[2].metric("Model RMSE (EUR/MWh)", f"{metrics['model_rmse']:.2f}")
        forecast_metric_cols[3].metric(
            "Baseline MAE (EUR/MWh)",
            f"{metrics['baseline_mae']:.2f}",
        )
        st.line_chart(forecast_chart, height=360, y_label="EUR/MWh")
        with st.expander("Forecast values"):
            st.dataframe(forecast.reset_index(), width="stretch", hide_index=True)

    daily_prices = (
        filtered.groupby("date", as_index=False)["price_eur_per_mwh"]
        .mean()
        .rename(columns={"date": "Date", "price_eur_per_mwh": "Average EUR/MWh"})
    )

    left_col, right_col = st.columns([1, 2])
    with left_col:
        st.subheader("Daily average")
        st.dataframe(daily_prices, width="stretch", hide_index=True)
    with right_col:
        st.subheader("Raw records")
        st.dataframe(filtered.drop(columns=["date"]), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
