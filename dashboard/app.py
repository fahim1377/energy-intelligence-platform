from __future__ import annotations

import os
from datetime import date

import pandas as pd
import psycopg
import streamlit as st
from dotenv import load_dotenv


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


def filter_prices(df: pd.DataFrame, selected_country: str, start_date: date, end_date: date) -> pd.DataFrame:
    filtered = df[df["country_code"] == selected_country].copy()
    filtered["date"] = filtered["timestamp_utc"].dt.date
    return filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]


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
