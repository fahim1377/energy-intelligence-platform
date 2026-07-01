# Energy Intelligence Platform

End-to-end European energy data platform for collecting, storing, transforming, visualizing, and forecasting electricity market data.

## Business Question

How do electricity prices, demand, renewable generation, and weather patterns interact across European energy markets?

The platform is designed to answer questions such as:

- How do electricity prices change over time?
- Which periods show high demand or price spikes?
- How much renewable generation contributes to the electricity mix?
- Can short-term electricity prices or demand be forecasted?

## Planned Architecture

```text
Public Energy APIs
        |
        v
Python Ingestion Jobs
        |
        v
PostgreSQL Raw Storage
        |
        v
Transformations
        |
        v
Analytics Tables
        |
        +--> FastAPI
        |
        +--> Streamlit Dashboard
        |
        +--> ML Forecasting
```

## Tech Stack

- Python
- SQL
- PostgreSQL
- Docker Compose
- FastAPI
- Streamlit
- Pandas
- Pytest

## Project Structure

```text
energy-intelligence-platform/
├── ingestion/
├── transformations/
├── warehouse/
├── dashboard/
├── ml/
├── api/
├── tests/
├── docs/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
└── pyproject.toml
```

## Local Setup

Install Python dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

Start PostgreSQL:

```bash
docker compose up -d postgres
```

Check the database container:

```bash
docker compose ps
```

Stop services:

```bash
docker compose down
```

## First Ingestion

Create a local environment file:

```bash
cp .env.example .env
```

Ingest German day-ahead electricity prices from Energy-Charts:

```bash
python3 -m ingestion.energy_charts_prices --start 2026-01-01 --end 2026-01-31
```

The job writes into `raw.energy_prices` and uses an upsert on `(source, country_code, timestamp_utc)`,
so repeated runs update existing rows instead of creating duplicates.

## Dashboard

Start the first Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard reads from `raw.energy_prices` using `DATABASE_URL` from your `.env` file.

## API

Start the FastAPI development server:

```bash
uvicorn api.main:app --reload
```

Check the service health at `http://127.0.0.1:8000/health`. Interactive API documentation is
available at `http://127.0.0.1:8000/docs`.

Query stored electricity prices by country and an optional inclusive date range:

```bash
curl "http://127.0.0.1:8000/prices?country_code=DE&start=2026-01-01&end=2026-01-31"
```

## MVP Roadmap

- [x] Set up project structure
- [x] Run PostgreSQL with Docker Compose
- [x] Create initial warehouse schema
- [x] Ingest first electricity market dataset
- [x] Build first Streamlit dashboard
- [x] Add FastAPI health endpoint
- Add forecasting model
- Deploy dashboard and API
