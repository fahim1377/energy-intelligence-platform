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

## MVP Roadmap

- Set up project structure
- Run PostgreSQL with Docker Compose
- Create initial warehouse schema
- Ingest first electricity market dataset
- Build first Streamlit dashboard
- Add FastAPI endpoints
- Add forecasting model
- Deploy dashboard and API

