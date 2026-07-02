import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import psycopg
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ml.features import FEATURE_COLUMNS, TARGET_COLUMN, build_hourly_features


DEFAULT_MODEL_PATH = Path("models/electricity_price_forecast.joblib")


@dataclass(frozen=True)
class TrainingMetrics:
    train_rows: int
    test_rows: int
    model_mae: float
    model_rmse: float
    baseline_mae: float
    baseline_rmse: float


def load_prices(database_url: str, country_code: str) -> pd.DataFrame:
    query = """
        SELECT timestamp_utc, price_eur_per_mwh
        FROM raw.energy_prices
        WHERE country_code = %s
          AND price_eur_per_mwh IS NOT NULL
        ORDER BY timestamp_utc
    """
    with psycopg.connect(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, (country_code.upper(),))
            rows = cursor.fetchall()

    return pd.DataFrame(rows, columns=["timestamp_utc", TARGET_COLUMN])


def train_model(
    prices: pd.DataFrame,
    *,
    test_size: float = 0.2,
) -> tuple[HistGradientBoostingRegressor, TrainingMetrics]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset = build_hourly_features(prices)
    split_index = int(len(dataset) * (1 - test_size))
    if split_index < 48 or len(dataset) - split_index < 24:
        raise ValueError("Not enough data: at least 48 training rows and 24 test rows are required")

    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        random_state=42,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    predictions = model.predict(test[FEATURE_COLUMNS])
    baseline_predictions = test["lag_24h"]
    metrics = TrainingMetrics(
        train_rows=len(train),
        test_rows=len(test),
        model_mae=float(mean_absolute_error(test[TARGET_COLUMN], predictions)),
        model_rmse=float(mean_squared_error(test[TARGET_COLUMN], predictions) ** 0.5),
        baseline_mae=float(mean_absolute_error(test[TARGET_COLUMN], baseline_predictions)),
        baseline_rmse=float(mean_squared_error(test[TARGET_COLUMN], baseline_predictions) ** 0.5),
    )
    return model, metrics


def save_model(
    model: HistGradientBoostingRegressor,
    metrics: TrainingMetrics,
    *,
    country_code: str,
    model_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "country_code": country_code.upper(),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "metrics": asdict(metrics),
        },
        model_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an hourly electricity price forecast.")
    parser.add_argument("--country-code", default="DE")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")

    prices = load_prices(database_url, args.country_code)
    model, metrics = train_model(prices, test_size=args.test_size)
    save_model(model, metrics, country_code=args.country_code, model_path=args.model_path)

    print(json.dumps(asdict(metrics), indent=2))
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
