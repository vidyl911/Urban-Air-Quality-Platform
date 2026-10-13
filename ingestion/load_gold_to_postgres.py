"""Load Gold air-quality and weather Parquet datasets into PostgreSQL."""

from pathlib import Path

import pyarrow.dataset as ds
import psycopg2
from psycopg2 import sql
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "gold_air_quality": ROOT / "data/gold/air_quality/daily_features",
    "gold_weather": ROOT / "data/gold/weather/daily_features",
}

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "urban_air_quality",
    "user": "postgres",
    # Set this environment variable in PowerShell before running.
    "password": None,
}

SCHEMA = "raw_gold"
BATCH_SIZE = 1000


def load_dataset(cursor, table_name: str, path: Path) -> None:
    """Replace a PostgreSQL table with a Parquet dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = ds.dataset(str(path), format="parquet")
    arrow_schema = dataset.schema
    columns = arrow_schema.names

    # Initial implementation supports the numeric, string, date and boolean
    # columns expected in these Gold datasets.
    pg_types = []

    for field in arrow_schema:
        arrow_type = field.type

        if pa.types.is_integer(arrow_type):
            pg_type = "BIGINT"
        elif pa.types.is_floating(arrow_type):
            pg_type = "DOUBLE PRECISION"
        elif pa.types.is_boolean(arrow_type):
            pg_type = "BOOLEAN"
        elif pa.types.is_timestamp(arrow_type):
            pg_type = "TIMESTAMP"
        elif pa.types.is_date(arrow_type):
            pg_type = "DATE"
        elif (
            pa.types.is_string(arrow_type)
            or pa.types.is_large_string(arrow_type)
        ):
            pg_type = "TEXT"
        else:
            raise TypeError(
                f"Unsupported Arrow type for column "
                f"{field.name}: {arrow_type}"
            )

        pg_types.append((field.name, pg_type))

    cursor.execute(
        sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
            sql.Identifier(SCHEMA), sql.Identifier(table_name)
        )
    )

    definitions = [
        sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(pg_type))
        for name, pg_type in pg_types
    ]
    cursor.execute(
        sql.SQL("CREATE TABLE {}.{} ({})").format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table_name),
            sql.SQL(", ").join(definitions),
        )
    )

    insert_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
        sql.Identifier(SCHEMA),
        sql.Identifier(table_name),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )

    total = 0
    for batch in dataset.to_batches(batch_size=BATCH_SIZE):
        rows = batch.to_pylist()
        cursor.executemany(
            insert_query.as_string(cursor.connection),
            [
                tuple(row.get(column) for column in columns)
                for row in rows
            ],
        )
        total += len(rows)

    print(f"{table_name}: loaded {total:,} rows")


def main() -> None:
    import os

    password = os.environ.get("URBAN_AQ_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "Set URBAN_AQ_DB_PASSWORD in PowerShell before running."
        )

    DB_CONFIG["password"] = password

    with psycopg2.connect(**DB_CONFIG) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(SCHEMA)
                )
            )
            for table_name, path in DATASETS.items():
                load_dataset(cursor, table_name, path)

    print("Gold datasets loaded successfully.")


if __name__ == "__main__":
    main()