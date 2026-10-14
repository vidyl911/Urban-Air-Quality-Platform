"""Load Gold Parquet datasets into PostgreSQL without breaking dbt views."""

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import psycopg2
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "gold_air_quality": ROOT / "data/gold/air_quality/daily_features",
    "gold_weather": ROOT / "data/gold/weather/daily_features",
    "station_landcover": (
        ROOT / "data/gold/spatial_context/station_landcover"
        / "station_landcover.parquet"
    ),
}

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "urban_air_quality",
    "user": "postgres",
}

SCHEMA = "raw_gold"
BATCH_SIZE = 1000


def get_postgres_type(arrow_type, column_name: str) -> str:
    """Map supported PyArrow types to PostgreSQL types."""
    if pa.types.is_integer(arrow_type):
        return "BIGINT"
    if pa.types.is_floating(arrow_type):
        return "DOUBLE PRECISION"
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP"
    if pa.types.is_date(arrow_type):
        return "DATE"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "TEXT"

    raise TypeError(
        f"Unsupported Arrow type for column {column_name}: {arrow_type}"
    )


def load_dataset(cursor, table_name: str, path: Path) -> None:
    """Create a table if needed, then refresh its rows transactionally."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = ds.dataset(str(path), format="parquet")
    arrow_schema = dataset.schema
    columns = arrow_schema.names

    if not columns:
        raise ValueError(f"No columns found in dataset: {path}")

    pg_types = [
        (field.name, get_postgres_type(field.type, field.name))
        for field in arrow_schema
    ]

    table = sql.Identifier(SCHEMA, table_name)

    # Check whether the target table already exists.
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name = %s
        )
        """,
        (SCHEMA, table_name),
    )
    table_exists = cursor.fetchone()[0]

    if table_exists:
        # Keep the existing table object so dependent dbt views remain valid.
        cursor.execute(
            sql.SQL("TRUNCATE TABLE {}").format(table)
        )

        # Fail clearly if the existing schema differs from the Parquet schema.
        cursor.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
            """,
            (SCHEMA, table_name),
        )
        existing_columns = cursor.fetchall()

        expected_columns = [name for name, _ in pg_types]
        actual_columns = [name for name, _ in existing_columns]

        if actual_columns != expected_columns:
            raise ValueError(
                f"Schema mismatch for {SCHEMA}.{table_name}. "
                f"PostgreSQL columns: {actual_columns}; "
                f"Parquet columns: {expected_columns}. "
                "No table was dropped. Resolve the schema difference "
                "before reloading."
            )
    else:
        definitions = [
            sql.SQL("{} {}").format(
                sql.Identifier(name), sql.SQL(pg_type)
            )
            for name, pg_type in pg_types
        ]
        cursor.execute(
            sql.SQL("CREATE TABLE {} ({})").format(
                table,
                sql.SQL(", ").join(definitions),
            )
        )

    insert_query = sql.SQL(
        "INSERT INTO {} ({}) VALUES ({})"
    ).format(
        table,
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
    password = os.environ.get("URBAN_AQ_DB_PASSWORD")
    if not password:
        raise RuntimeError(
            "Set URBAN_AQ_DB_PASSWORD in PowerShell before running."
        )

    DB_CONFIG["password"] = password

    # The connection context commits all three loads together.
    # If any load fails, the transaction rolls back.
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