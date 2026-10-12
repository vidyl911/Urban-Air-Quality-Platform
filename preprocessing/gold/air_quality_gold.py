
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "air_quality"
GOLD_DIR = PROJECT_ROOT / "data" / "gold" / "air_quality"

MEASUREMENTS_FILE = SILVER_DIR / "measurements.parquet"
STATIONS_FILE = SILVER_DIR / "stations.parquet"

OUTPUT_DIR = GOLD_DIR / "daily_features"

POLLUTANTS = [
    "NO2",
    "PM10",
    "PM2.5",
    "C6H6",
    "O3",
    "CO_8H",
    "SO2",
]


# --------------------------------------------------
# Main transformation
# --------------------------------------------------

def main() -> None:
    for path in [MEASUREMENTS_FILE, STATIONS_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Silver input: {path}"
            )

    spark = (
        SparkSession.builder
        .appName("UrbanAirQuality-Gold")
        .getOrCreate()
    )

    try:
        print("Reading Silver air-quality datasets...")

        measurements = spark.read.parquet(
            str(MEASUREMENTS_FILE)
        )

        stations = spark.read.parquet(
            str(STATIONS_FILE)
        )

        # Keep only the columns needed for the pivot.
        measurements = measurements.select(
            "station_id",
            "date",
            "pollutant",
            "concentration",
        )

        # Standardize data types and labels.
        measurements = (
            measurements
            .withColumn(
                "station_id",
                F.trim(F.col("station_id").cast("string")),
            )
            .withColumn(
                "date",
                F.to_date("date"),
            )
            .withColumn(
                "pollutant",
                F.upper(F.trim(F.col("pollutant"))),
            )
            .withColumn(
                "concentration",
                F.col("concentration").cast("double"),
            )
        )

        # Pivot from long format to one row per station and date.
        air_quality = (
            measurements
            .groupBy("station_id", "date")
            .pivot("pollutant", POLLUTANTS)
            .agg(F.first("concentration", ignorenulls=True))
        )

        # Rename pollutant columns to model-friendly feature names.
        for pollutant in POLLUTANTS:
            air_quality = air_quality.withColumnRenamed(
                pollutant,
                pollutant.lower()
                .replace(".", "_")
                + "_concentration",
            )

        feature_columns = [
            pollutant.lower().replace(".", "_")
            + "_concentration"
            for pollutant in POLLUTANTS
        ]

        # Count the available pollutant values per station and date.
        available_count = F.lit(0)

        for column in feature_columns:
            available_count += F.when(
                F.col(column).isNotNull(), 1
            ).otherwise(0)

        air_quality = air_quality.withColumn(
            "pollutant_count",
            available_count,
        )

        # Select station metadata and standardize the join key.
        station_metadata = stations.select(
            F.trim(F.col("station_id").cast("string")).alias(
                "station_id"
            ),
            "station_name",
            "arpa_station_id",
            "longitude",
            "latitude",
        )

        # Add metadata without duplicating measurement rows.
        gold = air_quality.join(
            station_metadata,
            on="station_id",
            how="left",
        )

        # Arrange the output columns.
        gold = gold.select(
            "station_id",
            "station_name",
            "arpa_station_id",
            "date",
            "longitude",
            "latitude",
            *feature_columns,
            "pollutant_count",
        ).orderBy("date", "station_id")

        # Materialize and inspect the final dataset.
        print("\nGold air-quality inspection")
        print(f"Rows: {gold.count():,}")
        print(f"Distinct stations: {gold.select('station_id').distinct().count():,}")
        print(f"Distinct dates: {gold.select('date').distinct().count():,}")

        print("\nSample Gold records:")
        gold.show(10, truncate=False)

        # Write the Gold dataset.
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        gold.write.mode("overwrite").parquet(
            str(OUTPUT_DIR)
        )

        print("\nGold air-quality transformation complete.")
        print(f"Output: {OUTPUT_DIR}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()