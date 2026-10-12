
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "silver" / "weather" / "weather.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "gold" / "weather" / "daily_features"

NUMERIC_COLUMNS = [
    "latitude",
    "longitude",
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "relative_humidity_2m_mean",
    "pressure_msl_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Silver weather dataset not found: {INPUT_PATH}"
        )

    spark = (
        SparkSession.builder
        .appName("UrbanAirQuality-Weather-Gold")
        .getOrCreate()
    )

    try:
        spark.sparkContext.setLogLevel("WARN")

        print("Reading Silver weather dataset...")

        weather = spark.read.parquet(str(INPUT_PATH))

        print("\nSilver weather inspection")
        print(f"Rows: {weather.count()}")
        print(f"Columns: {weather.columns}")

        # Validate the expected schema.
        required_columns = {"date", "latitude", "longitude", *NUMERIC_COLUMNS}
        missing_columns = required_columns - set(weather.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required weather columns: {sorted(missing_columns)}"
            )

        # Standardize dates and numeric columns.
        gold = weather.withColumn(
            "date",
            F.to_date(F.col("date").cast("string"))
        )

        for column in NUMERIC_COLUMNS:
            gold = gold.withColumn(
                column,
                F.col(column).cast("double")
            )

        # Remove rows without a valid date or grid-point coordinates.
        before_validation = gold.count()

        gold = gold.filter(
            F.col("date").isNotNull()
            & F.col("latitude").isNotNull()
            & F.col("longitude").isNotNull()
            & F.col("latitude").between(-90, 90)
            & F.col("longitude").between(-180, 180)
        )

        print(
            "Invalid date/coordinate rows removed:",
            before_validation - gold.count()
        )

        # Keep one observation per date and weather-grid point.
        before_deduplication = gold.count()

        gold = gold.dropDuplicates(["date", "latitude", "longitude"])

        print(
            "Duplicate grid-point observations removed:",
            before_deduplication - gold.count()
        )

        # Derive useful features for downstream analysis and modeling.
        gold = (
            gold
            .withColumn(
                "temperature_range",
                F.col("temperature_2m_max")
                - F.col("temperature_2m_min")
            )
            .withColumn(
                "wind_speed_range",
                F.col("wind_speed_10m_max")
                - F.col("wind_speed_10m_mean")
            )
            .withColumn("year", F.year("date"))
            .withColumn("month", F.month("date"))
            .withColumn("day_of_year", F.dayofyear("date"))
            .withColumn(
                "season",
                F.when(F.col("month").isin(12, 1, 2), "winter")
                .when(F.col("month").isin(3, 4, 5), "spring")
                .when(F.col("month").isin(6, 7, 8), "summer")
                .otherwise("autumn")
            )
        )

        # Retain the timezone metadata when available.
        output_columns = [
            "date",
            "latitude",
            "longitude",
            "timezone",
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "temperature_range",
            "relative_humidity_2m_mean",
            "pressure_msl_mean",
            "precipitation_sum",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "wind_speed_range",
            "wind_direction_10m_dominant",
            "year",
            "month",
            "day_of_year",
            "season",
        ]

        if "timezone" not in gold.columns:
            output_columns.remove("timezone")

        gold = gold.select(*output_columns)

        # Inspect output before writing.
        row_count = gold.count()
        grid_point_count = gold.select(
            "latitude", "longitude"
        ).distinct().count()

        date_stats = gold.agg(
            F.min("date").alias("min_date"),
            F.max("date").alias("max_date"),
        ).first()

        print("\nGold weather inspection")
        print(f"Rows: {row_count}")
        print(f"Distinct grid points: {grid_point_count}")
        print(f"Date range: {date_stats['min_date']} to {date_stats['max_date']}")
        print("\nSample Gold records:")

        gold.orderBy("date", "latitude", "longitude").show(10, truncate=False)

        # Write Gold Parquet dataset.
        gold.write.mode("overwrite").parquet(str(OUTPUT_PATH))

        print("\nGold weather transformation complete.")
        print(f"Output: {OUTPUT_PATH}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()