
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "air_quality"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "air_quality"

MEASUREMENTS_FILE = (
    RAW_DIR / "qaria_datoariagiornostazione_2024-12-24.csv"
)
STATIONS_FILE = RAW_DIR / "qaria_stazione.csv"


# --------------------------------------------------
# Air-quality measurements
# --------------------------------------------------

def process_measurements() -> pd.DataFrame:
    """Read and standardize daily pollutant measurements."""

    df = pd.read_csv(
        MEASUREMENTS_FILE,
        sep=";",
        encoding="utf-8-sig",
        dtype={
            "stazione_id": "string",
            "inquinante": "string",
        },
    )

    df.columns = df.columns.str.strip().str.lower()

    expected_columns = {
        "stazione_id",
        "data",
        "inquinante",
        "valore",
    }

    missing = expected_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing measurement columns: {sorted(missing)}"
        )

    df = df[
        ["stazione_id", "data", "inquinante", "valore"]
    ].copy()

    df["stazione_id"] = df["stazione_id"].str.strip()
    df["data"] = pd.to_datetime(
        df["data"],
        errors="coerce",
    ).dt.normalize()
    df["inquinante"] = (
        df["inquinante"].str.strip().str.upper()
    )
    df["valore"] = pd.to_numeric(
        df["valore"],
        errors="coerce",
    )

    print("\nMeasurement inspection")
    print(f"Rows: {len(df):,}")
    print(f"Invalid dates: {df['data'].isna().sum():,}")
    print(f"Invalid values: {df['valore'].isna().sum():,}")
    print("Pollutants:")
    print(df["inquinante"].value_counts(dropna=False))

    return df


# --------------------------------------------------
# Monitoring stations
# --------------------------------------------------

def process_stations() -> pd.DataFrame:
    """Read and standardize monitoring-station metadata."""

    station_columns = [
        "station_id",
        "station_name",
        "arpa_station_id",
        "operational_start",
        "operational_end",
        "pollutants_reported",
        "longitude",
        "latitude",
    ]

    # The source contains malformed CSV quoting.
    # Split on semicolons without interpreting quotation marks.
    rows = []

    with STATIONS_FILE.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file.readlines()[1:], start=2):
            line = line.strip()

            if not line:
                continue

            fields = line.split(";")

            if len(fields) < 8:
                print(f"Skipping malformed row {line_number}")
                continue

            # Keep the eight metadata fields; ignore Location columns.
            fields = fields[:8]

            fields = [
                value.strip().strip('"').strip()
                for value in fields
            ]

            rows.append(fields)

    df = pd.DataFrame(rows, columns=station_columns)

    for column in [
        "station_id",
        "station_name",
        "arpa_station_id",
        "pollutants_reported",
    ]:
        df[column] = df[column].astype("string").str.strip()

    for column in ["operational_start", "operational_end"]:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )

    for column in ["longitude", "latitude"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print("\nStation inspection")
    print(f"Rows: {len(df):,}")
    print(f"Invalid station IDs: {df['station_id'].isna().sum():,}")
    print(f"Invalid longitudes: {df['longitude'].isna().sum():,}")
    print(f"Invalid latitudes: {df['latitude'].isna().sum():,}")
    print("\nSample stations:")
    print(df.head().to_string(index=False))

    return df

# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    if not MEASUREMENTS_FILE.exists():
        raise FileNotFoundError(MEASUREMENTS_FILE)

    if not STATIONS_FILE.exists():
        raise FileNotFoundError(STATIONS_FILE)

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    measurements = process_measurements()
    stations = process_stations()

    measurements_path = BRONZE_DIR / "measurements.parquet"
    stations_path = BRONZE_DIR / "stations.parquet"

    measurements.to_parquet(
        measurements_path,
        index=False,
    )
    stations.to_parquet(
        stations_path,
        index=False,
    )

    print("\nBronze outputs saved")
    print(f"Measurements: {measurements_path}")
    print(f"Stations: {stations_path}")


if __name__ == "__main__":
    main()