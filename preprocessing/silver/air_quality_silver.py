
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_DIR = PROJECT_ROOT / "data" / "bronze" / "air_quality"
SILVER_DIR = PROJECT_ROOT / "data" / "silver" / "air_quality"

MEASUREMENTS_FILE = BRONZE_DIR / "measurements.parquet"
STATIONS_FILE = BRONZE_DIR / "stations.parquet"


def load_bronze_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the existing Bronze air-quality datasets."""
    for path in (MEASUREMENTS_FILE, STATIONS_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Bronze input: {path}. Run the Bronze script first."
            )

    measurements = pd.read_parquet(MEASUREMENTS_FILE)
    stations = pd.read_parquet(STATIONS_FILE)

    print(f"Bronze measurements: {len(measurements):,}")
    print(f"Bronze stations: {len(stations):,}")

    return measurements, stations


def clean_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize air-quality measurements."""
    df = df.copy()

    required = {"stazione_id", "data", "inquinante", "valore"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")

    df = df.rename(
        columns={
            "stazione_id": "station_id",
            "data": "date",
            "inquinante": "pollutant",
            "valore": "concentration",
        }
    )

    df["station_id"] = df["station_id"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["pollutant"] = df["pollutant"].astype("string").str.strip().str.upper()
    df["concentration"] = pd.to_numeric(
        df["concentration"], errors="coerce"
    )

    # Standardize the source's PM25 code.
    df["pollutant"] = df["pollutant"].replace({"PM25": "PM2.5"})

    before = len(df)

    valid = (
        df["station_id"].notna()
        & df["station_id"].ne("")
        & df["date"].notna()
        & df["pollutant"].notna()
        & df["pollutant"].ne("")
        & df["concentration"].notna()
        & df["concentration"].ge(0)
    )

    df = df.loc[valid].copy()

    removed_invalid = before - len(df)
    duplicate_count = int(
        df.duplicated(
            subset=["station_id", "date", "pollutant"],
            keep="first",
        ).sum()
    )

    df = df.drop_duplicates(
        subset=["station_id", "date", "pollutant"],
        keep="first",
    )

    df["station_id"] = df["station_id"].astype("string")
    df["pollutant"] = df["pollutant"].astype("string")

    print(f"Invalid measurement rows removed: {removed_invalid:,}")
    print(f"Duplicate measurements removed: {duplicate_count:,}")
    print("Pollutant counts:")
    print(df["pollutant"].value_counts().to_string())

    return df.sort_values(
        ["date", "station_id", "pollutant"]
    ).reset_index(drop=True)


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Validate station metadata and geographic coordinates."""
    df = df.copy()

    required = {
        "station_id",
        "station_name",
        "longitude",
        "latitude",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing station columns: {sorted(missing)}")

    df["station_id"] = df["station_id"].astype("string").str.strip()
    df["station_name"] = df["station_name"].astype("string").str.strip()

    for column in ("longitude", "latitude"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "operational_start" in df.columns:
        df["operational_start"] = pd.to_datetime(
            df["operational_start"], errors="coerce"
        )

    if "operational_end" in df.columns:
        df["operational_end"] = pd.to_datetime(
            df["operational_end"], errors="coerce"
        )

    before = len(df)

    valid = (
        df["station_id"].notna()
        & df["station_id"].ne("")
        & df["longitude"].between(-180, 180)
        & df["latitude"].between(-90, 90)
    )

    df = df.loc[valid].copy()
    print(f"Invalid station rows removed: {before - len(df):,}")

    duplicates = int(df["station_id"].duplicated().sum())
    if duplicates:
        print(f"Duplicate station IDs found: {duplicates:,}")
        df = df.drop_duplicates(subset=["station_id"], keep="first")

    print(f"Valid stations retained: {len(df):,}")

    return df.reset_index(drop=True)


def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    measurements, stations = load_bronze_data()

    measurements = clean_measurements(measurements)
    stations = clean_stations(stations)

    # Keep measurements even if station metadata is unavailable.
    # This makes unmatched station IDs visible for further investigation.
    joined = measurements.merge(
        stations,
        on="station_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = int(joined["_merge"].eq("left_only").sum())
    print(f"Measurements without matching station metadata: {unmatched:,}")

    joined = joined.drop(columns="_merge")

    measurements_path = SILVER_DIR / "measurements.parquet"
    stations_path = SILVER_DIR / "stations.parquet"

    joined.to_parquet(measurements_path, index=False)
    stations.to_parquet(stations_path, index=False)

    print("\nSilver air-quality processing complete.")
    print(f"Measurements with station metadata: {len(joined):,}")
    print(f"Measurements output: {measurements_path}")
    print(f"Stations output: {stations_path}")


if __name__ == "__main__":
    main()