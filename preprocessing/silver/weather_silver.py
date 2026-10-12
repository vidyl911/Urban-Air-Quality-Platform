from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_FILE = (
    PROJECT_ROOT / "data" / "bronze" / "weather" / "weather.parquet"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "silver" / "weather"


# --------------------------------------------------
# Load Bronze data
# --------------------------------------------------

def load_weather() -> pd.DataFrame:
    """Load Bronze weather observations."""

    if not BRONZE_FILE.exists():
        raise FileNotFoundError(
            f"Bronze weather file not found: {BRONZE_FILE}"
        )

    df = pd.read_parquet(BRONZE_FILE)

    required = {"date", "latitude", "longitude", "timezone"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required weather columns: {sorted(missing)}"
        )

    print("\nBronze weather inspection")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")

    return df


# --------------------------------------------------
# Clean and validate
# --------------------------------------------------

def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Validate weather observations and remove duplicates."""

    df = df.copy()

    # Standardize types.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    for column in ["latitude", "longitude"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timezone"] = df["timezone"].astype("string").str.strip()

    # Ensure weather variables are numeric.
    metadata_columns = {"date", "latitude", "longitude", "timezone"}

    weather_columns = [
        column for column in df.columns
        if column not in metadata_columns
    ]

    for column in weather_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove records without a valid date or geographic coordinates.
    before = len(df)

    valid = (
        df["date"].notna()
        & df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    df = df.loc[valid].copy()

    print(f"Invalid weather rows removed: {before - len(df):,}")

    # One observation per grid point and date.
    duplicate_subset = ["date", "latitude", "longitude"]

    duplicate_count = int(
        df.duplicated(subset=duplicate_subset, keep="first").sum()
    )

    if duplicate_count:
        print(f"Duplicate weather rows removed: {duplicate_count:,}")
        df = df.drop_duplicates(
            subset=duplicate_subset,
            keep="first",
        )

    # Report missing values without imputing them.
    print("\nMissing weather values:")
    print(df[weather_columns].isna().sum().to_string())

    # Report coverage.
    print(f"\nValid weather rows: {len(df):,}")
    print(
        "Unique grid points: "
        f"{df[['latitude', 'longitude']].drop_duplicates().shape[0]:,}"
    )
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df.sort_values(
        ["date", "latitude", "longitude"]
    ).reset_index(drop=True)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weather = load_weather()
    weather = clean_weather(weather)

    output_file = OUTPUT_DIR / "weather.parquet"

    weather.to_parquet(output_file, index=False)

    print("\nSilver weather processing complete.")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()