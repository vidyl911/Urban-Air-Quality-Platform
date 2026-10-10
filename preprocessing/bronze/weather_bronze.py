
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "weather"
    / "lombardy_weather_2024.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "weather"
)


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Weather file not found: {RAW_FILE}")

    with RAW_FILE.open("r", encoding="utf-8") as file:
        weather_points = json.load(file)

    frames = []

    for point in weather_points:
        latitude = point["latitude"]
        longitude = point["longitude"]
        weather = point["weather"]

        daily = weather.get("daily")
        if not daily or "time" not in daily:
            print(f"Skipping point without daily data: {latitude}, {longitude}")
            continue

        frame = pd.DataFrame(daily)
        frame["latitude"] = latitude
        frame["longitude"] = longitude
        frame["timezone"] = weather.get("timezone")
        frames.append(frame)

    if not frames:
        raise ValueError("No valid daily weather records found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in df.columns:
        if column not in {
            "date",
            "latitude",
            "longitude",
            "timezone",
        }:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df[
        [
            "date",
            "latitude",
            "longitude",
            "timezone",
            *[
                column
                for column in df.columns
                if column not in {
                    "date",
                    "latitude",
                    "longitude",
                    "timezone",
                }
            ],
        ]
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "weather.parquet"
    df.to_parquet(output_file, index=False)

    print(f"Weather rows: {len(df):,}")
    print(f"Weather points: {df[['latitude', 'longitude']].drop_duplicates().shape[0]}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Missing values by column:\n{df.isna().sum()}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()