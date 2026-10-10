import json
import time
from pathlib import Path

import requests


YEAR = 2024

OUTPUT_PATH = Path("data/raw/weather/lombardy_weather_2024.json")

WEATHER_POINTS = [
    (44.75, 8.75),
    (44.75, 9.25),
    (44.75, 9.75),
    (44.75, 10.25),
    (44.75, 10.75),
    (44.75, 11.25),
    (45.25, 8.75),
    (45.25, 9.25),
    (45.25, 9.75),
    (45.25, 10.25),
    (45.25, 10.75),
    (45.25, 11.25),
    (45.75, 8.75),
    (45.75, 9.25),
    (45.75, 9.75),
    (45.75, 10.25),
    (45.75, 10.75),
    (45.75, 11.25),
    (46.25, 8.75),
    (46.25, 9.25),
    (46.25, 9.75),
    (46.25, 10.25),
    (46.25, 10.75),
    (46.25, 11.25),
    (46.75, 8.75),
    (46.75, 9.25),
    (46.75, 9.75),
    (46.75, 10.25),
    (46.75, 10.75),
    (46.75, 11.25),
]


def fetch_weather(latitude: float, longitude: float) -> dict:
    """Fetch daily weather data for one geographic point."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{YEAR}-01-01",
        "end_date": f"{YEAR}-12-31",
        "daily": ",".join(
            [
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
        ),
        "timezone": "Europe/Rome",
    }

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                wait_time = 10 * (attempt + 1)
                print(
                    f"Rate limited. "
                    f"Waiting {wait_time}s before retry..."
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            # Avoid sending requests too quickly.
            time.sleep(2)

            return response.json()

        except requests.RequestException as error:
            if attempt == 2:
                raise RuntimeError(
                    f"Failed to fetch weather for "
                    f"{latitude}, {longitude}"
                ) from error

            wait_time = 5 * (attempt + 1)
            print(
                f"Request failed: {error}. "
                f"Retrying in {wait_time}s..."
            )
            time.sleep(wait_time)

    raise RuntimeError(
        f"Unable to fetch weather for "
        f"{latitude}, {longitude}"
    )


def main() -> None:
    """Fetch weather data for all Lombardy grid points."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    total_points = len(WEATHER_POINTS)

    for index, (latitude, longitude) in enumerate(
        WEATHER_POINTS,
        start=1,
    ):
        print(
            f"[{index}/{total_points}] "
            f"Fetching weather: "
            f"{latitude}, {longitude}"
        )

        data = fetch_weather(
            latitude,
            longitude,
        )

        results.append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "weather": data,
            }
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSuccessfully fetched "
        f"{len(results)} weather points."
    )

    print(
        f"Saved weather data to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()