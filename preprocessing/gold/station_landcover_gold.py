"""Calculate DUSAF land-cover composition around air-quality stations."""

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

POLYGONS_PATH = ROOT / "data/gold/spatial_context/dusaf7/polygons.parquet"
AIR_QUALITY_PATH = ROOT / "data/gold/air_quality/daily_features"
OUTPUT_DIR = ROOT / "data/gold/spatial_context/station_landcover"
OUTPUT_PATH = OUTPUT_DIR / "station_landcover.parquet"

PROJECTED_CRS = "EPSG:32632"
BUFFER_RADII_M = (500, 1000)


def main() -> None:
    print("Loading DUSAF polygons...")
    polygons = gpd.read_parquet(POLYGONS_PATH)

    if polygons.crs is None:
        raise ValueError("DUSAF polygons have no CRS metadata.")

    required_columns = {"COD_TOT", "DESCR", "geometry"}
    missing = required_columns.difference(polygons.columns)
    if missing:
        raise ValueError(f"Missing DUSAF columns: {sorted(missing)}")

    polygons = polygons[
        ["COD_TOT", "DESCR", "geometry"]
    ].copy()

    polygons = polygons[
        polygons.geometry.notna() & ~polygons.geometry.is_empty
    ].copy()

    polygons = polygons.to_crs(PROJECTED_CRS)

    print(f"Valid polygons: {len(polygons):,}")

    print("Loading air-quality station coordinates...")
    stations = pd.read_parquet(AIR_QUALITY_PATH)

    stations = stations[
        ["station_id", "station_name", "longitude", "latitude"]
    ].drop_duplicates("station_id").copy()

    if stations[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Station coordinates contain missing values.")

    station_points = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations["longitude"], stations["latitude"]
        ),
        crs="EPSG:4326",
    ).to_crs(PROJECTED_CRS)

    # Spatial index accelerates candidate polygon selection.
    spatial_index = polygons.sindex
    results = []

    for _, station in station_points.iterrows():
        for radius_m in BUFFER_RADII_M:
            buffer_geometry = station.geometry.buffer(radius_m)

            candidate_positions = list(
                spatial_index.query(buffer_geometry, predicate="intersects")
            )
            candidates = polygons.iloc[candidate_positions]

            if candidates.empty:
                raise ValueError(
                    f"No DUSAF polygons intersect station "
                    f"{station['station_id']} within {radius_m} m."
                )

            intersections = candidates.copy()
            intersections["intersection_area_m2"] = (
                intersections.geometry
                .intersection(buffer_geometry)
                .area
            )

            intersections = intersections[
                intersections["intersection_area_m2"] > 0
            ].copy()

            # A polygon can overlap the buffer only partially.
            # Aggregate its intersected area by DUSAF class.
            class_areas = (
                intersections.groupby(
                    ["COD_TOT", "DESCR"], dropna=False
                )["intersection_area_m2"]
                .sum()
                .reset_index()
            )

            # Normalize by the full circular buffer area, not by the
            # sum of mapped land-cover intersections. This preserves
            # any unmapped area as a coverage gap.
            buffer_area_m2 = buffer_geometry.area

            for _, land_class in class_areas.iterrows():
                results.append(
                    {
                        "station_id": str(station["station_id"]),
                        "station_name": station["station_name"],
                        "buffer_radius_m": radius_m,
                        "landcover_code": str(land_class["COD_TOT"]),
                        "landcover_description": land_class["DESCR"],
                        "intersection_area_m2": float(
                            land_class["intersection_area_m2"]
                        ),
                        "buffer_area_m2": float(buffer_area_m2),
                        "area_proportion": float(
                            land_class["intersection_area_m2"]
                            / buffer_area_m2
                        ),
                    }
                )

            print(
                f"Station {station['station_id']} "
                f"({station['station_name']}), {radius_m} m: "
                f"{len(class_areas)} land-cover classes"
            )

    output = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)

    print("\nLand-cover processing complete.")
    print(f"Stations: {station_points['station_id'].nunique()}")
    print(f"Buffer radii: {BUFFER_RADII_M}")
    print(f"Output rows: {len(output):,}")
    print(f"Output: {OUTPUT_PATH}")
    print("\nProportion sums by station and buffer:")
    print(
        output.groupby(
            ["station_id", "buffer_radius_m"]
        )["area_proportion"]
        .sum()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()