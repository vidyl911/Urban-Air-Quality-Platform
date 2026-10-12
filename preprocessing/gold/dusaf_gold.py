
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "spatial_context"
    / "dusaf7.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "spatial_context"
    / "dusaf7"
)

POLYGONS_OUTPUT = OUTPUT_DIR / "polygons.parquet"
SUMMARY_OUTPUT = OUTPUT_DIR / "class_summary.parquet"

AREA_CRS = "EPSG:32632"
OUTPUT_CRS = "EPSG:4326"

CLASS_COLUMNS = [
    "COD_TOT",
    "DESCR",
    "LIV_1",
    "LIV_2",
    "LIV_3",
    "LIV_4",
    "LIV_5",
]


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Silver DUSAF dataset not found: {INPUT_PATH}"
        )

    print("Reading Silver DUSAF dataset...")
    gdf = gpd.read_parquet(INPUT_PATH)

    if gdf.crs is None:
        raise ValueError("Silver DUSAF dataset has no CRS.")

    print("\nSilver DUSAF inspection")
    print(f"Features: {len(gdf):,}")
    print(f"CRS: {gdf.crs}")
    print(f"Columns: {gdf.columns.tolist()}")

    required_columns = {"geometry", "COD_TOT", "DESCR"}
    missing_columns = required_columns - set(gdf.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required DUSAF columns: {sorted(missing_columns)}"
        )

    # Keep polygon features with usable classification and geometry.
    gdf = gdf.loc[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf["COD_TOT"].notna()
        & gdf["DESCR"].notna()
    ].copy()

    gdf = gdf.loc[
        gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if gdf.empty:
        raise ValueError("No valid classified polygon features remain.")

    # Repair any remaining invalid geometries.
    invalid_before = int((~gdf.geometry.is_valid).sum())

    if invalid_before:
        invalid_mask = ~gdf.geometry.is_valid
        gdf.loc[invalid_mask, "geometry"] = (
            gdf.loc[invalid_mask, "geometry"].make_valid()
        )

    # make_valid() can return geometry collections; retain polygons only.
    gdf = gdf.loc[
        gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        & gdf.geometry.is_valid
        & ~gdf.geometry.is_empty
    ].copy()

    print(f"Invalid geometries repaired initially: {invalid_before:,}")
    print(f"Valid classified polygons retained: {len(gdf):,}")

    # Calculate areas in a projected CRS, not in longitude/latitude.
    projected = gdf.to_crs(AREA_CRS)
    gdf["area_m2"] = projected.geometry.area.to_numpy()
    gdf["area_ha"] = gdf["area_m2"] / 10_000
    gdf["area_km2"] = gdf["area_m2"] / 1_000_000

    # Avoid misleading area statistics if geometry processing failed.
    gdf = gdf.loc[
        gdf["area_m2"].notna() & (gdf["area_m2"] > 0)
    ].copy()

    total_area_m2 = float(gdf["area_m2"].sum())

    if total_area_m2 <= 0:
        raise ValueError("Total polygon area is not positive.")

    gdf["regional_area_share"] = gdf["area_m2"] / total_area_m2

    # Keep polygon geometry in geographic coordinates for interoperability.
    gdf = gdf.to_crs(OUTPUT_CRS)

    # Aggregate area by available DUSAF classification levels.
    class_columns = [
        column for column in CLASS_COLUMNS if column in gdf.columns
    ]

    summary = (
        gdf.groupby(class_columns, dropna=False, as_index=False)
        .agg(
            polygon_count=("area_m2", "size"),
            total_area_m2=("area_m2", "sum"),
            mean_polygon_area_m2=("area_m2", "mean"),
        )
    )

    summary["total_area_ha"] = summary["total_area_m2"] / 10_000
    summary["total_area_km2"] = summary["total_area_m2"] / 1_000_000
    summary["regional_area_share"] = (
        summary["total_area_m2"] / total_area_m2
    )
    summary["regional_area_share_pct"] = (
        summary["regional_area_share"] * 100
    )

    # Save both Gold datasets.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf.to_parquet(POLYGONS_OUTPUT, index=False)
    summary.to_parquet(SUMMARY_OUTPUT, index=False)

    print("\nGold DUSAF inspection")
    print(f"Polygon features: {len(gdf):,}")
    print(f"Land-cover classes: {len(summary):,}")
    print(f"Total mapped area: {total_area_m2 / 1_000_000:,.2f} km²")
    print(f"Output CRS: {gdf.crs}")

    print("\nLargest land-cover classes by area:")
    print(
        summary.sort_values("total_area_m2", ascending=False)
        .head(10)[
            [
                "COD_TOT",
                "DESCR",
                "polygon_count",
                "total_area_ha",
                "regional_area_share_pct",
            ]
        ].to_string(index=False)
    )

    print("\nGold DUSAF transformation complete.")
    print(f"Polygon output: {POLYGONS_OUTPUT}")
    print(f"Class summary output: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()