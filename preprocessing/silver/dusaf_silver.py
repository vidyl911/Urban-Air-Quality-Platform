from pathlib import Path
import geopandas as gpd
import pandas as pd


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_FILE = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "spatial_context"
    / "dusaf7"
    / "DUSAF7.shp"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "spatial_context"
)

OUTPUT_CRS = "EPSG:4326"


# --------------------------------------------------
# Load Bronze data
# --------------------------------------------------

def load_dusaf() -> gpd.GeoDataFrame:
    """Load the Bronze DUSAF land-use polygon layer."""

    if not BRONZE_FILE.exists():
        raise FileNotFoundError(
            f"DUSAF shapefile not found: {BRONZE_FILE}"
        )

    gdf = gpd.read_file(BRONZE_FILE)

    if gdf.crs is None:
        raise ValueError(
            "DUSAF source CRS is missing; refusing to assume one."
        )

    required = {"COD_TOT", "DESCR", "geometry"}
    missing = required - set(gdf.columns)

    if missing:
        raise ValueError(
            f"Missing required DUSAF fields: {sorted(missing)}"
        )

    print("\nBronze DUSAF inspection")
    print(f"Features loaded: {len(gdf):,}")
    print(f"Source CRS: {gdf.crs}")
    print(f"Columns: {gdf.columns.tolist()}")

    return gdf


# --------------------------------------------------
# Clean and standardize
# --------------------------------------------------

def clean_dusaf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate attributes and geometries and standardize the CRS."""

    gdf = gdf.copy()

    # Normalize classification and description fields.
    for column in ["COD_TOT", "DESCR"]:
        gdf[column] = gdf[column].astype("string").str.strip()

    # Normalize hierarchical land-use classification codes.
    for column in ["LIV_1", "LIV_2", "LIV_3", "LIV_4", "LIV_5"]:
        if column in gdf.columns:
            gdf[column] = gdf[column].astype("string").str.strip()

            # Treat blank strings as missing.
            gdf[column] = gdf[column].replace("", pd.NA)

    # Normalize numeric source attributes.
    for column in [
        "SHAPE_LENG",
        "SHAPE_AREA",
        "SHAPE_LEN",
    ]:
        if column in gdf.columns:
            gdf[column] = pd.to_numeric(
                gdf[column],
                errors="coerce",
            )

    # Remove records without geometry or essential classification.
    before = len(gdf)

    valid = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf["COD_TOT"].notna()
        & gdf["COD_TOT"].ne("")
        & gdf["DESCR"].notna()
        & gdf["DESCR"].ne("")
    )

    gdf = gdf.loc[valid].copy()

    print(f"Rows removed for missing geometry/classification: {before - len(gdf):,}")

    # Repair invalid geometries where possible.
    invalid_before = int((~gdf.geometry.is_valid).sum())
    print(f"Invalid geometries before repair: {invalid_before:,}")

    if invalid_before:
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = (
            gdf.loc[~gdf.geometry.is_valid, "geometry"].make_valid()
        )

    # Remove geometries that remain missing, empty, or invalid.
    geometry_valid = (
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.is_valid
    )

    removed_geometry = int((~geometry_valid).sum())
    gdf = gdf.loc[geometry_valid].copy()

    print(f"Rows removed after geometry validation: {removed_geometry:,}")

    # Retain only polygonal features.
    polygonal = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    removed_non_polygon = int((~polygonal).sum())
    gdf = gdf.loc[polygonal].copy()

    print(f"Non-polygon features removed: {removed_non_polygon:,}")

    # Drop exact duplicate records based on attributes and geometry.
    # Distinct polygons with the same classification are preserved.
    duplicate_columns = [
        column for column in gdf.columns
        if column != "geometry"
    ]

    duplicate_mask = gdf.drop(columns="geometry").duplicated(
        subset=duplicate_columns,
        keep="first",
    ) & gdf.geometry.to_wkb().duplicated(keep="first")

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        gdf = gdf.loc[~duplicate_mask].copy()

    print(f"Exact duplicate records removed: {duplicate_count:,}")

    # Reproject to geographic coordinates for integration with stations.
    gdf = gdf.to_crs(OUTPUT_CRS)

    print(f"Output CRS: {gdf.crs}")
    print(f"Valid features retained: {len(gdf):,}")
    print(f"Land-use classes: {gdf['COD_TOT'].nunique():,}")

    return gdf.reset_index(drop=True)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dusaf = load_dusaf()
    dusaf = clean_dusaf(dusaf)

    output_file = OUTPUT_DIR / "dusaf7.parquet"

    dusaf.to_parquet(
        output_file,
        index=False,
        compression="snappy",
    )

    print("\nSilver DUSAF processing complete.")
    print(f"Output: {output_file}")
    print(f"Output size: {output_file.stat().st_size / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()