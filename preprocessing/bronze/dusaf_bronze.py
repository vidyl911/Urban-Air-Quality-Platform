from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "spatial_context"
    / "regione_lombardia.zip"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "bronze"
    / "spatial_context"
    / "dusaf7"
)

REQUIRED_FILES = {
    "dusaf7.shp",
    "dusaf7.shx",
    "dusaf7.dbf",
    "dusaf7.prj",
    "dusaf7.cpg",
}


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(RAW_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with ZipFile(RAW_FILE, "r") as archive:
        members = archive.namelist()

        selected = {
            Path(name).name.lower(): name
            for name in members
            if Path(name).name.lower() in REQUIRED_FILES
        }

        missing = REQUIRED_FILES - set(selected)

        if missing:
            raise ValueError(
                f"Required shapefile components missing: {sorted(missing)}"
            )

        for filename in sorted(REQUIRED_FILES):
            archive.extract(
                selected[filename],
                path=OUTPUT_DIR,
            )
            print(f"Extracted: {filename}")

    print(f"\nMain DUSAF layer extracted to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()