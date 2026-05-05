from __future__ import annotations

from pathlib import Path
import sys

DATASET_REF = "nextmillionaire/car-accident-dataset"
OUTPUT_DIR = Path("data/raw")


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("The 'kaggle' package is not installed. Run: pip install kaggle")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local auth state
        print("Kaggle authentication failed.")
        print("Add kaggle.json to %USERPROFILE%/.kaggle or set KAGGLE_USERNAME and KAGGLE_KEY.")
        print(f"Details: {exc}")
        return 1

    print(f"Downloading '{DATASET_REF}' into '{OUTPUT_DIR}'...")
    api.dataset_download_files(DATASET_REF, path=str(OUTPUT_DIR), unzip=True)
    print("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
