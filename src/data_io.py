from pathlib import Path
import pandas as pd

COMPETITION = "kse-ua-location-extraction-2026"

def get_data_dir(environment: str) -> Path:
    environment = environment.strip().lower()

    if environment == "colab":
        import kagglehub
        data_dir = Path(kagglehub.competition_download(COMPETITION))
    elif environment == "kaggle":
        data_dir = Path("/kaggle/input/competitions") / COMPETITION
    else:
        raise ValueError(f"Unsupported environment: {environment}")

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    return data_dir

def load_validation_manifest(splits_dir):
    path = Path(splits_dir) / "validation_manifest.parquet"
    return pd.read_parquet(path)


def select_manifest_rows(df, manifest, source, split):
    selected = manifest.loc[
        (manifest["source"] == source) & (manifest["split"] == split),
        "row_id"
    ]

    if not df.index.is_unique:
        raise ValueError("Dataset index must be unique")

    missing = selected.difference(df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} row_ids not found in dataset")

    return df.loc[selected].copy()