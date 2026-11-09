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

def load_manifest(splits_dir, name="validation_manifest"):
    path = Path(splits_dir) / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_parquet(path)


def select_manifest_rows(df, manifest, source, split):
    selected = manifest.loc[
        (manifest["source"] == source) & (manifest["split"] == split),
        "row_id"
    ]

    if not df.index.is_unique:
        raise ValueError("Dataset index must be unique")

    missing = pd.Index(selected).difference(df.index)
    if len(missing):
        raise ValueError(f"{len(missing)} row_ids not found in dataset")

    return df.loc[selected].copy()


def load_data_splits(environment, splits_dir, sources=("uk", "ru"), chunksize=100_000):
    data_dir = get_data_dir(environment)
    manifest = load_manifest(splits_dir)
    result = {}

    for source in sources:
        if source not in ("uk", "ru"):
            raise ValueError(f"Unsupported source: {source}")

        source_manifest = manifest.loc[manifest["source"] == source]
        required_ids = pd.Index(source_manifest["row_id"])
        parts = []
        offset = 0

        reader = pd.read_csv(
            data_dir / f"{source}_geo_dataset.csv",
            chunksize=chunksize
        )

        for chunk in reader:
            chunk.index = pd.RangeIndex(offset, offset + len(chunk))
            selected = chunk.loc[chunk.index.isin(required_ids)]

            if not selected.empty:
                parts.append(selected)

            offset += len(chunk)

        df = pd.concat(parts)

        for split in ("train", "val"):
            result[f"{source}_{split}"] = select_manifest_rows(
                df, manifest, source, split
            )

        del df, parts

    return result