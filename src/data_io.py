import json
import hashlib
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

def load_manifest(splits_dir, name="validation_manifest_v2"):
    path = Path(splits_dir) / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_parquet(path)


def select_manifest_rows(df, manifest, source, split):
    selected = manifest.loc[(manifest["source"] == source) & (manifest["split"] == split), "row_id"]
    return df.loc[selected].copy()


def load_data_splits(environment, splits_dir, sources=("uk", "ru"),
                     chunksize=100_000, manifest_name="validation_manifest_v2"):
    data_dir = get_data_dir(environment)
    manifest = load_manifest(splits_dir, name=manifest_name)
    result = {}

    for source in sources:
        source_manifest = manifest[manifest["source"] == source]
        required_ids = pd.Index(source_manifest["row_id"])
        parts = []
        offset = 0

        for chunk in pd.read_csv(data_dir / f"{source}_geo_dataset.csv", chunksize=chunksize):
            chunk.index = pd.RangeIndex(offset, offset + len(chunk))
            selected = chunk.loc[chunk.index.isin(required_ids)]
            if not selected.empty:
                parts.append(selected)
            offset += len(chunk)

        df = pd.concat(parts)

        for split in ("train", "val", "holdout"):
            result[f"{source}_{split}"] = select_manifest_rows(
                df, source_manifest, source, split
            )

        del df, parts

    return result

def load_testlike(splits_dir):
    splits_dir = Path(splits_dir)
    original = pd.read_parquet(splits_dir / "uk_testlike_original_v2.parquet")
    synthetic = pd.read_parquet(splits_dir / "uk_testlike_v2.parquet")

    original = original.set_index("row_id", drop=False)
    return original, synthetic


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_predictions(path, records, predictions):
    rows = []
    for record in records:
        rows.append({
            "text_id": record["text_id"],
            "row_id": record["row_id"],
            "text": record["text"],
            "source": record["source"],
            "gold_locations": record.get("gold_locations"),
            "input_ids": record["input_ids"].tolist(),
            "offsets": record["offsets"].tolist(),
            **predictions[record["text_id"]]})
    pd.DataFrame(rows).to_json(path, orient="records", lines=True, force_ascii=False)
