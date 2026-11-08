from pathlib import Path

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