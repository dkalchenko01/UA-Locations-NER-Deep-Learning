"""Small opt-in helpers for BiGRU runs; does not change official metrics."""

import json
import shutil
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics import comp_metric
from src.postprocessing import postprocess



CLEAN_OPTIONS = {"expand": True, "trim": True, "empty": True}


def backup_files(destination, *files):
    """Copy a small list of completed artifacts to a mounted Google Drive path.

    Save locally first; Drive copies are made through a temporary filename.
    An exception is deliberately raised if the backup fails, rather than
    quietly claiming that the checkpoint is safe.
    """
    if destination is None:
        return
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    for source in files:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = folder / source.name
        temporary = folder / (source.name + ".tmp-copy")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()


def postprocessed_scores(records, predictions, clean_options=None):
    """Compute the unchanged official competition metric after span cleanup."""
    options = CLEAN_OPTIONS if clean_options is None else clean_options
    by_source = dict.fromkeys([r["source"] for r in records])
    truth, guess = {s: [] for s in by_source}, {s: [] for s in by_source}
    for record in records:
        source = record["source"]
        truth[source].append(record["gold_locations"])
        guess[source].append(postprocess(
            record["text"], predictions[record["text_id"]]["spans"], **options,
        ))
    scores = {f"{source}_comp_f1": comp_metric(truth[source], guess[source])
              for source in by_source}
    scores["combined_comp_f1"] = comp_metric(
        [item for source in by_source for item in truth[source]],
        [item for source in by_source for item in guess[source]],
    )
    return scores


@torch.no_grad()
def export_token_logits_jsonl(model, loader, device, path):
    """Store test-only emissions plus exact original-text character offsets.

    ByteLevel and Whitespace BPE produce different token sequences, so these
    logits MUST be aligned by offsets before any future LSTM/GRU ensemble.
    Do not serialize training text, gold labels or private Kaggle labels.
    """
    model.eval()
    path = Path(path)
    with path.open("w", encoding="utf-8") as stream:
        for batch in tqdm(loader, desc="Export logits"):
            logits = model(batch["input_ids"].to(device), batch["lengths"])
            lengths = batch["lengths"].tolist()
            for record, row, length in zip(batch["records"], logits.cpu(), lengths):
                n = len(record["input_ids"])
                if not n:
                    scores, offsets = [], []
                else:
                    scores = row[:n].tolist()
                    offsets = record["offsets"].tolist()
                stream.write(json.dumps({
                    "text_id": record["text_id"],
                    "offsets": offsets,
                    "logits": scores,
                }, ensure_ascii=False) + "\n")
    return path
