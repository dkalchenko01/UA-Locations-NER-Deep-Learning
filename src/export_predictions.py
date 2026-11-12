"""Export raw fixed-VAL and TEST spans from one checkpoint, without cleanup.

predict_fn(model, loader) must return (predictions_by_id, losses), as in the
repository notebooks. Records must contain original text and Python character
IDs/offsets. Use full, unfiltered VAL records and the model's own tokenizer.
"""

import json
from numbers import Integral
from pathlib import Path


def _prediction_rows(records, predictions):
    ids = [record["text_id"] for record in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Records must be nonempty and have unique IDs")
    if set(ids) != set(predictions):
        raise ValueError("Missing or extra prediction IDs")

    rows = []
    for record in records:
        text = record["text"]
        if not isinstance(text, str):
            raise ValueError("Records must contain original text strings")
        spans = []
        for start, end in predictions[record["text_id"]]["spans"]:
            if (isinstance(start, bool) or isinstance(end, bool)
                    or not isinstance(start, Integral) or not isinstance(end, Integral)
                    or not 0 <= start < end <= len(text)):
                raise ValueError(f"Invalid span for {record['text_id']!r}")
            spans.append([int(start), int(end)])

        text_id = record["text_id"]
        if isinstance(text_id, Integral) and not isinstance(text_id, bool):
            text_id = int(text_id)
        elif not isinstance(text_id, str):
            raise ValueError("IDs must be strings or integers")
        rows.append({"text_id": text_id, "text": text, "spans": spans})
    return rows


def export_predictions(output_dir, checkpoint_path, model, predict_fn,
                       val_records, val_loader, test_records, test_loader):
    """Load a trusted local checkpoint and export both splits to a NEW folder.

    predict_fn must use the supplied model and must not reload weights, train,
    clean predictions, or change text/offsets. Existing output folders are refused.
    Matching architecture and tokenizer must already be prepared by the notebook.
    """
    import torch

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new run folder: {output_dir}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Checkpoint:", checkpoint_path, "epoch:", checkpoint.get("epoch"))
    del checkpoint

    with torch.no_grad():
        val_predictions, _ = predict_fn(model, val_loader)
        val_rows = _prediction_rows(val_records, val_predictions)
        del val_predictions
        test_predictions, _ = predict_fn(model, test_loader)
        test_rows = _prediction_rows(test_records, test_predictions)
        del test_predictions

    # Validate both splits before creating any output files.
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, rows in [("val.jsonl", val_rows), ("test.jsonl", test_rows)]:
        with (output_dir / filename).open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(filename, len(rows), "texts")
    return output_dir
