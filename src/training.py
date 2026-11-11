"""Plain PyTorch training and full-text BIO inference."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from src.dataset import ID_TO_LABEL, IGNORE_INDEX
from src.preprocessing import bio_to_spans


def set_seed(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def train_epoch(model, loader, optimizer, device, gradient_clip=1.0):
    model.train()
    total_loss, total_tokens = 0.0, 0
    for batch in tqdm(loader, desc="Train"):
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        n_tokens = int(labels.ne(IGNORE_INDEX).sum().item())
        if not n_tokens:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, batch["lengths"])
        loss = F.cross_entropy(logits.transpose(1, 2), labels,
                               ignore_index=IGNORE_INDEX, reduction="sum")
        (loss / n_tokens).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        total_loss += loss.item()
        total_tokens += n_tokens
    return total_loss / total_tokens


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions = {}
    loss_sums, token_counts = {}, {}
    for batch in tqdm(loader, desc="Predict"):
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(ids, batch["lengths"])
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        losses = F.cross_entropy(logits.transpose(1, 2), labels,
                                 ignore_index=IGNORE_INDEX, reduction="none").sum(dim=1).cpu().tolist()
        counts = labels.ne(IGNORE_INDEX).sum(dim=1).cpu().tolist()
        for record, predicted, loss, count in zip(batch["records"], pred_ids, losses, counts):
            source = record["source"]
            loss_sums[source] = loss_sums.get(source, 0.0) + loss
            token_counts[source] = token_counts.get(source, 0) + count
            tags = [ID_TO_LABEL[value] for value in predicted[:len(record["input_ids"])]]
            spans = bio_to_spans(tags, record["offsets"].tolist())
            spans = [(a, b) for a, b in spans if a < b]
            predictions[record["text_id"]] = {
                "spans": spans, "locations": [record["text"][a:b] for a, b in spans]
            }
    losses = {source: loss_sums[source] / count if count else None
              for source, count in token_counts.items()}
    n_tokens = sum(token_counts.values())
    losses["combined"] = sum(loss_sums.values()) / n_tokens if n_tokens else None
    return predictions, losses


def save_checkpoint(path, model, optimizer, epoch, config, model_config, scores, provenance):
    checkpoint = {
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "config": config, "model_config": model_config,
        "scores": scores, "provenance": provenance,
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
