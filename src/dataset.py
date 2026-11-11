"""Compact prepared examples and deterministic, length-aware padded batches."""

import random
from functools import partial

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

from src.eda_scan import parse_spans
from src.preprocessing import prepare_bio_data

LABEL_TO_ID = {"O": 0, "B-LOC": 1, "I-LOC": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
IGNORE_INDEX = -100


class NERDataset(Dataset):
    """Records hold IDs, optional gold labels, and optional evaluation metadata."""

    def __init__(self, records):
        self.records = records
        # Packed RNNs need a positive length, including for empty inference texts.
        self.lengths = [max(1, len(record["input_ids"])) for record in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class LengthBatchSampler(Sampler):
    """Shuffle pools, sort locally, and bound padded tokens without truncating.

    A sequence longer than the token budget is emitted alone. The budget controls
    batching only; it never limits sequence length. set_epoch makes order repeatable.
    """

    def __init__(self, lengths, batch_size=128, shuffle=False, seed=17,
                 max_padded_tokens=16384, pool_size=4096):
        if batch_size < 1 or max_padded_tokens < 1 or pool_size < 1:
            raise ValueError("Batch size, token budget and pool size must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.max_padded_tokens = max_padded_tokens
        self.pool_size = pool_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.pool_size):
            pool = sorted(indices[start:start + self.pool_size], key=self.lengths.__getitem__)
            batch, longest = [], 0
            for index in pool:
                candidate_longest = max(longest, self.lengths[index])
                if batch and (len(batch) >= self.batch_size or
                              candidate_longest * (len(batch) + 1) > self.max_padded_tokens):
                    batches.append(batch)
                    batch, longest = [], 0
                batch.append(index)
                longest = max(longest, self.lengths[index])
            if batch:
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def collate_ner(records, pad_id=0):
    """Pad inputs and labels; lengths stay on CPU for pack_padded_sequence.

    Missing labels mean unavailable gold BIO alignment, never synthetic O labels.
    Metadata remains on CPU, in exactly the same order as the model batch.
    """
    ids, labels = [], []
    for record in records:
        token_ids = torch.as_tensor(record["input_ids"], dtype=torch.long)
        gold = record.get("labels")
        if gold is not None and len(gold) != len(token_ids):
            raise ValueError("Token and label lengths differ")
        if not len(token_ids):
            token_ids = torch.tensor([pad_id], dtype=torch.long)
        ids.append(token_ids)
        labels.append(
            torch.as_tensor(gold, dtype=torch.long) if gold is not None and len(gold)
            else torch.full((len(token_ids),), IGNORE_INDEX, dtype=torch.long)
        )
    lengths = torch.tensor([len(item) for item in ids], dtype=torch.long)
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    targets = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
    mask = torch.arange(input_ids.shape[1])[None, :] < lengths[:, None]
    return {"input_ids": input_ids, "labels": targets, "lengths": lengths,
            "mask": mask, "records": records}


def prepare_ner_records(tokenizer, df, source, split, batch_size=2000):
    records, skipped_ids = [], []
    for start in tqdm(range(0, len(df), batch_size), desc=f"Prepare {source}/{split}"):
        batch = df.iloc[start:start + batch_size]
        prepared, skipped = prepare_bio_data(tokenizer, batch, source)
        skipped_ids.extend(skipped)
        prepared = {record["text_id"]: record for record in prepared}
        for row_id, row in batch.iterrows():
            text_id = f"{source}_{row_id}"
            item = prepared.get(text_id)
            if item is None:
                if split == "train":
                    continue
                encoding = tokenizer.encode(row["text"], add_special_tokens=False)
                ids, offsets, labels = encoding.ids, encoding.offsets, None
            else:
                ids, offsets = item["input_ids"], item["offsets"]
                labels = np.asarray([LABEL_TO_ID[label] for label in item["labels"]], dtype=np.int8)
            record = {"text_id": text_id, "row_id": int(row_id), "source": source,
                      "input_ids": np.asarray(ids, dtype=np.int32), "labels": labels}
            if split != "train":
                spans = parse_spans(row["loc_markers"])
                record.update(text=row["text"], offsets=np.asarray(offsets, dtype=np.int32).reshape(-1, 2),
                              gold_locations=[row["text"][a:b] for a, b in spans])
            records.append(record)
    return records, skipped_ids


def prepare_inference_records(tokenizer, df, source="test", id_column="text_id"):
    text_ids = df[id_column].tolist()
    records = []
    for text_id, (row_id, row) in zip(text_ids, df.iterrows()):
        encoding = tokenizer.encode(row["text"], add_special_tokens=False)
        records.append({"text_id": text_id, "row_id": int(row_id), "source": source,
                        "text": row["text"], "input_ids": np.asarray(encoding.ids, dtype=np.int32),
                        "offsets": np.asarray(encoding.offsets, dtype=np.int32).reshape(-1, 2),
                        "labels": None})
    return records


def make_loader(records, pad_id=0, batch_size=128, shuffle=False, seed=17,
                max_padded_tokens=16384, pin_memory=False):
    dataset = NERDataset(records)
    sampler = LengthBatchSampler(dataset.lengths, batch_size=batch_size, shuffle=shuffle,
                                 seed=seed, max_padded_tokens=max_padded_tokens)
    return DataLoader(dataset, batch_sampler=sampler, pin_memory=pin_memory,
                      collate_fn=partial(collate_ner, pad_id=pad_id))
