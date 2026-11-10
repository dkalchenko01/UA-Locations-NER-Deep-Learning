"""Compact prepared examples and deterministic, length-aware padded batches."""

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler

LABEL_TO_ID = {"O": 0, "B-LOC": 1, "I-LOC": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
IGNORE_INDEX = -100


class NERDataset(Dataset):
    """Records hold IDs, optional gold labels, and optional evaluation metadata."""

    def __init__(self, records):
        self.records = records
        # Packed GRUs need a positive length, including for empty inference texts.
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
