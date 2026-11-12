"""BIO-constrained Viterbi decoding without learned transition weights."""

import numpy as np

from src.dataset import LABEL_TO_ID


def viterbi_decode(logits, lengths):
    emissions = logits.detach().float().cpu().numpy()
    lengths = np.asarray(lengths)
    batch_size, steps, num_labels = emissions.shape
    transitions = np.zeros((num_labels, num_labels), dtype=emissions.dtype)
    transitions[LABEL_TO_ID["O"], LABEL_TO_ID["I-LOC"]] = -np.inf

    scores = emissions[:, 0].copy()
    scores[:, LABEL_TO_ID["I-LOC"]] = -np.inf
    final_scores = scores.copy()
    backpointers = np.zeros((batch_size, steps, num_labels), dtype=np.int64)
    for step in range(1, steps):
        candidates = scores[:, :, None] + transitions[None, :, :]
        backpointers[:, step] = candidates.argmax(axis=1)
        scores = candidates.max(axis=1) + emissions[:, step]
        finished = lengths == step + 1
        final_scores[finished] = scores[finished]

    final_tags = final_scores.argmax(axis=1)
    paths = []
    for row, length in enumerate(lengths):
        path = []
        if length:
            tag = int(final_tags[row])
            path.append(tag)
            for step in range(int(length) - 1, 0, -1):
                tag = int(backpointers[row, step, tag])
                path.append(tag)
        paths.append(path[::-1])
    return paths
