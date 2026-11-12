"""Stacked bidirectional GRU token classifier, compatible with src.training.

Only the recurrent cell changes relative to BiLSTMTagger in src.model. The
classifier emits O/B-LOC/I-LOC logits per token; nothing is truncated/pooled.
"""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class StackedBiGRUTagger(nn.Module):
    def __init__(self, vocab_size, pad_id=0, embedding_dim=128,
                 hidden_size=256, num_layers=3, dropout=0.2, num_labels=3):
        super().__init__()
        if num_layers < 1 or vocab_size < 1 or embedding_dim < 1 or hidden_size < 1:
            raise ValueError("vocab_size, embedding_dim, hidden_size and num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=input_ids.size(1),
        )
        return self.classifier(self.dropout(encoded))
