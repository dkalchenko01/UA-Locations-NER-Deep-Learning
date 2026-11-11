"""Packed bidirectional GRU and LSTM token classifiers."""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiGRUTagger(nn.Module):
    def __init__(self, vocab_size, pad_id=0, embedding_dim=128,
                 hidden_size=128, dropout=0.2, num_labels=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers=1,
                          batch_first=True, bidirectional=True)
        # GRU internal dropout does not apply to a single recurrent layer.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)  # [batch, time, embedding_dim]
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True,
                                         total_length=input_ids.size(1))
        # pad_packed_sequence restores the original unsorted batch order.
        return self.classifier(self.dropout(encoded))  # [batch, time, 3]


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, pad_id=0, embedding_dim=128,
                 hidden_size=256, num_layers=2, dropout=0.2, num_labels=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True,
                                         total_length=input_ids.size(1))
        return self.classifier(self.dropout(encoded))
