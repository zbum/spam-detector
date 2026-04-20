"""Char-level BiLSTM for short-text spam classification.

Lightweight RNN variant — runs on CPU with low latency for short messages.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)
        mask = (input_ids != self.pad_id).unsqueeze(-1)   # (B, L, 1)
        x = self.embedding(input_ids)                     # (B, L, E)
        h, _ = self.lstm(x)                               # (B, L, H*dir)
        # Masked max pooling with a finite sentinel (ONNX-friendly)
        h = h.masked_fill(~mask, -1e4)
        pooled, _ = torch.max(h, dim=1)                   # (B, H*dir)
        pooled = self.dropout(pooled)
        return self.fc(pooled)                            # logits (B, num_classes)
