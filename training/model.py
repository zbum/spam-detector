"""Char-CNN for short-text spam classification.

Architecture is intentionally small (~100KB-1MB) so it runs on CPU at <5ms
per message — same philosophy as Magika's lightweight ONNX model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        conv_channels: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.3,
        num_classes: int = 2,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, conv_channels, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(conv_channels * len(kernel_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)
        x = self.embedding(input_ids)        # (B, L, E)
        x = x.transpose(1, 2)                # (B, E, L)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))              # (B, C, L)
            h, _ = torch.max(h, dim=2)       # (B, C)
            pooled.append(h)
        h = torch.cat(pooled, dim=1)         # (B, C*K)
        h = self.dropout(h)
        return self.fc(h)                    # logits (B, num_classes)