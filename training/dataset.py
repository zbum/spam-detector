"""CSV-backed dataset for spam detection.

Expected CSV columns:
- text  : the chat/comment string
- label : 0 (ham) or 1 (spam)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from tokenizer import CharTokenizer


class SpamDataset(Dataset):
    def __init__(self, csv_path: str | Path, tokenizer: CharTokenizer,
                 text_col: str = "text", label_col: str = "label"):
        df = pd.read_csv(csv_path)
        if text_col not in df.columns or label_col not in df.columns:
            raise ValueError(
                f"CSV must have columns '{text_col}' and '{label_col}', got {list(df.columns)}"
            )
        df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
        self.texts: list[str] = df[text_col].astype(str).tolist()
        self.labels: list[int] = df[label_col].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = self.tokenizer.encode(self.texts[idx])
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )