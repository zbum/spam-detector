"""Character-level tokenizer for chat/comment spam detection.

Uses raw Unicode characters as tokens. This is robust to:
- 우회 문자열 (e.g. ㅅ ㅍ ㅏ ㅁ)
- 신조어, 오타, 이모지
- 다국어 혼용
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_PHONE_RE = re.compile(r"\d{2,4}[-\s.]?\d{3,4}[-\s.]?\d{4}")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Normalize text before tokenization.

    Replaces URLs and phone numbers with placeholder tokens so the model
    learns "there is a URL here" rather than memorizing specific domains.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub(" <url> ", text)
    text = _PHONE_RE.sub(" <phone> ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


class CharTokenizer:
    PAD_ID = 0
    UNK_ID = 1
    SPECIAL_TOKENS = ["<pad>", "<unk>", "<url>", "<phone>"]

    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def fit(self, texts: Iterable[str], min_freq: int = 2) -> None:
        counter: Counter[str] = Counter()
        for raw in texts:
            text = normalize(raw)
            for token in self._iter_tokens(text):
                counter[token] += 1

        self.char_to_id = {tok: idx for idx, tok in enumerate(self.SPECIAL_TOKENS)}
        next_id = len(self.char_to_id)
        for char, freq in counter.most_common():
            if char in self.char_to_id:
                continue
            if freq < min_freq:
                break
            self.char_to_id[char] = next_id
            next_id += 1
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

    def _iter_tokens(self, normalized_text: str):
        i = 0
        while i < len(normalized_text):
            for sp in ("<url>", "<phone>"):
                if normalized_text.startswith(sp, i):
                    yield sp
                    i += len(sp)
                    break
            else:
                yield normalized_text[i]
                i += 1

    def encode(self, text: str) -> list[int]:
        text = normalize(text)
        ids = [self.char_to_id.get(tok, self.UNK_ID) for tok in self._iter_tokens(text)]
        ids = ids[: self.max_length]
        if len(ids) < self.max_length:
            ids.extend([self.PAD_ID] * (self.max_length - len(ids)))
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_length": self.max_length,
            "char_to_id": self.char_to_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(max_length=data["max_length"])
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {v: k for k, v in tok.char_to_id.items()}
        return tok