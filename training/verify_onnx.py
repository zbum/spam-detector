"""Verify ONNX and PyTorch produce matching outputs, then print sample inferences.

Usage:
    python verify_onnx.py --config config.yaml
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

from model import CharRNN
from tokenizer import CharTokenizer


SAMPLES = [
    "오늘 회의 몇 시였죠?",
    "★초대박★ 무료 이벤트 당첨! http://bit.ly/abcd",
    "내일 점심 같이 먹어요",
    "즉시대출 010-1234-5678 무심사 당일입금",
    "ㅅ.ㅍ.ㅏ.ㅁ 아닙니다 진짜 고수익 부업",
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(cfg["tokenizer"]["vocab_path"])

    torch_model = CharRNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        bidirectional=cfg["model"]["bidirectional"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        pad_id=CharTokenizer.PAD_ID,
    )
    torch_model.load_state_dict(torch.load(cfg["output"]["checkpoint"], map_location="cpu"))
    torch_model.eval()

    session = ort.InferenceSession(cfg["output"]["onnx"], providers=["CPUExecutionProvider"])

    ids = np.array([tokenizer.encode(s) for s in SAMPLES], dtype=np.int64)

    with torch.no_grad():
        pt_logits = torch_model(torch.from_numpy(ids)).numpy()
    onnx_logits = session.run(None, {"input_ids": ids})[0]

    max_diff = float(np.abs(pt_logits - onnx_logits).max())
    print(f"[info] max |pytorch - onnx| = {max_diff:.2e}")
    assert math.isclose(max_diff, 0, abs_tol=1e-4), "ONNX output drifted from PyTorch"

    probs = softmax(onnx_logits)
    for text, p in zip(SAMPLES, probs):
        verdict = "SPAM" if p[1] > p[0] else "HAM "
        print(f"  [{verdict}] p(spam)={p[1]:.4f}  {text}")


if __name__ == "__main__":
    main()
