"""Export trained PyTorch checkpoint to ONNX for Go inference.

Usage:
    python export_onnx.py --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from model import CharCNN
from tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = CharTokenizer.load(cfg["tokenizer"]["vocab_path"])

    model = CharCNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        conv_channels=cfg["model"]["conv_channels"],
        kernel_sizes=tuple(cfg["model"]["kernel_sizes"]),
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        pad_id=CharTokenizer.PAD_ID,
    )
    state = torch.load(cfg["output"]["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    onnx_path = Path(cfg["output"]["onnx"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, tokenizer.max_length, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path.as_posix(),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=20,
        dynamo=False,
    )
    print(f"[info] exported ONNX -> {onnx_path}")
    print(f"[info] vocab_size  = {tokenizer.vocab_size}")
    print(f"[info] max_length  = {tokenizer.max_length}")


if __name__ == "__main__":
    main()
