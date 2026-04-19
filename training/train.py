"""End-to-end training entry point.

Usage:
    python train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SpamDataset
from model import CharCNN
from tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(device)
            logits = model(ids)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=["ham", "spam"],
                                   digits=4, zero_division=0)
    return f1, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"]["device"])
    print(f"[info] device = {device}")

    # 1) Build / load tokenizer from training corpus
    train_df = pd.read_csv(cfg["data"]["train_csv"])
    tokenizer = CharTokenizer(max_length=cfg["tokenizer"]["max_length"])
    tokenizer.fit(train_df[cfg["data"]["text_column"]].astype(str).tolist(),
                  min_freq=cfg["tokenizer"]["min_freq"])
    tokenizer.save(cfg["tokenizer"]["vocab_path"])
    print(f"[info] vocab_size = {tokenizer.vocab_size}")

    # 2) Datasets / loaders
    train_ds = SpamDataset(cfg["data"]["train_csv"], tokenizer,
                           cfg["data"]["text_column"], cfg["data"]["label_column"])
    val_ds = SpamDataset(cfg["data"]["val_csv"], tokenizer,
                         cfg["data"]["text_column"], cfg["data"]["label_column"])
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"])

    # 3) Model / optimizer
    model = CharCNN(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        conv_channels=cfg["model"]["conv_channels"],
        kernel_sizes=tuple(cfg["model"]["kernel_sizes"]),
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
        pad_id=CharTokenizer.PAD_ID,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] params = {n_params:,}")

    # Class weights handle imbalance — typical chat data is heavily ham-biased
    label_counts = train_df[cfg["data"]["label_column"]].value_counts().to_dict()
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (2 * label_counts.get(c, 1)) for c in (0, 1)],
        dtype=torch.float, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_f1 = -1.0
    patience = cfg["train"]["early_stop_patience"]
    bad_epochs = 0
    ckpt_path = Path(cfg["output"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # 4) Train loop
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        for ids, labels in tqdm(train_loader, desc=f"epoch {epoch}"):
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * ids.size(0)
        train_loss = running / len(train_ds)

        f1, report = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f}  val_f1(spam)={f1:.4f}")
        print(report)

        if f1 > best_f1:
            best_f1 = f1
            bad_epochs = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"[info] saved best checkpoint -> {ckpt_path} (f1={f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[info] early stop at epoch {epoch}")
                break

    # 5) Final test eval (optional)
    metrics: dict = {"best_val_f1": best_f1, "vocab_size": tokenizer.vocab_size}
    test_csv = Path(cfg["data"]["test_csv"])
    if test_csv.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_ds = SpamDataset(test_csv, tokenizer,
                              cfg["data"]["text_column"], cfg["data"]["label_column"])
        test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"])
        f1, report = evaluate(model, test_loader, device)
        print(f"[test] f1(spam)={f1:.4f}\n{report}")
        metrics["test_f1"] = f1
        metrics["test_report"] = report

    metrics_path = Path(cfg["output"]["metrics"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[info] metrics written -> {metrics_path}")


if __name__ == "__main__":
    main()