"""Uniform node-level dataset for the learned ROOT classifier."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.models.relation_aware import render_role_text


class RootJsonlDataset(Dataset):
    def __init__(self, path: str | Path):
        self.rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


class RootCollator:
    def __init__(self, tokenizer, max_length: int = 192): self.tokenizer = tokenizer; self.max_length = max_length
    def __call__(self, rows):
        texts = [render_role_text(row["child"], row.get("child_gloss", ""), "child") for row in rows]
        return {"inputs": self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"),
                "labels": torch.tensor([row["is_root"] for row in rows], dtype=torch.float32)}
