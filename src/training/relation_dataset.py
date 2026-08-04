"""Relation dataset supporting zero-copy source manifests and tokenization."""

from __future__ import annotations
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src.models.relation_aware import labels_for_objective, render_role_text, target_for_objective


def _sources(path: Path) -> list[Path]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "relation_source_manifest": return [Path(source) for source in payload["sources"]]
    return [path]


class RelationJsonlDataset(Dataset):
    def __init__(self, path, root_path=None, relation_objective="typed"):
        self.sources = _sources(Path(path)); self.rows = [json.loads(line) for source in self.sources for line in source.read_text(encoding="utf-8").splitlines() if line]
        self.relation_objective = relation_objective
        self.relation_labels = labels_for_objective(relation_objective)
        self.relation_to_id = {label: index for index, label in enumerate(self.relation_labels)}
        self.root_labels = {}
        if root_path:
            for line in Path(root_path).read_text(encoding="utf-8").splitlines():
                row = json.loads(line); self.root_labels[row["child_synset"]] = float(row["is_root"])
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        target = target_for_objective(row["relation"], self.relation_objective)
        return {**row, "target_relation": target, "label_id": self.relation_to_id[target], "root_label": self.root_labels.get(row["child_synset"], -1.)}


class RelationCollator:
    def __init__(self, tokenizer, max_length=192): self.tokenizer = tokenizer; self.max_length = max_length
    def __call__(self, rows):
        children = [render_role_text(row["child"], row.get("child_gloss", ""), "child") for row in rows]; parents = [render_role_text(row["parent"], row.get("parent_gloss", ""), "parent") for row in rows]
        pair_inputs = self.tokenizer(children, parents, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        unique = {}
        for row in rows:
            if row["root_label"] >= 0: unique.setdefault(row["child_synset"], row)
        root_rows = list(unique.values()); root_children = [render_role_text(row["child"], row.get("child_gloss", ""), "child") for row in root_rows]
        child_inputs = self.tokenizer(root_children, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt") if root_children else None
        return {"pair_inputs": pair_inputs, "child_inputs": child_inputs, "labels": torch.tensor([row["label_id"] for row in rows], dtype=torch.long),
                "root_labels": torch.tensor([row["root_label"] for row in root_rows], dtype=torch.float32)}
