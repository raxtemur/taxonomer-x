"""Symmetric semantic parent-pair scoring over frozen edge representations."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import torch
from torch import nn

from src.eval.score_calibration import LogisticCalibration
from src.models.multiparent_evidence import _pair_features, _ranked_rows


class SymmetricPairMLP(nn.Module):
    def __init__(self, edge_dim: int = 768, hidden_dim: int = 96, scalar_dim: int = 8):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.GELU())
        self.out = nn.Sequential(
            nn.Linear(3 * hidden_dim + scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, left, right, scalar):
        a, b = self.edge(left.float()), self.edge(right.float())
        features = torch.cat((a + b, torch.abs(a - b), a * b, scalar.float()), dim=-1)
        return self.out(features).squeeze(-1)


def edge_repr_maps(records, score_path):
    payload = torch.load(score_path, map_location="cpu", weights_only=False)
    by_graph = {row["subgraph_id"]: row for row in payload["graphs"]}
    output = {}
    for record in records:
        graph = by_graph[record.key.split(":", 1)[1]]
        # Score caches are persisted in FP16. Keep the shared CPU view in that
        # dtype and let the MLP cast only the active GPU batch to FP32. The old
        # eager conversion doubled resident memory before converting back.
        flat = graph["edge_repr"]
        cursor = 0
        mapping = {}
        for child, rows in enumerate(record.retrieved):
            for rank in range(len(rows)):
                mapping[(child, rank)] = flat[cursor]
                cursor += 1
        if cursor != len(flat):
            raise ValueError("edge representation/cache mismatch")
        output[record.key] = mapping
    return output


def external_edge_repr_map(record, payload):
    flat = payload["edge_repr"]
    mapping, cursor = {}, 0
    for child, rows in enumerate(record.retrieved):
        for rank in range(len(rows)):
            mapping[(child, rank)] = flat[cursor]
            cursor += 1
    if cursor != len(flat):
        raise ValueError("external edge representation/cache mismatch")
    return mapping


def _pair_examples(records, score_maps, repr_maps, edge_calibration, q, negatives):
    examples = []
    for record, score_map in zip(records, score_maps):
        gold_by_child = {}
        for child, parent in record.gold:
            gold_by_child.setdefault(child, set()).add(parent)
        representations = repr_maps[record.key]
        for child in range(len(record.nodes)):
            rows = _ranked_rows(record, score_map, child, q)
            if len(rows) < 2:
                continue
            logits = edge_calibration.log_odds([[row["raw_score"]] for row in rows]).tolist()
            ranked = sorted(zip(rows, logits), key=lambda item: -item[1])[:q]
            positive, negative = [], []
            for (left, ll), (right, rl) in combinations(ranked, 2):
                label = int({left["parent"], right["parent"]}.issubset(gold_by_child.get(record.nodes[child], set())))
                item = (
                    representations[(child, left["rank"])],
                    representations[(child, right["rank"])],
                    torch.tensor(_pair_features(left, right, ll, rl), dtype=torch.float32),
                    label,
                    ll + rl,
                )
                (positive if label else negative).append(item)
            negative.sort(key=lambda item: -item[-1])
            selected = positive + negative[:negatives]
            examples.extend(item[:4] for item in selected)
    return examples


def fit_semantic_pair_model(records, score_maps, repr_maps, edge_calibration, *, q=20, negatives=12, hidden_dim=96, epochs=20, lr=1e-3, weight_decay=1e-4, seed=0, device="auto"):
    torch.manual_seed(seed)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
    examples = _pair_examples(records, score_maps, repr_maps, edge_calibration, q, negatives)
    if not examples or not any(item[3] for item in examples):
        raise ValueError("semantic pair training requires positive and negative examples")
    model = SymmetricPairMLP(examples[0][0].numel(), hidden_dim).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    labels = torch.tensor([item[3] for item in examples], dtype=torch.float32)
    pos_weight = torch.tensor([(len(labels) - labels.sum()) / labels.sum().clamp_min(1)], device=dev)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator)
        model.train()
        for start in range(0, len(order), 256):
            batch = [examples[int(index)] for index in order[start:start + 256]]
            left = torch.stack([item[0] for item in batch]).to(dev)
            right = torch.stack([item[1] for item in batch]).to(dev)
            scalar = torch.stack([item[2] for item in batch]).to(dev)
            target = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=dev)
            loss = nn.functional.binary_cross_entropy_with_logits(model(left, right, scalar), target, pos_weight=pos_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    return model, {"examples": len(examples), "positive": int(labels.sum()), "epochs": epochs, "device": str(dev), "symmetric_by_construction": True}


@dataclass
class PairFrontierEvidence:
    base: object
    repr_maps: dict
    kind: str
    q: int
    m: int
    semantic_model: SymmetricPairMLP | None = None
    margin_calibration: LogisticCalibration | None = None
    _transform_cache: dict = field(default_factory=dict, repr=False)
    _pair_cache: dict = field(default_factory=dict, repr=False)

    @property
    def edge(self):
        return self.base.edge

    def _pair_rows(self, record, score_map):
        cache_key = (record.key, self.kind, self.q)
        if cache_key in self._pair_cache:
            full, all_scores = self._pair_cache[cache_key]
            return {child: pairs[:self.m] for child, pairs in full.items()}, all_scores

        representations = self.repr_maps[record.key]
        output = {}
        all_scores = {}
        for child in range(len(record.nodes)):
            rows = _ranked_rows(record, score_map, child, self.base.max_k)
            logits = self.base.edge.log_odds([[row["raw_score"]] for row in rows]).tolist() if rows else []
            ranked = sorted(zip(rows, logits), key=lambda item: -item[1])[:self.q]
            raw_pairs = []
            for (left, ll), (right, rl) in combinations(ranked, 2):
                scalar = _pair_features(left, right, ll, rl)
                raw_pairs.append((left, ll, right, rl, scalar))
            if self.kind == "scalar" and raw_pairs:
                values = self.base.pair.log_odds([item[4] for item in raw_pairs]).tolist()
            elif raw_pairs:
                left_repr = torch.stack([representations[(child, item[0]["rank"])] for item in raw_pairs])
                right_repr = torch.stack([representations[(child, item[2]["rank"])] for item in raw_pairs])
                scalar = torch.tensor([item[4] for item in raw_pairs], dtype=torch.float32)
                device = next(self.semantic_model.parameters()).device
                with torch.inference_mode():
                    values = self.semantic_model(left_repr.to(device), right_repr.to(device), scalar.to(device)).cpu().tolist()
            else:
                values = []
            pairs = [
                (float(value), left, ll, right, rl)
                for value, (left, ll, right, rl, _) in zip(values, raw_pairs)
            ]
            pairs.sort(key=lambda item: (-item[0], item[1]["parent"], item[3]["parent"]))
            all_scores[record.nodes[child]] = [item[0] for item in pairs]
            output[child] = pairs
        self._pair_cache[cache_key] = (output, all_scores)
        return {child: pairs[:self.m] for child, pairs in output.items()}, all_scores

    def transform(self, record, score_map):
        if record.key in self._transform_cache:
            return self._transform_cache[record.key]
        base = self.base.transform(record, score_map)
        selected, all_scores = self._pair_rows(record, score_map)
        pair_scores, allowed = {}, set()
        for child, pairs in selected.items():
            for value, left, ll, right, rl in pairs:
                key = (record.nodes[child], *sorted((left["parent"], right["parent"])))
                pair_scores[key] = value - ll - rl
                allowed.update(((record.nodes[child], left["parent"]), (record.nodes[child], right["parent"])))
        cardinality = base["cardinality_scores"]
        if self.margin_calibration is not None:
            cardinality = {}
            for child, node in enumerate(record.nodes):
                rows = _ranked_rows(record, score_map, child, self.base.max_k)
                logits = self.base.edge.log_odds([[row["raw_score"]] for row in rows]).tolist() if rows else []
                best_single = max(logits, default=0.0)
                best_pair = max(all_scores[node], default=best_single)
                logit = float(self.margin_calibration.log_odds([[best_pair - best_single]])[0])
                cardinality[node] = {0: 0.0, 1: 0.0, 2: logit}
        transformed = {**base, "pair_scores": pair_scores, "multi_parent_candidate_edges": allowed, "cardinality_scores": cardinality}
        self._transform_cache[record.key] = transformed
        return transformed

    def ranking_audit(self, record, score_map):
        selected, _ = self._pair_rows(record, score_map)
        gold = {}
        for child, parent in record.gold: gold.setdefault(child, set()).add(parent)
        eligible = hit = 0; reciprocal = 0.0
        for child, pairs in selected.items():
            node = record.nodes[child]
            if len(gold.get(node, set())) != 2:
                continue
            available = {record.nodes[p] for p, _ in record.retrieved[child][:self.base.max_k]}
            if not gold[node] <= available:
                continue
            eligible += 1
            for rank, (_, left, _, right, _) in enumerate(pairs, 1):
                if {left["parent"], right["parent"]} == gold[node]:
                    hit += 1; reciprocal += 1.0 / rank; break
        return {"eligible_exact_pairs": eligible, "gold_pair_in_top_m": hit, "gold_pair_recall_at_m": hit / eligible if eligible else 0.0, "gold_pair_mrr": reciprocal / eligible if eligible else 0.0}
